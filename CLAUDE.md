# CLAUDE.md — Fast Markky Fund

> **Read this first.** Quick context map so you (Claude or any new contributor) can answer questions and make design decisions without re-analyzing the whole codebase. Keep this file pruned and accurate — if it stops being trustworthy, it stops being useful.
>
> Last updated: 2026-04-26 (during test suite work).

---

## What this project is

A **regime-based tactical allocation backtester** for QQQ-family ETFs. The portfolio rotates between three risk allocations based on QQQ's drawdown from a rolling all-time high. Production target: $10k starting capital, 1999-01-04 → 2026-03-27, instant rebalancing.

```
QQQ drawdown from rolling 1y peak ──→ Market regime ──→ (asymmetric rules) ──→ Portfolio regime ──→ Allocation
                                       (R1/R2/R3)         per_regime hold/match    target weights      shares
```

**Three regimes** (current defaults — subject to change):
- **R1 — Ride High** (0–6% dd): 100% TQQQ
- **R2 — Cautious Defense** (6–20% dd): 70% XLU + 30% QQQ
- **R3 — Contrarian Buyback** (20%+ dd): 50% TQQQ + 50% QQQ

These are sensible defaults, not tuned production weights. Treat them as a starting point.

**Signal overrides** sit on top: VIX z-score (L1) + MACD (L2) + MA50/200 (L3) produce a composite signal total in `[−6, +6]`. Each regime declares an upside override (when signal > threshold) and a protection override (when signal < threshold). All 6 overrides enabled in defaults:

| Regime | Upside (signal >) | Protection (signal <) |
|---|---|---|
| R1 | +2 → 100% TQQQ ("Max Bull") | −2 → 100% QQQ ("Bull Fading") |
| R2 | +2 → 70% QQQ + 30% XLU ("Recovery Confirmed") | −3 → 100% XLU ("Deteriorating Fast") |
| R3 | +3 → 80% TQQQ + 20% QQQ ("Capitulation Reversal") | −4 → 100% XLU ("Crisis Deepening") |

---

## Architecture (data flow)

```
yfinance ──┐                                                              ┌──→ exporter.py (Excel)
           ├──→ data_loader.py ──→ run_backtest() ──→ equity_df ──→ ─────┤
yfinance ──┘   (no caching!)        in backtest.py    (DataFrame)         └──→ dashboard.py (Streamlit)
                                                                                  │
                                                                                  └─→ calculate_metrics() → CAGR, Sharpe, etc.
```

`run_backtest` (`backtest.py:542`) is the orchestrator. Per-day loop:

1. Compute drawdown from rolling/standard ATH (`compute_rolling_ath_and_dd` / `compute_drawdown_from_ath`)
2. **T+1 lag** the drawdown signal (`build_regime_signal_drawdown`, `backtest.py:158`)
3. Map drawdown → market regime (`regime_engine.determine_regime`)
4. Decide if portfolio follows market (`apply_per_regime_direction_strategy` / `_down_only` / `_up_only`)
5. Apply signal overrides if any (`signal_override_engine`)
6. Execute rebalance if frequency triggers (`rebalance_engine.rebalance_portfolio`)

---

## Key APIs (call signatures)

```python
# backtest.py:542 — orchestrator. Returns (equity_df, rebalance_df, dividend_df).
run_backtest(
    price_data: pd.DataFrame,           # OHLCV-style closes; columns = config["tickers"], DatetimeIndex
    config: dict,                       # CONFIG dict (or test config with same shape)
    dd_fn,                              # callable(series) -> (dd_series, ath_series)
    regime_detector,                    # callable(dd_value, config) -> regime label or None
    rebalance_fn,                       # callable(value, allocations, prices) -> shares dict or None
    dividend_data=None,                 # optional pd.DataFrame of dividends per ticker
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

# regime_engine.py:13 — pure boundary detection.
determine_regime(dd_value, config, date=None) -> str | None
# Non-last regimes: low <= dd < high. Last regime: low <= dd <= high. NaN/None → None.

# backtest.py:117 — rolling N-year peak with cummax fallback.
compute_rolling_ath_and_dd(
    series: pd.Series, n_calendar_years: int
) -> tuple[pd.Series, pd.Series]   # (ath_series, dd_series)

# regime_engine.py:5 — standard cummax-based drawdown.
compute_drawdown_from_ath(series: pd.Series) -> tuple[pd.Series, pd.Series]   # (dd, ath)

# allocation_engine.py:6 — regime → {ticker: weight} dict (raw target).
get_allocation_for_regime(regime: str, config: dict) -> dict[str, float]

# allocation_engine.py:32 — filter+renormalize to executable weights.
tradable_allocation(alloc: dict, prices: pd.Series, config: dict) -> dict[str, float]
# Fallback ladder when prices have NaNs:
#   1. Keep allocation_tickers with weight > 0 AND priced; renormalize to 1.
#   2. If empty AND TQQQ was targeted but unpriced: 100% QQQ proxy.
#   3. Else: 100% drawdown_ticker if priced.
#   4. Else: equal weight across any priced allocation_ticker.
#   5. Else: empty dict (skip rebalance, stay in cash).

# rebalance_engine.py:5 — convert target weights + prices to share counts.
rebalance_portfolio(
    portfolio_value: float, allocations: dict, prices: pd.Series
) -> dict[str, float] | None         # None if any targeted ticker has NaN price

# dashboard.py:144 — metrics dict from equity_df.
calculate_metrics(equity_df: pd.DataFrame, config=None) -> dict
# Keys: cagr, volatility, sharpe_ratio, sortino_ratio, max_drawdown,
#       total_return, beta, beta_benchmark, start_value, end_value,
#       years, start_date, end_date. All floats (post-1818ea7 fix).

# utils.py:6 — non-positive max drawdown from a value series.
max_drawdown_from_equity_curve(values) -> float   # e.g. -0.35 for -35% drawdown
```

---

## `equity_df` schema (output of `run_backtest`)

Returned by `run_backtest`. **Note:** signal-layer columns (Signal_L1, MACD_*, etc.) are NOT included here — they are appended later by `compute_signal_layer_columns` (`signal_layers.py:318`), which is called from `main.py:57` and `app.py:849`. Tests that bypass entry points only see the core columns.

### Core columns (every row)

| Column | Type | Meaning |
|---|---|---|
| `Date` | Timestamp | Trading day |
| `Value` | float | Portfolio total value (positions + cash) |
| `Cash` | float | Cash balance (only used when dividend_reinvestment_target="cash") |
| `Market_Regime` | str/None | Regime determined from market drawdown signal (T+1-lagged) |
| `Portfolio_Regime` | str | Regime the portfolio is currently allocated to |
| `Regime_Trajectory` | str | "Upward" / "Downward" / "Flat" / "" (vs prev day) |
| `Prev_Market_Regime` | str/None | Prior trading day's market regime |
| `Rebalanced` | str | "Rebalanced" if a rebalance fired today, else "" |
| `Pct_Growth` | float | Cumulative return from starting_balance: `Value/start - 1` |
| `Signal_override_active` | str | "upside" / "protection" / "none" |
| `Signal_override_label` | str | Human label of the override (e.g. "Max Bull") |
| `Signal_override_allocation` | str | Pretty-printed weights for the active override |

### Drawdown columns (every row)

| Column | Type | Meaning |
|---|---|---|
| `{drawdown_ticker}_ATH_raw` | float | Reference high used by regime engine (rolling 1y if `drawdown_window_enabled=True`) |
| `{drawdown_ticker}_DD_raw` | float | `(ATH - close) / ATH`, in `[0, 1]` |
| `Portfolio_ATH` | float | Portfolio's own running peak |
| `Portfolio_DD` | float | Portfolio drawdown from its peak (negative or zero) |

### Per-ticker columns (for each ticker in `config["tickers"]`)

| Column pattern | Type | Meaning |
|---|---|---|
| `{ticker}_price` | float | Daily close (NaN before listing) |
| `{ticker}_norm` | float | Normalized to starting_balance from this ticker's first valid date |
| `{ticker}_shares` | float | Held share count |
| `{ticker}_value` | float | `shares * price` |

### Signal columns (appended by `compute_signal_layer_columns`)

`VIX`, `VIX_252d_mean`, `VIX_252d_stdev`, `VIX_zscore`, `VIX_zscore_direction`, `VIX_regime_label`, `Signal_L1`, `MACD_12ema`, `MACD_26ema`, `MACD_line`, `MACD_signal`, `MACD_histogram`, `MACD_histogram_delta`, `Signal_L2`, `MA_50`, `MA_200`, `MA_regime_label`, `Signal_L3`, `Signal_total`, `Signal_label`

---

## File map

### Core logic (`*.py` at repo root)

| File | One-line description |
|---|---|
| `config.py` | **Production config dict.** Single source of truth for tickers, regimes, strategy, dates. |
| `regime_engine.py` | `determine_regime(dd, config)` and `compute_drawdown_from_ath(series)`. Pure functions. |
| `backtest.py` (1196 lines) | Orchestrator (`run_backtest`) + regime transition logic + duplicate `compute_drawdown_from_ath` (`backtest.py:107` — drift risk). |
| `allocation_engine.py` | Maps regime → ticker weights. Handles TQQQ→QQQ proxy for pre-2010 (TQQQ IPO date). |
| `rebalance_engine.py` | Converts target weights + prices → share counts. Returns `None` if any allocated price is NaN. |
| `signal_layers.py` | VIX z-score (L1), MACD (L2), MA50/200 (L3) → composite signal total in `[-6, +6]`. |
| `signal_override_engine.py` | Within-regime allocation overrides triggered by signal total thresholds. |
| `utils.py` | `max_drawdown_from_equity_curve()` (returns non-positive float), `log()`, helpers. |
| `data_loader.py` | yfinance wrappers; routes every download through `data_cache.cached_yf_download`. |
| `data_cache.py` | On-disk cache for yfinance downloads. Three modes via `FMF_DATA_MODE`: `auto` (cache or fetch), `frozen` (cache only, raises on miss), `refresh` (always fetch). Manifest at `data_cache/manifest.json` records SHA256, fetch timestamp, yfinance version. |

### Entry points / UI / output

| File | One-line description |
|---|---|
| `app.py` | Streamlit web app (recommended production runner). |
| `main.py` | CLI runner. |
| `dashboard.py` | Streamlit dashboard + `calculate_metrics()`. **Imports streamlit at module level — coupling issue.** |
| `dashboard_runner.py` | Entry script for the dashboard. |
| `console_ui.py` | Terminal display logic. |
| `exporter.py` | Excel export for backtest results. |
| `worst_case_simulator.py` | Synthetic worst-case data scenarios (separate concern from main backtest). |

### Tests (153 passing, all under `tests/`)

| File | Count | Covers |
|---|---|---|
| `conftest.py` | — | Shared fixtures: `price_fixture` (seed=42), `minimal_config`, `production_per_regime_config`, equity helpers |
| `test_regime_boundaries.py` | 17 | `determine_regime` boundary edge cases + Hypothesis property |
| `test_regime_transitions.py` | 23 | `apply_per_regime`, asymmetric down_only / up_only + full sequence |
| `test_financial_metrics.py` | 28 | CAGR / Sharpe / Sortino / max DD / volatility — formula + quantstats + type contracts |
| `test_portfolio_math.py` | 9 | Value invariants, allocation sums, TQQQ→QQQ proxy |
| `test_drawdown_window.py` | 6 | Rolling ATH scenarios + Hypothesis (overlaps with `test_rolling_drawdown_window.py`) |
| `test_regression_ground_truth.py` | 7 | **Locked anchor** — but uses `down_only`, no overrides, synthetic data (NOT production) |
| `test_regime_signal_lag.py` | (existing) | T+1 lag verification |
| `test_rolling_drawdown_window.py` | (existing) | Rolling window correctness |
| `test_signal_layers.py` | (existing) | Signal computation |
| `test_signal_override_engine.py` | (existing) | Override decision unit tests (not integration) |
| `test_tradable_allocation_and_backtest.py` | (existing) | Pre-IPO TQQQ proxy paths |
| `test_cash_allocation.py` | 7 | CASH series compounding + end-to-end backtest + `enable_cash_in_regimes` + `forced_base_allocations` |

### Other

| File | Purpose |
|---|---|
| `validate_tests.py` | Older test harness using yfinance with `test_data_cache/` (predates pytest suite, still functional but legacy) |
| `pyproject.toml` | pytest config + markers (`unit`, `integration`, `metrics`) |
| `requirements.txt` | pytest, hypothesis, quantstats, streamlit, plotly, yfinance, pandas, numpy |
| `docs/superpowers/specs/` | Design docs |
| `docs/superpowers/plans/` | Implementation plans |
| `test_data_cache/` | Pickle cache for `validate_tests.py` only — **NOT used by the production backtest** |

---

## Non-obvious concepts (read these before changing related code)

### 1. T+1 signal lag
The regime decision for trading day D uses the **prior** trading day's drawdown signal. Execution still happens at today's prices. Prevents look-ahead bias. See `build_regime_signal_drawdown` (`backtest.py:158`). Do NOT "simplify" by using same-day drawdown.

### 2. Rolling ATH (production default)
With `drawdown_window_enabled=True` + `drawdown_window_years=1`, the "all-time high" is the max close in the trailing 1 calendar year, not all-time. So a 2000 dot-com peak doesn't suppress today's drawdown. Bootstrap: until 1y of history exists, falls back to standard cummax. See `compute_rolling_ath_and_dd` (`backtest.py:117`).

### 3. Per-regime hold/match — THE TRICKY ONE
`apply_per_regime_direction_strategy` (`backtest.py:327`) checks the **TARGET** regime's `rebalance_on_upward`/`downward`, not the source's.

So `R2.rebalance_on_upward = "hold"` would mean **"when market arrives AT R2 from R3, hold"** — NOT "when leaving R2 going up, hold". This is non-obvious and easy to flip when reading config.

Current defaults set every regime to `match/match` (always follow), so the trickiness doesn't bite right now. But if anyone sets `hold` in the future, this rule applies.

Example with all-match defaults:
| Transition | Target regime | Setting checked | Result |
|---|---|---|---|
| R1→R2 (down) | R2 | downward=match | Follow to R2 (defense) |
| R2→R3 (down) | R3 | downward=match | Follow to R3 (contrarian buyback) |
| R3→R2 (up, partial recovery) | R2 | upward=match | Follow to R2 (back to defense) |
| R2→R1 (up, full recovery) | R1 | upward=match | Follow to R1 (calm) |

### 4. Pre-portfolio simulation
Before the first backtest day, `run_backtest` walks pre-history (`backtest.py:660-720`) — from the earliest available date in the merged drawdown_ticker series up to `panel_start` — applying the chosen strategy to determine what regime the portfolio should *open* in.

**Important caveat:** with current production config (`start_date=1999-01-04`, `drawdown_ticker=QQQ`, QQQ inception=1999-03-10), this walk is essentially zero-length: yfinance returns no QQQ data before 1999-03-10, so `panel_start` and `pre_portfolio_start_date` are nearly the same. The walk only meaningfully runs when `start_date` is later than the drawdown ticker's inception, OR when a different drawdown_ticker has older history than the portfolio start. **No test covers this — see Open Issues.**

### 5. TQQQ pre-IPO handling
TQQQ launched 2010-02-09. For dates before, R1's "100% TQQQ" target is renormalized to "100% QQQ" via `tradable_allocation` (`allocation_engine.py:32`). This is why the panel can start in 1999.

### 6. Sharpe / Sortino are non-standard
`calculate_metrics` uses `CAGR / annualized_vol` instead of standard `(annualized_mean - rf) / annualized_std`. Diverges from quantstats by ~0.04 (Sharpe) and up to ~0.55 (Sortino) on 5y series. Internally consistent for relative comparisons; **not directly comparable to externally published Sharpe numbers.**

### 7. CAGR uses calendar days
`years = (end_date - start_date).days / 365.25`. quantstats uses `len(returns) / 252`. Tiny divergence on long series, larger on short ones.

---

## Signal layer rules (reference)

`Signal_total = L1 + L2 + L3`, range `[−6, +6]`. Each layer outputs an integer in `{−2, −1, 0, +1, +2}` per its rule. All rules in `signal_layers.py`.

### L1 — VIX z-score (252-day rolling)

`z = (VIX − VIX_252d_mean) / VIX_252d_stdev`. Contrarian: high VIX (fear) → high positive signal.

| z range | L1 | Label |
|---|---|---|
| `z > 2` | +2 | Extreme Fear |
| `1 < z ≤ 2` | +1 | Elevated |
| `−1 < z ≤ 1` | 0 | Normal |
| `−2 < z ≤ −1` | −1 | Complacent |
| `z ≤ −2` | −2 | Extreme Complacency |

### L2 — SPY MACD(12, 26, 9)

EMA12 − EMA26 = MACD line. EMA9 of MACD line = signal line. Histogram = line − signal. First matching rule wins:

| Rule (in priority order) | L2 |
|---|---|
| Bullish cross (line crossed above signal) AND `histogram_delta > 0` | +2 |
| `MACD_line > 0` AND `histogram > 0` | +1 |
| `histogram > 0` AND `histogram_delta < 0` (rolling over) | 0 |
| Bearish cross (line crossed below signal) AND `histogram < 0` | −2 |
| Bearish divergence (SPY at 55d high, MACD below prior peak) | −1 |
| Otherwise | 0 |

### L3 — SPY 50/200 MA crossover

`spread = MA50 − MA200`. First matching rule wins:

| Rule (in priority order) | L3 |
|---|---|
| `MA50 > MA200` AND `spread > spread_prev` (golden cross strengthening) | +2 |
| `SPY > MA200` AND `MA50 < MA200` (price above 200, MAs not yet crossed) | +1 |
| `SPY > MA50` AND `SPY < MA200` (mixed) | 0 |
| `MA50 < MA200` AND `spread < spread_prev` (death cross deepening) | −2 |
| Otherwise | 0 |

### `Signal_label` thresholds (display only)

| Total | Label |
|---|---|
| `≥ +5` | Strong Buy |
| `+3 to +5` | Buy |
| `+1 to +3` | Lean Long |
| `−1 to +1` | Neutral |
| `−3 to −1` | Reduce |
| `< −3` | Strong Sell |

The override engine uses raw `Signal_total` against per-regime `threshold` values, NOT these labels.

---

## Numeric tolerance conventions (used in tests)

| Comparison | Tolerance | Rationale |
|---|---|---|
| Boundary equality (`dd == 0.08`) | `abs=1e-9` | Exact float comparison |
| Max drawdown vs quantstats | `< 1e-6` | Same algorithm |
| Volatility vs quantstats | `< 1e-4` | Same algorithm |
| CAGR vs quantstats | `< 0.01` | Calendar-day vs trading-day years |
| Sharpe vs quantstats | `< 0.1` | CAGR-based vs mean-based formula |
| Sortino vs quantstats | `< 0.7` | Larger methodology divergence |
| Value invariants (shares × price) | `< 1e-6` | Float roundoff |

NaN policy: `determine_regime(NaN)` → None; `max_drawdown_from_equity_curve` drops NaNs; `rebalance_portfolio` returns None if any allocated ticker is NaN.

---

## Default config (`config.py`) — subject to change

```python
start_date:                "1999-01-04"
end_date:                  "2026-03-27"
starting_balance:          10_000
tickers:                   ["QQQ", "TQQQ", "XLU", "SPY"]
allocation_tickers:        ["QQQ", "TQQQ", "XLU"]
drawdown_ticker:           "QQQ"
rebalance_strategy:        "per_regime"     # NOT down_only — locked test uses down_only!
rebalance_frequency:       "instant"
drawdown_window_enabled:   True
drawdown_window_years:     1
dividend_reinvestment:     False
```

**Allocation universe:** the production config holds three risk sleeves
(TQQQ / QQQ / XLU). The optimizer can additionally include a synthetic
**CASH** ticker as a 4th allocation option — see `optimizer/parameter_space.py`
(`CASH_TICKER`, `CASH_APY`). CASH is a daily-compounded risk-free series
(default 4% APY) generated in `optimizer/score._build_cash_series`; it is
opt-in via `IterationConstraints.enable_cash_in_regimes` (search) or
`forced_base_allocations` (pin). CASH never appears in the base production
config — it only enters when the optimizer explicitly asks for it.

Regime thresholds (all match/match for clean follow logic):
- **R1**: dd ∈ [0.00, 0.06) — 100% TQQQ
- **R2**: dd ∈ [0.06, 0.20) — 70% XLU + 30% QQQ
- **R3**: dd ∈ [0.20, 1.00] — 50% TQQQ + 50% QQQ

All 6 signal overrides enabled (see Architecture section for the full table). These are defaults, not tuned values — expect to iterate.

---

## Bugs fixed during test suite work

- **2026-04-26** — `dashboard.calculate_metrics` returned `numpy.float64` (and `int 0` for zero-vol/zero-years edge cases) instead of consistent Python `float`. Fixed in commit `1818ea7` by coercing all metric returns to `float()`. Caught by type-contract tests in `tests/test_financial_metrics.py`.

---

## Open issues / tech debt (severity high → low)

1. ~~**Non-reproducible backtests.**~~ Resolved by `data_cache.py`. Run with `FMF_DATA_MODE=frozen` for guaranteed reproducibility against the committed snapshot.
2. **Locked regression doesn't cover production code path.** `test_regression_ground_truth.py` uses `down_only` + no overrides + no drawdown window. Production uses `per_regime` + 6 overrides + drawdown window + 25y of real history. Will be addressed by Phase 2E of the optimizer plan.
3. **Duplicate `compute_drawdown_from_ath`** — `regime_engine.py:5` AND `backtest.py:107`. Will silently drift if either changes.
4. **`dashboard.py` imports streamlit at module level** — forces metric-only tests to install streamlit/plotly. `calculate_metrics` should live in a `metrics.py` with no UI deps.
5. **`run_backtest` has hidden network calls** — yfinance + `signal_layers.load_spy_series` + `signal_layers.load_vix_series`. Tests must patch 3 functions to mock.
6. **Signal layer computation runs even when all overrides disabled.** Wasted CPU; forces test mocks. Easy guard: skip the block when no regime has an enabled override.
7. **No end-to-end test of signal overrides enabled.** `test_signal_override_engine.py` covers unit logic only.
8. **Pre-portfolio simulation untested.** `backtest.py:660-720` walks regime through ~25y of pre-history but no test verifies the output.
9. **No end-to-end test of dividend reinvestment.** Disabled in production but code path untested.
10. **`test_rolling_drawdown_window.py` and `test_drawdown_window.py` overlap.** Consider merging.

---

## Working agreements with user

- **Tests over coverage %.** User explicitly wants targeted critical-path coverage, not 80% blanket coverage.
- **Trust over speed.** "I need this to be so secure the data has to be so trusted." Reproducibility matters.
- **Verify metrics two ways.** quantstats AND first-principles formula. Both must agree within documented tolerances.
- **Don't change LOCKED constants silently.** If a refactor moves a locked value, update the constant with an explicit comment explaining what changed.
- **Architecture cleanup is queued AFTER tests are solid.** UI redesign comes last (frontend-design skill).
- **No date prefixes on file names.** User prefers `test-suite-design.md` over `2026-04-26-test-suite-design.md`. Same for plans.
- **Browser/computer-use tools only for UI testing.** Not for token-heavy investigation work.

---

## Common commands

```bash
# Run all tests
python3 -m pytest tests/ -q

# Verbose
python3 -m pytest tests/ -v

# Single file
python3 -m pytest tests/test_regime_boundaries.py -v

# By marker
python3 -m pytest -m metrics
python3 -m pytest -m unit

# Collection-only sanity check
python3 -m pytest tests/ --collect-only -q

# Run the backtest (CLI)
python3 main.py

# Run the Streamlit app
streamlit run app.py
```

---

## Pinned design docs and plans

- **Test suite design** — `docs/superpowers/specs/test-suite-design.md`
- **Test suite implementation** — `docs/superpowers/plans/test-suite-implementation.md`
- **Data provenance + production regression** (resolved by `data_cache.py`) — `docs/superpowers/plans/data-provenance-and-production-regression.md`
- **Iterative config search methodology** — `docs/superpowers/methodologies/iterative-config-search.md`. The standard playbook for tuning regime allocations: 50-trial Monte Carlo batches, analyze trends, narrow constraints, repeat 5–7+ times. Run via `python3 scripts/iterate.py --study iterN`.

---

## When to update this file

Update CLAUDE.md when:
- A new file is added to the repo (add it to the file map with one-line description)
- A bug is fixed that changes how a documented concept works (update Non-obvious concepts)
- A locked test value changes (note it under Bugs fixed)
- An open issue is resolved (move it from Open issues to Bugs fixed, with commit SHA)
- The production config changes (update Production config section)
- A new working agreement is established with the user

Do NOT update CLAUDE.md for:
- Routine commits, refactors that don't change behavior, or test additions that don't reveal new code semantics
- Anything derivable from `git log` or `grep` (those are authoritative)
