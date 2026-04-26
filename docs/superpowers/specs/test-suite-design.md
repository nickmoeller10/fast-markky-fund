# Test Suite Design — Fast Markky Fund
**Date:** 2026-04-26  
**Scope:** Critical logic correctness + financial math verification  
**Goal:** Trust the numbers the backtest produces. Every critical path has a test that would catch a silent regression.

---

## 1. Philosophy

Coverage percentage is not the goal. **Trust** is. A test that checks the wrong thing at 100% coverage is worse than no test. Every test in this suite answers a specific question:

> "If I change this code, will I know if something financially meaningful broke?"

We verify financial metrics two ways: against `quantstats` (reference implementation) and from first principles (start/end values + formula). Both must agree.

---

## 2. Infrastructure

### 2.1 `pytest.ini` (or `pyproject.toml` `[tool.pytest]` section)
- Test discovery: `tests/`
- No coverage threshold enforced (intentional — quality over quantity)
- Markers: `unit`, `integration`, `metrics`

### 2.2 `tests/conftest.py` — shared fixtures
All tests draw from the same controlled building blocks:

**`price_fixture(n_days, start, tickers, seed)`**  
Deterministic synthetic OHLCV-style close prices. Seeded so tests are reproducible. No yfinance calls.

**`minimal_config(regimes, strategy, window_enabled)`**  
Minimal valid config dict. Defaults to 3-regime R1/R2/R3 with thresholds matching production (0%, 8%, 28%). Parameterizable.

**`equity_series_fixture(start_val, daily_returns)`**  
Build an equity curve `pd.Series` from a list of daily returns. Used to feed metrics functions without running a full backtest.

**`known_backtest_result`**  
A frozen, manually-verified ground truth: 5 years of synthetic prices with a known regime sequence, known final value, known CAGR/Sharpe/max drawdown. Locked as a regression anchor — any change to core logic that moves these numbers fails the suite.

---

## 3. Test Modules

### 3.1 `test_regime_boundaries.py` — Regime detection edge cases

**What it protects:** `regime_engine.determine_regime()` and `compute_drawdown_from_ath()`

| Test | Scenario | Why critical |
|---|---|---|
| `test_exact_lower_boundary` | dd = 0.08 exactly → R2 | Boundary is `low <= dd < high`; off-by-one sends wrong allocation |
| `test_exact_upper_boundary` | dd = 0.28 exactly → R3 | Last regime is inclusive `<=`; different rule than others |
| `test_zero_drawdown` | dd = 0.0 → R1 | Clean ATH state, must always be R1 |
| `test_full_drawdown` | dd = 1.0 → R3 | Total loss scenario, last regime must catch it |
| `test_nan_drawdown_returns_none` | dd = NaN → None | NaN must never silently resolve to a regime |
| `test_boundary_r1_r2_below` | dd = 0.0799 → R1 | One tick below R2, still R1 |
| `test_boundary_r2_r3_below` | dd = 0.2799 → R2 | One tick below R3, still R2 |
| `test_drawdown_monotonic_ath` | Rising prices → dd always 0 | ATH must track correctly |
| `test_drawdown_recovery` | Price falls then fully recovers → dd returns to 0 | Recovery path |
| `test_drawdown_partial_recovery` | Falls 20%, recovers to -10% → dd = 0.10 | Math check |

**Hypothesis property test:**  
`dd ∈ [0.0, 1.0]` always returns a non-None regime when regimes cover [0, 1.0] completely.

---

### 3.2 `test_regime_transitions.py` — Asymmetric transition logic

**What it protects:** `apply_per_regime_direction_strategy`, `apply_asymmetric_rules_down_only`, `apply_asymmetric_rules_up_only`

This is the most financially critical untested area. The `R2 rebalance_on_upward: hold` rule means the portfolio stays in XLU even when QQQ recovers above 8% drawdown. A bug here silently holds the wrong allocation for extended periods.

| Test | Scenario | Expected |
|---|---|---|
| `test_per_regime_downward_match` | R1→R2, R2 has `rebalance_on_downward: match` | Portfolio moves to R2 |
| `test_per_regime_upward_hold` | R2→R1, R2 has `rebalance_on_upward: hold` | Portfolio stays R2 |
| `test_per_regime_upward_match` | R2→R1, R2 has `rebalance_on_upward: match` | Portfolio moves to R1 |
| `test_per_regime_no_change` | R1→R1 (flat) | Portfolio unchanged |
| `test_per_regime_first_day_aligns` | prev_market=None | Portfolio adopts market regime |
| `test_down_only_follows_down` | R1→R2 | Portfolio follows to R2 |
| `test_down_only_holds_on_partial_recovery` | R2→R2 (still stressed) | Portfolio stays R2 |
| `test_down_only_only_recovers_at_r1` | R2→R1 | Portfolio returns to R1 |
| `test_down_only_r1_hold_when_not_r1` | Portfolio R2, market R2 | Stays R2 |
| `test_up_only_follows_up` | R2→R1 | Portfolio follows |
| `test_up_only_holds_on_down` | R1→R2 | Portfolio stays R1 |
| `test_up_only_bottom_regime_rebalances` | R3→R2 | Always follows up from bottom |
| `test_sequence_r1_to_r2_to_r1` | Full round trip with `down_only` | Ends at R1 only after full recovery |
| `test_sequence_per_regime_production_config` | R1→R2→R1 with production config (hold on upward) | Stays R2 until explicit match |

**Hypothesis property test:**  
For any sequence of market regimes, portfolio regime is always a valid regime label (never None, never unknown string).

---

### 3.3 `test_financial_metrics.py` — CAGR, Sharpe, Sortino, max drawdown

**What it protects:** `dashboard.calculate_metrics()`, `utils.max_drawdown_from_equity_curve()`

Each metric verified two ways: (A) formula from first principles, (B) against `quantstats`.

**CAGR:**
| Test | Input | Output type | Method A (formula) | Method B (quantstats) |
|---|---|---|---|---|
| `test_cagr_doubles_in_one_year` | $10k → $20k over 1yr | `float` | 1.0 (100%) | `qs.stats.cagr` |
| `test_cagr_ten_years` | $10k → $16.1k over 10yr | `float` | ~0.05 | `qs.stats.cagr` |
| `test_cagr_flat` | $10k → $10k | `float` | 0.0 | `qs.stats.cagr` |
| `test_cagr_loss` | $10k → $5k over 2yr | `float` | ~-0.293 | `qs.stats.cagr` |
| `test_cagr_returns_python_float` | Any valid equity_df | `float` (not np.float64, not Series) | — | — |
| `test_cagr_zero_years` | start_date == end_date | `float` = 0.0, no ZeroDivisionError | — | — |
| `test_cagr_single_day` | 1 row equity_df | `float`, no crash | — | — |
| `test_cagr_negative_start_value` | start_val = 0 | `float` = 0.0, no ZeroDivisionError | — | — |

**Volatility:**
| Test | Input | Output type | Expected |
|---|---|---|---|
| `test_volatility_returns_float` | 252-day equity curve | `float` (not Series) | annualized std × √252 |
| `test_volatility_flat_equity` | Constant value series | `float` = 0.0 | no division by zero |
| `test_volatility_single_return` | 2-row equity_df | `float` = 0.0 or handled | no crash |
| `test_volatility_vs_quantstats` | 252-day random walk | within 1e-4 of `qs.stats.volatility` | annualized, 252 periods |
| `test_volatility_all_nan_returns` | All NaN pct_change | `float` = 0.0, no crash | NaN propagation safe |

**Max Drawdown:**
| Test | Input | Output type | Expected |
|---|---|---|---|
| `test_max_drawdown_single_drop` | 100 → 60 → 80 | `float` | -0.40 |
| `test_max_drawdown_new_ath_resets` | 100 → 120 → 90 | `float` | -0.25 (from 120) |
| `test_max_drawdown_no_decline` | Monotonically rising | `float` | 0.0 |
| `test_max_drawdown_vs_quantstats` | 252-day random walk | within 1e-6 of `qs.stats.max_drawdown` | — |
| `test_max_drawdown_returns_float` | Any valid series | `float` (not Series, not np.float64) | — |
| `test_max_drawdown_nan_handling` | Series with NaNs | `float`, no crash | NaNs dropped before calculation |
| `test_max_drawdown_single_value` | 1-element series | `float` = 0.0 | no crash |
| `test_max_drawdown_is_nonpositive` | Any series | `float` ≤ 0.0 | by definition |

**Sharpe:**
| Test | Input | Output type | Expected |
|---|---|---|---|
| `test_sharpe_positive_returns` | Consistent +0.1% daily | `float` > 0 | positive Sharpe |
| `test_sharpe_vs_quantstats` | 252-day series, rf=0 | within 0.01 of `qs.stats.sharpe` | — |
| `test_sharpe_zero_volatility` | Flat equity curve | `float` = 0.0, no ZeroDivisionError | — |
| `test_sharpe_returns_float` | Any valid equity_df | `float` (not Series) | — |
| `test_sharpe_negative_cagr` | Losing portfolio | `float` < 0 | negative Sharpe |

**Sortino:**
| Test | Input | Output type | Expected |
|---|---|---|---|
| `test_sortino_no_negative_returns` | All positive daily returns | `float` = `inf` or large positive | no crash |
| `test_sortino_vs_quantstats` | Mixed returns, 252 days | within 0.01 of `qs.stats.sortino` | — |
| `test_sortino_returns_float` | Any valid equity_df | `float` (not Series, not np.float64) | — |
| `test_sortino_all_negative_returns` | All negative daily returns | `float` < 0 | high downside deviation |

---

### 3.4 `test_portfolio_math.py` — Value calculation and rebalance execution

**What it protects:** Share counting, portfolio value, tradable allocation proxy

| Test | Scenario | Why |
|---|---|---|
| `test_portfolio_value_equals_shares_times_price` | After rebalance, value = Σ(shares × price) | Fundamental invariant |
| `test_total_allocation_sums_to_one` | Any regime config | Allocations must sum to 1.0 |
| `test_rebalance_preserves_value` | Before/after rebalance on same prices | No value destroyed by rebalancing |
| `test_proxy_qqq_when_tqqq_nan` | TQQQ price = NaN | Uses QQQ, allocation renormalized |
| `test_proxy_all_missing` | All allocation tickers NaN | Graceful — no crash, no phantom value |
| `test_starting_balance_preserved_day_one` | First day, no price change | Portfolio = starting_balance |

---

### 3.5 `test_drawdown_window.py` — Rolling ATH vs standard ATH

**What it protects:** `compute_rolling_ath_and_dd`, the production config path (`drawdown_window_enabled: True, years: 1`)

| Test | Scenario |
|---|---|
| `test_rolling_matches_cummax_before_window_fills` | First N days → rolling = cummax |
| `test_rolling_peak_scrolls_out` | Price 18 months ago > current prices → rolling ATH lower than cummax ATH |
| `test_rolling_and_standard_agree_on_fresh_ath` | New all-time high → both = 0% drawdown |
| `test_rolling_produces_higher_dd_than_standard` | After peak scrolls out → rolling dd > standard dd for same prices |
| `test_drawdown_always_in_0_1_range` | Hypothesis: any price series with window ≥ 1 | dd ∈ [0, 1] |

---

### 3.6 `test_regression_ground_truth.py` — Locked regression anchor

A single frozen test with a hand-verified synthetic dataset:

- **Input:** 5-year deterministic price series (seeded), standard 3-regime config, instant rebalance, `down_only` strategy
- **Locked outputs:** final portfolio value, CAGR, max drawdown, regime sequence for first 20 days
- **Purpose:** If any refactor, cleanup, or new feature changes these numbers without an explicit unlock, the test fails

This test is the safety net for the architecture cleanup phase.

---

## 4. What We Are NOT Testing (Intentionally)

- `exporter.py` — Excel formatting, no financial logic
- `dashboard.py` UI rendering — Streamlit widget layout
- `data_loader.py` live yfinance calls — mocked at boundary, not integration tested
- `console_ui.py` — display logic only
- `worst_case_simulator.py` — synthetic data generation, separate concern

---

## 5. Test Infrastructure Files to Create

```
tests/
  conftest.py                          ← shared fixtures
  test_regime_boundaries.py            ← new
  test_regime_transitions.py           ← new (most critical)
  test_financial_metrics.py            ← new (quantstats + formula)
  test_portfolio_math.py               ← new
  test_drawdown_window.py              ← extends existing
  test_regression_ground_truth.py      ← new (locked anchor)
  test_regime_signal_lag.py            ← exists, keep
  test_rolling_drawdown_window.py      ← exists, keep
  test_signal_layers.py                ← exists, keep
  test_signal_override_engine.py       ← exists, keep
  test_tradable_allocation_and_backtest.py ← exists, keep
pyproject.toml                         ← pytest config + markers
```

---

## 6. Dependencies to Add to `requirements.txt`

```
pytest>=9.0
pytest-cov>=7.0
hypothesis>=6.0
quantstats>=0.0.81
```

---

## 7. Success Criteria

- All existing 25 tests continue to pass
- New tests cover all 6 critical areas above
- `test_regression_ground_truth.py` locked and passing
- Every financial metric (CAGR, Sharpe, Sortino, max drawdown) verified against both quantstats and formula
- `pytest tests/` completes clean with no warnings on financial logic
