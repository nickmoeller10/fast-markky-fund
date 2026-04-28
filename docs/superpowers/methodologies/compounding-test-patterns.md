---
name: Compounding Test Patterns
description: Patterns for writing tests that verify the fast-markky-fund backtest compounds correctly. Use when adding tests to tests/test_compounding_correctness.py, when investigating a suspected accounting bug in run_backtest's daily loop, or when modifying anything in backtest.py / rebalance_engine.py / signal_override_engine.py / allocation_engine.py that could affect equity-curve calculation.
---

# Compounding Test Patterns

> Why this exists: the backtest had a silent compounding bug where rebalance days used yesterday's NAV as the share basis instead of today's mark, destroying the day-of-rebalance return. The bug only showed up under careful paired-config tests; standard end-to-end tests passed because absolute equity values "looked plausible." These patterns are the toolkit for catching the next bug like that.

The tests live in **`tests/test_compounding_correctness.py`**. Read that file first — every test there is an instance of one of the patterns below.

---

## The compounding contract

The fast-markky-fund daily loop must satisfy these invariants on every trading day **D**:

1. **Mark invariant**: `equity_df.Value[D] == sum(shares[t] × price[t][D]) + cash_balance[D]`. Both rebalance days and non-rebalance days. No exceptions.
2. **No-op rebalance is identity**: rebalancing to the *same* target weights the portfolio already holds (within the tradable subset) must not change `Value[D]`.
3. **Today's return is preserved through rebalances**: a rebalance on day D uses today's mark as the NAV basis, so `Value[D]` reflects today's intraday price moves whether or not a rebalance fired.
4. **Daily-return sum equals total return**: `prod(1 + daily_return[d] for d) ≈ Value[end] / Value[start]`, within float tolerance.
5. **Cash sleeve is non-decreasing**: a 100% `$` portfolio has zero drawdown and grows at exactly `(1 + CASH_APY)^(years)`.
6. **DD monotonicity in R1 width**: with R2/R3 in safe assets, max drawdown must non-decrease as the R1 dd_high boundary widens (more time leveraged = more drawdown).

If any of those breaks, there's a compounding bug.

---

## The five test patterns

### 1. **Paired-config equivalence** (the "two configs that should be the same")

Run two backtests where the only structural difference is something that **must not** affect the equity curve. Assert curves match within ~1e-9 relative tolerance.

Use this pattern when:
- An override panel has weights identical to its regime base (enabled vs disabled must match).
- Two regimes have identical allocations (splitting one regime into two must not change anything).
- A threshold is unreachable (enabled-but-never-fires must equal disabled).
- An R2/R3 boundary is moved while their allocations are identical (boundary is cosmetic).

Tests A01, A04, E16, E17, E18 in the existing file are this pattern.

```python
eq_a, _, _ = _run(cfg_a, panel)
eq_b, _, _ = _run(cfg_b, panel)
_equity_close(eq_a, eq_b, rel=1e-9)
```

### 2. **Algebraic invariant** (the "mark must always equal shares × price")

For every row in `equity_df`, manually recompute `Value` from `shares × price + cash` and assert equality.

Use this pattern as a *general-purpose tripwire* — it catches any future bug where `Value` is updated asynchronously from share counts.

Test A02 is this pattern. The helper is `_mark_invariant(eq, panel, allocation_tickers)`.

### 3. **Closed-form expected value** (the "I can compute the right answer by hand")

For a single-asset portfolio with no rebalance, the final equity is exactly `start_value × price_end / price_start`. For 100% `$`, the final equity is exactly `start_value × (1 + CASH_APY)^(years)`.

Use this pattern when:
- A regime is forced into a single asset and the allocation is sticky.
- The synthetic data is deterministic (your fixture, not real yfinance).

Tests B05–B08, F19, F20, G21–G24 are this pattern.

### 4. **Monotonicity / sensitivity** (the "knob X must move outcome Y in direction Z")

Run a sweep across a config knob and assert the output is monotonic in the expected direction. The user's specific concern — "R1 dd_high 6% / 8% / 10% must give monotonically worse max DD when R2 = 100% cash" — is exactly this pattern.

Use this pattern when a knob *must* change something but a bug could mask the change. The bug we found made R1 width insensitive to DD because each rebalance "snapped back" today's loss.

Test D13 is this pattern. Note: design the synthetic price panel carefully — flat segment-boundary days will collapse multiple thresholds onto the same rebalance day and defeat the monotonicity test. The deep_dd_panel fixture uses a strictly monotonic decline to avoid this.

### 5. **Real-data sanity** (the "this still works on actual yfinance data")

Run with `FMF_DATA_MODE=frozen` and `data_loader.load_price_data()` to pull real cached TQQQ / XLU / SPY history, force a single-asset config, and assert the equity curve matches buy-and-hold. This catches data-pipeline / cache-loading regressions that synthetic-only tests would miss.

Tests G22–G24 are this pattern.

---

## How to write a new test

1. **Pick the pattern**. Most new compounding tests fit pattern 1 (paired equivalence) or pattern 3 (closed-form). If you're adding a sensitivity test, use pattern 4. For invariants, pattern 2.

2. **Choose synthetic vs real data**:
   - **Synthetic** (deterministic, fast, exact assertions): use `small_panel` or `deep_dd_panel` fixtures. **Pair with `isolate_from_yfinance`** — without it, run_backtest pulls real QQQ/SPY/VIX history and the merge contaminates synthetic dd. The fixture monkey-patches `cached_yf_download` to return empty.
   - **Real cached data** (slower, looser assertions): use `data_loader.load_price_data()` with `FMF_DATA_MODE=frozen`. Don't use `isolate_from_yfinance` — you want the cache.

3. **Use date 1995-01-02 onward for synthetic panels**. QQQ inception is 1999-03-10. Earlier dates avoid the cached-yfinance merge polluting the dd signal even when you forget the isolation fixture.

4. **Always include `$` in the panel**. The production allocation_tickers list includes `"$"`, so omitting it triggers a column-key error. Use `_add_cash_column(panel)` (it attaches `data_loader._build_cash_series` to the index).

5. **Avoid flat segment-boundary days in synthetic price panels**. `np.concatenate([np.linspace(100, 90, 5), np.linspace(90, 70, 10)])` repeats price 90 across the boundary. With T+1 lag and discrete dd_high thresholds, multiple thresholds collapse onto the same rebalance day. Use a single monotonic linspace, or `np.linspace(100, 65, 18, endpoint=True)[:-1]` then continue.

6. **Match override panel weights to regime base when testing identity**. Easy mistake: forgetting that `_simple_config` puts leftover weight into `$`. Verify the panels you compare actually have identical weights.

7. **Tolerance discipline**:
   - Paired-config equivalence: `rel=1e-9`.
   - Algebraic invariant: `rel=5e-6` (some float roundoff in share computations).
   - Real-data buy-and-hold ratio: `rel=2e-3` (T+1 lag + initial-allocation roundoff).
   - APY compounding: `rel=5e-3` (calendar-day vs trading-day mismatch).

---

## Files & helpers

| File | Purpose |
|---|---|
| `tests/test_compounding_correctness.py` | All 24 compounding tests. Read first. |
| `tests/conftest.py` | `price_fixture` (5y synthetic, seed=42), `minimal_config`, `production_per_regime_config`. |
| `_simple_config(...)` | 2-regime config builder local to the file. Defaults to `$` for leftover weight. |
| `_add_cash_column(panel)` | Attaches the synthetic `$` sleeve to a price panel. |
| `_run(cfg, panel)` | Thin wrapper around `run_backtest` with default args. |
| `_equity_close(eq_a, eq_b, rel=...)` | Compare two equity curves by Date+Value. |
| `_mark_invariant(eq, panel, allocation_tickers)` | Pattern-2 helper for the mark invariant. |
| `isolate_from_yfinance` (fixture) | Monkey-patches `cached_yf_download` to return empty. Use on synthetic-data tests. |

---

## What a compounding bug actually looks like

The fix in commit history (2026-04-28): added 6 lines at the top of the daily loop in `backtest.py` (after `skip_day` check, before dividend handling) that mark `portfolio_value` to today's prices. Before the fix, the rebalance blocks at `backtest.py:1078` and `backtest.py:1132` started from yesterday's close NAV, so:

```text
day N before rebalance:  shares × today_prices = $11,000   (today's mark, NOT used)
day N rebalance basis:   yesterday_close_NAV   = $10,000   (used by rebalance_fn)
day N after rebalance:   sum(new_shares × today_prices) = $10,000   (today's $1,000 gain destroyed)
day N+1 onward:          compounds from $10,000 instead of $11,000
```

That single 1000-dollar miss compounded across 25 years of backtests turned 30%+ CAGR into 28% and made max-DD numbers underestimate the true loss because rebalance-into-cash on a crash day "snapped back" to the prior NAV.

**Symptom checklist for the next bug like this**:
- Override panel weights match base, but enabling/disabling shifts the equity curve.
- Max-DD is identical (or near-identical) across structurally-different configs that should produce different DDs.
- A buy-and-hold single-asset config doesn't match `start × price_ratio`.
- The mark invariant fails on rebalance days but holds on non-rebalance days.

When you see any of those, write a paired-config or invariant test that pins the symptom, *then* fix.
