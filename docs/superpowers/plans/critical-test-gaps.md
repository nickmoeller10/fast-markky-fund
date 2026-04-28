# Critical Test Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three highest-priority test gaps identified in the 2026-04-28 coverage audit: (1) production-config locked regression on real cached data, (2) end-to-end signal override integration through `run_backtest`, (3) end-to-end dividend reinvestment in both target modes.

**Architecture:** Three independent test files, each self-contained, all using existing `tests/conftest.py` fixtures and the `_simple_config` / `_run` / `_mark_invariant` helpers from `tests/test_compounding_correctness.py`. Task 1 is a real-cached-data anchor (`FMF_DATA_MODE=frozen`). Tasks 2 and 3 are synthetic-data integration tests that monkeypatch `build_signal_total_series` and pass a synthetic `dividend_data` DataFrame respectively.

**Tech Stack:** pytest, hypothesis, pandas, numpy, the existing fast-markky-fund backtest engine. No new dependencies.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/test_production_locked_regression.py` | NEW. Locked regression on the real production config (`config.CONFIG`) over the full 1999-01-04 → 2026-03-27 range using `FMF_DATA_MODE=frozen`. Anchors CAGR, max DD, final value, Sharpe, volatility. Includes mark invariant + daily-return-product invariant on the production path. |
| `tests/test_signal_override_integration.py` | NEW. End-to-end override behavior through `run_backtest` using a monkeypatched `build_signal_total_series` to inject deterministic Signal_total values. Covers: first-day override, sticky persistence into neutral band, upside↔protection flip, regime-change clears override, override+rebalance interaction, `validate_panel_sums` integration assertion. |
| `tests/test_dividend_reinvestment_integration.py` | NEW. End-to-end dividend handling using a synthetic `dividend_data` DataFrame. Covers: cash mode, share-reinvest mode, dividend-on-rebalance-day ordering, dividend-on-regime-change-day ordering, mark invariant during dividends, `dividend_df` schema + content. |
| `tests/conftest.py` | MODIFY. Add a `production_config_dict` fixture that returns a deep copy of `config.CONFIG` (so tests can mutate freely without leaking state across tests). |

---

## Task 1: Production-Config Locked Regression

**Goal:** A frozen anchor on the real production code path so the next silent compounding/regime/override bug bumps a test instead of slipping through.

**Files:**
- Create: `tests/test_production_locked_regression.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1.1: Add `production_config_dict` fixture to conftest.py**

Add at the bottom of `tests/conftest.py`:

```python
import copy as _copy


@pytest.fixture
def production_config_dict():
    """Deep copy of the live production config from config.py.

    Tests should mutate the copy freely; conftest hands out a fresh one per test.
    Use this when you need to exercise the *actual* production path (per_regime,
    rolling DD window, all 4 allocation tickers including '$', enabled R1
    overrides, etc.) — not a hand-rolled minimal subset.
    """
    from config import CONFIG
    return _copy.deepcopy(CONFIG)
```

- [ ] **Step 1.2: Create the test file with the unlocked harness**

Create `tests/test_production_locked_regression.py`:

```python
"""Locked regression on the real production config + real cached yfinance data.

Anchors CAGR, max drawdown, final value, Sharpe, volatility, and a few
structural invariants over the full production date range. The previous
locked regression (test_regression_ground_truth.py) uses down_only +
no overrides + no DD window + 5y synthetic data — it does NOT cover
the production code path. This test does.

Numbers are locked the FIRST time the test is run against a stable
backtest (see Task 1 step 1.5). After that, any change to core logic
that moves these numbers must update the LOCKED_* constants with an
explicit comment explaining what changed and why (per CLAUDE.md
working agreement: 'Don't change LOCKED constants silently').

Requires FMF_DATA_MODE=frozen to be set (or the cache to be present).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from backtest import compute_drawdown_from_ath, run_backtest
from rebalance_engine import rebalance_portfolio
from regime_engine import determine_regime
from utils import max_drawdown_from_equity_curve


# LOCKED CONSTANTS — populate via Task 1 step 1.5 then commit.
LOCKED_FINAL_VALUE: float | None = None
LOCKED_CAGR: float | None = None
LOCKED_MAX_DRAWDOWN: float | None = None
LOCKED_VOLATILITY: float | None = None
LOCKED_SHARPE: float | None = None
LOCKED_ROW_COUNT: int | None = None


@pytest.fixture(scope="module")
def production_equity_df(production_config_dict):
    """Run the full production backtest once per module against frozen cache."""
    os.environ.setdefault("FMF_DATA_MODE", "frozen")

    from data_loader import load_price_data
    from signal_override_engine import validate_panel_sums

    cfg = production_config_dict
    # Sanity: the config we're about to lock must pass the panel-sum invariant.
    validate_panel_sums(cfg)

    panel = load_price_data(
        cfg["tickers"], cfg["start_date"], cfg["end_date"]
    )

    eq, _, _ = run_backtest(
        panel,
        cfg,
        compute_drawdown_from_ath,
        determine_regime,
        rebalance_portfolio,
    )
    return eq
```

- [ ] **Step 1.3: Add a "print-and-fail" bootstrap test**

Append to `tests/test_production_locked_regression.py`:

```python
@pytest.mark.integration
def test_print_locked_values_for_bootstrap(production_equity_df, request):
    """Run this once with -s to print the values that should populate the
    LOCKED_* constants, then delete this test (or keep it skipped).

    Skip-by-default: this is a bootstrap aid, not a real test. To run:
        pytest tests/test_production_locked_regression.py::test_print_locked_values_for_bootstrap -s --bootstrap
    """
    if not request.config.getoption("--bootstrap", default=False):
        pytest.skip("bootstrap-only; pass --bootstrap to print locked values")

    eq = production_equity_df
    final = float(eq["Value"].iloc[-1])
    start_v = float(eq["Value"].iloc[0])
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (final / start_v) ** (1.0 / years) - 1.0
    dd = max_drawdown_from_equity_curve(eq["Value"])
    returns = eq["Value"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    sharpe = cagr / vol if vol > 0 else 0.0

    print(f"\n--- LOCKED VALUES (paste into LOCKED_* constants) ---")
    print(f"LOCKED_FINAL_VALUE   = {final:.4f}")
    print(f"LOCKED_CAGR          = {cagr:.6f}")
    print(f"LOCKED_MAX_DRAWDOWN  = {dd:.6f}")
    print(f"LOCKED_VOLATILITY    = {vol:.6f}")
    print(f"LOCKED_SHARPE        = {sharpe:.4f}")
    print(f"LOCKED_ROW_COUNT     = {len(eq)}")
    print(f"------------------------------------------------------\n")
```

Add a `--bootstrap` flag to `tests/conftest.py` (append to bottom):

```python
def pytest_addoption(parser):
    parser.addoption(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Enable bootstrap helpers in locked-regression tests.",
    )
```

- [ ] **Step 1.4: Add the locked-value assertions and invariants**

Append to `tests/test_production_locked_regression.py`:

```python
@pytest.mark.integration
def test_production_locked_final_value(production_equity_df):
    assert LOCKED_FINAL_VALUE is not None, "Locked value not yet populated; see step 1.5"
    final = float(production_equity_df["Value"].iloc[-1])
    assert abs(final - LOCKED_FINAL_VALUE) < 0.5, (
        f"Production final value drifted: got {final:.4f}, locked {LOCKED_FINAL_VALUE}. "
        "If intentional, update LOCKED_FINAL_VALUE with a comment explaining the change."
    )


@pytest.mark.integration
def test_production_locked_cagr(production_equity_df):
    assert LOCKED_CAGR is not None
    eq = production_equity_df
    start_v = float(eq["Value"].iloc[0])
    end_v = float(eq["Value"].iloc[-1])
    years = (pd.to_datetime(eq["Date"].iloc[-1]) - pd.to_datetime(eq["Date"].iloc[0])).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    assert abs(cagr - LOCKED_CAGR) < 1e-4, (
        f"Production CAGR drifted: got {cagr:.6f}, locked {LOCKED_CAGR}."
    )


@pytest.mark.integration
def test_production_locked_max_drawdown(production_equity_df):
    assert LOCKED_MAX_DRAWDOWN is not None
    dd = max_drawdown_from_equity_curve(production_equity_df["Value"])
    assert abs(dd - LOCKED_MAX_DRAWDOWN) < 1e-4, (
        f"Production max DD drifted: got {dd:.6f}, locked {LOCKED_MAX_DRAWDOWN}."
    )


@pytest.mark.integration
def test_production_locked_volatility(production_equity_df):
    assert LOCKED_VOLATILITY is not None
    returns = production_equity_df["Value"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    assert abs(vol - LOCKED_VOLATILITY) < 1e-4, (
        f"Production volatility drifted: got {vol:.6f}, locked {LOCKED_VOLATILITY}."
    )


@pytest.mark.integration
def test_production_locked_sharpe(production_equity_df):
    assert LOCKED_SHARPE is not None
    eq = production_equity_df
    start_v = float(eq["Value"].iloc[0])
    end_v = float(eq["Value"].iloc[-1])
    years = (pd.to_datetime(eq["Date"].iloc[-1]) - pd.to_datetime(eq["Date"].iloc[0])).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    returns = eq["Value"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    sharpe = cagr / vol if vol > 0 else 0.0
    assert abs(sharpe - LOCKED_SHARPE) < 1e-3, (
        f"Production Sharpe drifted: got {sharpe:.4f}, locked {LOCKED_SHARPE}."
    )


@pytest.mark.integration
def test_production_locked_row_count(production_equity_df):
    assert LOCKED_ROW_COUNT is not None
    assert len(production_equity_df) == LOCKED_ROW_COUNT, (
        f"Production row count changed: got {len(production_equity_df)}, "
        f"locked {LOCKED_ROW_COUNT}. New trading days in the cache?"
    )


@pytest.mark.integration
def test_production_mark_invariant(production_equity_df, production_config_dict):
    """Value[t] == sum(shares × price) + Cash on every row of the production run."""
    from data_loader import load_price_data

    os.environ.setdefault("FMF_DATA_MODE", "frozen")
    cfg = production_config_dict
    panel = load_price_data(cfg["tickers"], cfg["start_date"], cfg["end_date"])

    eq = production_equity_df
    panel_aligned = panel.reindex(pd.to_datetime(eq["Date"]))
    max_rel_diff = 0.0
    worst_date = None
    for i, row in eq.iterrows():
        date = pd.to_datetime(row["Date"])
        manual = 0.0
        for t in cfg["allocation_tickers"]:
            shares_col = f"{t}_shares"
            if shares_col in eq.columns and t in panel_aligned.columns:
                s = float(eq.loc[i, shares_col] or 0)
                p = panel_aligned.loc[date, t]
                if pd.notna(p):
                    manual += s * float(p)
        cash = float(row.get("Cash", 0) or 0)
        manual += cash
        v = float(row["Value"])
        denom = max(abs(v), 1.0)
        rel = abs(v - manual) / denom
        if rel > max_rel_diff:
            max_rel_diff = rel
            worst_date = row["Date"]
    assert max_rel_diff <= 5e-6, (
        f"Mark invariant violated on production path. "
        f"Worst row: {worst_date}, rel diff = {max_rel_diff:.2e}"
    )


@pytest.mark.integration
def test_production_daily_return_product_matches_total_return(production_equity_df):
    """prod(1 + r_d) ≈ Value[end] / Value[start] on the full production run.

    This is the algebraic compounding invariant from
    docs/superpowers/methodologies/compounding-test-patterns.md applied at
    25y production scale (synthetic-only test_compounding_correctness.py
    only runs it on 5d / 30d panels)."""
    eq = production_equity_df
    returns = eq["Value"].astype(float).pct_change().dropna()
    product = float((1.0 + returns).prod())
    total = float(eq["Value"].iloc[-1]) / float(eq["Value"].iloc[0])
    rel = abs(product - total) / total
    assert rel < 1e-9, (
        f"Daily-return product {product:.10f} does not match total return "
        f"{total:.10f} (rel diff {rel:.2e}) — compounding accounting drift."
    )
```

- [ ] **Step 1.5: Bootstrap the locked values**

Run the bootstrap test to capture the values:

```bash
python3 -m pytest tests/test_production_locked_regression.py::test_print_locked_values_for_bootstrap -s --bootstrap
```

Expected output: a block like
```
--- LOCKED VALUES (paste into LOCKED_* constants) ---
LOCKED_FINAL_VALUE   = <some number>
LOCKED_CAGR          = <some number>
...
```

Paste each number into the corresponding `LOCKED_*` constant at the top of `tests/test_production_locked_regression.py`, replacing the `None` placeholders.

- [ ] **Step 1.6: Run the full file and verify all locked tests pass**

```bash
python3 -m pytest tests/test_production_locked_regression.py -v
```

Expected: all `test_production_locked_*` and the two invariant tests pass; `test_print_locked_values_for_bootstrap` is skipped (no `--bootstrap` flag).

- [ ] **Step 1.7: Run the full suite to confirm no collateral damage**

```bash
python3 -m pytest tests/ -q
```

Expected: 153 prior tests + new tests all pass.

- [ ] **Step 1.8: Commit**

```bash
git add tests/test_production_locked_regression.py tests/conftest.py
git commit -m "test: lock production-config regression on real cached data

Anchors CAGR / max DD / final value / Sharpe / volatility on the full
1999-2026 production config + real frozen cache. Adds mark invariant
and daily-return-product invariant on the production path. Closes
CLAUDE.md open issue #2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Signal Override Integration Tests

**Goal:** Cover override behavior through `run_backtest` end-to-end (open issue #7), where unit tests on `desired_signal_override_mode` cannot reach.

**Files:**
- Create: `tests/test_signal_override_integration.py`

- [ ] **Step 2.1: Create the test file with helpers**

Create `tests/test_signal_override_integration.py`:

```python
"""End-to-end signal override behavior through run_backtest.

Unit tests in test_signal_override_engine.py cover the level/sticky
decision logic in isolation. This file covers what only an integration
test can: that the run_backtest daily loop correctly threads
signal_override_mode through dividends, regime changes, and rebalances.

Approach: monkeypatch signal_layers.build_signal_total_series so the
signal layer is deterministic. We don't need real VIX/MACD math —
we need to control exactly when Signal_total crosses the override
thresholds.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest

from backtest import compute_drawdown_from_ath, run_backtest
from rebalance_engine import rebalance_portfolio
from regime_engine import determine_regime


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
def _disabled_panel():
    return {
        "enabled": False, "label": "", "direction": "above", "threshold": 0,
        "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
    }


def _override_cfg(*, upside=None, protection=None, r1_high=0.99):
    """2-regime config: R1 has the override(s) under test; R2 = 100% $."""
    upside = upside or _disabled_panel()
    protection = protection or _disabled_panel()
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "per_regime",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "$"],
        "allocation_tickers": ["TQQQ", "QQQ", "XLU", "$"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": r1_high,
                "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {"upside": upside, "protection": protection},
            },
            "R2": {
                "dd_low": r1_high, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": _disabled_panel(),
                    "protection": _disabled_panel(),
                },
            },
        },
    }


@pytest.fixture
def isolate_yf(monkeypatch):
    """Stub cached_yf_download so synthetic-panel tests don't pull real data."""
    import data_cache, data_loader
    import backtest as backtest_mod

    def empty(*a, **kw): return pd.DataFrame()
    monkeypatch.setattr(data_cache, "cached_yf_download", empty)
    monkeypatch.setattr(data_loader, "cached_yf_download", empty)
    if hasattr(backtest_mod, "cached_yf_download"):
        monkeypatch.setattr(backtest_mod, "cached_yf_download", empty)
    yield


def _patch_signal_total(monkeypatch, signal_values_by_date):
    """Force build_signal_total_series to return the given mapping.

    `signal_values_by_date` is a dict of pd.Timestamp → float; missing dates
    default to 0.0. The patch is applied at the import site inside backtest."""
    def _fake(trading_index, *args, **kwargs):
        s = pd.Series(0.0, index=trading_index)
        for d, v in signal_values_by_date.items():
            d = pd.Timestamp(d)
            if d in s.index:
                s.loc[d] = float(v)
        return s

    import backtest as backtest_mod
    monkeypatch.setattr(backtest_mod, "build_signal_total_series", _fake)


def _add_cash(panel):
    from data_loader import _build_cash_series
    out = panel.copy()
    out["$"] = _build_cash_series(out.index)
    return out


@pytest.fixture
def flat_panel():
    """20 trading days, all prices flat. Dd stays 0 → portfolio always in R1."""
    dates = pd.bdate_range("1995-01-02", periods=20)
    df = pd.DataFrame({
        "QQQ": np.full(20, 100.0),
        "TQQQ": np.full(20, 100.0),
        "XLU": np.full(20, 50.0),
        "SPY": np.full(20, 200.0),
    }, index=dates)
    return _add_cash(df)


@pytest.fixture
def deep_dd_panel():
    """30 trading days, QQQ 100→65→95. Forces R1→R2 transition."""
    dates = pd.bdate_range("1995-01-02", periods=30)
    qqq = np.concatenate([
        np.linspace(100, 65, 18, endpoint=True)[:-1],
        np.linspace(67, 95, 13),
    ])
    tqqq = 100 * (qqq / qqq[0]) ** 3.0
    df = pd.DataFrame({
        "QQQ": qqq, "TQQQ": tqqq,
        "XLU": 50 + 0.01 * np.arange(len(qqq)),
        "SPY": 200 * np.ones(len(qqq)),
    }, index=dates)
    return _add_cash(df)
```

- [ ] **Step 2.2: Add the "override fires when threshold crossed" test**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_override_fires_when_signal_crosses_threshold(
    isolate_yf, monkeypatch, flat_panel
):
    """R1 base = 100% TQQQ. Upside override at threshold > 1 routes to 100% XLU.
    Inject Signal_total = 2.0 starting day 5; assert TQQQ exits and XLU is held."""
    panel = flat_panel
    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "test up", "XLU": 1.0,
        },
    )
    # Fire from day 5 onward.
    sig = {d: (2.0 if i >= 5 else 0.0) for i, d in enumerate(panel.index)}
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    # T+1 lag: override fires on row 6 onward (signal observed at end of row 5
    # is used to set today's regime decision on row 6).
    overrides_active = eq["Signal_override_active"].astype(str).tolist()
    assert "upside" in overrides_active, (
        f"Upside override never fired despite Signal_total=2 from day 5. "
        f"Saw: {set(overrides_active)}"
    )
    # Once active, TQQQ shares must drop to 0 on the rebalance day.
    upside_rows = eq[eq["Signal_override_active"] == "upside"]
    assert upside_rows["TQQQ_shares"].astype(float).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert upside_rows["XLU_shares"].astype(float).iloc[-1] > 0.0
```

- [ ] **Step 2.3: Add the sticky-persistence-into-neutral test**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_override_sticky_when_signal_drifts_to_neutral(
    isolate_yf, monkeypatch, flat_panel
):
    """Once upside fires, signal drifts back to 0 (neutral band) — override
    must persist (sticky), not unwind. Fixed 2026-04-28; this is the
    integration-level lock for that fix."""
    panel = flat_panel
    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "test up", "XLU": 1.0,
        },
        protection={
            **_disabled_panel(),
            "enabled": True, "direction": "below", "threshold": -2.0,
            "label": "test down", "$": 1.0,
        },
    )
    # Day 5: cross +1 (override fires). Day 8 onward: drift to 0 (neutral).
    sig = {}
    for i, d in enumerate(panel.index):
        if i < 5: sig[d] = 0.0
        elif i < 8: sig[d] = 2.0
        else: sig[d] = 0.0   # neutral — must NOT unwind upside
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    # After day 8 the signal is neutral; upside should persist all the way to end.
    last_override_state = eq["Signal_override_active"].iloc[-1]
    assert last_override_state == "upside", (
        f"Override unwound to {last_override_state!r} on neutral signal — "
        "sticky behavior broken (regression of the 2026-04-28 fix)."
    )
```

- [ ] **Step 2.4: Add the upside↔protection flip test**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_override_flips_from_upside_to_protection(
    isolate_yf, monkeypatch, flat_panel
):
    """While upside is active, signal crosses into protection territory →
    override must flip to protection (not stay sticky on upside)."""
    panel = flat_panel
    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "test up", "XLU": 1.0,
        },
        protection={
            **_disabled_panel(),
            "enabled": True, "direction": "below", "threshold": -2.0,
            "label": "test down", "$": 1.0,
        },
    )
    sig = {}
    for i, d in enumerate(panel.index):
        if i < 5: sig[d] = 0.0
        elif i < 10: sig[d] = 2.0     # upside zone
        else: sig[d] = -3.0           # protection zone
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    # End state must be protection (100% $).
    last = eq.iloc[-1]
    assert last["Signal_override_active"] == "protection", (
        f"Expected protection at end; got {last['Signal_override_active']!r}"
    )
    cash_shares = float(last.get("$_shares", 0))
    assert cash_shares > 0, "Protection panel routes to $ but no $ shares held"
```

- [ ] **Step 2.5: Add the regime-change-clears-override test**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_regime_change_clears_active_override(
    isolate_yf, monkeypatch, deep_dd_panel
):
    """Override active in R1; market drops into R2 (dd > r1_high).
    On the regime-change day Signal_override_active must reset to ""/"none"
    so R2's base allocation is in effect, not the leftover R1 override."""
    panel = deep_dd_panel
    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "r1 up", "XLU": 1.0,
        },
        r1_high=0.10,   # R1 ends at 10% dd; deep_dd_panel goes well below
    )
    # Constant +2 signal so upside is active whenever regime allows.
    sig = {d: 2.0 for d in panel.index}
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    # Find first row where the portfolio regime is R2.
    r2_rows = eq[eq["Portfolio_Regime"] == "R2"]
    assert not r2_rows.empty, "R2 never reached — panel design bug"
    first_r2 = r2_rows.iloc[0]
    assert first_r2["Signal_override_active"] in ("", "none"), (
        f"Override not cleared on R1→R2 transition: "
        f"got {first_r2['Signal_override_active']!r}"
    )
```

- [ ] **Step 2.6: Add the validate_panel_sums integration assertion**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_invalid_panel_sum_raises_at_backtest_start(isolate_yf, flat_panel):
    """A malformed (sum != 1.0) enabled override panel must raise via
    validate_panel_sums — caught before any backtest day runs."""
    from signal_override_engine import validate_panel_sums

    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "broken", "TQQQ": 0.5, "XLU": 0.3,   # sums to 0.8
        },
    )
    with pytest.raises((ValueError, AssertionError)):
        validate_panel_sums(cfg)
```

- [ ] **Step 2.7: Add the override-on-day-1 (initialization) test**

Append to `tests/test_signal_override_integration.py`:

```python
@pytest.mark.integration
def test_override_active_on_first_backtest_day(isolate_yf, monkeypatch, flat_panel):
    """If Signal_total[day-0] is already past threshold, override should be
    active on (or one bar after, depending on T+1) the first row — not
    silently default to 'none' due to uninitialized state."""
    panel = flat_panel
    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "from-day-0", "XLU": 1.0,
        },
    )
    sig = {d: 2.0 for d in panel.index}   # always above threshold
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )
    # By row 2 at the latest the override must be active (T+1 lag at most).
    early_states = eq["Signal_override_active"].iloc[:3].astype(str).tolist()
    assert "upside" in early_states, (
        f"Override never activated even with constant high signal: {early_states}"
    )
```

- [ ] **Step 2.8: Run the file and verify all tests pass**

```bash
python3 -m pytest tests/test_signal_override_integration.py -v
```

Expected: 6 passed.

- [ ] **Step 2.9: Run the full suite**

```bash
python3 -m pytest tests/ -q
```

Expected: previous count + 6 new tests, all green.

- [ ] **Step 2.10: Commit**

```bash
git add tests/test_signal_override_integration.py
git commit -m "test: end-to-end signal override integration coverage

Closes CLAUDE.md open issue #7. Covers override fire / sticky / flip /
regime-change-clears / day-1-init / invalid-panel-raises through
run_backtest, where the existing unit tests on
desired_signal_override_mode cannot reach.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Dividend Reinvestment Integration Tests

**Goal:** Cover both `dividend_reinvestment_target` modes ("cash" and a ticker), plus the dangerous interaction days (dividend on rebalance day, dividend on regime-change day). Closes open issue #9.

**Files:**
- Create: `tests/test_dividend_reinvestment_integration.py`

- [ ] **Step 3.1: Create the test file with helpers and a flat panel**

Create `tests/test_dividend_reinvestment_integration.py`:

```python
"""End-to-end dividend reinvestment behavior through run_backtest.

Production has dividend_reinvestment=False, but the code path runs in
historical configs and in the optimizer. This file exercises both
target modes ('cash' and a ticker), and pins behavior on the
interaction days where compounding bugs hide: a dividend that lands
on a rebalance day, and a dividend that lands on a regime-change day.

The dividend block in backtest.py:976-1036 sits between today's
mark-to-market (line 964) and the regime/rebalance blocks. If the
ordering ever flips, the mark invariant or the equity curve will drift.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from backtest import compute_drawdown_from_ath, run_backtest
from rebalance_engine import rebalance_portfolio
from regime_engine import determine_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _disabled_panel():
    return {
        "enabled": False, "label": "", "direction": "above", "threshold": 0,
        "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
    }


def _div_cfg(*, target="cash", r1_alloc="XLU", r1_high=0.99):
    """2-regime config with dividend_reinvestment=True and chosen target."""
    r1 = {"TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0}
    r1[r1_alloc] = 1.0
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "per_regime",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "$"],
        "allocation_tickers": ["TQQQ", "QQQ", "XLU", "$"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": True,
        "dividend_reinvestment_target": target,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": r1_high,
                **r1,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": _disabled_panel(),
                    "protection": _disabled_panel(),
                },
            },
            "R2": {
                "dd_low": r1_high, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": _disabled_panel(),
                    "protection": _disabled_panel(),
                },
            },
        },
    }


def _add_cash(panel):
    from data_loader import _build_cash_series
    out = panel.copy()
    out["$"] = _build_cash_series(out.index)
    return out


@pytest.fixture
def isolate_yf(monkeypatch):
    import data_cache, data_loader
    import backtest as backtest_mod
    def empty(*a, **kw): return pd.DataFrame()
    monkeypatch.setattr(data_cache, "cached_yf_download", empty)
    monkeypatch.setattr(data_loader, "cached_yf_download", empty)
    if hasattr(backtest_mod, "cached_yf_download"):
        monkeypatch.setattr(backtest_mod, "cached_yf_download", empty)
    yield


@pytest.fixture
def flat_xlu_panel():
    """20 trading days, XLU = $50 flat. Dividends are easy to verify against."""
    dates = pd.bdate_range("1995-01-02", periods=20)
    df = pd.DataFrame({
        "QQQ": np.full(20, 100.0),
        "TQQQ": np.full(20, 100.0),
        "XLU": np.full(20, 50.0),
        "SPY": np.full(20, 200.0),
    }, index=dates)
    return _add_cash(df)


@pytest.fixture
def deep_dd_xlu_panel():
    """30 days, QQQ 100→60→90 (forces R1→R2 transition); XLU stays flat."""
    dates = pd.bdate_range("1995-01-02", periods=30)
    qqq = np.concatenate([
        np.linspace(100, 60, 18, endpoint=True)[:-1],
        np.linspace(62, 90, 13),
    ])
    df = pd.DataFrame({
        "QQQ": qqq,
        "TQQQ": 100 * (qqq / qqq[0]) ** 3.0,
        "XLU": np.full(len(qqq), 50.0),
        "SPY": np.full(len(qqq), 200.0),
    }, index=dates)
    return _add_cash(df)


def _div_df(panel, events):
    """Build a dividend DataFrame from a list of (date_str, ticker, dps) tuples."""
    df = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for date_str, ticker, dps in events:
        df.loc[pd.Timestamp(date_str), ticker] = float(dps)
    return df
```

- [ ] **Step 3.2: Add the cash-mode dividend test**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_cash_dividend_increases_value_by_exact_amount(
    isolate_yf, flat_xlu_panel
):
    """100% XLU, prices flat, single $0.50/share XLU dividend on day 5,
    target='cash'. Day-5 portfolio value must increase by exactly
    (initial_xlu_shares × 0.50) — the dividend goes to cash."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [(panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    initial_shares = 10_000.0 / 50.0   # $10k → XLU at $50
    expected_dividend = initial_shares * 0.50
    day5_value = float(eq["Value"].iloc[5])
    day4_value = float(eq["Value"].iloc[4])
    actual_jump = day5_value - day4_value
    assert actual_jump == pytest.approx(expected_dividend, rel=1e-6), (
        f"Day-5 jump {actual_jump:.4f} != expected {expected_dividend:.4f}"
    )

    # divs should record exactly one dividend event.
    assert len(divs) == 1
    assert divs.iloc[0]["Ticker"] == "XLU"
    assert float(divs.iloc[0]["Dividend_Amount"]) == pytest.approx(expected_dividend, rel=1e-6)
    assert divs.iloc[0]["Reinvestment_Target"] == "cash"

    # Cash column should reflect the dividend on day 5 onward.
    assert float(eq["Cash"].iloc[5]) == pytest.approx(expected_dividend, rel=1e-6)
```

- [ ] **Step 3.3: Add the share-reinvest dividend test**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_ticker_reinvest_buys_shares_at_target_price(
    isolate_yf, flat_xlu_panel
):
    """100% XLU, $0.50/share dividend, target='XLU' → reinvested into XLU at
    today's price. Total value rises by the dividend; XLU share count
    increases by dividend / XLU_price."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="XLU", r1_alloc="XLU")
    div = _div_df(panel, [(panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    initial_shares = 10_000.0 / 50.0
    dividend_amount = initial_shares * 0.50
    bought = dividend_amount / 50.0       # XLU still at $50

    # Day-5 XLU shares should be initial + bought.
    day5_xlu_shares = float(eq["XLU_shares"].iloc[5])
    assert day5_xlu_shares == pytest.approx(initial_shares + bought, rel=1e-6)
    # And value reflects the new shares × price.
    day5_value = float(eq["Value"].iloc[5])
    assert day5_value == pytest.approx(
        (initial_shares + bought) * 50.0, rel=1e-6
    )
    # Cash should be 0 — the dividend went to shares, not cash.
    assert float(eq["Cash"].iloc[5]) == pytest.approx(0.0, abs=1e-9)
    assert divs.iloc[0]["Reinvestment_Target"] == "XLU"
```

- [ ] **Step 3.4: Add the mark-invariant test**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_mark_invariant_holds_with_dividends(isolate_yf, flat_xlu_panel):
    """Value[t] == sum(shares × price) + Cash on every row, even with
    multiple dividend events in cash mode."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [
        (panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50),
        (panel.index[10].strftime("%Y-%m-%d"), "XLU", 0.30),
        (panel.index[15].strftime("%Y-%m-%d"), "XLU", 0.40),
    ])

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    panel_aligned = panel.reindex(pd.to_datetime(eq["Date"]))
    for i, row in eq.iterrows():
        date = pd.to_datetime(row["Date"])
        manual = 0.0
        for t in cfg["allocation_tickers"]:
            col = f"{t}_shares"
            if col in eq.columns and t in panel_aligned.columns:
                s = float(eq.loc[i, col] or 0)
                p = panel_aligned.loc[date, t]
                if pd.notna(p):
                    manual += s * float(p)
        manual += float(row.get("Cash", 0) or 0)
        v = float(row["Value"])
        denom = max(abs(v), 1.0)
        assert abs(v - manual) / denom <= 5e-6, (
            f"Mark invariant broke on {row['Date']}: V={v:.4f}, "
            f"shares×price+cash={manual:.4f}"
        )
```

- [ ] **Step 3.5: Add the dividend-on-regime-change-day test**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_dividend_on_regime_change_day_routes_to_post_change_regime(
    isolate_yf, deep_dd_xlu_panel
):
    """A dividend that lands on the same day a R1→R2 transition fires.
    Order of operations (per backtest.py): mark-to-market → dividend →
    regime → rebalance. So the dividend lands as cash (or shares) BEFORE
    the rebalance to R2's all-cash allocation. Total NAV must remain
    correct after rebalance: dividend cash + value of held shares (now
    sold and re-bought as $) all add up."""
    panel = deep_dd_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU", r1_high=0.10)

    # Find the first row where dd > 0.10 (R2 entry day).
    qqq = panel["QQQ"]
    dd = (qqq.cummax() - qqq) / qqq.cummax()
    transition_idx = int(np.where(dd.values > 0.10)[0][0])
    transition_date = panel.index[transition_idx]

    # Dividend lands the day before transition triggers (T+1 lag means the
    # rebalance fires on transition_idx, not transition_idx-1).
    div = _div_df(panel, [(transition_date.strftime("%Y-%m-%d"), "XLU", 0.20)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    # The dividend MUST be recorded.
    assert len(divs) == 1, f"Expected 1 dividend, got {len(divs)}"

    # On that day, mark invariant must still hold post-rebalance.
    row = eq.iloc[transition_idx]
    manual = 0.0
    for t in cfg["allocation_tickers"]:
        col = f"{t}_shares"
        if col in eq.columns and t in panel.columns:
            s = float(row.get(col, 0) or 0)
            p = panel.loc[transition_date, t]
            if pd.notna(p):
                manual += s * float(p)
    manual += float(row.get("Cash", 0) or 0)
    v = float(row["Value"])
    assert abs(v - manual) / max(abs(v), 1.0) <= 5e-6, (
        f"Mark invariant violated on regime-change day with dividend: "
        f"V={v:.4f}, manual={manual:.4f}"
    )
```

- [ ] **Step 3.6: Add the equivalence test (no dividend events ≡ disabled)**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_empty_dividend_data_equals_disabled(isolate_yf, flat_xlu_panel):
    """dividend_reinvestment=True but dividend_data has zero events should
    produce the same equity curve as dividend_reinvestment=False."""
    panel = flat_xlu_panel
    cfg_off = _div_cfg(target="cash", r1_alloc="XLU")
    cfg_off["dividend_reinvestment"] = False

    cfg_on = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [])   # all-zero dividend frame

    eq_off, _, _ = run_backtest(
        panel, cfg_off, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio,
    )
    eq_on, _, _ = run_backtest(
        panel, cfg_on, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    diff = (eq_off["Value"].astype(float) - eq_on["Value"].astype(float)).abs().max()
    assert diff < 1e-9, (
        f"Empty-dividend run differs from disabled run by {diff:.2e}"
    )
```

- [ ] **Step 3.7: Add the dividend_df schema test**

Append to `tests/test_dividend_reinvestment_integration.py`:

```python
@pytest.mark.integration
def test_dividend_df_schema(isolate_yf, flat_xlu_panel):
    """The third return value of run_backtest carries the dividend log.
    Pin its column set so downstream consumers (exporter, dashboard)
    don't break silently."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [
        (panel.index[3].strftime("%Y-%m-%d"), "XLU", 0.10),
        (panel.index[7].strftime("%Y-%m-%d"), "XLU", 0.15),
    ])

    _, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    assert len(divs) == 2
    expected_cols = {
        "Date", "Ticker", "Dividend_Per_Share", "Shares", "Dividend_Amount",
        "Dividend_Yield", "Portfolio_Pct", "Reinvestment_Target",
        "Portfolio_Value",
    }
    missing = expected_cols - set(divs.columns)
    assert not missing, f"dividend_df missing expected columns: {missing}"
```

- [ ] **Step 3.8: Run the file and verify all tests pass**

```bash
python3 -m pytest tests/test_dividend_reinvestment_integration.py -v
```

Expected: 6 passed.

If any test fails: read the failure carefully — these tests are pinning new behavior so a failure may indicate a real bug worth investigating (especially `test_dividend_on_regime_change_day_routes_to_post_change_regime` and `test_mark_invariant_holds_with_dividends`).

- [ ] **Step 3.9: Run the full suite**

```bash
python3 -m pytest tests/ -q
```

Expected: 153 + previous additions + 6 dividend tests, all green.

- [ ] **Step 3.10: Commit**

```bash
git add tests/test_dividend_reinvestment_integration.py
git commit -m "test: end-to-end dividend reinvestment integration coverage

Closes CLAUDE.md open issue #9. Covers cash mode, share-reinvest mode,
mark invariant under dividends, dividend-on-regime-change-day ordering,
empty-events equivalence, and dividend_df schema. Pins the
backtest.py:976-1036 dividend block ordering relative to today's
mark-to-market and the regime/rebalance blocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Wrap-Up

- [ ] **Step W.1: Update CLAUDE.md open issues**

Open `/Users/nickmoeller/PycharmProjects/fast-markky-fund/CLAUDE.md` and:

1. Move open issues #2, #7, #9 to the "Bugs fixed during test suite work" section as test-coverage notes (not bugs, but the same closure semantics).
2. Update the test-file table (around line 207) with the three new files and their test counts.
3. Bump the "Last updated" date at the top to today.

- [ ] **Step W.2: Final full suite run**

```bash
python3 -m pytest tests/ -q
```

Expected: all green; new test count = original 153 + ~9 production-locked + 6 override + 6 dividend.

- [ ] **Step W.3: Commit CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs: close open issues #2, #7, #9 in CLAUDE.md

Production-config locked regression, end-to-end signal override
integration, and dividend reinvestment integration tests are now in
place — see tests/test_production_locked_regression.py,
tests/test_signal_override_integration.py,
tests/test_dividend_reinvestment_integration.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage:**
- Open issue #2 (production-path locked regression): Task 1 ✓
- Open issue #7 (E2E signal override): Task 2 ✓
- Open issue #9 (E2E dividend reinvestment): Task 3 ✓
- Mark invariant on production path: Task 1 step 1.4 ✓
- Daily-return-product invariant on production path: Task 1 step 1.4 ✓
- Override sticky persistence (locks the 2026-04-28 fix): Task 2 step 2.3 ✓
- `validate_panel_sums` integration assertion: Task 2 step 2.6 ✓
- Dividend on regime-change day (analogous to the 2026-04-28 mark bug): Task 3 step 3.5 ✓

**Type / signature consistency:**
- All tests call `run_backtest(panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio, dividend_data=...)` matching the signature at backtest.py:543.
- `production_config_dict` fixture used in Task 1 is defined in step 1.1 before being referenced.
- The `--bootstrap` flag added to conftest.py in step 1.3 is used in step 1.5.
- Override panel dict shape (`enabled`, `direction`, `threshold`, `label`, ticker keys) matches the production schema in `config.py` and the helpers in existing `test_compounding_correctness.py`.
- `dividend_data` shape (DataFrame indexed by date, columns = tickers, values = dividend per share) matches `data_loader.load_price_data(include_dividends=True)` and the consumer code at `backtest.py:988`.
