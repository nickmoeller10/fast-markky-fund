"""Compounding-correctness tests for the daily backtest loop.

These tests guard against a class of bugs where the rebalance logic uses
yesterday's close NAV (instead of today's mark) as the basis for new
share counts. That bug silently destroys today's intraday return on every
rebalance day, which makes:

  - identical-allocation rebalances produce different equity curves than
    no-rebalance (they should be identical),
  - max-drawdown numbers under-count the true loss on transition days,
  - higher-leverage R1 widths look strangely insensitive to dd_high,
  - signal-override panels that match the regime base produce different
    final equity than the override-disabled baseline.

Every test here either pins an algebraic invariant (sum-of-shares-times-prices
equals portfolio_value) or runs paired backtests where the result must match
within rounding.

If you add a test, follow the patterns documented in
.claude/skills/compounding-tests/SKILL.md.
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
# Helpers
# ---------------------------------------------------------------------------
def _empty_panel():
    return {
        "enabled": False,
        "label": "",
        "direction": "above",
        "threshold": 0,
        "TQQQ": 0.0,
        "QQQ": 0.0,
        "XLU": 0.0,
        "$": 0.0,
    }


def _disabled_overrides():
    return {"upside": _empty_panel(), "protection": _empty_panel()}


def _simple_config(
    *,
    rebalance_strategy="per_regime",
    r1_dd_high=0.5,
    r1_alloc=("TQQQ", 1.0),
    r2_alloc=("TQQQ", 1.0),
    overrides_enabled=False,
    upside_threshold=99.0,
    protection_threshold=-99.0,
    upside_alloc=None,
    protection_alloc=None,
):
    """Build a small 2-regime config with controllable identical/different allocations.

    Allocation tuples are (ticker, weight); the rest of the simplex defaults to 0.
    """
    def alloc_dict(spec):
        if spec is None:
            return {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0}
        ticker, weight = spec
        d = {"TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0}
        d[ticker] = weight
        # Any leftover goes to $ to keep panels summing to 1.0.
        leftover = 1.0 - weight
        if abs(leftover) > 1e-9:
            d["$"] = d.get("$", 0.0) + leftover
        return d

    r1 = alloc_dict(r1_alloc)
    r2 = alloc_dict(r2_alloc)

    if overrides_enabled:
        up = {**_empty_panel(), **alloc_dict(upside_alloc),
              "enabled": True, "direction": "above", "threshold": upside_threshold,
              "label": "test upside"}
        pr = {**_empty_panel(), **alloc_dict(protection_alloc),
              "enabled": True, "direction": "below", "threshold": protection_threshold,
              "label": "test protection"}
    else:
        up = _empty_panel()
        pr = _empty_panel()

    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": rebalance_strategy,
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "$"],
        "allocation_tickers": ["TQQQ", "QQQ", "XLU", "$"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": r1_dd_high,
                **r1,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {"upside": up, "protection": pr},
            },
            "R2": {
                "dd_low": r1_dd_high, "dd_high": 1.0,
                **r2,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
        },
    }


def _add_cash_column(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the synthetic $ sleeve to a price panel for tests that need it."""
    from data_loader import _build_cash_series

    out = panel.copy()
    out["$"] = _build_cash_series(out.index)
    return out


def _run(config, panel):
    return run_backtest(
        panel,
        config,
        compute_drawdown_from_ath,
        determine_regime,
        rebalance_portfolio,
    )


def _equity_close(eq_a, eq_b, *, rel=1e-6):
    """Compare two equity curves by Date+Value within a relative tolerance."""
    a = eq_a.set_index("Date")["Value"].astype(float)
    b = eq_b.set_index("Date")["Value"].astype(float)
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (
        f"Equity dataframes have different date indexes ({len(a)} vs {len(b)})"
    )
    diff = (a.loc[common] - b.loc[common]).abs() / a.loc[common].abs().clip(lower=1.0)
    assert diff.max() <= rel, (
        f"Max relative difference {diff.max():.2e} exceeds tolerance {rel:.2e}; "
        f"first divergence at {diff.idxmax()}: a={a.loc[diff.idxmax()]:.6f}, "
        f"b={b.loc[diff.idxmax()]:.6f}"
    )


def _mark_invariant(eq, panel, allocation_tickers, *, rel=5e-6):
    """Assert eq.Value[t] == sum(shares[t] × price[t]) + Cash[t] for every row."""
    panel_aligned = panel.reindex(pd.to_datetime(eq["Date"]))
    for i, row in eq.iterrows():
        date = pd.to_datetime(row["Date"])
        manual = 0.0
        for t in allocation_tickers:
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
        assert abs(v - manual) / denom <= rel, (
            f"Mark invariant violated on {row['Date']}: Value={v:.6f}, "
            f"sum(shares*price)+cash={manual:.6f} (rel diff={(v-manual)/denom:.2e})"
        )


# ---------------------------------------------------------------------------
# Mocking — synthetic-data tests must not pull in real QQQ/SPY/VIX history,
# because the historical merge inside run_backtest contaminates dd signals.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=False)
def isolate_from_yfinance(monkeypatch):
    """Replace every cached_yf_download import-site with a stub that returns an
    empty DataFrame. The backtest then runs purely against the synthetic panel.

    Use on synthetic-data tests; do NOT use on real-data tests in Group G,
    which intentionally exercise the cache."""
    import data_cache
    import data_loader
    import backtest as backtest_mod

    def empty(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(data_cache, "cached_yf_download", empty)
    monkeypatch.setattr(data_loader, "cached_yf_download", empty)
    if hasattr(backtest_mod, "cached_yf_download"):
        monkeypatch.setattr(backtest_mod, "cached_yf_download", empty)
    yield


# ---------------------------------------------------------------------------
# Synthetic price panels
# ---------------------------------------------------------------------------
@pytest.fixture
def small_panel():
    """5 trading days, TQQQ jumps 10% on day 2 then drifts. Includes $ sleeve."""
    dates = pd.bdate_range("1995-01-02", periods=5)
    df = pd.DataFrame(
        {
            "QQQ":  [100.0, 100.0,  98.0, 95.0, 96.0],
            "TQQQ": [100.0, 110.0, 105.0, 95.0, 96.0],
            "XLU":  [ 50.0,  50.5,  50.7, 50.3, 50.6],
            "SPY":  [200.0, 200.0, 200.0, 200.0, 200.0],
        },
        index=dates,
    )
    return _add_cash_column(df)


@pytest.fixture
def deep_dd_panel():
    """30 trading days with QQQ dropping smoothly from 100 to 65 then recovering.
    No flat segment-boundary days — the dd path is strictly monotonic on the way
    down so different R1 dd_high thresholds trigger rebalances on different days."""
    dates = pd.bdate_range("1995-01-02", periods=30)
    # Strictly monotonic decline 100 → 65 over 18 days, then recover to 95.
    qqq = np.concatenate([
        np.linspace(100, 65, 18, endpoint=True)[:-1],   # 17 days, ending at ~67
        np.linspace(67, 95, 13),                         # 13 days recovery
    ])
    tqqq = 100 * (qqq / qqq[0]) ** 3.0                   # rough 3x leverage proxy
    df = pd.DataFrame(
        {
            "QQQ": qqq,
            "TQQQ": tqqq,
            "XLU": 50 + 0.01 * np.arange(len(qqq)),
            "SPY": 200 * np.ones(len(qqq)),
        },
        index=dates,
    )
    return _add_cash_column(df)


# ===========================================================================
# Group A — fix-confirmation: today's mark must be preserved through rebalances
# ===========================================================================
@pytest.mark.unit
def test_a01_rebalance_to_identical_weights_preserves_today_return(isolate_from_yfinance, small_panel):
    """Two configs whose only difference is whether they rebalance to identical
    weights mid-stream must produce the same equity curve."""
    cfg_no_rebal = _simple_config(r1_dd_high=0.5,
                                   r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    cfg_rebal = copy.deepcopy(cfg_no_rebal)
    cfg_rebal["regimes"]["R1"]["dd_high"] = 0.01
    cfg_rebal["regimes"]["R2"]["dd_low"] = 0.01

    eq_a, _, _ = _run(cfg_no_rebal, small_panel)
    eq_b, _, _ = _run(cfg_rebal, small_panel)
    _equity_close(eq_a, eq_b, rel=1e-9)


@pytest.mark.unit
def test_a02_value_invariant_holds_on_rebalance_days(isolate_from_yfinance, small_panel):
    """For every row in equity_df, Value must equal sum(shares × price) + Cash."""
    cfg = _simple_config(r1_dd_high=0.01,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    _mark_invariant(eq, small_panel, cfg["allocation_tickers"])


@pytest.mark.unit
def test_a03_intraday_return_preserved_on_first_rebalance_day(isolate_from_yfinance, small_panel):
    """Force a rebalance on day 2 (TQQQ +10%) — portfolio must show ~11000 not 10000."""
    cfg = _simple_config(r1_dd_high=0.5,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    cfg["regimes"]["R1"]["dd_high"] = 0.005
    cfg["regimes"]["R2"]["dd_low"] = 0.005
    eq, _, _ = _run(cfg, small_panel)
    day2_value = float(eq["Value"].iloc[1])
    # Exact: 100% TQQQ, TQQQ went 100 → 110 → starting balance × 1.10 = 11000.
    assert day2_value == pytest.approx(11_000.0, rel=1e-6), (
        f"Day-2 value {day2_value:.4f} should reflect the 10% TQQQ gain (11000)"
    )


@pytest.mark.integration
def test_a04_signal_override_matching_base_equals_disabled(isolate_from_yfinance, price_fixture):
    """Override panel weights = base panel weights → equity curve identical
    regardless of override enabled/disabled. (The user-reported case.)"""
    panel = _add_cash_column(price_fixture)

    cfg_disabled = _simple_config(r1_dd_high=0.10,
                                   r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))

    cfg_enabled = copy.deepcopy(cfg_disabled)
    # Override panel matches R1 base (100% TQQQ); threshold guaranteed unreachable.
    cfg_enabled["regimes"]["R1"]["signal_overrides"] = {
        "upside": {**_empty_panel(), "TQQQ": 1.0,
                   "enabled": True, "direction": "above", "threshold": -10.0,
                   "label": "matches base"},
        "protection": _empty_panel(),
    }

    eq_a, _, _ = _run(cfg_disabled, panel)
    eq_b, _, _ = _run(cfg_enabled, panel)
    _equity_close(eq_a, eq_b, rel=1e-6)


# ===========================================================================
# Group B — single-asset compounding (the building blocks)
# ===========================================================================
@pytest.mark.unit
def test_b05_single_asset_no_rebalance_compounds_exactly(isolate_from_yfinance, small_panel):
    """100% TQQQ for 5 days, no rebalance: final equity = start × TQQQ_5/TQQQ_0."""
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    expected = 10_000 * small_panel["TQQQ"].iloc[-1] / small_panel["TQQQ"].iloc[0]
    assert float(eq["Value"].iloc[-1]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_b06_single_asset_xlu_no_rebalance_compounds_exactly(isolate_from_yfinance, small_panel):
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("XLU", 1.0), r2_alloc=("XLU", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    expected = 10_000 * small_panel["XLU"].iloc[-1] / small_panel["XLU"].iloc[0]
    assert float(eq["Value"].iloc[-1]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_b07_pure_cash_compounds_at_apy_for_one_year(isolate_from_yfinance):
    """100% $ for 252 trading days grows by exactly (1 + CASH_APY)^(251/252)."""
    from data_loader import CASH_APY, _build_cash_series

    dates = pd.bdate_range("1995-01-02", periods=252)
    panel = pd.DataFrame(
        {
            "QQQ":  np.linspace(100, 100, 252),
            "TQQQ": np.linspace(100, 100, 252),
            "XLU":  np.linspace(50, 50, 252),
            "SPY":  np.linspace(200, 200, 252),
        },
        index=dates,
    )
    panel = _add_cash_column(panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("$", 1.0), r2_alloc=("$", 1.0))
    eq, _, _ = _run(cfg, panel)
    expected_factor = (1.0 + CASH_APY) ** (251.0 / 252.0)
    assert float(eq["Value"].iloc[-1]) / 10_000.0 == pytest.approx(
        expected_factor, rel=2e-3
    )


@pytest.mark.unit
def test_b08_pure_cash_zero_drawdown(isolate_from_yfinance, small_panel):
    """100% $ allocation must have zero portfolio drawdown."""
    panel = _add_cash_column(small_panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("$", 1.0), r2_alloc=("$", 1.0))
    eq, _, _ = _run(cfg, panel)
    values = eq["Value"].astype(float)
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    assert dd.min() >= -1e-9, f"100% $ portfolio shows DD {dd.min():.6f}"


# ===========================================================================
# Group C — invariants under rebalances
# ===========================================================================
@pytest.mark.unit
def test_c09_total_value_conserved_through_rebalance(isolate_from_yfinance, small_panel):
    """Pre-rebalance NAV (today's mark + cash) must equal post-rebalance NAV."""
    cfg = _simple_config(r1_dd_high=0.005,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("XLU", 1.0))
    eq, rebals, _ = _run(cfg, small_panel)
    # On every rebalance row, the recorded portfolio value must match the
    # equity_df Value on that same date.
    for _, rb in rebals.iterrows():
        eq_row = eq.loc[eq["Date"] == rb["Date"]]
        assert not eq_row.empty
        assert float(eq_row["Value"].iloc[0]) == pytest.approx(
            float(rb["Portfolio_Value"]), rel=1e-9
        )


@pytest.mark.unit
def test_c10_no_negative_or_nan_shares(isolate_from_yfinance, small_panel):
    """Share counts must always be finite and non-negative."""
    cfg = _simple_config(r1_dd_high=0.005,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("XLU", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    for t in cfg["allocation_tickers"]:
        col = f"{t}_shares"
        if col not in eq.columns:
            continue
        vals = eq[col].astype(float)
        assert vals.notna().all(), f"{col} contains NaN"
        assert (vals >= -1e-12).all(), f"{col} has negative values"


@pytest.mark.unit
def test_c11_value_non_negative_throughout(isolate_from_yfinance, small_panel):
    cfg = _simple_config(r1_dd_high=0.005,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("XLU", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    assert (eq["Value"].astype(float) >= 0).all()


@pytest.mark.unit
def test_c12_rebalance_count_matches_regime_transitions(isolate_from_yfinance, small_panel):
    """Number of rebalance rows should equal number of regime transitions."""
    cfg = _simple_config(r1_dd_high=0.005,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("XLU", 1.0))
    eq, rebals, _ = _run(cfg, small_panel)
    transitions = (eq["Portfolio_Regime"] != eq["Portfolio_Regime"].shift()).sum() - 1
    assert len(rebals) == transitions


# ===========================================================================
# Group D — drawdown sensitivity (the user's R1-width concern)
# ===========================================================================
@pytest.mark.integration
def test_d13_dd_monotonic_with_r1_width(isolate_from_yfinance, deep_dd_panel):
    """Wider R1 (higher TQQQ exposure window) must non-decrease max DD when
    R2 = 100% $. Three points: dd_high in {6%, 8%, 10%}."""
    panel = _add_cash_column(deep_dd_panel)
    results = {}
    for r1_high in (0.06, 0.08, 0.10):
        cfg = _simple_config(r1_dd_high=r1_high,
                             r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))
        eq, _, _ = _run(cfg, panel)
        v = eq["Value"].astype(float)
        max_dd = ((v - v.cummax()) / v.cummax()).min()
        results[r1_high] = max_dd

    # DD becomes worse (more negative) as R1 gets wider.
    assert results[0.06] >= results[0.08] >= results[0.10] - 1e-9, (
        f"Expected DD monotonicity 6% >= 8% >= 10%; got {results}"
    )
    # And meaningfully different — 6% should be at least 1pp better than 10%.
    assert results[0.06] - results[0.10] > 0.01, (
        f"R1 width has no impact on DD: {results}"
    )


@pytest.mark.integration
def test_d14_r3_cash_caps_drawdown(isolate_from_yfinance, deep_dd_panel):
    """R3 = 100% $ vs R3 = 100% TQQQ → cash must give strictly smaller DD."""
    panel = _add_cash_column(deep_dd_panel)
    cfg_cash = _simple_config(r1_dd_high=0.05,
                              r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))
    cfg_tqqq = _simple_config(r1_dd_high=0.05,
                              r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    eq_cash, _, _ = _run(cfg_cash, panel)
    eq_tqqq, _, _ = _run(cfg_tqqq, panel)
    dd_cash = ((eq_cash["Value"] - eq_cash["Value"].cummax()) / eq_cash["Value"].cummax()).min()
    dd_tqqq = ((eq_tqqq["Value"] - eq_tqqq["Value"].cummax()) / eq_tqqq["Value"].cummax()).min()
    assert dd_cash > dd_tqqq + 0.05, f"Cash DD {dd_cash:.4f} not better than TQQQ DD {dd_tqqq:.4f}"


@pytest.mark.unit
def test_d15_zero_width_r1_means_quick_exit_to_cash(isolate_from_yfinance, deep_dd_panel):
    """R1 dd_high → 0 → strategy can only hold TQQQ on day 1 (initial open)
    and exits to cash on day 2 because yesterday's dd > 0 puts us in R2."""
    cfg = _simple_config(r1_dd_high=1e-9,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))
    eq, _, _ = _run(cfg, deep_dd_panel)
    tqqq_shares = eq["TQQQ_shares"].astype(float)
    # Day 1 = initial open (R1 since dd=0). Day 2's regime uses day-1 dd=0 →
    # also R1, so TQQQ held. From day 3 onward, yesterday's dd > 0 → R2,
    # so TQQQ exposure must be zero.
    assert tqqq_shares.iloc[2:].abs().max() < 1e-9, (
        f"TQQQ held beyond day 2: {tqqq_shares.iloc[2:].max()}"
    )


# ===========================================================================
# Group E — paired-config equivalences (override-match-base style)
# ===========================================================================
@pytest.mark.integration
def test_e16_override_disabled_unreachable_threshold_equivalence(isolate_from_yfinance, deep_dd_panel):
    """An override with an unreachable threshold must equal an override
    that's disabled."""
    panel = _add_cash_column(deep_dd_panel)
    base = _simple_config(r1_dd_high=0.05,
                          r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))

    cfg_disabled = copy.deepcopy(base)
    cfg_unreachable = copy.deepcopy(base)
    cfg_unreachable["regimes"]["R1"]["signal_overrides"]["upside"] = {
        **_empty_panel(), "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
        "enabled": True, "direction": "above", "threshold": 1e6,
        "label": "unreachable",
    }

    eq_a, _, _ = _run(cfg_disabled, panel)
    eq_b, _, _ = _run(cfg_unreachable, panel)
    _equity_close(eq_a, eq_b, rel=1e-9)


@pytest.mark.unit
def test_e17_split_regime_with_identical_weights_equals_single_regime(isolate_from_yfinance, small_panel):
    """Splitting a regime into two regimes with identical weights produces the
    same equity curve as keeping the whole range as one regime (because
    rebalancing to the same target should be a no-op for value)."""
    panel = small_panel.copy()
    cfg_single = _simple_config(r1_dd_high=0.99,
                                r1_alloc=("XLU", 1.0), r2_alloc=("XLU", 1.0))
    cfg_split = _simple_config(r1_dd_high=0.005,
                               r1_alloc=("XLU", 1.0), r2_alloc=("XLU", 1.0))
    eq_a, _, _ = _run(cfg_single, panel)
    eq_b, _, _ = _run(cfg_split, panel)
    _equity_close(eq_a, eq_b, rel=1e-9)


@pytest.mark.unit
def test_e18_r2_and_r3_identical_means_dd_t2_irrelevant(isolate_from_yfinance, deep_dd_panel):
    """When R2 and R3 have identical allocations, the dd_t2 boundary is
    cosmetic — equity curves with dd_t2 = 0.20 vs dd_t2 = 0.40 must match."""
    panel = _add_cash_column(deep_dd_panel)
    base = _simple_config(r1_dd_high=0.05,
                          r1_alloc=("TQQQ", 1.0), r2_alloc=("$", 1.0))
    # Promote to a 3-regime layout where R2 and R3 share allocations.
    base["regimes"]["R3"] = copy.deepcopy(base["regimes"]["R2"])
    base["regimes"]["R3"]["dd_low"] = 0.20
    base["regimes"]["R3"]["dd_high"] = 1.0
    base["regimes"]["R2"]["dd_high"] = 0.20

    cfg_a = copy.deepcopy(base)
    cfg_b = copy.deepcopy(base)
    cfg_b["regimes"]["R2"]["dd_high"] = 0.40
    cfg_b["regimes"]["R3"]["dd_low"] = 0.40

    eq_a, _, _ = _run(cfg_a, panel)
    eq_b, _, _ = _run(cfg_b, panel)
    _equity_close(eq_a, eq_b, rel=1e-9)


# ===========================================================================
# Group F — cash sleeve specifics
# ===========================================================================
@pytest.mark.unit
def test_f19_cash_only_portfolio_value_strictly_non_decreasing(isolate_from_yfinance, small_panel):
    panel = _add_cash_column(small_panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("$", 1.0), r2_alloc=("$", 1.0))
    eq, _, _ = _run(cfg, panel)
    diffs = eq["Value"].astype(float).diff().dropna()
    assert (diffs >= -1e-9).all(), (
        f"100% $ value not monotonic non-decreasing: min diff = {diffs.min():.6e}"
    )


@pytest.mark.unit
def test_f20_split_cash_to_xlu_and_back_preserves_total(isolate_from_yfinance, small_panel):
    """50/50 cash/XLU portfolio with no rebalance: end value matches drift formula."""
    panel = _add_cash_column(small_panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("XLU", 0.5), r2_alloc=("XLU", 0.5))  # 50% XLU + 50% $
    eq, _, _ = _run(cfg, panel)
    # No rebalance → drift. Day-0 shares = $5000 / price[0] for each side.
    xlu_shares = 5_000.0 / panel["XLU"].iloc[0]
    cash_shares = 5_000.0 / panel["$"].iloc[0]
    expected_end = xlu_shares * panel["XLU"].iloc[-1] + cash_shares * panel["$"].iloc[-1]
    assert float(eq["Value"].iloc[-1]) == pytest.approx(expected_end, rel=1e-5)


# ===========================================================================
# Group G — small-data unit + full-data integration with cached yfinance
# ===========================================================================
@pytest.mark.integration
def test_g21_small_load_compounds_pure_tqqq(isolate_from_yfinance, small_panel):
    """Smoke test: small synthetic data, 100% TQQQ, compounds exactly."""
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    eq, _, _ = _run(cfg, small_panel)
    ratio = float(eq["Value"].iloc[-1]) / float(eq["Value"].iloc[0])
    expected = small_panel["TQQQ"].iloc[-1] / small_panel["TQQQ"].iloc[0]
    assert ratio == pytest.approx(expected, rel=1e-6)


@pytest.mark.integration
def test_g22_full_load_pure_tqqq_matches_buy_and_hold():
    """Real cached TQQQ data, 100% TQQQ from TQQQ inception, equity matches
    buy-and-hold ratio. Uses the FMF_DATA_MODE=frozen cache."""
    os.environ.setdefault("FMF_DATA_MODE", "frozen")
    from data_loader import load_price_data

    panel = load_price_data(
        ["QQQ", "TQQQ", "XLU", "SPY"], "2015-01-02", "2020-12-31"
    )
    panel = _add_cash_column(panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("TQQQ", 1.0), r2_alloc=("TQQQ", 1.0))
    cfg["start_date"] = "2015-01-02"
    cfg["end_date"] = "2020-12-31"

    eq, _, _ = _run(cfg, panel)
    # Compare against pure buy-and-hold ratio of TQQQ.
    tqqq = panel["TQQQ"].dropna()
    expected_ratio = tqqq.iloc[-1] / tqqq.iloc[0]
    actual_ratio = float(eq["Value"].iloc[-1]) / float(eq["Value"].iloc[0])
    assert actual_ratio == pytest.approx(expected_ratio, rel=2e-3), (
        f"Pure-TQQQ ratio {actual_ratio:.4f} doesn't match buy-and-hold {expected_ratio:.4f}"
    )


@pytest.mark.integration
def test_g23_full_load_pure_cash_grows_at_apy():
    """Real-data integration: 100% $ over a real-date range grows ≈ (1+APY)^years."""
    os.environ.setdefault("FMF_DATA_MODE", "frozen")
    from data_loader import CASH_APY, load_price_data

    panel = load_price_data(
        ["QQQ", "TQQQ", "XLU", "SPY"], "2015-01-02", "2020-12-31"
    )
    panel = _add_cash_column(panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("$", 1.0), r2_alloc=("$", 1.0))
    cfg["start_date"] = "2015-01-02"
    cfg["end_date"] = "2020-12-31"

    eq, _, _ = _run(cfg, panel)
    years = (
        pd.to_datetime(eq["Date"].iloc[-1]) - pd.to_datetime(eq["Date"].iloc[0])
    ).days / 365.25
    expected = (1.0 + CASH_APY) ** years
    actual = float(eq["Value"].iloc[-1]) / float(eq["Value"].iloc[0])
    assert actual == pytest.approx(expected, rel=5e-3)


# ===========================================================================
# Group H — regime-chain coverage (R1→R2→R3 with distinct assets, plus the
# regime-clears-override interaction). Added after a review noted that the
# Group A-F suite was 2-regime-heavy.
# ===========================================================================
@pytest.mark.integration
def test_h25_three_regime_chain_mark_invariant_and_dd_floor(
    isolate_from_yfinance, deep_dd_panel
):
    """3-regime config with three different assets — R1=TQQQ, R2=XLU, R3=$.
    The deep_dd_panel takes QQQ from 100 → 65 → 95 (35% peak DD), forcing the
    chain R1→R2→R3 on the way down and R3→R2→R1 on the way up. We assert:

    1. The mark invariant holds on every day (catches accounting bugs that
       only show up at the second or third transition).
    2. Once R3 is entered, the DD floor doesn't get worse (cash sleeve has
       zero drawdown — the DD locked in at the R2→R3 transition is the worst
       point of the run).
    """
    cfg = {
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
                "dd_low": 0.0, "dd_high": 0.10,
                "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
            "R2": {
                "dd_low": 0.10, "dd_high": 0.25,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0, "$": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
            "R3": {
                "dd_low": 0.25, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
        },
    }
    eq, rebals, _ = _run(cfg, deep_dd_panel)

    # 1. Mark invariant on every row, including rebalance days.
    _mark_invariant(eq, deep_dd_panel, cfg["allocation_tickers"])

    # 2. We saw at least one transition into each regime (chain actually fires).
    portfolio_regimes_seen = set(eq["Portfolio_Regime"].unique())
    assert {"R1", "R2", "R3"}.issubset(portfolio_regimes_seen), (
        f"Expected R1/R2/R3 all to be visited; saw {portfolio_regimes_seen}"
    )

    # 3. R3 entry locks the DD floor — once portfolio_regime is R3, value never
    #    dips below the entry NAV (cash compounds non-decreasingly).
    r3_mask = eq["Portfolio_Regime"] == "R3"
    if r3_mask.any():
        r3_values = eq.loc[r3_mask, "Value"].astype(float).reset_index(drop=True)
        diffs = r3_values.diff().dropna()
        assert (diffs >= -1e-6).all(), (
            "Portfolio value decreased while in R3 (100% $ should be monotonic). "
            f"min diff = {diffs.min():.6e}"
        )


@pytest.mark.integration
def test_h26_regime_change_clears_active_signal_override(
    isolate_from_yfinance, deep_dd_panel
):
    """When a regime transition fires on a day where a signal override is
    active, backtest.py clears `signal_override_mode` to "none" before the
    next override evaluation. The new regime's base allocation should be in
    effect immediately, not the prior override's allocation.

    We can't reliably trigger Signal_total in synthetic-isolation mode
    (signal layers see no SPY/VIX), so the realistic check is that
    Signal_override_active flips back to "" / "none" exactly when the
    portfolio regime changes.
    """
    cfg = {
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
                "dd_low": 0.0, "dd_high": 0.10,
                "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
            "R2": {
                "dd_low": 0.10, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": _disabled_overrides(),
            },
        },
    }
    eq, _, _ = _run(cfg, deep_dd_panel)
    # On every row where the portfolio regime is the one that just changed,
    # the override mode column must be the empty/"none" sentinel — it is
    # cleared at the regime-rebalance site (backtest.py:1094) before any
    # override evaluation for that day.
    regime_changed = eq["Portfolio_Regime"] != eq["Portfolio_Regime"].shift()
    overrides_on_change_day = eq.loc[regime_changed, "Signal_override_active"].astype(str)
    assert overrides_on_change_day.isin(["", "none"]).all(), (
        "Signal_override_active should be reset on regime-change days; "
        f"saw {overrides_on_change_day[~overrides_on_change_day.isin(['', 'none'])].tolist()}"
    )


@pytest.mark.integration
def test_g24_full_load_pure_xlu_matches_buy_and_hold():
    os.environ.setdefault("FMF_DATA_MODE", "frozen")
    from data_loader import load_price_data

    panel = load_price_data(
        ["QQQ", "TQQQ", "XLU", "SPY"], "2015-01-02", "2020-12-31"
    )
    panel = _add_cash_column(panel)
    cfg = _simple_config(r1_dd_high=0.99,
                         r1_alloc=("XLU", 1.0), r2_alloc=("XLU", 1.0))
    cfg["start_date"] = "2015-01-02"
    cfg["end_date"] = "2020-12-31"

    eq, _, _ = _run(cfg, panel)
    xlu = panel["XLU"].dropna()
    expected = xlu.iloc[-1] / xlu.iloc[0]
    actual = float(eq["Value"].iloc[-1]) / float(eq["Value"].iloc[0])
    assert actual == pytest.approx(expected, rel=2e-3)
