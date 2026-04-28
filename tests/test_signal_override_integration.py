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
    import data_cache
    import data_loader
    import backtest as backtest_mod

    def empty(*a, **kw):
        return pd.DataFrame()

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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
    sig = {d: (2.0 if i >= 5 else 0.0) for i, d in enumerate(panel.index)}
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    overrides_active = eq["Signal_override_active"].astype(str).tolist()
    assert "upside" in overrides_active, (
        f"Upside override never fired despite Signal_total=2 from day 5. "
        f"Saw: {set(overrides_active)}"
    )
    upside_rows = eq[eq["Signal_override_active"] == "upside"]
    assert upside_rows["TQQQ_shares"].astype(float).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert upside_rows["XLU_shares"].astype(float).iloc[-1] > 0.0


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
    sig = {}
    for i, d in enumerate(panel.index):
        if i < 5:
            sig[d] = 0.0
        elif i < 8:
            sig[d] = 2.0
        else:
            sig[d] = 0.0
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    last_override_state = eq["Signal_override_active"].iloc[-1]
    assert last_override_state == "upside", (
        f"Override unwound to {last_override_state!r} on neutral signal — "
        "sticky behavior broken (regression of the 2026-04-28 fix)."
    )


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
        if i < 5:
            sig[d] = 0.0
        elif i < 10:
            sig[d] = 2.0
        else:
            sig[d] = -3.0
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    last = eq.iloc[-1]
    assert last["Signal_override_active"] == "protection", (
        f"Expected protection at end; got {last['Signal_override_active']!r}"
    )
    cash_shares = float(last.get("$_shares", 0))
    assert cash_shares > 0, "Protection panel routes to $ but no $ shares held"


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
        r1_high=0.10,
    )
    sig = {d: 2.0 for d in panel.index}
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )

    r2_rows = eq[eq["Portfolio_Regime"] == "R2"]
    assert not r2_rows.empty, "R2 never reached — panel design bug"
    first_r2 = r2_rows.iloc[0]
    assert first_r2["Signal_override_active"] in ("", "none"), (
        f"Override not cleared on R1→R2 transition: "
        f"got {first_r2['Signal_override_active']!r}"
    )


@pytest.mark.integration
def test_invalid_panel_sum_raises_at_backtest_start(isolate_yf, flat_panel):
    """A malformed (sum != 1.0) enabled override panel must raise via
    validate_panel_sums — caught before any backtest day runs."""
    from signal_override_engine import validate_panel_sums

    cfg = _override_cfg(
        upside={
            **_disabled_panel(),
            "enabled": True, "direction": "above", "threshold": 1.0,
            "label": "broken", "TQQQ": 0.5, "XLU": 0.3,
        },
    )
    with pytest.raises((ValueError, AssertionError)):
        validate_panel_sums(cfg)


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
    sig = {d: 2.0 for d in panel.index}
    _patch_signal_total(monkeypatch, sig)

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime, rebalance_portfolio
    )
    early_states = eq["Signal_override_active"].iloc[:3].astype(str).tolist()
    assert "upside" in early_states, (
        f"Override never activated even with constant high signal: {early_states}"
    )
