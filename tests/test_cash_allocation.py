"""
Tests for CASH as a first-class allocation option throughout the optimizer.

Covers:
  1. The synthetic CASH price series compounds at exactly CASH_APY/yr.
  2. A config with CASH in allocation_tickers runs end-to-end through
     run_backtest and produces a non-empty equity_df where the CASH-heavy
     regime actually accumulates the risk-free return.
  3. `enable_cash_in_regimes` extends the simplex to 4 tickers for the listed
     regimes only — other regimes keep the legacy 3-ticker simplex.
  4. `forced_base_allocations` still works as the stricter override and adds
     CASH to allocation_tickers / panel tickers automatically.
  5. With no CASH constraints, the search space is unchanged (back-compat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. CASH series compounds at the right APY
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_cash_series_compounds_at_target_apy():
    from optimizer.parameter_space import CASH_APY
    from data_loader import _build_cash_series

    # 252 trading days = exactly 1 year of compounding
    idx = pd.bdate_range("2000-01-03", periods=252)
    series = _build_cash_series(idx)
    assert series.iloc[0] == pytest.approx(1.0, abs=1e-12)
    # After 252 compounded days at (1+APY)^(1/252), price should be (1+APY)^(251/252)
    # since day 0 holds factor^0 = 1.0 and day 251 holds factor^251.
    expected_last = (1.0 + CASH_APY) ** (251.0 / 252.0)
    assert series.iloc[-1] == pytest.approx(expected_last, abs=1e-9)

    # Over exactly 253 entries (day 0 .. day 252) the last entry should hit
    # 1+APY exactly — that's the property the user cares about.
    idx_full = pd.bdate_range("2000-01-03", periods=253)
    series_full = _build_cash_series(idx_full)
    assert series_full.iloc[-1] == pytest.approx(1.0 + CASH_APY, abs=1e-6)


@pytest.mark.unit
def test_cash_series_is_monotonically_increasing_zero_drawdown():
    """CASH must never lose value — drawdown is identically zero."""
    from data_loader import _build_cash_series

    idx = pd.bdate_range("2000-01-03", periods=2000)
    series = _build_cash_series(idx)
    diffs = series.diff().dropna()
    assert (diffs > 0).all(), "CASH series must be strictly increasing"


# ---------------------------------------------------------------------------
# 2. End-to-end backtest with CASH
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_run_backtest_handles_cash_in_allocation(price_fixture):
    """
    A config that lists CASH in tickers + allocation_tickers must run cleanly
    through run_backtest. We pin a regime to 100% CASH and verify the equity
    curve is non-empty and grows at roughly the risk-free rate while in that
    regime.
    """
    from backtest import run_backtest, compute_drawdown_from_ath
    from optimizer.parameter_space import CASH_APY, CASH_TICKER
    from data_loader import _build_cash_series
    from rebalance_engine import rebalance_portfolio
    from regime_engine import determine_regime

    panel = price_fixture.copy()
    panel[CASH_TICKER] = _build_cash_series(panel.index)

    # Force every drawdown bucket into CASH so the portfolio earns only carry.
    config = {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "per_regime",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", CASH_TICKER],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU", CASH_TICKER],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": 0.08,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "CASH": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R2": {
                "dd_low": 0.08, "dd_high": 0.28,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "CASH": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R3": {
                "dd_low": 0.28, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "CASH": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
        },
    }

    eq, _, _ = run_backtest(
        panel,
        config,
        compute_drawdown_from_ath,
        determine_regime,
        rebalance_portfolio,
    )
    assert eq is not None
    assert not eq.empty
    # 100% CASH every day: portfolio value should be strictly non-decreasing
    # (modulo float roundoff at the rebalance bar). We check that final value
    # is approximately starting_balance * (1+APY)^years — within a small
    # tolerance to allow for rebalance roundoff and the t+1 lag.
    start_v = float(eq["Value"].iloc[0])
    end_v = float(eq["Value"].iloc[-1])
    years = (pd.to_datetime(eq["Date"].iloc[-1]) - pd.to_datetime(eq["Date"].iloc[0])).days / 365.25
    expected_factor = (1.0 + CASH_APY) ** years
    actual_factor = end_v / start_v
    # Loose tolerance: rebalance roundoff + the trading-day vs calendar-year mismatch.
    assert actual_factor == pytest.approx(expected_factor, rel=0.02), (
        f"100% CASH portfolio grew {actual_factor:.4f}x; expected ~{expected_factor:.4f}x"
    )


# ---------------------------------------------------------------------------
# 3. enable_cash_in_regimes extends the simplex
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_enable_cash_in_regimes_adds_cash_simplex_param():
    """When R2 is in enable_cash_in_regimes, R2 has a w_cash_raw param sampled."""
    import optuna
    from optimizer.parameter_space import (
        CASH_TICKER, IterationConstraints, suggest_config,
    )

    cons = IterationConstraints(enable_cash_in_regimes=["R2"])
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    config = suggest_config(trial, constraints=cons)

    # CASH should now be in tickers + allocation_tickers
    assert CASH_TICKER in config["tickers"]
    assert CASH_TICKER in config["allocation_tickers"]

    # The trial must have sampled R2_base_w_cash_raw (cash-enabled regime)
    # but NOT R1_base_w_cash_raw (regime not in enable list).
    sampled = set(trial.params.keys())
    assert "R2_base_w_cash_raw" in sampled
    assert "R2_upside_w_cash_raw" in sampled
    assert "R2_protection_w_cash_raw" in sampled
    assert "R1_base_w_cash_raw" not in sampled
    assert "R3_base_w_cash_raw" not in sampled

    # R2's regime block carries a CASH weight in [0, 1] as part of a
    # 4-ticker simplex; R1's block omits CASH (legacy 3-ticker shape).
    if "R2" in config["regimes"]:
        r2 = config["regimes"]["R2"]
        assert CASH_TICKER in r2
        s = sum(float(r2.get(t, 0.0)) for t in ("TQQQ", "QQQ", "XLU", "CASH"))
        assert s == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_no_cash_constraint_keeps_three_ticker_simplex():
    """Default search space (no CASH constraints) is unchanged."""
    import optuna
    from optimizer.parameter_space import CASH_TICKER, suggest_config

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    config = suggest_config(trial)

    # No CASH anywhere in the panel/allocation universe by default
    assert CASH_TICKER not in config["tickers"]
    assert CASH_TICKER not in config["allocation_tickers"]

    # No CASH params sampled
    assert not any(k.endswith("_w_cash_raw") for k in trial.params.keys())

    # Each regime's weights still sum to 1 over the core 3 tickers
    for regime_block in config["regimes"].values():
        s = sum(float(regime_block.get(t, 0.0)) for t in ("TQQQ", "QQQ", "XLU"))
        assert s == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. forced_base_allocations still works
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_forced_base_allocations_with_cash():
    """A regime forced to {XLU: 0.5, CASH: 0.5} ignores the simplex and pins those weights."""
    import optuna
    from optimizer.parameter_space import (
        CASH_TICKER, IterationConstraints, suggest_config,
    )

    cons = IterationConstraints(
        n_regimes_choices=[3],
        forced_base_allocations={"R2": {"XLU": 0.5, CASH_TICKER: 0.5}},
    )
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    config = suggest_config(trial, constraints=cons)

    # CASH propagates into the universe even though enable_cash_in_regimes is empty
    assert CASH_TICKER in config["tickers"]
    assert CASH_TICKER in config["allocation_tickers"]

    r2 = config["regimes"]["R2"]
    assert r2["XLU"] == pytest.approx(0.5)
    assert r2[CASH_TICKER] == pytest.approx(0.5)
    assert r2["TQQQ"] == pytest.approx(0.0)
    assert r2["QQQ"] == pytest.approx(0.0)


@pytest.mark.unit
def test_forced_allocation_overrides_enable_cash_simplex():
    """When both are set for the same regime, the forced allocation wins for the base."""
    import optuna
    from optimizer.parameter_space import (
        CASH_TICKER, IterationConstraints, suggest_config,
    )

    cons = IterationConstraints(
        n_regimes_choices=[3],
        enable_cash_in_regimes=["R2"],
        forced_base_allocations={"R2": {"XLU": 0.5, CASH_TICKER: 0.5}},
    )
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    config = suggest_config(trial, constraints=cons)

    r2 = config["regimes"]["R2"]
    # Base is the forced allocation, NOT the simplex-sampled values
    assert r2["TQQQ"] == pytest.approx(0.0)
    assert r2["QQQ"] == pytest.approx(0.0)
    assert r2["XLU"] == pytest.approx(0.5)
    assert r2[CASH_TICKER] == pytest.approx(0.5)
