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


# LOCKED CONSTANTS — bootstrapped 2026-04-28 against frozen cache,
# production config (clean baseline set 2026-04-28: R1=75%TQQQ/25%XLU,
# R2=100%XLU, R3=100%$, drawdown_window_enabled=True/1y, per_regime,
# R1 signal overrides enabled). Full date range 1999-01-04 → 2026-03-27.
LOCKED_FINAL_VALUE: float = 862569.7271
LOCKED_CAGR: float = 0.177905
LOCKED_MAX_DRAWDOWN: float = -0.518655
LOCKED_VOLATILITY: float = 0.281996
LOCKED_SHARPE: float = 0.6309
LOCKED_ROW_COUNT: int = 6849


@pytest.fixture(scope="module")
def production_panel(production_config_dict):
    """Load the production panel once per module."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FMF_DATA_MODE", os.environ.get("FMF_DATA_MODE", "frozen"))
        from data_loader import load_price_data
        cfg = production_config_dict
        return load_price_data(
            cfg["tickers"], cfg["start_date"], cfg["end_date"]
        )


@pytest.fixture(scope="module")
def production_equity_df(production_config_dict, production_panel):
    """Run the full production backtest once per module against frozen cache."""
    from signal_override_engine import validate_panel_sums

    cfg = production_config_dict
    # Sanity: the config we're about to lock must pass the panel-sum invariant.
    validate_panel_sums(cfg)

    eq, _, _ = run_backtest(
        production_panel,
        cfg,
        compute_drawdown_from_ath,
        determine_regime,
        rebalance_portfolio,
    )
    return eq


@pytest.mark.integration
def test_print_locked_values_for_bootstrap(production_equity_df, request):
    """Run this once with -s --bootstrap to print the values that should populate
    the LOCKED_* constants. Skip-by-default: this is a bootstrap aid."""
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
def test_production_mark_invariant(production_equity_df, production_config_dict, production_panel):
    """Value[t] == sum(shares × price) + Cash on every row of the production run."""
    cfg = production_config_dict

    eq = production_equity_df
    panel_aligned = production_panel.reindex(pd.to_datetime(eq["Date"]))
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
