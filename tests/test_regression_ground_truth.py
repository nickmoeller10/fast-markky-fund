"""
Locked regression anchor test.

A single frozen test over a deterministic 5-year synthetic backtest. Any change
to core logic that moves these numbers without an explicit constant update fails.

DO NOT change the LOCKED_* constants without understanding what changed and why.
Config: 3-regime, down_only, drawdown_window_enabled=False, all signal_overrides disabled.
Seed: 42 (price_fixture). Generated: 2026-04-26.
"""
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from backtest import run_backtest, compute_drawdown_from_ath
from regime_engine import determine_regime
from rebalance_engine import rebalance_portfolio
from utils import max_drawdown_from_equity_curve


LOCKED_FINAL_VALUE   = 10949.7909
LOCKED_CAGR          = 0.018976
LOCKED_SHARPE        = 0.1552
LOCKED_MAX_DRAWDOWN  = -0.290396
LOCKED_VOLATILITY    = 0.122285
LOCKED_FIRST_20_REGIMES = ["R1"] * 20


def _build_price_df():
    """Deterministic 5yr price panel. seed=42, do not change."""
    np.random.seed(42)
    n_days = 1260
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    returns = np.random.normal(0.0004, 0.010, n_days)
    returns[200:260] = np.random.normal(-0.006, 0.014, 60)
    returns[700:820] = np.random.normal(-0.009, 0.018, 120)

    qqq  = 100.0 * np.exp(np.cumsum(returns))
    tqqq = 30.0  * np.exp(3.0 * np.cumsum(returns * 0.9))
    xlu  = 50.0  * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    spy  = 250.0 * np.exp(np.cumsum(0.7 * returns + np.random.normal(0, 0.004, n_days)))

    return pd.DataFrame({"QQQ": qqq, "TQQQ": tqqq, "XLU": xlu, "SPY": spy}, index=dates)


def _build_config():
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "down_only",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": 0.08,
                "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R2": {
                "dd_low": 0.08, "dd_high": 0.28,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R3": {
                "dd_low": 0.28, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
        },
    }


@pytest.fixture(scope="module")
def locked_equity_df():
    """Run the full backtest once per module with all network calls mocked."""
    price_df = _build_price_df()
    config = _build_config()

    qqq_prices = price_df["QQQ"].values
    dates = price_df.index
    empty_series = pd.Series(dtype=float)

    def _mock_yf_download(*args, **kwargs):
        df = pd.DataFrame({"Close": qqq_prices}, index=dates)
        df.columns = pd.MultiIndex.from_tuples([("Close", "QQQ")])
        return df

    # Patch the cache wrapper at its source module. The local `from data_cache
    # import cached_yf_download` inside run_backtest re-runs each call, so the
    # patched value is picked up. yfinance.download is also patched in case any
    # path bypasses the cache. data_loader signal loaders are patched for the
    # same reason — they're called from signal_layers, not run_backtest directly.
    with (
        patch("data_cache.cached_yf_download", side_effect=_mock_yf_download),
        patch("yfinance.download", side_effect=_mock_yf_download),
        patch("data_loader.load_spy_series", return_value=empty_series),
        patch("data_loader.load_vix_series", return_value=empty_series),
    ):
        equity_df, _, _ = run_backtest(
            price_df,
            config,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
        )
    return equity_df


@pytest.mark.integration
def test_locked_final_value(locked_equity_df):
    final = float(locked_equity_df["Value"].iloc[-1])
    assert abs(final - LOCKED_FINAL_VALUE) < 0.01, (
        f"Final value changed: got {final:.4f}, locked {LOCKED_FINAL_VALUE}. "
        "If this is intentional, update LOCKED_FINAL_VALUE."
    )


@pytest.mark.integration
def test_locked_row_count(locked_equity_df):
    assert len(locked_equity_df) == 1260, (
        f"Row count changed: got {len(locked_equity_df)}, expected 1260"
    )


@pytest.mark.integration
def test_locked_first_20_regimes(locked_equity_df):
    first_20 = list(locked_equity_df["Market_Regime"].iloc[:20])
    assert first_20 == LOCKED_FIRST_20_REGIMES, (
        f"First 20 regimes changed: got {first_20}"
    )


@pytest.mark.integration
def test_locked_cagr(locked_equity_df):
    vals = locked_equity_df["Value"]
    start_v, end_v = float(vals.iloc[0]), float(vals.iloc[-1])
    start_d = pd.to_datetime(locked_equity_df["Date"].iloc[0])
    end_d   = pd.to_datetime(locked_equity_df["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    assert abs(cagr - LOCKED_CAGR) < 1e-4, (
        f"CAGR changed: got {cagr:.6f}, locked {LOCKED_CAGR}. "
        "Update LOCKED_CAGR if intentional."
    )


@pytest.mark.integration
def test_locked_max_drawdown(locked_equity_df):
    dd = max_drawdown_from_equity_curve(locked_equity_df["Value"])
    assert abs(dd - LOCKED_MAX_DRAWDOWN) < 1e-4, (
        f"Max drawdown changed: got {dd:.6f}, locked {LOCKED_MAX_DRAWDOWN}. "
        "Update LOCKED_MAX_DRAWDOWN if intentional."
    )


@pytest.mark.integration
def test_locked_volatility(locked_equity_df):
    returns = locked_equity_df["Value"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    assert abs(vol - LOCKED_VOLATILITY) < 1e-4, (
        f"Volatility changed: got {vol:.6f}, locked {LOCKED_VOLATILITY}. "
        "Update LOCKED_VOLATILITY if intentional."
    )


@pytest.mark.integration
def test_locked_sharpe(locked_equity_df):
    vals = locked_equity_df["Value"]
    returns = vals.pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    start_v, end_v = float(vals.iloc[0]), float(vals.iloc[-1])
    start_d = pd.to_datetime(locked_equity_df["Date"].iloc[0])
    end_d   = pd.to_datetime(locked_equity_df["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    sharpe = cagr / vol if vol > 0 else 0.0
    assert abs(sharpe - LOCKED_SHARPE) < 1e-3, (
        f"Sharpe changed: got {sharpe:.4f}, locked {LOCKED_SHARPE}. "
        "Update LOCKED_SHARPE if intentional."
    )
