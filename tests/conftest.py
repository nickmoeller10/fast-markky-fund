"""Shared pytest fixtures for the Fast Markky Fund test suite."""
import numpy as np
import pandas as pd
import pytest


def _base_config():
    """3-regime config matching production thresholds, all signal_overrides disabled."""
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
                "dd_low": 0.0,
                "dd_high": 0.08,
                "TQQQ": 1.0,
                "QQQ": 0.0,
                "XLU": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R2": {
                "dd_low": 0.08,
                "dd_high": 0.28,
                "TQQQ": 0.0,
                "QQQ": 0.0,
                "XLU": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R3": {
                "dd_low": 0.28,
                "dd_high": 1.0,
                "TQQQ": 0.0,
                "QQQ": 0.0,
                "XLU": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
        },
    }


@pytest.fixture
def minimal_config():
    """Minimal 3-regime config with thresholds matching production (0%, 8%, 28%)."""
    return _base_config()


@pytest.fixture
def production_per_regime_config():
    """Production per_regime config: R2 has rebalance_on_upward=hold."""
    cfg = _base_config()
    cfg["rebalance_strategy"] = "per_regime"
    cfg["regimes"]["R2"]["rebalance_on_upward"] = "hold"
    return cfg


@pytest.fixture
def price_fixture():
    """
    Deterministic OHLCV-style close prices for 5 years (1260 trading days).
    seed=42 — do not change without updating locked regression values.

    TQQQ starts at 30.0 (approximate historical price at 2018 launch, not 3x QQQ).
    TQQQ path uses log-return approximation: exp(3 * cumsum(returns * 0.9)).
    For returns < ±2% daily the error vs true daily-rebalanced 3x is negligible.
    """
    np.random.seed(42)
    n_days = 1260
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    returns = np.random.normal(0.0004, 0.010, n_days)
    returns[200:260] = np.random.normal(-0.006, 0.014, 60)
    returns[700:820] = np.random.normal(-0.009, 0.018, 120)

    qqq = 100.0 * np.exp(np.cumsum(returns))
    tqqq = 30.0 * np.exp(3.0 * np.cumsum(returns * 0.9))
    xlu = 50.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    spy = 250.0 * np.exp(np.cumsum(0.7 * returns + np.random.normal(0, 0.004, n_days)))

    return pd.DataFrame(
        {"QQQ": qqq, "TQQQ": tqqq, "XLU": xlu, "SPY": spy}, index=dates
    )


def make_equity_df(start_val, daily_returns, start_date="2020-01-02"):
    """Build an equity_df from a list of daily returns (pct as decimals, e.g. 0.01 = 1%)."""
    dates = pd.bdate_range(start_date, periods=len(daily_returns) + 1)
    vals = [float(start_val)]
    for r in daily_returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.DataFrame({"Date": dates, "Value": vals})


def make_equity_df_from_values(values, start_date="2020-01-02"):
    """Build an equity_df from an explicit sequence of portfolio values."""
    dates = pd.bdate_range(start_date, periods=len(values))
    return pd.DataFrame({"Date": list(dates), "Value": list(map(float, values))})


@pytest.fixture
def equity_series_fixture():
    """Factory: returns make_equity_df so tests can call it with their own params."""
    return make_equity_df


@pytest.fixture
def equity_from_values():
    """Factory: returns make_equity_df_from_values."""
    return make_equity_df_from_values
