"""
Portfolio value calculation and rebalance execution invariants.
Protects: rebalance_engine.rebalance_portfolio(),
          allocation_engine.tradable_allocation(), get_allocation_for_regime()
"""
import numpy as np
import pandas as pd
import pytest

from allocation_engine import get_allocation_for_regime, tradable_allocation
from rebalance_engine import rebalance_portfolio


@pytest.fixture
def base_config():
    return {
        "starting_balance": 10_000.0,
        "drawdown_ticker": "QQQ",
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {"dd_low": 0.0, "dd_high": 0.08, "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0},
            "R2": {"dd_low": 0.08, "dd_high": 0.28, "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0},
            "R3": {"dd_low": 0.28, "dd_high": 1.0, "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0},
        },
    }


@pytest.mark.unit
def test_portfolio_value_equals_shares_times_price(base_config):
    prices = pd.Series({"QQQ": 100.0, "TQQQ": 30.0, "XLU": 50.0})
    alloc = get_allocation_for_regime("R2", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)

    computed_value = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(computed_value - 10_000.0) < 1e-6


@pytest.mark.unit
def test_r1_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R1", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_r2_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R2", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_r3_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R3", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_rebalance_preserves_value_same_prices(base_config):
    prices = pd.Series({"QQQ": 100.0, "TQQQ": 30.0, "XLU": 50.0})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)
    value_after = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(value_after - 10_000.0) < 1e-6


@pytest.mark.unit
def test_proxy_qqq_when_tqqq_nan(base_config):
    prices = pd.Series({"QQQ": 100.0, "TQQQ": float("nan"), "XLU": 50.0})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    assert list(alloc_t.keys()) == ["QQQ"]
    assert abs(alloc_t["QQQ"] - 1.0) < 1e-9


@pytest.mark.unit
def test_proxy_all_missing_returns_empty(base_config):
    prices = pd.Series({"QQQ": float("nan"), "TQQQ": float("nan"), "XLU": float("nan")})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    assert alloc_t == {}


@pytest.mark.unit
def test_starting_balance_preserved_day_one(base_config):
    prices = pd.Series({"QQQ": 50.0, "TQQQ": 15.0, "XLU": 25.0})
    alloc = get_allocation_for_regime("R2", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)
    value = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(value - 10_000.0) < 1e-6


@pytest.mark.unit
def test_rebalance_returns_none_on_nan_price(base_config):
    prices = pd.Series({"XLU": float("nan")})
    result = rebalance_portfolio(10_000.0, {"XLU": 1.0}, prices)
    assert result is None
