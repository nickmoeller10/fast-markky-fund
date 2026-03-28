"""
Tradable allocation (pre-IPO / missing prices) and backtest deploy from year-2000 panels.
"""

import unittest
import numpy as np
import pandas as pd

from allocation_engine import get_allocation_for_regime, tradable_allocation
from backtest import run_backtest
from regime_engine import compute_drawdown_from_ath, determine_regime
from rebalance_engine import rebalance_portfolio


def _minimal_config():
    return {
        "starting_balance": 10_000.0,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "none",
        "rebalance_strategy": "always",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "drawdown_window_enabled": False,
        "dividend_reinvestment": False,
        "regimes": {
            "R1": {"dd_low": 0.0, "dd_high": 0.06, "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0},
            "R2": {"dd_low": 0.06, "dd_high": 0.28, "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0},
            "R3": {"dd_low": 0.28, "dd_high": 1.0, "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0},
        },
    }


class TestTradableAllocation(unittest.TestCase):
    def test_renormalizes_when_one_ticker_missing_price(self):
        cfg = _minimal_config()
        alloc = {"QQQ": 0.5, "TQQQ": 0.5, "XLU": 0.0}
        prices = pd.Series({"QQQ": 100.0, "TQQQ": np.nan, "XLU": 40.0})
        out = tradable_allocation(alloc, prices, cfg)
        self.assertAlmostEqual(out["QQQ"], 1.0)

    def test_tqqq_target_maps_to_qqq_when_tqqq_nan(self):
        cfg = _minimal_config()
        alloc = get_allocation_for_regime("R1", cfg)
        prices = pd.Series({"QQQ": 50.0, "TQQQ": np.nan, "XLU": 30.0})
        out = tradable_allocation(alloc, prices, cfg)
        self.assertEqual(list(out.keys()), ["QQQ"])
        self.assertAlmostEqual(out["QQQ"], 1.0)

    def test_r2_xlu_when_priced(self):
        cfg = _minimal_config()
        alloc = get_allocation_for_regime("R2", cfg)
        prices = pd.Series({"QQQ": 100.0, "TQQQ": np.nan, "XLU": 25.0})
        out = tradable_allocation(alloc, prices, cfg)
        self.assertAlmostEqual(out["XLU"], 1.0)


class TestBacktestYear2000Deploy(unittest.TestCase):
    """Portfolio must hold risk assets from the first bar when QQQ/XLU price but TQQQ is pre-IPO (NaN)."""

    def test_first_row_has_qqq_shares_when_regime_wants_tqqq_only(self):
        idx = pd.bdate_range("2000-01-03", periods=8)
        price_data = pd.DataFrame(
            {
                "QQQ": np.linspace(100.0, 108.0, len(idx)),
                "TQQQ": [np.nan] * len(idx),
                "XLU": np.linspace(24.0, 24.5, len(idx)),
                "SPY": np.linspace(140.0, 142.0, len(idx)),
            },
            index=idx,
        )
        cfg = _minimal_config()
        cfg["rebalance_frequency"] = "none"

        equity_df, _, _ = run_backtest(
            price_data,
            cfg,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
            dividend_data=None,
        )

        self.assertEqual(pd.to_datetime(equity_df["Date"].iloc[0]).date(), idx[0].date())
        q0 = float(equity_df["QQQ_shares"].iloc[0])
        self.assertGreater(
            q0,
            0.0,
            "Expected QQQ proxy position from day one when TQQQ has no price",
        )
        t0 = equity_df["TQQQ_shares"].iloc[0]
        self.assertTrue(
            pd.isna(t0) or float(t0) == 0.0,
            "TQQQ should not have shares while price is missing",
        )
        self.assertGreater(float(equity_df["Value"].iloc[0]), 0.0)

    def test_instant_rebalance_uses_qqq_before_tqqq_lists(self):
        idx = pd.bdate_range("2000-06-01", periods=12)
        price_data = pd.DataFrame(
            {
                "QQQ": np.linspace(95.0, 98.0, len(idx)),
                "TQQQ": [np.nan] * len(idx),
                "XLU": np.linspace(22.0, 22.4, len(idx)),
                "SPY": np.linspace(130.0, 131.0, len(idx)),
            },
            index=idx,
        )
        cfg = _minimal_config()
        cfg["rebalance_frequency"] = "instant"

        equity_df, _, _ = run_backtest(
            price_data,
            cfg,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
            dividend_data=None,
        )

        self.assertGreater(float(equity_df["QQQ_shares"].fillna(0).iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
