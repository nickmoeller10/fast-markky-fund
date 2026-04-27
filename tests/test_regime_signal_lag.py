"""
Regime uses prior trading day's drawdown; execution still uses same-day prices.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import build_regime_signal_drawdown, run_backtest
from regime_engine import compute_drawdown_from_ath, determine_regime
from rebalance_engine import rebalance_portfolio


def _cfg_instant_always():
    return {
        "starting_balance": 10_000.0,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
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


class TestBuildRegimeSignalDrawdown(unittest.TestCase):
    def test_second_day_uses_prior_close_dd(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        # Day0 ATH 100 dd 0; day1 price 100 dd 0; day2 price 85 → same-day dd 0.15 (R2), signal on day2 = 0 (R1)
        qqq = pd.Series([100.0, 100.0, 85.0], index=idx)
        dd_raw, _ = compute_drawdown_from_ath(qqq)
        dd_raw = (
            dd_raw.reindex(idx)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        sig = build_regime_signal_drawdown(dd_raw, idx, idx, None)
        self.assertAlmostEqual(float(sig.loc[idx[0]]), 0.0, places=6)
        self.assertAlmostEqual(float(sig.loc[idx[1]]), 0.0, places=6)
        self.assertAlmostEqual(float(sig.loc[idx[2]]), 0.0, places=6)
        self.assertAlmostEqual(float(dd_raw.loc[idx[2]]), 0.15, places=6)


class TestBacktestRegimeLag(unittest.TestCase):
    @patch("data_cache.cached_yf_download")
    @patch("yfinance.download")
    def test_instant_no_same_day_regime_flip(self, mock_download, mock_cached):
        # Patch BOTH the raw yfinance call and the cache wrapper so a populated
        # real-data cache doesn't leak into this synthetic 3-day scenario.
        mock_download.return_value = pd.DataFrame()
        mock_cached.return_value = pd.DataFrame()
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        price_data = pd.DataFrame(
            {
                "QQQ": [100.0, 100.0, 85.0],
                "TQQQ": [np.nan] * 3,
                "XLU": [24.0, 24.0, 24.0],
                "SPY": [300.0, 300.0, 300.0],
            },
            index=idx,
        )
        cfg = _cfg_instant_always()
        equity_df, _, _ = run_backtest(
            price_data,
            cfg,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
            dividend_data=None,
        )
        self.assertEqual(equity_df["Market_Regime"].iloc[2], "R1")
        self.assertEqual(equity_df["Portfolio_Regime"].iloc[2], "R1")


if __name__ == "__main__":
    unittest.main()
