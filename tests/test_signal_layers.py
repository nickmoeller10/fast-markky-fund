"""Unit tests for VIX z-score buckets, composite labels, and MACD/MA wiring."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from signal_layers import (
    _signal_l1,
    _vix_regime_from_z,
    _signal_label_from_total,
    build_signal_total_series,
    compute_signal_layer_columns,
)


class TestSignalL1AndVIXRegime(unittest.TestCase):
    def test_l1_thresholds(self):
        z = pd.Series([3.0, 2.0, 1.5, 0.0, -1.0, -1.5, -2.0, -3.0])
        s = _signal_l1(z)
        # z=-1 is not > -1 → L1 = -1. z=2 is not > 2 → >1 → +1.
        np.testing.assert_array_equal(
            s.values,
            np.array([2.0, 1.0, 1.0, 0.0, -1.0, -1.0, -2.0, -2.0]),
        )

    def test_vix_regime_labels(self):
        z = pd.Series([2.5, 1.5, 0.0, -1.5, -2.5])
        r = _vix_regime_from_z(z)
        self.assertEqual(r.iloc[0], "Extreme Fear")
        self.assertEqual(r.iloc[1], "Elevated")
        self.assertEqual(r.iloc[2], "Normal")
        self.assertEqual(r.iloc[3], "Complacent")
        self.assertEqual(r.iloc[4], "Extreme Complacency")


class TestSignalLabel(unittest.TestCase):
    def test_total_buckets(self):
        t = pd.Series([6.0, 4.0, 2.0, 0.0, -2.0, -4.0])
        lab = _signal_label_from_total(t)
        self.assertEqual(lab.iloc[0], "Strong Buy")
        self.assertEqual(lab.iloc[1], "Buy")
        self.assertEqual(lab.iloc[2], "Lean Long")
        self.assertEqual(lab.iloc[3], "Neutral")
        self.assertEqual(lab.iloc[4], "Reduce")
        self.assertEqual(lab.iloc[5], "Strong Sell")


class TestComputeColumns(unittest.TestCase):
    @patch("signal_layers.load_vix_series")
    @patch("signal_layers.load_spy_series")
    def test_spy_macd_ma_columns_exist(self, mock_spy, mock_vix):
        n = 260
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        spy = 100 + np.cumsum(np.random.randn(n) * 0.3)
        ext = pd.date_range("2017-01-01", periods=900, freq="B")
        mock_spy.return_value = pd.Series(
            100 + np.cumsum(np.random.randn(len(ext)) * 0.2), index=ext
        )
        mock_vix.return_value = pd.Series(18 + np.random.randn(len(ext)), index=ext)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Value": np.linspace(10000, 11000, n),
                "VIX": 20 + np.random.randn(n) * 2,
                "SPY_price": spy,
            }
        )
        out = compute_signal_layer_columns(df)
        for c in (
            "VIX_252d_mean",
            "VIX_zscore",
            "Signal_L1",
            "MACD_line",
            "Signal_L2",
            "MA_50",
            "MA_200",
            "Signal_L3",
            "Signal_total",
            "Signal_label",
        ):
            self.assertIn(c, out.columns)


class TestExtendedHistoryWarmStart(unittest.TestCase):
    def test_signal_total_valid_on_first_trading_day(self):
        """Rollings use pre-panel history; first portfolio bar should not be all-NaN."""
        full = pd.bdate_range("2018-06-04", periods=320)
        ti = full[-60:]
        rng = np.random.default_rng(0)
        spy = pd.Series(280 + np.cumsum(rng.normal(0, 0.5, len(full))), index=full)
        vix = pd.Series(18 + rng.normal(0, 1, len(full)), index=full)
        tot = build_signal_total_series(ti, spy, vix)
        self.assertTrue(bool(pd.notna(tot.iloc[0])), "first Signal_total should be defined")
        self.assertLess(tot.isna().sum(), len(tot) * 0.5)


if __name__ == "__main__":
    unittest.main()
