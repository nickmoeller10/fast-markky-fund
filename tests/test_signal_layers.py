"""Unit tests for VIX z-score buckets, composite labels, and MACD/MA wiring."""

import unittest
import numpy as np
import pandas as pd

from signal_layers import (
    _signal_l1,
    _vix_regime_from_z,
    _signal_label_from_total,
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
    def test_spy_macd_ma_columns_exist(self):
        n = 260
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        spy = 100 + np.cumsum(np.random.randn(n) * 0.3)
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


if __name__ == "__main__":
    unittest.main()
