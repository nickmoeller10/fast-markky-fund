"""
Unit tests for compute_rolling_ath_and_dd (rolling N-calendar-year reference high).

The brute-force oracle repeats the documented rules in code so we verify the
optimized implementation matches independent math.
"""

import unittest
import numpy as np
import pandas as pd

from backtest import compute_rolling_ath_and_dd


def brute_rolling_ath_and_dd(series: pd.Series, n_calendar_years: int):
    """
    Reference implementation: same semantics as backtest.compute_rolling_ath_and_dd.
    O(n^2) per slice max — only for testing.
    """
    s = series.sort_index().dropna()
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if n_calendar_years <= 0:
        ath = s.cummax()
        dd = (ath - s) / ath
        return ath, dd

    idx = s.index
    values = s.to_numpy(dtype=float, copy=False)
    cummax_vals = np.maximum.accumulate(values)
    first_ts = idx[0]
    need_until = first_ts + pd.DateOffset(years=n_calendar_years)
    ref = np.empty(len(s), dtype=float)

    for i, ts in enumerate(idx):
        if ts < need_until:
            ref[i] = cummax_vals[i]
            continue
        win_start = ts - pd.DateOffset(years=n_calendar_years)
        mask = (idx >= win_start) & (idx <= ts)
        ref[i] = float(values[mask].max())

    ath_series = pd.Series(ref, index=idx, dtype=float)
    dd_series = (ath_series - s) / ath_series
    return ath_series, dd_series


class TestRollingDrawdownWindow(unittest.TestCase):
    def assert_series_close(self, a: pd.Series, b: pd.Series, msg=None):
        self.assertTrue(a.index.equals(b.index), msg or "index mismatch")
        np.testing.assert_allclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            err_msg=msg,
        )

    def test_empty_series(self):
        s = pd.Series(dtype=float)
        ath, dd = compute_rolling_ath_and_dd(s, 5)
        self.assertTrue(ath.empty)
        self.assertTrue(dd.empty)

    def test_non_positive_n_matches_cummax(self):
        idx = pd.date_range("2019-01-02", periods=30, freq="B")
        s = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
        for n in (0, -1):
            ath_impl, dd_impl = compute_rolling_ath_and_dd(s, n)
            ath_brute, dd_brute = brute_rolling_ath_and_dd(s, n)
            self.assert_series_close(ath_impl, ath_brute)
            self.assert_series_close(dd_impl, dd_brute)
        expected_ath = s.cummax()
        self.assert_series_close(ath_impl, expected_ath)

    def test_matches_brute_random_walk_daily(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2015-03-04", periods=800, freq="B")
        # Log-normal style positive prices
        steps = rng.normal(0, 0.008, len(idx))
        prices = 100 * np.exp(np.cumsum(steps))
        s = pd.Series(prices, index=idx)
        for n_years in (1, 3, 5):
            impl_a, impl_d = compute_rolling_ath_and_dd(s, n_years)
            brute_a, brute_d = brute_rolling_ath_and_dd(s, n_years)
            self.assert_series_close(impl_a, brute_a, msg=f"n={n_years} ath")
            self.assert_series_close(impl_d, brute_d, msg=f"n={n_years} dd")

    def test_matches_brute_weekly_irregular_gaps(self):
        """Index need not be business-daily; algorithm uses actual timestamps."""
        idx = pd.date_range("2012-06-01", periods=400, freq="W-WED")
        rng = np.random.default_rng(7)
        prices = 50 + np.cumsum(rng.normal(0, 1.0, len(idx)))
        s = pd.Series(np.maximum(prices, 1.0), index=idx)
        impl_a, impl_d = compute_rolling_ath_and_dd(s, 2)
        brute_a, brute_d = brute_rolling_ath_and_dd(s, 2)
        self.assert_series_close(impl_a, brute_a)
        self.assert_series_close(impl_d, brute_d)

    def test_bootstrap_uses_cummax_until_n_calendar_years(self):
        """
        Before first_ts + N years, reference high equals cummax (standard ATH).
        """
        first = pd.Timestamp("2021-01-04")
        idx = pd.bdate_range(first, periods=600, freq="B")
        # Dip then recover so cummax is not always last price
        prices = np.concatenate(
            [
                np.linspace(100, 120, 200),
                np.linspace(120, 90, 150),
                np.linspace(90, 125, 250),
            ]
        )
        s = pd.Series(prices[: len(idx)], index=idx)
        n = 1
        need_until = first + pd.DateOffset(years=n)
        ath, _ = compute_rolling_ath_and_dd(s, n)
        cummax = s.cummax()
        mask = s.index < need_until
        self.assertTrue(mask.any(), "fixture should include bootstrap dates")
        self.assert_series_close(ath.loc[mask], cummax.loc[mask])

    def test_peak_scrolls_out_of_window(self):
        """
        After bootstrap, a large peak that falls outside the trailing window
        must stop anchoring drawdown — reference high drops to max inside window.
        """
        # ~6 years of weeklies so a 2-year window can exclude an early spike
        idx = pd.date_range("2015-01-07", periods=320, freq="W-FRI")
        values = np.full(len(idx), 100.0)
        # Early spike only in first year of data
        values[0:15] = 200.0
        s = pd.Series(values, index=idx)
        ath, dd = compute_rolling_ath_and_dd(s, 2)
        last_ts = idx[-1]
        win_start = last_ts - pd.DateOffset(years=2)
        in_window = s.loc[(s.index >= win_start) & (s.index <= last_ts)]
        expected_ref = float(in_window.max())
        self.assertAlmostEqual(ath.loc[last_ts], expected_ref, places=10)
        self.assertLess(ath.loc[last_ts], 200.0 - 1e-6)
        expected_dd = (expected_ref - s.loc[last_ts]) / expected_ref
        self.assertAlmostEqual(dd.loc[last_ts], expected_dd, places=10)

    def test_unsorted_input_sorted_like_brute(self):
        idx = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-03", "2020-01-07"])
        s = pd.Series([10.0, 40.0, 25.0, 35.0], index=idx)
        impl_a, impl_d = compute_rolling_ath_and_dd(s, 1)
        brute_a, brute_d = brute_rolling_ath_and_dd(s, 1)
        self.assert_series_close(impl_a, brute_a)
        self.assert_series_close(impl_d, brute_d)

    def test_nan_dropped_consistently(self):
        idx = pd.bdate_range("2019-01-02", periods=20, freq="B")
        vals = np.linspace(100, 120, len(idx)).astype(float)
        vals[5] = np.nan
        s = pd.Series(vals, index=idx)
        impl_a, impl_d = compute_rolling_ath_and_dd(s, 1)
        brute_a, brute_d = brute_rolling_ath_and_dd(s, 1)
        self.assert_series_close(impl_a, brute_a)
        self.assert_series_close(impl_d, brute_d)
        self.assertEqual(len(impl_a), len(s.dropna()))
        self.assertTrue(impl_a.index.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
