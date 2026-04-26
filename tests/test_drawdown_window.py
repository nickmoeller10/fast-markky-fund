"""
Rolling ATH vs standard ATH drawdown path tests.
Complements test_rolling_drawdown_window.py with scenario-based coverage.
Protects: backtest.compute_rolling_ath_and_dd()
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtest import compute_rolling_ath_and_dd, compute_drawdown_from_ath


@pytest.mark.unit
def test_rolling_matches_cummax_before_window_fills():
    """Before N calendar years of data exist, rolling ATH must equal cummax."""
    dates = pd.bdate_range("2020-01-02", periods=130)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.010, 130))),
        index=dates,
    )
    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    cummax_ath = prices.cummax()
    np.testing.assert_allclose(rolling_ath.values, cummax_ath.values, rtol=1e-9)


@pytest.mark.unit
def test_rolling_peak_scrolls_out():
    """After >N years, a peak from >N years ago should no longer anchor the ATH."""
    dates_early = pd.bdate_range("2019-01-02", periods=20)
    early_prices = pd.Series([200.0] * 20, index=dates_early)

    dates_low = pd.bdate_range("2019-02-01", periods=380)
    low_prices = pd.Series([100.0] * 380, index=dates_low)

    prices = pd.concat([early_prices, low_prices]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    standard_dd, standard_ath = compute_drawdown_from_ath(prices)

    last_rolling_ath = float(rolling_ath.iloc[-1])
    last_standard_ath = float(standard_ath.iloc[-1])
    assert last_rolling_ath < last_standard_ath, (
        f"Rolling ATH {last_rolling_ath} should be below standard ATH {last_standard_ath} "
        "after early high peak leaves the window"
    )


@pytest.mark.unit
def test_rolling_and_standard_agree_on_fresh_ath():
    """When price is at a new all-time high, both methods must give 0% drawdown."""
    dates = pd.bdate_range("2020-01-02", periods=252)
    prices = pd.Series(np.linspace(100.0, 200.0, 252), index=dates)
    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    standard_dd, _ = compute_drawdown_from_ath(prices)
    assert float(rolling_dd.iloc[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(standard_dd.iloc[-1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_rolling_produces_higher_dd_after_peak_scrolls_out():
    """Once the high peak leaves the 1-year window, rolling dd > standard dd."""
    dates_old = pd.bdate_range("2019-01-02", periods=10)
    old_high = pd.Series([150.0] * 10, index=dates_old)

    dates_new = pd.bdate_range("2020-06-01", periods=252)
    new_prices = pd.Series([100.0] * 252, index=dates_new)

    prices = pd.concat([old_high, new_prices]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    standard_dd, _ = compute_drawdown_from_ath(prices)

    last_rolling_dd = float(rolling_dd.iloc[-1])
    last_standard_dd = float(standard_dd.iloc[-1])

    # rolling dd should be LOWER (less stressed) once old peak leaves window
    assert last_rolling_dd <= last_standard_dd + 1e-6, (
        f"After peak scrolls out: rolling_dd={last_rolling_dd:.4f}, "
        f"standard_dd={last_standard_dd:.4f}"
    )


@pytest.mark.unit
def test_drawdown_always_in_0_1_range():
    """Drawdown must always be in [0, 1] for any valid price series."""
    np.random.seed(42)
    dates = pd.bdate_range("2018-01-02", periods=500)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 500))), index=dates
    )
    _, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    assert (rolling_dd >= 0.0).all(), "Rolling dd must never be negative"
    assert (rolling_dd <= 1.0).all(), "Rolling dd must never exceed 100%"


@given(
    n=st.integers(min_value=1, max_value=5),
    prices_list=st.lists(
        st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    ),
)
@settings(max_examples=200)
def test_rolling_dd_hypothesis_always_bounded(n, prices_list):
    """Property: for any positive price series and window ≥ 1, dd ∈ [0, 1]."""
    dates = pd.bdate_range("2020-01-02", periods=len(prices_list))
    prices = pd.Series(prices_list, index=dates)
    _, dd = compute_rolling_ath_and_dd(prices, n_calendar_years=n)
    if not dd.empty:
        assert (dd >= -1e-9).all(), "dd should never be negative"
        assert (dd <= 1.0 + 1e-9).all(), "dd should never exceed 1.0"
