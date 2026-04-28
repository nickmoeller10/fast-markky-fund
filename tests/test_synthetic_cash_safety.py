"""
Safety guards for the synthetic CASH ticker.

"CASH" is a real yfinance ticker (Pathward Financial Inc.) and "$" might also
be picked up by yfinance fuzzy matching. These tests pin down the contract:

  1. Synthetic tickers in `data_loader.SYNTHETIC_TICKERS` are NEVER sent to
     yfinance — load_price_data filters them out before the network call.
  2. The synthetic CASH price series is monotonically non-decreasing
     (a money market fund proxy: zero drawdown, daily-compounded carry).
  3. Both "CASH" and "$" produce equivalent series, named per the request.
  4. End-to-end through run_backtest, the synthetic ticker's normalized
     value series is also monotonically non-decreasing — i.e. you can never
     observe a drawdown on the cash sleeve in dashboard / Excel output.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    CASH_ALIASES,
    CASH_APY,
    CASH_TICKER,
    DOLLAR_TICKER,
    SYNTHETIC_TICKERS,
    _build_cash_series,
    _split_real_and_synthetic,
    load_price_data,
)


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cash_series_starts_at_one_and_is_monotonic():
    idx = pd.bdate_range("2010-01-04", periods=252)
    s = _build_cash_series(idx)
    assert s.iloc[0] == pytest.approx(1.0)
    diffs = s.diff().dropna()
    assert (diffs >= 0).all(), "CASH must be monotonically non-decreasing"


@pytest.mark.unit
def test_cash_series_compounds_to_apy_over_one_year():
    idx = pd.bdate_range("2010-01-04", periods=252 + 1)  # 252 returns over 253 prices
    s = _build_cash_series(idx)
    realized = s.iloc[-1] / s.iloc[0] - 1
    assert realized == pytest.approx(CASH_APY, abs=1e-6)


@pytest.mark.unit
def test_cash_series_named_per_request():
    from data_loader import LEGACY_CASH_TICKER

    idx = pd.bdate_range("2020-01-02", periods=10)
    s_legacy = _build_cash_series(idx, name=LEGACY_CASH_TICKER)
    s_dollar = _build_cash_series(idx, name=DOLLAR_TICKER)
    assert s_legacy.name == "CASH"
    assert s_dollar.name == "$"
    # Same numeric values regardless of label
    np.testing.assert_array_equal(s_legacy.values, s_dollar.values)


# ---------------------------------------------------------------------------
# load_price_data filtering tests — the critical bug guards
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_filters_known_synthetic_aliases():
    real, syn = _split_real_and_synthetic(["QQQ", "CASH", "TQQQ", "$"])
    assert real == ["QQQ", "TQQQ"]
    assert syn == ["CASH", "$"]


@pytest.mark.unit
def test_load_price_data_does_not_send_cash_to_yfinance():
    """
    The smoking-gun guard. If load_price_data ever sends "CASH" to yfinance
    we'll silently get Pathward Financial Inc. equity data with normal
    drawdowns. Patch the cache wrapper and verify the synthetic ticker
    never appears in the call args.
    """
    fake = pd.DataFrame(
        {("Close", "QQQ"): np.linspace(100.0, 110.0, 10)},
        index=pd.bdate_range("2020-01-02", periods=10),
    )
    fake.columns = pd.MultiIndex.from_tuples([("Close", "QQQ")])

    with patch("data_loader.cached_yf_download", return_value=fake) as mock:
        result = load_price_data(["QQQ", "CASH"], "2020-01-02", end_date="2020-01-15")

    args, kwargs = mock.call_args
    requested_tickers = args[0] if args else kwargs.get("tickers")
    assert "CASH" not in requested_tickers
    assert "QQQ" in requested_tickers
    # Result must contain the synthetic CASH column
    assert "CASH" in result.columns
    # CASH series must be monotonic
    assert (result["CASH"].diff().dropna() >= 0).all()


@pytest.mark.unit
def test_load_price_data_does_not_send_dollar_alias_to_yfinance():
    """Same guard for the '$' alias."""
    fake = pd.DataFrame(
        {("Close", "QQQ"): np.linspace(100.0, 110.0, 10)},
        index=pd.bdate_range("2020-01-02", periods=10),
    )
    fake.columns = pd.MultiIndex.from_tuples([("Close", "QQQ")])

    with patch("data_loader.cached_yf_download", return_value=fake) as mock:
        result = load_price_data(["QQQ", "$"], "2020-01-02", end_date="2020-01-15")

    args, kwargs = mock.call_args
    requested_tickers = args[0] if args else kwargs.get("tickers")
    assert "$" not in requested_tickers
    assert "$" in result.columns


@pytest.mark.unit
def test_load_price_data_all_synthetic_skips_yfinance_entirely():
    """When the only tickers requested are synthetic, no network call happens."""
    with patch("data_loader.cached_yf_download") as mock:
        result = load_price_data(["CASH", "$"], "2020-01-02", end_date="2020-12-31")
    mock.assert_not_called()
    assert "CASH" in result.columns
    assert "$" in result.columns


# ---------------------------------------------------------------------------
# End-to-end: cash normalized in equity_df is monotonic
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cash_normalized_is_monotonic_through_run_backtest():
    """
    The original bug report: 'normalized version of CASH is sometimes going
    down.' That can only happen if CASH ever holds real-equity prices.
    This test runs a full backtest with CASH in the panel and asserts the
    `CASH_norm` column never decreases.
    """
    from backtest import run_backtest, compute_drawdown_from_ath
    from regime_engine import determine_regime
    from rebalance_engine import rebalance_portfolio

    # Build a deterministic synthetic price panel and inject CASH via the
    # data_loader pathway so the test exercises real plumbing.
    fake_yf = pd.DataFrame(
        {
            ("Close", "QQQ"): np.linspace(100.0, 200.0, 252),
            ("Close", "TQQQ"): np.linspace(50.0, 150.0, 252),
            ("Close", "XLU"): np.linspace(60.0, 75.0, 252),
            ("Close", "SPY"): np.linspace(300.0, 400.0, 252),
        },
        index=pd.bdate_range("2020-01-02", periods=252),
    )

    with patch("data_loader.cached_yf_download", return_value=fake_yf):
        panel = load_price_data(
            ["QQQ", "TQQQ", "XLU", "SPY", "CASH"],
            "2020-01-02",
            end_date="2020-12-31",
        )

    cfg = {
        "starting_balance": 10_000.0,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "always",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "CASH"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU", "CASH"],
        "drawdown_window_enabled": False,
        "dividend_reinvestment": False,
        "regimes": {
            # Force the portfolio into 100% CASH so we can isolate the CASH path.
            "R1": {
                "dd_low": 0.0, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "CASH": 1.0,
            },
        },
    }

    # Patch the cached download AND the inline drawdown-history call inside
    # run_backtest so the test is fully isolated.
    with (
        patch("data_loader.cached_yf_download", return_value=fake_yf),
        patch("data_cache.cached_yf_download", return_value=fake_yf),
        patch("data_loader.load_spy_series", return_value=pd.Series(dtype=float)),
        patch("data_loader.load_vix_series", return_value=pd.Series(dtype=float)),
    ):
        equity_df, _, _ = run_backtest(
            panel,
            cfg,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
        )

    assert "CASH_norm" in equity_df.columns, (
        "CASH_norm column must be present in equity_df"
    )
    cash_norm = equity_df["CASH_norm"].dropna()
    assert len(cash_norm) > 0
    diffs = cash_norm.diff().dropna()
    # Allow a tiny floating-point tolerance — but no real drop.
    assert (diffs >= -1e-9).all(), (
        f"CASH_norm must be monotonically non-decreasing. "
        f"Worst diff: {diffs.min():.6e} on {diffs.idxmin()}"
    )


# ---------------------------------------------------------------------------
# worst_case_simulator path — the second leak source
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_worst_case_simulator_does_not_send_cash_to_yfinance():
    """The worst-case simulator (default app entry point) used to forward
    every ticker (including CASH/$) to yf.download, returning Pathward equity
    instead of synthetic MMF. This guard fails the test if any synthetic
    alias appears in a yfinance call."""
    import worst_case_simulator
    from unittest.mock import MagicMock

    real_yf_download = worst_case_simulator.yf.download
    leaked: list[list[str]] = []

    def tracking_download(tickers, *args, **kwargs):
        if isinstance(tickers, str):
            tlist = [tickers]
        else:
            tlist = list(tickers)
        for t in tlist:
            if t in SYNTHETIC_TICKERS:
                leaked.append(tlist)
                raise AssertionError(
                    f"LEAK: yf.download called with synthetic ticker(s) {tlist}"
                )
        return real_yf_download(tickers, *args, **kwargs)

    mock_yf = MagicMock()
    mock_yf.download = tracking_download
    with patch.object(worst_case_simulator, "yf", mock_yf):
        df, earliest_dates = worst_case_simulator.generate_worst_case_prices(
            config={},
            requested_tickers=["QQQ", "TQQQ", "XLU", DOLLAR_TICKER],
            start_date="2010-01-04",
            end_date="2012-12-31",
        )

    assert leaked == [], f"Synthetic ticker leaked to yfinance: {leaked}"
    assert DOLLAR_TICKER in df.columns
    # The synthetic series MUST be monotonically non-decreasing
    diffs = df[DOLLAR_TICKER].diff().dropna()
    assert (diffs >= -1e-12).all(), (
        f"$ column must be monotonically non-decreasing in worst-case sim "
        f"(worst diff: {diffs.min():.6e})"
    )
    assert earliest_dates.get(DOLLAR_TICKER) == pd.Timestamp("1980-01-01")


@pytest.mark.unit
def test_worst_case_simulator_get_earliest_date_short_circuits_synthetic():
    """get_earliest_date must return the sentinel pre-1985 date for synthetic
    tickers without touching yfinance — used by the Streamlit UI's inception
    panel via app._cached_ticker_earliest_date."""
    import worst_case_simulator

    with patch.object(worst_case_simulator, "yf") as mock_yf:
        mock_yf.download.side_effect = AssertionError("yfinance must not be called")
        ts_cash = worst_case_simulator.get_earliest_date("CASH")
        ts_dollar = worst_case_simulator.get_earliest_date("$")

    assert ts_cash == pd.Timestamp("1980-01-01")
    assert ts_dollar == pd.Timestamp("1980-01-01")


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_synthetic_ticker_registry_contains_both_aliases():
    assert CASH_TICKER in SYNTHETIC_TICKERS
    assert DOLLAR_TICKER in SYNTHETIC_TICKERS
    from data_loader import LEGACY_CASH_TICKER
    assert CASH_ALIASES == frozenset({CASH_TICKER, LEGACY_CASH_TICKER})
