"""
Financial metric correctness: CAGR, volatility, max drawdown, Sharpe, Sortino.
Each metric verified against: (A) first-principles formula, (B) quantstats reference.
All functions must return Python float.

Protects: metrics.calculate_metrics(), utils.max_drawdown_from_equity_curve()
"""
import math
import numpy as np
import pandas as pd
import pytest
import quantstats as qs

from metrics import calculate_metrics
from utils import max_drawdown_from_equity_curve


def make_equity_df(start_val, daily_returns, start_date="2020-01-02"):
    dates = pd.bdate_range(start_date, periods=len(daily_returns) + 1)
    vals = [float(start_val)]
    for r in daily_returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.DataFrame({"Date": list(dates), "Value": vals})


def make_equity_df_from_values(values, start_date="2020-01-02"):
    dates = pd.bdate_range(start_date, periods=len(values))
    return pd.DataFrame({"Date": list(dates), "Value": list(map(float, values))})


def _random_returns(n, seed, mean=0.0004, std=0.012):
    np.random.seed(seed)
    return list(np.random.normal(mean, std, n))


@pytest.mark.metrics
def test_cagr_doubles_in_one_year():
    eq = make_equity_df(10_000, [0.0] * 251)
    eq["Value"] = np.linspace(10_000, 20_000, 252)
    m = calculate_metrics(eq)
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    expected = (20_000 / 10_000) ** (1.0 / years) - 1.0
    assert abs(m["cagr"] - expected) < 1e-9
    assert isinstance(m["cagr"], float)


@pytest.mark.metrics
def test_cagr_ten_years():
    n_days = 252 * 10
    eq = make_equity_df(10_000, [0.0] * (n_days - 1))
    eq["Value"] = np.linspace(10_000, 16_289.0, n_days)
    m = calculate_metrics(eq)
    assert abs(m["cagr"] - 0.05) < 0.005


@pytest.mark.metrics
def test_cagr_flat():
    eq = make_equity_df(10_000, [0.0] * 251)
    m = calculate_metrics(eq)
    assert abs(m["cagr"]) < 1e-9
    assert isinstance(m["cagr"], float)


@pytest.mark.metrics
def test_cagr_loss():
    n = 252 * 2
    eq = make_equity_df(10_000, [0.0] * (n - 1))
    eq["Value"] = np.linspace(10_000, 5_000, n)
    m = calculate_metrics(eq)
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    expected = (5_000 / 10_000) ** (1.0 / years) - 1.0
    assert abs(m["cagr"] - expected) < 1e-9


@pytest.mark.metrics
def test_cagr_returns_python_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=1))
    m = calculate_metrics(eq)
    assert type(m["cagr"]) is float, f"Expected float, got {type(m['cagr'])}"


@pytest.mark.metrics
def test_cagr_single_day_no_crash():
    eq = make_equity_df_from_values([10_000])
    m = calculate_metrics(eq)
    assert isinstance(m["cagr"], float)
    assert m["cagr"] == 0.0


@pytest.mark.metrics
def test_cagr_vs_quantstats():
    eq = make_equity_df(10_000, _random_returns(251, seed=99, mean=0.0004, std=0.010))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_cagr = float(qs.stats.cagr(returns, rf=0))
    assert abs(m["cagr"] - qs_cagr) < 0.01


@pytest.mark.metrics
def test_volatility_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=2))
    m = calculate_metrics(eq)
    assert type(m["volatility"]) is float


@pytest.mark.metrics
def test_volatility_flat_equity_is_zero():
    eq = make_equity_df_from_values([10_000] * 252)
    m = calculate_metrics(eq)
    assert m["volatility"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_volatility_single_return_no_crash():
    eq = make_equity_df_from_values([10_000, 10_100])
    m = calculate_metrics(eq)
    assert isinstance(m["volatility"], float)


@pytest.mark.metrics
def test_volatility_vs_quantstats():
    eq = make_equity_df(10_000, _random_returns(251, seed=77, std=0.015))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_vol = float(qs.stats.volatility(returns, periods=252, annualize=True))
    assert abs(m["volatility"] - qs_vol) < 1e-4


@pytest.mark.metrics
def test_max_drawdown_single_drop():
    s = pd.Series([100.0, 80.0, 60.0, 80.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.40, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_new_ath_resets():
    s = pd.Series([100.0, 120.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.25, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_no_decline():
    s = pd.Series([100.0, 110.0, 120.0, 130.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_is_nonpositive():
    np.random.seed(5)
    s = pd.Series(np.exp(np.cumsum(np.random.normal(0.001, 0.015, 500))) * 10_000)
    result = max_drawdown_from_equity_curve(s)
    assert result <= 0.0


@pytest.mark.metrics
def test_max_drawdown_returns_float():
    s = pd.Series([100.0, 80.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert type(result) is float


@pytest.mark.metrics
def test_max_drawdown_single_value():
    result = max_drawdown_from_equity_curve(pd.Series([10_000.0]))
    assert result == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_nan_handling():
    s = pd.Series([100.0, float("nan"), 80.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.20, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_vs_quantstats():
    np.random.seed(33)
    prices = pd.Series(
        10_000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 1261))),
        index=pd.bdate_range("2015-01-02", periods=1261),
    )
    result = max_drawdown_from_equity_curve(prices)
    returns = prices.pct_change().dropna()
    qs_dd = float(qs.stats.max_drawdown(returns))
    assert abs(result - qs_dd) < 1e-6


@pytest.mark.metrics
def test_sharpe_positive_returns():
    # Returns must vary — constant returns have zero volatility → Sharpe defaults to 0
    np.random.seed(11)
    daily = list(np.random.normal(0.0008, 0.008, 251))
    eq = make_equity_df(10_000, daily)
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] > 0
    assert isinstance(m["sharpe_ratio"], float)


@pytest.mark.metrics
def test_sharpe_zero_volatility():
    eq = make_equity_df_from_values([10_000] * 252)
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_sharpe_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=3))
    m = calculate_metrics(eq)
    assert type(m["sharpe_ratio"]) is float


@pytest.mark.metrics
def test_sharpe_negative_cagr():
    eq = make_equity_df(10_000, [-0.001] * 251)
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] < 0


@pytest.mark.metrics
def test_sharpe_vs_quantstats():
    eq = make_equity_df(10_000, _random_returns(1259, seed=77, mean=0.0003, std=0.012))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_sharpe = float(qs.stats.sharpe(returns, rf=0, periods=252))
    assert abs(m["sharpe_ratio"] - qs_sharpe) < 0.1


@pytest.mark.metrics
def test_sortino_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=4))
    m = calculate_metrics(eq)
    assert type(m["sortino_ratio"]) is float


@pytest.mark.metrics
def test_sortino_no_negative_returns_is_inf():
    # Use varying-but-strictly-positive returns so volatility > 0 but no downside dev
    np.random.seed(13)
    daily = list(np.abs(np.random.normal(0.001, 0.0005, 251)) + 1e-6)
    eq = make_equity_df(10_000, daily)
    m = calculate_metrics(eq)
    assert m["sortino_ratio"] == float("inf") or m["sortino_ratio"] > 100


@pytest.mark.metrics
def test_sortino_all_negative_returns():
    eq = make_equity_df(10_000, [-0.002] * 251)
    m = calculate_metrics(eq)
    assert m["sortino_ratio"] < 0


@pytest.mark.metrics
def test_sortino_vs_quantstats():
    eq = make_equity_df(10_000, _random_returns(1259, seed=88, mean=0.0003, std=0.012))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_sortino = float(qs.stats.sortino(returns, rf=0, periods=252))
    # Methodologies differ (project: CAGR/downside_dev with calendar-day CAGR;
    # qs: annualized_mean/downside_dev). 0.7 absorbs typical drift on 5y series.
    assert abs(m["sortino_ratio"] - qs_sortino) < 0.7
