"""
Unit tests for the optimizer's scoring + entry-point sampling.
Avoids running real backtests — patches score._single_run_metrics with a
deterministic stub so we can verify the aggregation math precisely.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.mark.unit
def test_sample_entry_points_returns_n_dates_within_window():
    from optimizer.score import _sample_entry_points

    dates = pd.bdate_range("2010-01-04", periods=4000)
    rng = np.random.default_rng(0)
    chosen = _sample_entry_points(dates, n=10, rng=rng, min_years_remaining=5.0)
    assert len(chosen) == 10
    # Every chosen date must leave at least 5 years before the last date
    cutoff = dates[-1] - pd.DateOffset(years=5)
    assert all(d <= cutoff for d in chosen)


@pytest.mark.unit
def test_sample_entry_points_handles_short_history():
    from optimizer.score import _sample_entry_points

    short = pd.bdate_range("2024-01-02", periods=200)
    rng = np.random.default_rng(0)
    chosen = _sample_entry_points(short, n=10, rng=rng, min_years_remaining=5.0)
    assert chosen == []


@pytest.mark.unit
def test_sample_entry_points_deterministic_with_seed():
    from optimizer.score import _sample_entry_points

    dates = pd.bdate_range("2010-01-04", periods=4000)
    a = _sample_entry_points(dates, n=20, rng=np.random.default_rng(42), min_years_remaining=5.0)
    b = _sample_entry_points(dates, n=20, rng=np.random.default_rng(42), min_years_remaining=5.0)
    assert a == b


@pytest.mark.unit
def test_score_aggregation_with_stub_metrics(tmp_path):
    """
    Replace _single_run_metrics + _load_panel with stubs so we test the
    aggregation math without touching the cache or running a real backtest.
    """
    from optimizer import score as score_mod

    # Stub panel: just needs an index. Content doesn't matter (single-run is stubbed).
    fake_panel = pd.DataFrame(
        {"QQQ": np.linspace(100, 200, 4000)},
        index=pd.bdate_range("2010-01-04", periods=4000),
    )

    fake_metrics_per_run = [
        {"cagr": 0.20, "max_dd": -0.30, "rebalance_count": 100, "years": 25.0},  # full
        {"cagr": 0.15, "max_dd": -0.25, "rebalance_count": 80,  "years": 20.0},
        {"cagr": 0.10, "max_dd": -0.40, "rebalance_count": 90,  "years": 18.0},
        {"cagr": 0.05, "max_dd": -0.20, "rebalance_count": 50,  "years": 15.0},
        {"cagr": 0.25, "max_dd": -0.15, "rebalance_count": 30,  "years": 10.0},
    ]
    runs_iter = iter(fake_metrics_per_run)

    def fake_single_run(panel, config):
        return next(runs_iter)

    config = {
        "start_date": "2010-01-04",
        "end_date": None,
        "tickers": ["QQQ"],
    }

    with (
        patch("optimizer.score._load_panel", return_value=fake_panel),
        patch("optimizer.score._single_run_metrics", side_effect=fake_single_run),
    ):
        result = score_mod.score_config(config, n_entry_points=4, rng_seed=0)

    # 1 full + 4 entry-point runs
    assert result["n_runs"] == 5
    # CAGRs: [0.20, 0.15, 0.10, 0.05, 0.25] → median 0.15, p05 ≈ 0.06
    assert result["median_cagr"] == pytest.approx(0.15, abs=1e-9)
    assert result["best_cagr"] == pytest.approx(0.25, abs=1e-9)
    assert result["worst_cagr"] == pytest.approx(0.05, abs=1e-9)
    # max_dds: [-0.30, -0.25, -0.40, -0.20, -0.15] → worst (min) = -0.40, median = -0.25
    assert result["worst_max_dd"] == pytest.approx(-0.40, abs=1e-9)
    assert result["median_max_dd"] == pytest.approx(-0.25, abs=1e-9)
    # rebalances per year: [4.0, 4.0, 5.0, 3.33, 3.0] → median ~4.0
    assert result["median_rebalances_per_year"] == pytest.approx(4.0, abs=1e-2)
    # Score sanity: median_cagr / |worst_max_dd| - reb penalty - tail penalty (p05 > 0)
    expected_calmar = 0.15 / max(abs(-0.40), 0.05)  # = 0.375
    expected_score = expected_calmar - 0.10 * result["median_rebalances_per_year"]  # tail penalty = 0
    assert result["score"] == pytest.approx(expected_score, abs=1e-3)


@pytest.mark.unit
def test_score_returns_error_on_empty_panel():
    from optimizer import score as score_mod

    config = {"start_date": "2010-01-04", "end_date": None, "tickers": ["QQQ"]}
    with patch("optimizer.score._load_panel", return_value=pd.DataFrame()):
        result = score_mod.score_config(config, n_entry_points=4)
    assert result["score"] == pytest.approx(-999.0)
    assert result["error"] == "empty_panel"


@pytest.mark.unit
def test_parameter_space_produces_valid_config():
    """A trial drawn from the search space must be shape-compatible with run_backtest."""
    import optuna
    from optimizer.parameter_space import suggest_config

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    config = suggest_config(trial)

    # Required top-level keys
    for k in ("starting_balance", "drawdown_ticker", "tickers", "allocation_tickers", "regimes"):
        assert k in config

    # Three regimes with proper structure
    assert set(config["regimes"]) == {"R1", "R2", "R3"}
    for r in ("R1", "R2", "R3"):
        block = config["regimes"][r]
        assert "dd_low" in block and "dd_high" in block
        assert block["rebalance_on_downward"] in ("match", "hold")
        assert block["rebalance_on_upward"] in ("match", "hold")
        assert "signal_overrides" in block
        # Core ticker weights present
        for t in ("TQQQ", "QQQ", "XLU"):
            assert t in block

    # Thresholds non-overlapping
    assert config["regimes"]["R1"]["dd_high"] == config["regimes"]["R2"]["dd_low"]
    assert config["regimes"]["R2"]["dd_high"] == config["regimes"]["R3"]["dd_low"]
    assert config["regimes"]["R3"]["dd_high"] == 1.0

    # drawdown_window_years from the categorical set
    assert config["drawdown_window_years"] in (1, 2, 3, 5)

    # Strategy fixed
    assert config["rebalance_strategy"] == "per_regime"
    assert config["rebalance_frequency"] == "instant"

    # Allocation tickers always include the core 3
    for t in ("TQQQ", "QQQ", "XLU"):
        assert t in config["allocation_tickers"]
