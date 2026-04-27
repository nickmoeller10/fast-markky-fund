"""
Tests for optimizer.results — Pareto frontier, stability filter, config parser.
"""
import json

import pandas as pd
import pytest

from optimizer.results import (
    pareto_frontier,
    parse_config,
    stability_filter,
    top_n_by,
)


def _trial_row(num, cagr, dd, rebs, p05=0.05, value=None):
    return {
        "number": num,
        "value": value if value is not None else cagr,
        "median_cagr": cagr,
        "p05_cagr": p05,
        "worst_max_dd": dd,
        "median_rebalances_per_year": rebs,
    }


@pytest.mark.unit
def test_pareto_frontier_picks_non_dominated():
    """
    Maximize CAGR + worst_max_dd (closer to 0 is better); minimize rebalances.
    Trial A is dominated by B (B is better on every axis).
    """
    df = pd.DataFrame([
        _trial_row(0, 0.10, -0.30, 5.0),     # A — dominated by B
        _trial_row(1, 0.15, -0.20, 3.0),     # B — non-dominated
        _trial_row(2, 0.20, -0.40, 2.0),     # C — non-dominated (highest CAGR, lowest rebs)
        _trial_row(3, 0.05, -0.10, 4.0),     # D — non-dominated (best DD)
    ])
    pf = pareto_frontier(df)
    keep_numbers = sorted(pf["number"].tolist())
    assert keep_numbers == [1, 2, 3]


@pytest.mark.unit
def test_pareto_frontier_empty_input():
    df = pd.DataFrame(columns=["median_cagr", "worst_max_dd", "median_rebalances_per_year"])
    pf = pareto_frontier(df)
    assert pf.empty


@pytest.mark.unit
def test_top_n_by_sorts_descending_by_default():
    df = pd.DataFrame([
        _trial_row(0, 0.10, -0.30, 5.0),
        _trial_row(1, 0.20, -0.20, 4.0),
        _trial_row(2, 0.05, -0.10, 3.0),
    ])
    top = top_n_by(df, "median_cagr", n=2)
    assert top["number"].tolist() == [1, 0]


@pytest.mark.unit
def test_stability_filter_drops_unstable_trials():
    df = pd.DataFrame([
        _trial_row(0, 0.20, -0.60, 5.0, p05=-0.05),   # p05 negative → drop
        _trial_row(1, 0.15, -0.30, 4.0, p05=0.02),    # keep
        _trial_row(2, 0.10, -0.95, 3.0, p05=0.01),    # too-deep DD → drop
    ])
    out = stability_filter(df, min_p05_cagr=0.0, max_worst_dd=-0.50)
    assert out["number"].tolist() == [1]


@pytest.mark.unit
def test_parse_config_returns_dict_from_user_attrs_json():
    cfg = {"starting_balance": 10000, "regimes": {"R1": {"dd_low": 0.0}}}
    row = pd.Series({"user_attrs_config_json": json.dumps(cfg, default=str)})
    parsed = parse_config(row)
    assert parsed["starting_balance"] == 10000
    assert "R1" in parsed["regimes"]


@pytest.mark.unit
def test_parse_config_returns_empty_on_missing_or_invalid():
    assert parse_config(pd.Series({})) == {}
    assert parse_config(pd.Series({"config_json": "not-json"})) == {}
