"""
Monte Carlo entry-point evaluator + aggregate scoring.

For a given candidate config, run the full-history backtest plus N random
entry-point backtests, aggregate metrics, and return a dict that includes
the primary score (used by Optuna) and the Pareto axes (saved for analysis).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

# Force frozen mode for the optimizer — every backtest reads from the
# committed snapshot, never the network. Set BEFORE importing run_backtest.
os.environ.setdefault("FMF_DATA_MODE", "frozen")

from backtest import run_backtest, compute_drawdown_from_ath  # noqa: E402
from data_loader import load_price_data  # noqa: E402
from optimizer.parameter_space import ALL_PANEL_TICKERS  # noqa: E402
from regime_engine import determine_regime  # noqa: E402
from rebalance_engine import rebalance_portfolio  # noqa: E402
from utils import max_drawdown_from_equity_curve  # noqa: E402


# Scoring coefficients (exposed for tuning)
SCORE_DD_FLOOR = 0.05            # robust Calmar denominator floor
SCORE_REBAL_PENALTY = 0.10        # turnover penalty per (median rebalances / year)
SCORE_TAIL_PENALTY = 0.50         # weight on p05_cagr below threshold
SCORE_P05_THRESHOLD = 0.0          # configs with p05_cagr < 0 get penalized


def _load_panel(config: dict, full_start: str, full_end: str | None) -> pd.DataFrame:
    """
    Cached load of the panel. Always loads the full optimizer pool so the cache
    has a single entry per (start, end) shape — run_backtest will index just
    the tickers listed in config["tickers"], so extra columns are harmless.
    """
    return load_price_data(ALL_PANEL_TICKERS, full_start, end_date=full_end)


def _single_run_metrics(
    price_panel: pd.DataFrame, config: dict
) -> dict[str, float]:
    """Run one backtest, return CAGR / max_dd / rebalance_count / years."""
    eq, _, _ = run_backtest(
        price_panel,
        config,
        lambda s: compute_drawdown_from_ath(s),
        lambda dd, c: determine_regime(dd, c),
        rebalance_portfolio,
    )
    if eq is None or eq.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "rebalance_count": 0, "years": 0.0}

    vals = eq["Value"]
    start_v = float(vals.iloc[0])
    end_v = float(vals.iloc[-1])
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0 if years > 0 and start_v > 0 else 0.0
    max_dd = float(max_drawdown_from_equity_curve(vals))
    rebalance_count = int((eq.get("Rebalanced", pd.Series(dtype=str)) == "Rebalanced").sum())
    return {
        "cagr": float(cagr),
        "max_dd": max_dd,
        "rebalance_count": rebalance_count,
        "years": float(years),
    }


def _sample_entry_points(
    panel_dates: pd.DatetimeIndex,
    n: int,
    rng: np.random.Generator,
    min_years_remaining: float = 5.0,
) -> list[pd.Timestamp]:
    """
    Stratified-uniform sample of n entry-date indices from panel_dates,
    constrained so every sampled date has at least min_years_remaining
    years of subsequent data.
    """
    if len(panel_dates) == 0 or n <= 0:
        return []
    last = panel_dates[-1]
    cutoff = last - pd.DateOffset(years=int(np.ceil(min_years_remaining)))
    eligible = panel_dates[panel_dates <= cutoff]
    if len(eligible) == 0:
        return []
    if n >= len(eligible):
        return list(eligible)

    # Stratified: split into n equal-size buckets, pick one date per bucket
    indices = np.linspace(0, len(eligible) - 1, n + 1).astype(int)
    chosen: list[pd.Timestamp] = []
    for i in range(n):
        lo, hi = indices[i], indices[i + 1]
        if hi <= lo:
            hi = lo + 1
        pick = int(rng.integers(lo, hi))
        chosen.append(pd.Timestamp(eligible[pick]))
    return chosen


def score_config(
    config: dict[str, Any],
    n_entry_points: int = 30,
    rng_seed: int = 0,
    min_years_remaining: float = 5.0,
) -> dict[str, float]:
    """
    Evaluate a config across (1) the full history + (n_entry_points) random
    entry-date sub-runs. Returns a dict of aggregated metrics + the score.
    """
    full_start = str(config["start_date"])
    full_end = str(config["end_date"]) if config.get("end_date") else None
    panel = _load_panel(config, full_start, full_end)
    if panel is None or panel.empty:
        return {"score": -999.0, "error": "empty_panel"}

    rng = np.random.default_rng(rng_seed)
    panel_dates = pd.DatetimeIndex(panel.index)
    entry_dates = _sample_entry_points(
        panel_dates, n_entry_points, rng, min_years_remaining=min_years_remaining
    )

    # Full-history run uses the panel as-is
    runs: list[dict] = []
    full_metrics = _single_run_metrics(panel, config)
    runs.append({"start_date": panel_dates[0], "kind": "full", **full_metrics})

    # Per-entry-point runs use a sub-panel from each random start
    for d in entry_dates:
        sub = panel.loc[d:].copy()
        if len(sub) < 50:
            continue
        sub_cfg = dict(config)
        sub_cfg["start_date"] = d.strftime("%Y-%m-%d")
        m = _single_run_metrics(sub, sub_cfg)
        runs.append({"start_date": d, "kind": "entry", **m})

    if not runs:
        return {"score": -999.0, "error": "no_runs"}

    df = pd.DataFrame(runs)
    cagrs = df["cagr"].to_numpy(dtype=float)
    max_dds = df["max_dd"].to_numpy(dtype=float)
    rebs = df["rebalance_count"].to_numpy(dtype=float)
    years = df["years"].to_numpy(dtype=float)
    rebs_per_year = np.where(years > 0, rebs / years, 0.0)

    median_cagr = float(np.median(cagrs))
    p05_cagr = float(np.percentile(cagrs, 5))
    p95_cagr = float(np.percentile(cagrs, 95))
    best_cagr = float(np.max(cagrs))
    worst_cagr = float(np.min(cagrs))
    worst_max_dd = float(np.min(max_dds))   # most negative
    median_max_dd = float(np.median(max_dds))
    median_rebs_per_year = float(np.median(rebs_per_year))
    worst_rebs_per_year = float(np.max(rebs_per_year))

    score = (
        median_cagr / max(abs(worst_max_dd), SCORE_DD_FLOOR)
        - SCORE_REBAL_PENALTY * median_rebs_per_year
        - SCORE_TAIL_PENALTY * max(0.0, SCORE_P05_THRESHOLD - p05_cagr)
    )

    return {
        "score": float(score),
        "median_cagr": median_cagr,
        "p05_cagr": p05_cagr,
        "p95_cagr": p95_cagr,
        "best_cagr": best_cagr,
        "worst_cagr": worst_cagr,
        "worst_max_dd": worst_max_dd,
        "median_max_dd": median_max_dd,
        "median_rebalances_per_year": median_rebs_per_year,
        "worst_rebalances_per_year": worst_rebs_per_year,
        "n_runs": int(len(df)),
    }
