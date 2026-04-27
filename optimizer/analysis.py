"""
Batch-level analysis tools for the iterative config search.

Designed to be called between iteration runs. Surfaces:
  - top_configs(study) — best N trials by primary score
  - trend_summary(study, top_n) — for each parameter, distribution among top N
  - drawdown_profile(study) — how many trials breach the DD floor
  - hypothesis_check(study, predicate) — fraction of top trials that match a rule
  - load_run_details(study) — long-format DataFrame of all per-run results
  - propose_constraints(study) — heuristic suggestions for next iteration's bounds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from optimizer.parameter_space import IterationConstraints
from optimizer.results import load_results


def top_configs(study_name: str, n: int = 10, metric: str = "value") -> pd.DataFrame:
    """Return top-N trials by metric (default: optimizer score)."""
    df = load_results(study_name)
    df = df[df["state"] == "COMPLETE"].copy()
    return df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(n)


def load_run_details(study_name: str) -> pd.DataFrame:
    """
    Long-format DataFrame: one row per (trial, run) covering each Monte Carlo
    sub-backtest. Columns: trial_number, score, run_index, start_date, kind,
    cagr, max_dd, rebalance_count, years, final_value.
    """
    df = load_results(study_name)
    df = df[df["state"] == "COMPLETE"].copy()
    if df.empty or "runs_json" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, trow in df.iterrows():
        try:
            runs = json.loads(trow["runs_json"]) if isinstance(trow["runs_json"], str) else []
        except (json.JSONDecodeError, TypeError):
            continue
        for i, r in enumerate(runs):
            rows.append({
                "trial_number": int(trow["number"]),
                "score": float(trow["value"]) if pd.notna(trow["value"]) else None,
                "run_index": i,
                "start_date": r.get("start_date"),
                "kind": r.get("kind"),
                "cagr": r.get("cagr"),
                "max_dd": r.get("max_dd"),
                "rebalance_count": r.get("rebalance_count"),
                "years": r.get("years"),
                "final_value": r.get("final_value"),
            })
    return pd.DataFrame(rows)


def trend_summary(study_name: str, top_n: int = 10) -> dict[str, Any]:
    """
    For each parameter, summarize its distribution among the top-N trials.
    Numeric params: median, p25, p75, min, max. Categorical: most common value + count.
    """
    df = top_configs(study_name, n=top_n)
    if df.empty:
        return {}

    param_cols = [c for c in df.columns if c.startswith("params_")]
    out: dict[str, Any] = {"top_n": top_n, "n_actual": len(df)}
    out["top_score_range"] = (
        float(df["value"].min()) if "value" in df else None,
        float(df["value"].max()) if "value" in df else None,
    )

    summary: dict[str, dict[str, Any]] = {}
    for c in param_cols:
        p = c[len("params_"):]
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            arr = col.dropna().to_numpy(dtype=float)
            if len(arr) == 0:
                continue
            summary[p] = {
                "kind": "numeric",
                "min": float(arr.min()),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(arr.max()),
            }
        else:
            counts = col.dropna().astype(str).value_counts()
            if counts.empty:
                continue
            summary[p] = {
                "kind": "categorical",
                "most_common": str(counts.index[0]),
                "frequency": int(counts.iloc[0]),
                "distribution": {str(k): int(v) for k, v in counts.items()},
            }
    out["params"] = summary
    return out


def drawdown_profile(study_name: str, dd_floor: float = -0.40) -> dict[str, Any]:
    """How many trials' worst run breached the floor? Min/median/max worst_max_dd."""
    df = load_results(study_name)
    df = df[df["state"] == "COMPLETE"].copy()
    if df.empty or "worst_max_dd" not in df.columns:
        return {}
    arr = df["worst_max_dd"].dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "n_trials": int(len(arr)),
        "n_breaching_floor": int((arr < dd_floor).sum()),
        "breach_rate": float((arr < dd_floor).mean()),
        "min_worst_dd": float(arr.min()),
        "median_worst_dd": float(np.median(arr)),
        "max_worst_dd": float(arr.max()),
        "dd_floor": dd_floor,
    }


def hypothesis_check(
    study_name: str,
    predicate: Callable[[pd.Series], bool],
    top_n: int = 10,
    label: str = "predicate",
) -> dict[str, Any]:
    """
    Run a row-wise predicate on the top-N trials. Returns the fraction
    matching plus the trials that did. Use to validate hypotheses like
    'top configs have R2_base_w_tqqq_raw < 0.05'.
    """
    df = top_configs(study_name, n=top_n)
    if df.empty:
        return {"label": label, "n_top": 0, "n_match": 0, "match_rate": 0.0}
    matched = df.apply(predicate, axis=1)
    return {
        "label": label,
        "n_top": int(len(df)),
        "n_match": int(matched.sum()),
        "match_rate": float(matched.mean()),
        "matching_trials": df.loc[matched, "number"].tolist(),
    }


def regime_usage_summary(study_name: str, top_n: int = 10) -> pd.DataFrame:
    """
    For top-N trials, average % of trading days spent in each regime
    (across that trial's full-history run). Helps spot 'this config never
    actually used R3' or 'R5 is dead weight'.
    """
    df = top_configs(study_name, n=top_n)
    if df.empty or "runs_json" not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, trow in df.iterrows():
        try:
            runs = json.loads(trow["runs_json"]) if isinstance(trow["runs_json"], str) else []
        except (json.JSONDecodeError, TypeError):
            continue
        # Use the full-history run for stable %-of-time stats
        full = next((r for r in runs if r.get("kind") == "full"), None)
        if not full:
            continue
        dist = full.get("regime_distribution", {})
        rows.append({
            "trial_number": int(trow["number"]),
            "score": float(trow["value"]) if pd.notna(trow["value"]) else None,
            **{f"pct_in_{k}": float(v) for k, v in dist.items()},
        })
    return pd.DataFrame(rows).fillna(0.0)


def propose_constraints(
    study_name: str,
    top_n: int = 10,
    narrow_factor: float = 0.5,
) -> IterationConstraints:
    """
    Heuristic: build IterationConstraints that NARROW each numeric param
    around the top-N median ± narrow_factor*IQR. Categorical params get
    pinned to the dominant choice if it appears in >=70% of top trials.
    """
    summary = trend_summary(study_name, top_n=top_n)
    if not summary or "params" not in summary:
        return IterationConstraints()

    cons = IterationConstraints(
        notes=f"Auto-derived from top-{top_n} of '{study_name}', narrow_factor={narrow_factor}"
    )
    for p, info in summary["params"].items():
        if info["kind"] == "numeric":
            iqr = max(info["p75"] - info["p25"], 1e-3)
            half = max(narrow_factor * iqr, 0.005)
            lo = max(0.0, info["median"] - half)
            hi = min(info["max"] + half, info["median"] + half)
            # Only narrow dd thresholds (`dd_t*`) — leave per-regime weights
            # free so allocation tweaks remain wide. Threshold tightening is
            # the highest-leverage constraint.
            if p.startswith("dd_t"):
                cons.threshold_bounds[p] = (round(lo, 4), round(hi, 4))
        elif info["kind"] == "categorical":
            if info["frequency"] / max(1, summary["n_actual"]) >= 0.70:
                # Dominant choice — pin it
                if p == "n_regimes":
                    cons.n_regimes_choices = [int(info["most_common"])]
                elif p == "drawdown_window_years":
                    cons.drawdown_window_choices = [int(info["most_common"])]
                elif p.endswith("_rebalance_on_downward") or p.endswith("_rebalance_on_upward"):
                    cons.rebalance_choices[p] = [info["most_common"]]
    return cons
