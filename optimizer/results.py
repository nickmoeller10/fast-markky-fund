"""
Loaders + analysis helpers for optimizer study results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(study_name: str, output_dir: Path | str | None = None) -> pd.DataFrame:
    """Load a study's per-trial results parquet."""
    base = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parent.parent / "optimizer_runs"
    )
    path = base / f"{study_name}_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No results file at {path}")
    df = pd.read_parquet(path)

    # Promote user_attrs.* columns into top-level
    for c in list(df.columns):
        if c.startswith("user_attrs_"):
            df[c[len("user_attrs_"):]] = df[c]
    return df


def parse_config(row: pd.Series) -> dict:
    """Recover the full config dict from a row's stashed JSON."""
    raw = row.get("config_json") or row.get("user_attrs_config_json")
    if raw is None or not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return non-dominated trials on (median_cagr, worst_max_dd, median_rebalances_per_year).
    Maximize median_cagr and worst_max_dd (closer to 0 is better);
    minimize median_rebalances_per_year.
    """
    keep_cols = {"median_cagr", "worst_max_dd", "median_rebalances_per_year"}
    df = df.dropna(subset=list(keep_cols)).reset_index(drop=True)
    if df.empty:
        return df

    pts = df[["median_cagr", "worst_max_dd", "median_rebalances_per_year"]].to_numpy()
    n = len(pts)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (
                pts[j, 0] >= pts[i, 0]
                and pts[j, 1] >= pts[i, 1]
                and pts[j, 2] <= pts[i, 2]
                and (
                    pts[j, 0] > pts[i, 0]
                    or pts[j, 1] > pts[i, 1]
                    or pts[j, 2] < pts[i, 2]
                )
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def top_n_by(df: pd.DataFrame, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    return df.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(n)


def stability_filter(
    df: pd.DataFrame, min_p05_cagr: float | None = None, max_worst_dd: float | None = None
) -> pd.DataFrame:
    out = df
    if min_p05_cagr is not None and "p05_cagr" in out.columns:
        out = out[out["p05_cagr"] >= min_p05_cagr]
    if max_worst_dd is not None and "worst_max_dd" in out.columns:
        out = out[out["worst_max_dd"] >= max_worst_dd]
    return out.reset_index(drop=True)
