"""
Streamlit page for browsing optimizer results.

Activate by launching `streamlit run app.py` and selecting "optimizer_results"
from the sidebar (Streamlit auto-discovers files under `pages/`).

Reads <project_root>/optimizer_runs/<study>_results.parquet plus the SQLite
study DB. Provides:

  - Run summary (n_trials, best score, time)
  - Pareto frontier scatter on (median_cagr, worst_max_dd, rebalances/year)
  - Top-N trial table
  - Single-trial config detail viewer
  - Stability filter (min p05_cagr, max worst_max_dd)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from optimizer.results import (
    load_results,
    pareto_frontier,
    parse_config,
    stability_filter,
    top_n_by,
)


OPTIMIZER_DIR = Path(__file__).resolve().parent.parent / "optimizer_runs"


def _list_studies() -> list[str]:
    if not OPTIMIZER_DIR.exists():
        return []
    studies = set()
    for f in OPTIMIZER_DIR.glob("*_results.parquet"):
        studies.add(f.name.replace("_results.parquet", ""))
    return sorted(studies)


st.set_page_config(page_title="Optimizer Results", layout="wide")
st.title("🔍 Config Optimizer Results")

studies = _list_studies()
if not studies:
    st.warning(
        f"No optimizer studies found in `{OPTIMIZER_DIR}`. "
        "Run `python scripts/run_optimizer.py --study v1 --trials 200` first."
    )
    st.stop()

study_name = st.sidebar.selectbox("Study", studies, index=len(studies) - 1)
df = load_results(study_name)

# Drop incomplete / failed trials
df = df[df["state"] == "COMPLETE"].copy()
if df.empty:
    st.error(f"Study `{study_name}` has no completed trials.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### Stability filter")
min_p05 = st.sidebar.number_input(
    "Min p05 CAGR (worst-5% entry-point CAGR)",
    value=-1.0,
    step=0.05,
    format="%.2f",
)
max_dd_floor = st.sidebar.number_input(
    "Max acceptable worst drawdown (closer to 0 = stricter)",
    value=-0.95,
    step=0.05,
    format="%.2f",
)
filtered = stability_filter(df, min_p05_cagr=min_p05, max_worst_dd=max_dd_floor)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total trials", int(len(df)))
c1.caption(f"after filter: {len(filtered)}")
c2.metric("Best score", f"{df['value'].max():.4f}")
c3.metric("Median CAGR (best)", f"{df['median_cagr'].max():.2%}")
c4.metric("Worst DD (most-conservative)", f"{df['worst_max_dd'].max():.2%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------
st.subheader("Pareto frontier — robustness vs return vs turnover")
pf = pareto_frontier(filtered)

required_pareto_cols = {"median_cagr", "worst_max_dd", "median_rebalances_per_year"}
if pf.empty or not required_pareto_cols.issubset(pf.columns):
    st.info("Not enough completed trials with metrics to draw a Pareto frontier yet.")
else:
    fig = px.scatter(
        filtered,
        x="median_cagr",
        y="worst_max_dd",
        color="median_rebalances_per_year",
        size_max=12,
        hover_data=["number", "value", "p05_cagr", "best_cagr", "worst_cagr"],
        color_continuous_scale="RdYlGn_r",
        title="Each dot = one trial. Pareto frontier highlighted in red.",
        labels={
            "median_cagr": "Median CAGR (across all entry points)",
            "worst_max_dd": "Worst max drawdown (closer to 0 = better)",
            "median_rebalances_per_year": "Rebalances / year",
        },
    )
    fig.add_scatter(
        x=pf["median_cagr"],
        y=pf["worst_max_dd"],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star"),
        name="Pareto frontier",
        hoverinfo="skip",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top trials by score
# ---------------------------------------------------------------------------
st.subheader("Top 10 trials by score")
display_cols = [
    "number",
    "value",
    "median_cagr",
    "p05_cagr",
    "p95_cagr",
    "best_cagr",
    "worst_cagr",
    "worst_max_dd",
    "median_rebalances_per_year",
]
display_cols = [c for c in display_cols if c in filtered.columns]
sort_metric = st.selectbox(
    "Sort by",
    options=display_cols,
    index=display_cols.index("value") if "value" in display_cols else 0,
)
ascending = st.checkbox("Ascending", value=False)
top = top_n_by(filtered, sort_metric, n=10, ascending=ascending)
st.dataframe(top[display_cols], use_container_width=True)

# ---------------------------------------------------------------------------
# Single-trial detail
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Trial detail")
trial_numbers = filtered["number"].tolist()
if not trial_numbers:
    st.info("No trials match the current filter.")
else:
    chosen = st.selectbox("Trial number", trial_numbers, index=0)
    row = filtered[filtered["number"] == chosen].iloc[0]

    left, right = st.columns(2)
    with left:
        st.markdown("**Score & metrics**")
        for k in display_cols:
            if k == "number":
                continue
            val = row.get(k)
            if pd.notna(val):
                if "cagr" in k.lower() or "dd" in k.lower():
                    st.text(f"{k}: {float(val):.2%}")
                else:
                    st.text(f"{k}: {float(val):.4f}")

    with right:
        st.markdown("**Full config**")
        cfg = parse_config(row)
        if cfg:
            st.json(cfg, expanded=False)
        else:
            st.text("(no config_json stored on this row)")
