"""
Run one iteration of the iterative config search and print the analysis report.

Each iteration:
  1. Runs a small batch (default 50 trials) of Optuna search
  2. Stores per-run details on each trial for offline analysis
  3. Prints a report:
       - Top-10 trials by score
       - Trend summary (parameter distributions in top-10)
       - Drawdown profile (how many trials breached the -40% floor)
       - Hypothesis checks (e.g., R2.TQQQ ≈ 0)
       - Regime usage (% time in each regime for the top configs)
       - Proposed constraints for the next iteration

You then either accept the proposal, override it, or skip — and run the next
iteration with whatever constraints you want.

Examples:
    # Iter 1: baseline, no constraints, default DD floor
    python3 scripts/iterate.py --study iter1

    # Iter 2: load constraints from a JSON file
    python3 scripts/iterate.py --study iter2 --constraints iter2_constraints.json

    # Iter 3: tighter DD floor (more risk-averse)
    python3 scripts/iterate.py --study iter3 --dd-floor -0.35
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pin the cache mode for the entire run
os.environ["FMF_DATA_MODE"] = "frozen"

import pandas as pd

from optimizer.analysis import (
    drawdown_profile,
    hypothesis_check,
    propose_constraints,
    regime_usage_summary,
    top_configs,
    trend_summary,
)
from optimizer.parameter_space import IterationConstraints
from optimizer.runner import OPTIMIZER_DIR, run_study


CONSTRAINTS_DIR = ROOT / "optimizer_runs" / "constraints"


def _load_constraints(path: str | None) -> IterationConstraints | None:
    if not path:
        return None
    p = Path(path) if Path(path).is_absolute() else CONSTRAINTS_DIR / path
    if not p.exists():
        raise FileNotFoundError(f"No constraints file at {p}")
    with open(p) as f:
        d = json.load(f)
    cons = IterationConstraints(
        n_regimes_choices=d.get("n_regimes_choices"),
        drawdown_window_choices=d.get("drawdown_window_choices"),
        threshold_bounds={k: tuple(v) for k, v in (d.get("threshold_bounds") or {}).items()},
        upside_threshold_bounds={
            k: tuple(v) for k, v in (d.get("upside_threshold_bounds") or {}).items()
        },
        protection_threshold_bounds={
            k: tuple(v) for k, v in (d.get("protection_threshold_bounds") or {}).items()
        },
        rebalance_choices=d.get("rebalance_choices") or {},
        force_zero_params=d.get("force_zero_params") or [],
        weight_bounds={k: tuple(v) for k, v in (d.get("weight_bounds") or {}).items()},
        forced_base_allocations=d.get("forced_base_allocations") or {},
        enable_cash_in_regimes=d.get("enable_cash_in_regimes") or [],
        notes=d.get("notes", ""),
    )
    return cons


def _save_proposed_constraints(study: str, cons: IterationConstraints) -> Path:
    CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONSTRAINTS_DIR / f"{study}_proposed.json"
    with open(path, "w") as f:
        json.dump(cons.to_dict(), f, indent=2, default=str)
    return path


def _print_report(study_name: str, dd_floor: float) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("\n" + "=" * 78)
    print(f"REPORT — study '{study_name}'")
    print("=" * 78)

    top = top_configs(study_name, n=10)
    if top.empty:
        print("(no completed trials)")
        return

    show_cols = [
        c for c in [
            "number", "value",
            "user_attrs_median_cagr", "user_attrs_p05_cagr", "user_attrs_p95_cagr",
            "user_attrs_worst_max_dd", "user_attrs_median_max_dd",
            "user_attrs_median_rebalances_per_year",
            "user_attrs_dd_floor_breach_count",
            "params_n_regimes", "params_drawdown_window_years",
        ] if c in top.columns
    ]
    print("\nTop 10 trials by score:")
    print(top[show_cols].to_string(index=False))

    print("\n--- Drawdown profile ---")
    dp = drawdown_profile(study_name, dd_floor=dd_floor)
    if dp:
        print(f"  {dp['n_trials']} trials, {dp['n_breaching_floor']} breached "
              f"DD floor of {dd_floor:.2%} ({dp['breach_rate']:.1%})")
        print(f"  worst_max_dd: min={dp['min_worst_dd']:.2%}, "
              f"median={dp['median_worst_dd']:.2%}, max={dp['max_worst_dd']:.2%}")

    print("\n--- Trend summary (top 10) ---")
    ts = trend_summary(study_name, top_n=10)
    if ts and "params" in ts:
        for p, info in sorted(ts["params"].items()):
            if info["kind"] == "numeric":
                print(f"  {p:40s}  median={info['median']:.4f}  "
                      f"[p25={info['p25']:.4f}, p75={info['p75']:.4f}]  "
                      f"range=[{info['min']:.4f}, {info['max']:.4f}]")
            else:
                print(f"  {p:40s}  most_common={info['most_common']!r:6} "
                      f"freq={info['frequency']}/{ts['n_actual']}")

    print("\n--- Hypothesis checks ---")
    checks = [
        ("R2_base TQQQ ≈ 0 in top 10",
         lambda r: float(r.get("params_R2_base_w_tqqq_raw", 1.0)) < 0.10),
        ("R3_base TQQQ ≈ 0 in top 10",
         lambda r: float(r.get("params_R3_base_w_tqqq_raw", 1.0)) < 0.10),
        ("R1_base XLU = 0 in top 10",
         lambda r: float(r.get("params_R1_base_w_xlu_raw", 1.0)) < 0.10),
        ("Worst DD better than -35%",
         lambda r: float(r.get("user_attrs_worst_max_dd", -1.0)) >= -0.35),
        ("Median CAGR > 12%",
         lambda r: float(r.get("user_attrs_median_cagr", 0.0)) > 0.12),
    ]
    for label, pred in checks:
        h = hypothesis_check(study_name, pred, top_n=10, label=label)
        print(f"  {label:45s}  {h['n_match']}/{h['n_top']} matched ({h['match_rate']:.0%})")

    print("\n--- Regime usage (top 10, % of trading days in each regime) ---")
    ru = regime_usage_summary(study_name, top_n=10)
    if not ru.empty:
        cols = sorted([c for c in ru.columns if c.startswith("pct_in_")])
        if cols:
            print(ru[["trial_number", "score"] + cols].to_string(index=False))

    print("\n--- Proposed constraints for next iteration ---")
    cons = propose_constraints(study_name, top_n=10, narrow_factor=0.5)
    saved = _save_proposed_constraints(study_name, cons)
    print(f"  Saved to: {saved}")
    print(f"  Notes: {cons.notes}")
    if cons.n_regimes_choices:
        print(f"  Pin n_regimes to: {cons.n_regimes_choices}")
    if cons.drawdown_window_choices:
        print(f"  Pin drawdown_window_years to: {cons.drawdown_window_choices}")
    for k, (lo, hi) in cons.threshold_bounds.items():
        print(f"  Narrow {k}: [{lo:.4f}, {hi:.4f}]")
    for k, choices in cons.rebalance_choices.items():
        print(f"  Pin {k}: {choices}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Run one iteration of the config search.")
    p.add_argument("--study", required=True, help="Iteration study name (e.g. 'iter1')")
    p.add_argument("--trials", type=int, default=50, help="Trials in this iteration")
    p.add_argument("--entry-points", type=int, default=10, help="Monte Carlo entry points per trial")
    p.add_argument("--jobs", type=int, default=2, help="Parallel trial workers")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument("--constraints", default=None,
                   help="Path to constraints JSON (relative to optimizer_runs/constraints/)")
    p.add_argument("--dd-floor", type=float, default=-0.40,
                   help="Hard penalty kicks in when worst_max_dd < this floor (default -0.40)")
    p.add_argument("--report-only", action="store_true",
                   help="Skip the search, just print the analysis report")
    args = p.parse_args()

    constraints = _load_constraints(args.constraints)

    if not args.report_only:
        print("=" * 78)
        print(f"Iteration: study={args.study} trials={args.trials} "
              f"entry_points={args.entry_points} jobs={args.jobs} dd_floor={args.dd_floor}")
        if constraints:
            print(f"Constraints: {constraints.notes or '(no notes)'}")
            print(f"  details: {json.dumps(constraints.to_dict(), default=str, indent=2)}")
        print("=" * 78)

        run_study(
            study_name=args.study,
            n_trials=args.trials,
            n_entry_points=args.entry_points,
            n_jobs=args.jobs,
            rng_seed_base=args.seed,
            constraints=constraints,
            dd_hard_floor=args.dd_floor,
        )

    _print_report(args.study, dd_floor=args.dd_floor)


if __name__ == "__main__":
    main()
