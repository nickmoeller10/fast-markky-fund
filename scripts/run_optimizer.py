"""
CLI entry point for the config optimizer.

Examples:
    # Smoke test (5 trials, 5 entry points each):
    python3 scripts/run_optimizer.py --study smoke --trials 5 --entry-points 5

    # Full v1 run (500 trials, 30 entry points each, 4 parallel workers):
    python3 scripts/run_optimizer.py --study v1 --trials 500 --entry-points 30 --jobs 4

The script forces FMF_DATA_MODE=frozen so every backtest reads from the
committed cache snapshot — guarantees reproducibility across runs.
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pin the cache mode for the entire optimizer run
os.environ["FMF_DATA_MODE"] = "frozen"

from optimizer.runner import run_study   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Fast Markky Fund config optimizer.")
    p.add_argument("--study", default="v1", help="Optuna study name (used for the SQLite db)")
    p.add_argument("--trials", type=int, default=500, help="Number of optimizer trials")
    p.add_argument("--entry-points", type=int, default=30, help="Monte Carlo entry points per trial")
    p.add_argument("--jobs", type=int, default=1, help="Parallel trial workers (joblib)")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = p.parse_args()

    print("=" * 70)
    print(f"Optimizer: study={args.study} trials={args.trials} "
          f"entry_points={args.entry_points} jobs={args.jobs}")
    print(f"FMF_DATA_MODE={os.environ['FMF_DATA_MODE']}")
    print("=" * 70)

    study, df = run_study(
        study_name=args.study,
        n_trials=args.trials,
        n_entry_points=args.entry_points,
        n_jobs=args.jobs,
        rng_seed_base=args.seed,
    )

    # Print top 5 by score
    if "value" in df.columns:
        top = df.dropna(subset=["value"]).sort_values("value", ascending=False).head(5)
        print("\nTop 5 trials by score:")
        cols = [c for c in [
            "number", "value", "user_attrs_median_cagr",
            "user_attrs_worst_max_dd", "user_attrs_median_rebalances_per_year"
        ] if c in top.columns]
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
