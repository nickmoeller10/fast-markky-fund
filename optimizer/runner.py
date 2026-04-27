"""
Optuna study driver. Wires parameter_space.suggest_config to score.score_config
and persists per-trial results (full config + metrics) to a Parquet file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from optimizer.parameter_space import suggest_config
from optimizer.score import score_config


OPTIMIZER_DIR = Path(__file__).resolve().parent.parent / "optimizer_runs"


def _objective(trial: optuna.Trial, n_entry_points: int, rng_seed_base: int) -> float:
    config = suggest_config(trial)
    metrics = score_config(
        config,
        n_entry_points=n_entry_points,
        rng_seed=rng_seed_base + trial.number,
    )
    if "error" in metrics:
        # Penalize empty/broken configs but don't crash the study
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                trial.set_user_attr(k, float(v))
            else:
                trial.set_user_attr(k, str(v))
        return -1e6

    # Stash metrics + the config dict on the trial
    for k, v in metrics.items():
        trial.set_user_attr(k, float(v))
    trial.set_user_attr("config_json", json.dumps(config, default=str))
    return float(metrics["score"])


def run_study(
    study_name: str,
    n_trials: int,
    n_entry_points: int = 30,
    n_jobs: int = 1,
    rng_seed_base: int = 0,
    output_dir: Path | str | None = None,
) -> tuple[optuna.Study, pd.DataFrame]:
    """
    Run an Optuna study and return (study, results_df).

    Persists:
      <output_dir>/<study_name>.db          Optuna SQLite (resumable)
      <output_dir>/<study_name>_results.parquet   per-trial metrics + config
    """
    out_dir = Path(output_dir) if output_dir else OPTIMIZER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir}/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=rng_seed_base),
    )

    started = time.time()
    study.optimize(
        lambda trial: _objective(trial, n_entry_points, rng_seed_base),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    elapsed = time.time() - started

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.to_parquet(out_dir / f"{study_name}_results.parquet", index=False)

    print(f"\nStudy '{study_name}' complete: {len(study.trials)} trials in {elapsed:.1f}s")
    if study.best_trial is not None:
        print(f"Best score: {study.best_value:.4f} (trial {study.best_trial.number})")

    return study, df
