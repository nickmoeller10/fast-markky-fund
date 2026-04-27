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


def _objective(
    trial: optuna.Trial,
    n_entry_points: int,
    rng_seed_base: int,
    constraints=None,
    dd_hard_floor: float | None = None,
) -> float:
    from optimizer.parameter_space import suggest_config as _suggest

    if constraints is None:
        config = _suggest(trial)
    else:
        config = _suggest(trial, constraints=constraints)

    score_kwargs = {"n_entry_points": n_entry_points, "rng_seed": rng_seed_base + trial.number}
    if dd_hard_floor is not None:
        score_kwargs["dd_hard_floor"] = dd_hard_floor

    metrics = score_config(config, **score_kwargs)
    if "error" in metrics:
        # Penalize empty/broken configs but don't crash the study
        for k, v in metrics.items():
            if k == "runs":
                continue
            if isinstance(v, (int, float)):
                trial.set_user_attr(k, float(v))
            else:
                trial.set_user_attr(k, str(v))
        return -1e6

    # Stash scalar metrics + the config + the per-run details
    runs = metrics.pop("runs", [])
    for k, v in metrics.items():
        trial.set_user_attr(k, float(v))
    trial.set_user_attr("config_json", json.dumps(config, default=str))
    trial.set_user_attr("runs_json", json.dumps(runs, default=str))
    return float(metrics["score"])


def run_study(
    study_name: str,
    n_trials: int,
    n_entry_points: int = 30,
    n_jobs: int = 1,
    rng_seed_base: int = 0,
    output_dir: Path | str | None = None,
    constraints=None,
    dd_hard_floor: float | None = None,
) -> tuple[optuna.Study, pd.DataFrame]:
    """
    Run an Optuna study and return (study, results_df).

    Persists:
      <output_dir>/<study_name>.db                 Optuna SQLite (resumable, includes
                                                    per-trial config_json + runs_json)
      <output_dir>/<study_name>_results.parquet    flattened per-trial snapshot
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

    if constraints is not None or dd_hard_floor is not None:
        study.set_user_attr(
            "iteration_metadata",
            json.dumps(
                {
                    "constraints": (constraints.to_dict() if constraints is not None else None),
                    "dd_hard_floor": dd_hard_floor,
                },
                default=str,
            ),
        )

    started = time.time()
    study.optimize(
        lambda trial: _objective(
            trial, n_entry_points, rng_seed_base, constraints, dd_hard_floor
        ),
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
