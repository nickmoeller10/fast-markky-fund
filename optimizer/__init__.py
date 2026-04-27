"""
Config-search optimizer for Fast Markky Fund.

Public surface:
    parameter_space.suggest_config(trial)   -> dict
    score.score_config(config, ...)         -> dict
    runner.run_study(...)                   -> optuna.Study
    results.load_results(study_path)        -> pd.DataFrame
"""
