"""
Defines the search space for the config optimizer.

Suggests a complete CONFIG dict from an Optuna trial. The returned config
is shape-compatible with config.CONFIG and consumed by run_backtest.

Search axes (v2):
  - n_regimes ∈ {2, 3, 4, 5}: how many regimes the strategy uses
  - drawdown_window_years ∈ {1, 2, 3, 5}: rolling-peak window length
  - Up to 4 monotonically-increasing dd thresholds (only first n_regimes-1 used)
  - Per-regime base allocation across {TQQQ, QQQ, XLU}
  - Per-regime upside override (threshold + 3-ticker allocation)
  - Per-regime protection override (threshold + 3-ticker allocation)
  - Per-regime rebalance_on_downward / rebalance_on_upward ∈ {match, hold}

Fixed:
  - rebalance_strategy = "per_regime"
  - rebalance_frequency = "instant"
  - drawdown_ticker = "QQQ"
  - allocation universe = {TQQQ, QQQ, XLU} only
    (alt tickers like GLD/TLT/SPLV/BIL excluded — they don't backtest to
    1999, which would bias the Monte Carlo entry-point evaluator.)
"""

from __future__ import annotations

from typing import Any

import optuna


CORE_TICKERS = ("TQQQ", "QQQ", "XLU")
# All possible regime counts; ALWAYS sample params for MAX_REGIMES so the
# search space is shape-stable. We just use only the first n_regimes worth.
MIN_REGIMES = 2
MAX_REGIMES = 5
# Panel tickers that the optimizer's score module loads. Includes SPY for
# signal-layer warmup; allocation_tickers stays at CORE_TICKERS only.
ALL_PANEL_TICKERS = ["QQQ", "TQQQ", "XLU", "SPY"]


def _suggest_simplex_3(trial: optuna.Trial, prefix: str) -> dict[str, float]:
    """Suggest 3 weights in {TQQQ, QQQ, XLU} that sum to 1.0."""
    a = trial.suggest_float(f"{prefix}_w_tqqq_raw", 0.0, 1.0)
    b = trial.suggest_float(f"{prefix}_w_qqq_raw", 0.0, 1.0)
    c = trial.suggest_float(f"{prefix}_w_xlu_raw", 0.0, 1.0)
    s = a + b + c
    if s <= 1e-9:
        return {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0}
    return {"TQQQ": a / s, "QQQ": b / s, "XLU": c / s}


def _suggest_regime_block(
    trial: optuna.Trial,
    regime: str,
    dd_low: float,
    dd_high: float,
) -> dict[str, Any]:
    base = _suggest_simplex_3(trial, f"{regime}_base")
    upside = _suggest_simplex_3(trial, f"{regime}_upside")
    protection = _suggest_simplex_3(trial, f"{regime}_protection")

    upside_threshold = trial.suggest_int(f"{regime}_upside_threshold", 1, 5)
    protection_threshold = trial.suggest_int(f"{regime}_protection_threshold", -5, -1)

    rb_down = trial.suggest_categorical(f"{regime}_rebalance_on_downward", ["match", "hold"])
    rb_up = trial.suggest_categorical(f"{regime}_rebalance_on_upward", ["match", "hold"])

    block: dict[str, Any] = {
        "dd_low": dd_low,
        "dd_high": dd_high,
        "rebalance_on_downward": rb_down,
        "rebalance_on_upward": rb_up,
        "signal_overrides": {
            "upside": {
                "enabled": True,
                "label": f"{regime} Upside",
                "direction": "above",
                "threshold": upside_threshold,
                **{t: float(upside.get(t, 0.0)) for t in CORE_TICKERS},
            },
            "protection": {
                "enabled": True,
                "label": f"{regime} Protection",
                "direction": "below",
                "threshold": protection_threshold,
                **{t: float(protection.get(t, 0.0)) for t in CORE_TICKERS},
            },
        },
    }
    for t in CORE_TICKERS:
        block[t] = float(base.get(t, 0.0))
    return block


def _suggest_thresholds(trial: optuna.Trial) -> list[float]:
    """
    Always suggest MAX_REGIMES-1 monotonically-increasing dd boundaries.
    Caller uses only the first n_regimes-1 of these. TPE learns to ignore
    unused tail thresholds.
    """
    t1 = trial.suggest_float("dd_t1", 0.03, 0.12)
    t2 = trial.suggest_float("dd_t2", t1 + 0.03, 0.30)
    t3 = trial.suggest_float("dd_t3", t2 + 0.03, 0.55)
    t4 = trial.suggest_float("dd_t4", t3 + 0.03, 0.80)
    return [t1, t2, t3, t4]


def suggest_config(trial: optuna.Trial) -> dict[str, Any]:
    """Build a complete CONFIG dict from an Optuna trial."""
    n_regimes = int(trial.suggest_categorical("n_regimes", list(range(MIN_REGIMES, MAX_REGIMES + 1))))
    drawdown_window_years = int(trial.suggest_categorical("drawdown_window_years", [1, 2, 3, 5]))

    all_thresholds = _suggest_thresholds(trial)
    thresholds = all_thresholds[: n_regimes - 1]

    # Always sample regime blocks for MAX_REGIMES; use first n_regimes
    sampled_blocks: list[dict[str, Any]] = []
    for i in range(MAX_REGIMES):
        regime_name = f"R{i + 1}"
        # dd bounds will be adjusted before the block is used
        sampled_blocks.append(
            _suggest_regime_block(trial, regime_name, dd_low=0.0, dd_high=1.0)
        )

    regimes: dict[str, dict[str, Any]] = {}
    for i in range(n_regimes):
        block = sampled_blocks[i]
        block["dd_low"] = 0.0 if i == 0 else thresholds[i - 1]
        block["dd_high"] = thresholds[i] if i < n_regimes - 1 else 1.0
        regimes[f"R{i + 1}"] = block

    return {
        "starting_balance": 10_000,
        "start_date": "1999-01-04",
        "end_date": "2026-03-27",
        "drawdown_ticker": "QQQ",
        "drawdown_window_enabled": True,
        "drawdown_window_years": drawdown_window_years,
        "rebalance_frequency": "instant",
        "rebalance_holiday_rule": "next_trading_day",
        "rebalance_strategy": "per_regime",
        "dividend_reinvestment": False,
        "dividend_reinvestment_target": "cash",
        "tickers": list(ALL_PANEL_TICKERS),
        "allocation_tickers": list(CORE_TICKERS),
        "minimum_allocation": 0.0,
        "regimes": regimes,
    }
