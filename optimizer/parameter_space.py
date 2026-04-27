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

from dataclasses import dataclass, field, asdict
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


@dataclass
class IterationConstraints:
    """
    Optional narrower bounds applied on top of the default parameter space.
    Use to focus a follow-up batch on a specific region of the search space
    after analyzing prior results.

    Examples:
      - n_regimes_choices=[3]                       # fix at 3 regimes
      - drawdown_window_choices=[1]                 # fix 1y rolling window
      - threshold_bounds={"dd_t1": (0.05, 0.08)}    # narrow R1/R2 boundary
      - force_zero_params=["R2_base_w_tqqq_raw"]    # zero out TQQQ in R2 base
      - upside_threshold_bounds={"R1": (3, 5)}      # require strong signal for R1 upside
      - rebalance_choices={"R2_rebalance_on_upward": ["hold"]}   # pin R2 hold-on-upward
      - notes="Iter 2: validate user hypothesis that TQQQ should be 0 in R2 base"
    """
    n_regimes_choices: list[int] | None = None
    drawdown_window_choices: list[int] | None = None
    threshold_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    upside_threshold_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    protection_threshold_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    rebalance_choices: dict[str, list[str]] = field(default_factory=dict)
    force_zero_params: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple bounds to lists for JSON serialization
        d["threshold_bounds"] = {k: list(v) for k, v in d["threshold_bounds"].items()}
        d["upside_threshold_bounds"] = {k: list(v) for k, v in d["upside_threshold_bounds"].items()}
        d["protection_threshold_bounds"] = {
            k: list(v) for k, v in d["protection_threshold_bounds"].items()
        }
        return d


def _bounds(name: str, default: tuple[float, float], constraints: IterationConstraints | None) -> tuple[float, float]:
    if constraints is None or name not in constraints.threshold_bounds:
        return default
    return constraints.threshold_bounds[name]


def _int_bounds(
    role: str, regime: str, default: tuple[int, int], constraints: IterationConstraints | None,
) -> tuple[int, int]:
    if constraints is None:
        return default
    table = (
        constraints.upside_threshold_bounds
        if role == "upside"
        else constraints.protection_threshold_bounds
    )
    return tuple(table.get(regime, default))


def _suggest_weight(
    trial: optuna.Trial, name: str, constraints: IterationConstraints | None,
) -> float:
    """Suggest a raw weight in [0, 1], honoring `force_zero_params`."""
    if constraints is not None and name in constraints.force_zero_params:
        return trial.suggest_float(name, 0.0, 0.0)
    return trial.suggest_float(name, 0.0, 1.0)


def _suggest_simplex_3(
    trial: optuna.Trial, prefix: str, constraints: IterationConstraints | None,
) -> dict[str, float]:
    """Suggest 3 weights in {TQQQ, QQQ, XLU} that sum to 1.0."""
    a = _suggest_weight(trial, f"{prefix}_w_tqqq_raw", constraints)
    b = _suggest_weight(trial, f"{prefix}_w_qqq_raw", constraints)
    c = _suggest_weight(trial, f"{prefix}_w_xlu_raw", constraints)
    s = a + b + c
    if s <= 1e-9:
        # If TQQQ was forced to zero AND simplex collapsed, fall back to QQQ-only
        if constraints and f"{prefix}_w_tqqq_raw" in constraints.force_zero_params:
            return {"TQQQ": 0.0, "QQQ": 1.0, "XLU": 0.0}
        return {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0}
    return {"TQQQ": a / s, "QQQ": b / s, "XLU": c / s}


def _suggest_regime_block(
    trial: optuna.Trial,
    regime: str,
    dd_low: float,
    dd_high: float,
    constraints: IterationConstraints | None = None,
) -> dict[str, Any]:
    base = _suggest_simplex_3(trial, f"{regime}_base", constraints)
    upside = _suggest_simplex_3(trial, f"{regime}_upside", constraints)
    protection = _suggest_simplex_3(trial, f"{regime}_protection", constraints)

    up_lo, up_hi = _int_bounds("upside", regime, (1, 5), constraints)
    pr_lo, pr_hi = _int_bounds("protection", regime, (-5, -1), constraints)
    upside_threshold = trial.suggest_int(f"{regime}_upside_threshold", up_lo, up_hi)
    protection_threshold = trial.suggest_int(f"{regime}_protection_threshold", pr_lo, pr_hi)

    rb_down_choices = (
        constraints.rebalance_choices.get(f"{regime}_rebalance_on_downward", ["match", "hold"])
        if constraints else ["match", "hold"]
    )
    rb_up_choices = (
        constraints.rebalance_choices.get(f"{regime}_rebalance_on_upward", ["match", "hold"])
        if constraints else ["match", "hold"]
    )
    rb_down = trial.suggest_categorical(f"{regime}_rebalance_on_downward", rb_down_choices)
    rb_up = trial.suggest_categorical(f"{regime}_rebalance_on_upward", rb_up_choices)

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


def _suggest_thresholds(
    trial: optuna.Trial, constraints: IterationConstraints | None,
) -> list[float]:
    """
    Always suggest MAX_REGIMES-1 monotonically-increasing dd boundaries.
    Caller uses only the first n_regimes-1 of these. TPE learns to ignore
    unused tail thresholds. Constraints can narrow each bound.
    """
    t1_lo, t1_hi = _bounds("dd_t1", (0.03, 0.12), constraints)
    t1 = trial.suggest_float("dd_t1", t1_lo, t1_hi)
    t2_lo, t2_hi = _bounds("dd_t2", (t1 + 0.03, 0.30), constraints)
    t2 = trial.suggest_float("dd_t2", max(t2_lo, t1 + 0.03), max(t2_hi, t1 + 0.04))
    t3_lo, t3_hi = _bounds("dd_t3", (t2 + 0.03, 0.55), constraints)
    t3 = trial.suggest_float("dd_t3", max(t3_lo, t2 + 0.03), max(t3_hi, t2 + 0.04))
    t4_lo, t4_hi = _bounds("dd_t4", (t3 + 0.03, 0.80), constraints)
    t4 = trial.suggest_float("dd_t4", max(t4_lo, t3 + 0.03), max(t4_hi, t3 + 0.04))
    return [t1, t2, t3, t4]


def suggest_config(
    trial: optuna.Trial, constraints: IterationConstraints | None = None,
) -> dict[str, Any]:
    """Build a complete CONFIG dict from an Optuna trial."""
    n_regimes_choices = (
        constraints.n_regimes_choices
        if constraints and constraints.n_regimes_choices
        else list(range(MIN_REGIMES, MAX_REGIMES + 1))
    )
    drawdown_window_choices = (
        constraints.drawdown_window_choices
        if constraints and constraints.drawdown_window_choices
        else [1, 2, 3, 5]
    )

    n_regimes = int(trial.suggest_categorical("n_regimes", n_regimes_choices))
    drawdown_window_years = int(
        trial.suggest_categorical("drawdown_window_years", drawdown_window_choices)
    )

    all_thresholds = _suggest_thresholds(trial, constraints)
    thresholds = all_thresholds[: n_regimes - 1]

    # Always sample regime blocks for MAX_REGIMES; use first n_regimes
    sampled_blocks: list[dict[str, Any]] = []
    for i in range(MAX_REGIMES):
        regime_name = f"R{i + 1}"
        sampled_blocks.append(
            _suggest_regime_block(
                trial, regime_name, dd_low=0.0, dd_high=1.0, constraints=constraints
            )
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
