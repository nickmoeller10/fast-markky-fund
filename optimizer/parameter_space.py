"""
Defines the search space for the config optimizer.

Suggests a complete CONFIG dict from an Optuna trial. The returned config
is shape-compatible with config.CONFIG and consumed by run_backtest.

Search axes (v3):
  - n_regimes ∈ {2, 3, 4, 5}: how many regimes the strategy uses
  - drawdown_window_years ∈ {1, 2, 3, 5}: rolling-peak window length
  - Up to 4 monotonically-increasing dd thresholds (only first n_regimes-1 used)
  - Per-regime base allocation across {TQQQ, QQQ, XLU} — optionally extended
    to include CASH when the regime is listed in
    `IterationConstraints.enable_cash_in_regimes`
  - Per-regime upside override (threshold + simplex allocation)
  - Per-regime protection override (threshold + simplex allocation)
  - Per-regime rebalance_on_downward / rebalance_on_upward ∈ {match, hold}

Fixed:
  - rebalance_strategy = "per_regime"
  - rebalance_frequency = "instant"
  - drawdown_ticker = "QQQ"
  - allocation universe = {TQQQ, QQQ, XLU} by default; CASH joins per-regime
    when explicitly enabled. (Alt tickers like GLD/TLT/SPLV/BIL excluded —
    they don't backtest to 1999, which would bias the entry-point evaluator.)

CASH semantics:
  - CASH is a synthetic ticker (`CASH_TICKER = "CASH"`) with a daily-compounded
    risk-free price series. Annualized return is `CASH_APY` (default 4%).
  - The price series is generated in `optimizer.score._load_panel`. From the
    backtester's perspective CASH is "always priced" so `tradable_allocation`
    treats it like any other asset.
  - There are TWO ways to put CASH in a regime:
      1. `forced_base_allocations` — pin specific weights including CASH (strict).
      2. `enable_cash_in_regimes` — add CASH as a 4th simplex coordinate so the
         optimizer searches over CASH weight together with TQQQ/QQQ/XLU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import optuna


# CASH constants live in data_loader (single source of truth) — both "CASH"
# and "$" are recognized as synthetic MMF aliases. Re-export here for
# backward-compat callers that imported from parameter_space.
from data_loader import CASH_TICKER, CASH_APY, CASH_ALIASES, DOLLAR_TICKER  # noqa: E402

CORE_TICKERS = ("TQQQ", "QQQ", "XLU")
ALL_TICKERS_INCL_CASH = CORE_TICKERS + (CASH_TICKER,)
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
      - enable_cash_in_regimes=["R2", "R3"]         # add CASH to the simplex for R2/R3
      - forced_base_allocations={"R2": {"XLU": 0.5, "CASH": 0.5}}  # pin a regime's base
      - notes="Iter 10: search over CASH in R2/R3, look for tail-risk hedge"
    """
    n_regimes_choices: list[int] | None = None
    drawdown_window_choices: list[int] | None = None
    threshold_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    upside_threshold_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    protection_threshold_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    rebalance_choices: dict[str, list[str]] = field(default_factory=dict)
    force_zero_params: list[str] = field(default_factory=list)
    weight_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Forced-allocation override per regime. Keys are regime names ("R1"..); values
    # are dicts of ticker→weight (must sum to ~1.0). When set for a regime, the
    # search-suggested base allocation is replaced with this fixed allocation.
    # CASH is recognized as a pass-through ticker.
    forced_base_allocations: dict[str, dict[str, float]] = field(default_factory=dict)
    # Regimes for which the simplex sampling should include CASH as a 4th
    # coordinate (TQQQ/QQQ/XLU/CASH instead of TQQQ/QQQ/XLU). Applies to base,
    # upside override, and protection override panels for the listed regimes.
    # Regimes NOT listed keep the legacy 3-ticker simplex (back-compat).
    enable_cash_in_regimes: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple bounds to lists for JSON serialization
        d["threshold_bounds"] = {k: list(v) for k, v in d["threshold_bounds"].items()}
        d["upside_threshold_bounds"] = {k: list(v) for k, v in d["upside_threshold_bounds"].items()}
        d["protection_threshold_bounds"] = {
            k: list(v) for k, v in d["protection_threshold_bounds"].items()
        }
        d["weight_bounds"] = {k: list(v) for k, v in d["weight_bounds"].items()}
        # forced_base_allocations and enable_cash_in_regimes are already JSON-friendly
        return d

    def _uses_cash(self) -> bool:
        """True if any regime references CASH (forced or via enable_cash_in_regimes)."""
        if self.enable_cash_in_regimes:
            return True
        return any(
            CASH_TICKER in alloc for alloc in self.forced_base_allocations.values()
        )

    def _cash_enabled_for(self, regime: str) -> bool:
        """True if CASH should be a sampleable simplex coordinate for this regime."""
        return regime in (self.enable_cash_in_regimes or [])


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
    """Suggest a raw weight in [0, 1], honoring `force_zero_params` and `weight_bounds`."""
    if constraints is not None:
        if name in constraints.force_zero_params:
            return trial.suggest_float(name, 0.0, 0.0)
        if name in constraints.weight_bounds:
            lo, hi = constraints.weight_bounds[name]
            return trial.suggest_float(name, float(lo), float(hi))
    return trial.suggest_float(name, 0.0, 1.0)


# Param-name suffix per ticker used in the simplex. Stable so Optuna's TPE
# can transfer learning across trials and batches.
_TICKER_SUFFIX = {
    "TQQQ": "tqqq",
    "QQQ": "qqq",
    "XLU": "xlu",
    "CASH": "cash",
}


def _suggest_simplex(
    trial: optuna.Trial,
    prefix: str,
    constraints: IterationConstraints | None,
    include_cash: bool,
) -> dict[str, float]:
    """
    Suggest weights over {TQQQ, QQQ, XLU} (and CASH when include_cash=True)
    that sum to 1.0. Returned dict always includes all sampled tickers; CASH
    key is omitted when include_cash=False (back-compat with legacy 3-ticker
    blocks downstream).
    """
    tickers: tuple[str, ...] = CORE_TICKERS + ((CASH_TICKER,) if include_cash else ())
    raw: dict[str, float] = {}
    for t in tickers:
        param_name = f"{prefix}_w_{_TICKER_SUFFIX[t]}_raw"
        raw[t] = _suggest_weight(trial, param_name, constraints)
    s = sum(raw.values())
    if s <= 1e-9:
        # Degenerate sample (e.g. all forced to zero). Pick a deterministic fallback.
        if constraints and f"{prefix}_w_tqqq_raw" in constraints.force_zero_params:
            fallback = "QQQ"
        else:
            fallback = "TQQQ"
        return {t: (1.0 if t == fallback else 0.0) for t in tickers}
    return {t: raw[t] / s for t in tickers}


def _suggest_simplex_3(
    trial: optuna.Trial, prefix: str, constraints: IterationConstraints | None,
) -> dict[str, float]:
    """Back-compat shim: 3-ticker simplex over {TQQQ, QQQ, XLU}."""
    return _suggest_simplex(trial, prefix, constraints, include_cash=False)


def _suggest_regime_block(
    trial: optuna.Trial,
    regime: str,
    dd_low: float,
    dd_high: float,
    constraints: IterationConstraints | None = None,
) -> dict[str, Any]:
    cash_enabled = constraints is not None and constraints._cash_enabled_for(regime)

    # Always sample (keeps Optuna's search space shape stable across trials,
    # even when forced_base_allocations would discard the base sample).
    base = _suggest_simplex(trial, f"{regime}_base", constraints, include_cash=cash_enabled)
    upside = _suggest_simplex(trial, f"{regime}_upside", constraints, include_cash=cash_enabled)
    protection = _suggest_simplex(trial, f"{regime}_protection", constraints, include_cash=cash_enabled)

    # If a forced base allocation is configured for this regime, replace the
    # sampled base with the fixed weights. Optuna's TPE doesn't care that the
    # sampled raw weights are unused — it just learns to ignore them.
    if constraints and regime in constraints.forced_base_allocations:
        base = dict(constraints.forced_base_allocations[regime])

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

    # Carry CASH onto every panel (base + overrides) when this regime references it.
    # CASH may show up via:
    #   - the simplex (cash_enabled): all three panels sampled it
    #   - a forced_base_allocations entry: only the base has it, but to keep the
    #     schema consistent we add 0.0 onto the override panels too.
    base_has_cash = CASH_TICKER in base
    if cash_enabled or base_has_cash:
        block[CASH_TICKER] = float(base.get(CASH_TICKER, 0.0))
        block["signal_overrides"]["upside"][CASH_TICKER] = float(
            upside.get(CASH_TICKER, 0.0)
        )
        block["signal_overrides"]["protection"][CASH_TICKER] = float(
            protection.get(CASH_TICKER, 0.0)
        )
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

    uses_cash = constraints is not None and constraints._uses_cash()
    panel_tickers = list(ALL_PANEL_TICKERS)
    alloc_tickers = list(CORE_TICKERS)
    if uses_cash:
        panel_tickers.append(CASH_TICKER)
        alloc_tickers.append(CASH_TICKER)

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
        "tickers": panel_tickers,
        "allocation_tickers": alloc_tickers,
        "minimum_allocation": 0.0,
        "regimes": regimes,
    }
