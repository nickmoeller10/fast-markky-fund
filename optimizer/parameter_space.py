"""
Defines the search space for the config optimizer.

Suggests a complete CONFIG dict from an Optuna trial. The returned config
is shape-compatible with config.CONFIG and consumed by run_backtest.

Search space:
  - Regime drawdown thresholds (R1/R2, R2/R3 boundaries)
  - drawdown_window_years (categorical: 1, 2, 3, 5)
  - Per-regime base allocations across {TQQQ, QQQ, XLU} + optional alt sleeve
  - Per-regime upside override (threshold + allocation)
  - Per-regime protection override (threshold + allocation)
  - Per-regime rebalance_on_downward / rebalance_on_upward (match | hold)
  - Optional alternative ticker per regime (TLT | GLD | SPLV | BIL | none)

Fixed (not searched):
  - rebalance_strategy = "per_regime"
  - rebalance_frequency = "instant"
  - drawdown_ticker = "QQQ"
  - starting_balance, dates: same as production CONFIG
"""

from __future__ import annotations

from typing import Any

import optuna


CORE_TICKERS = ("TQQQ", "QQQ", "XLU")
ALT_TICKERS = (None, "TLT", "GLD", "SPLV", "BIL")
ALL_PANEL_TICKERS = ["QQQ", "TQQQ", "XLU", "SPY", "TLT", "GLD", "SPLV", "BIL"]


def _suggest_simplex_3(trial: optuna.Trial, prefix: str) -> tuple[float, float, float]:
    """
    Suggest 3 weights that sum to 1.0 using a Dirichlet-style decomposition.
    Each call produces a triplet (w_tqqq, w_qqq, w_xlu) — independent of any alt.
    Returns weights normalized to sum to 1.0 with at most one alt added separately.
    """
    a = trial.suggest_float(f"{prefix}_w_tqqq_raw", 0.0, 1.0)
    b = trial.suggest_float(f"{prefix}_w_qqq_raw", 0.0, 1.0)
    c = trial.suggest_float(f"{prefix}_w_xlu_raw", 0.0, 1.0)
    s = a + b + c
    if s <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (a / s, b / s, c / s)


def _suggest_alloc_with_optional_alt(
    trial: optuna.Trial, prefix: str
) -> dict[str, float]:
    """
    Suggest a normalized allocation across CORE_TICKERS + at most one alt sleeve.
    Returns a dict of weights summing to ~1.0; tickers with weight 0 are still
    keyed (zero) so the regime block is always shape-stable.
    """
    alt = trial.suggest_categorical(f"{prefix}_alt", ALT_TICKERS)

    if alt is None:
        w_t, w_q, w_x = _suggest_simplex_3(trial, prefix)
        return {"TQQQ": w_t, "QQQ": w_q, "XLU": w_x}

    # With an alt: 4-weight simplex
    a = trial.suggest_float(f"{prefix}_w_tqqq_raw", 0.0, 1.0)
    b = trial.suggest_float(f"{prefix}_w_qqq_raw", 0.0, 1.0)
    c = trial.suggest_float(f"{prefix}_w_xlu_raw", 0.0, 1.0)
    d = trial.suggest_float(f"{prefix}_w_alt_raw", 0.0, 1.0)
    s = a + b + c + d
    if s <= 1e-9:
        return {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0, alt: 0.0}
    return {"TQQQ": a / s, "QQQ": b / s, "XLU": c / s, alt: d / s}


def _suggest_regime_block(
    trial: optuna.Trial,
    regime: str,
    dd_low: float,
    dd_high: float,
) -> dict[str, Any]:
    base_alloc = _suggest_alloc_with_optional_alt(trial, f"{regime}_base")
    upside_alloc = _suggest_alloc_with_optional_alt(trial, f"{regime}_upside")
    protection_alloc = _suggest_alloc_with_optional_alt(trial, f"{regime}_protection")

    upside_threshold = trial.suggest_int(f"{regime}_upside_threshold", 1, 5)
    protection_threshold = trial.suggest_int(f"{regime}_protection_threshold", -5, -1)

    rb_down = trial.suggest_categorical(f"{regime}_rebalance_on_downward", ["match", "hold"])
    rb_up = trial.suggest_categorical(f"{regime}_rebalance_on_upward", ["match", "hold"])

    block = {
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
                **{t: float(upside_alloc.get(t, 0.0)) for t in CORE_TICKERS},
            },
            "protection": {
                "enabled": True,
                "label": f"{regime} Protection",
                "direction": "below",
                "threshold": protection_threshold,
                **{t: float(protection_alloc.get(t, 0.0)) for t in CORE_TICKERS},
            },
        },
    }
    # Base regime weights for core tickers
    for t in CORE_TICKERS:
        block[t] = float(base_alloc.get(t, 0.0))

    # Pull through any alt-ticker weights into the block AND override panels
    alt_tickers_used: set[str] = set()
    for alloc in (base_alloc, upside_alloc, protection_alloc):
        for t in alloc:
            if t not in CORE_TICKERS:
                alt_tickers_used.add(t)
    for t in alt_tickers_used:
        block[t] = float(base_alloc.get(t, 0.0))
        block["signal_overrides"]["upside"][t] = float(upside_alloc.get(t, 0.0))
        block["signal_overrides"]["protection"][t] = float(protection_alloc.get(t, 0.0))
    return block, alt_tickers_used


def suggest_config(trial: optuna.Trial) -> dict[str, Any]:
    """Build a complete CONFIG dict from an Optuna trial."""
    # Boundaries: keep R1 narrower than R2, R2 narrower than R3
    r1_high = trial.suggest_float("r1_dd_high", 0.04, 0.10)
    r2_high = trial.suggest_float("r2_dd_high", r1_high + 0.05, 0.30)
    drawdown_window_years = trial.suggest_categorical("drawdown_window_years", [1, 2, 3, 5])

    r1_block, r1_alts = _suggest_regime_block(trial, "R1", 0.0, r1_high)
    r2_block, r2_alts = _suggest_regime_block(trial, "R2", r1_high, r2_high)
    r3_block, r3_alts = _suggest_regime_block(trial, "R3", r2_high, 1.0)

    alt_tickers_used = sorted(r1_alts | r2_alts | r3_alts)
    allocation_tickers = list(CORE_TICKERS) + alt_tickers_used
    panel_tickers = sorted(set(allocation_tickers) | {"SPY"})

    return {
        "starting_balance": 10_000,
        "start_date": "1999-01-04",
        "end_date": "2026-03-27",
        "drawdown_ticker": "QQQ",
        "drawdown_window_enabled": True,
        "drawdown_window_years": int(drawdown_window_years),
        "rebalance_frequency": "instant",
        "rebalance_holiday_rule": "next_trading_day",
        "rebalance_strategy": "per_regime",
        "dividend_reinvestment": False,
        "dividend_reinvestment_target": "cash",
        "tickers": panel_tickers,
        "allocation_tickers": allocation_tickers,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": r1_block,
            "R2": r2_block,
            "R3": r3_block,
        },
    }
