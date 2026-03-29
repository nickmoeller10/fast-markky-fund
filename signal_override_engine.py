# signal_override_engine.py
# ======================================================================
# Signal_total level rules → optional allocation overlay per regime.
# direction "above": active while Signal_total >= threshold; "below": while <= threshold.
# Protection wins if both panels are active. Rebalance only when desired mode changes
# (handled in backtest). Regime rebalance clears override state.
# ======================================================================

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import pandas as pd

from allocation_engine import get_allocation_for_regime


def default_signal_override_panel() -> Dict[str, Any]:
    return {
        "enabled": False,
        "label": "",
        "direction": "above",
        "threshold": 0,
        "TQQQ": 0.0,
        "QQQ": 0.0,
        "XLU": 0.0,
    }


def default_signal_overrides_block() -> Dict[str, Any]:
    up = default_signal_override_panel()
    pr = default_signal_override_panel()
    pr["direction"] = "below"
    return {"upside": up, "protection": pr}


def ensure_regime_signal_overrides(regime_params: Dict[str, Any]) -> None:
    """Mutate regime dict in place with defaults if missing."""
    if "signal_overrides" not in regime_params:
        regime_params["signal_overrides"] = default_signal_overrides_block()
        return
    so = regime_params["signal_overrides"]
    if not isinstance(so, dict):
        regime_params["signal_overrides"] = default_signal_overrides_block()
        return
    for key in ("upside", "protection"):
        if key not in so or not isinstance(so[key], dict):
            so[key] = default_signal_override_panel()
        else:
            raw = so[key]
            p = default_signal_override_panel()
            for k in p:
                if k in raw:
                    p[k] = raw[k]
            # Config may omit enabled; treat labeled panels as on (UI/export friendly).
            if "enabled" not in raw:
                p["enabled"] = bool(str(p.get("label") or "").strip())
            so[key] = p


def _panel_allocation(panel: Dict[str, Any], allocation_tickers: list) -> Dict[str, float]:
    return {t: float(panel.get(t, 0) or 0) for t in allocation_tickers}


def allocation_human_readable(alloc: Dict[str, float], allocation_tickers: list) -> str:
    parts = [f"{t} {alloc.get(t, 0):.0%}" for t in allocation_tickers if float(alloc.get(t, 0) or 0) > 1e-9]
    return ", ".join(parts) if parts else ""


def _panel_active_at_level(panel: Dict[str, Any], s_curr: float) -> bool:
    """Whether this panel's level rule is satisfied at s_curr."""
    if not panel.get("enabled"):
        return False
    try:
        T = float(panel["threshold"])
    except (TypeError, ValueError):
        return False
    d = str(panel.get("direction", "above")).strip().lower()
    if d == "above":
        return bool(s_curr >= T)
    if d == "below":
        return bool(s_curr <= T)
    return False


def desired_signal_override_mode(
    s_curr: float,
    regime_params: Dict[str, Any],
    current_mode: str = "none",
) -> str:
    """
    Desired overlay from today's Signal_total only (level rules, not crosses).
    - direction \"above\": active while Signal_total >= threshold
    - direction \"below\": active while Signal_total <= threshold
    If both upside and protection are active, returns \"protection\".
    If s_curr is NaN, returns current_mode (no change).
    """
    ensure_regime_signal_overrides(regime_params)
    so = regime_params.get("signal_overrides", {})
    up = so.get("upside", {})
    pr = so.get("protection", {})

    if not up.get("enabled") and not pr.get("enabled"):
        return "none"

    if pd.isna(s_curr):
        return current_mode
    try:
        sc = float(s_curr)
    except (TypeError, ValueError):
        return current_mode
    if not math.isfinite(sc):
        return current_mode

    prot_on = _panel_active_at_level(pr, sc)
    up_on = _panel_active_at_level(up, sc)
    if prot_on and up_on:
        return "protection"
    if prot_on:
        return "protection"
    if up_on:
        return "upside"
    return "none"


def get_target_allocation_for_override(
    portfolio_regime: str,
    override_mode: str,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Weights for rebalance: regime default or upside/protection panel."""
    if override_mode == "none":
        return get_allocation_for_regime(portfolio_regime, config)

    ensure_regime_signal_overrides(config["regimes"][portfolio_regime])
    so = config["regimes"][portfolio_regime]["signal_overrides"]
    panel = so["upside"] if override_mode == "upside" else so["protection"]
    alloc_tickers = config["allocation_tickers"]
    raw = _panel_allocation(panel, alloc_tickers)
    s = sum(raw.values())
    if s <= 0:
        return get_allocation_for_regime(portfolio_regime, config)
    return {t: raw[t] / s for t in alloc_tickers}


def describe_signal_override_row(
    portfolio_regime: str,
    override_mode: str,
    config: Dict[str, Any],
) -> Tuple[str, str, str]:
    """(active, label, allocation_str) for daily snapshot."""
    if override_mode == "none":
        return "none", "", ""

    ensure_regime_signal_overrides(config["regimes"][portfolio_regime])
    so = config["regimes"][portfolio_regime]["signal_overrides"]
    panel = so["upside"] if override_mode == "upside" else so["protection"]
    label = str(panel.get("label") or "")
    alloc = _panel_allocation(panel, config["allocation_tickers"])
    return override_mode, label, allocation_human_readable(alloc, config["allocation_tickers"])
