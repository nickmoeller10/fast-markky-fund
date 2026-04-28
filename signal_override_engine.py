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


_PANEL_META_KEYS = frozenset({"enabled", "label", "direction", "threshold"})
_PANEL_SUM_TOL = 1e-3


def ensure_regime_signal_overrides(regime_params: Dict[str, Any]) -> None:
    """Mutate regime dict in place with defaults if missing.

    Preserves any ticker-weight keys present in panels (e.g. TQQQ/QQQ/XLU/CASH/$).
    Only the meta keys (enabled/label/direction/threshold) and the legacy
    TQQQ/QQQ/XLU schema get filled in with defaults — extra ticker keys passed
    in by the optimizer or the user are kept verbatim.
    """
    if "signal_overrides" not in regime_params or not isinstance(
        regime_params.get("signal_overrides"), dict
    ):
        regime_params["signal_overrides"] = default_signal_overrides_block()
        return
    so = regime_params["signal_overrides"]
    for key in ("upside", "protection"):
        if key not in so or not isinstance(so[key], dict):
            new_panel = default_signal_override_panel()
            if key == "protection":
                new_panel["direction"] = "below"
            so[key] = new_panel
            continue
        raw_in = so[key]
        merged = dict(raw_in)
        defaults = default_signal_override_panel()
        if key == "protection":
            defaults["direction"] = "below"
        for meta in _PANEL_META_KEYS:
            if meta not in merged:
                merged[meta] = defaults[meta]
        for tk in ("TQQQ", "QQQ", "XLU"):
            if tk not in merged:
                merged[tk] = 0.0
        if "enabled" not in raw_in:
            merged["enabled"] = bool(str(merged.get("label") or "").strip())
        so[key] = merged


def validate_panel_sums(config: Dict[str, Any], *, atol: float = _PANEL_SUM_TOL) -> None:
    """Raise ValueError if any enabled panel does not sum to 1.0.

    Walks every regime's base allocation and each *enabled* override panel,
    summing weights across ``config["allocation_tickers"]``. Disabled override
    panels are skipped (they're often left at all-zeros).

    Catches the iter-25-era bug where override panels were missing a CASH key,
    silently summing to ~0.31 and being renormalized at execution time —
    masking the fact that an entire ticker dimension had been dropped.
    """
    alloc_tickers = config.get("allocation_tickers", [])
    if not alloc_tickers:
        return
    regimes = config.get("regimes", {})
    for regime_name, params in regimes.items():
        if not isinstance(params, dict):
            continue
        base_weights = {t: float(params.get(t, 0) or 0) for t in alloc_tickers}
        base_sum = sum(base_weights.values())
        if abs(base_sum - 1.0) > atol:
            raise ValueError(
                f"Regime {regime_name} base allocation sums to {base_sum:.4f}, "
                f"expected 1.0 ± {atol}. Weights over {alloc_tickers}: {base_weights}"
            )
        so = params.get("signal_overrides", {}) or {}
        for panel_kind in ("upside", "protection"):
            panel = so.get(panel_kind, {}) or {}
            if not panel.get("enabled"):
                continue
            panel_weights = {t: float(panel.get(t, 0) or 0) for t in alloc_tickers}
            psum = sum(panel_weights.values())
            if abs(psum - 1.0) > atol:
                raise ValueError(
                    f"Regime {regime_name} {panel_kind} override panel sums to "
                    f"{psum:.4f}, expected 1.0 ± {atol}. "
                    f"Weights over {alloc_tickers}: {panel_weights}. "
                    f"Likely missing a ticker key on this panel — check that "
                    f"every allocation_ticker has an explicit weight."
                )


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
    Desired overlay for today, taking the prior day's override mode into account.
    - direction \"above\": active while Signal_total >= threshold
    - direction \"below\": active while Signal_total <= threshold
    - If both upside and protection are active, returns \"protection\".
    - If s_curr is NaN/non-finite, returns current_mode (no change).
    - Stickiness: once current_mode is \"upside\" or \"protection\", it is held
      until either a different override fires (this function returns the new
      one) or the regime changes (the caller clears the mode to \"none\" before
      invoking this function — see backtest.py regime-rebalance block).
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

    # New override fires → switch (protection wins ties).
    if prot_on and up_on:
        return "protection"
    if prot_on:
        return "protection"
    if up_on:
        return "upside"

    # Neither panel active right now. The override is sticky: stay in the
    # existing override until either (a) a regime change clears it externally
    # (handled in backtest.py at the regime-rebalance site), or (b) a new
    # override fires (handled above).
    if current_mode in ("upside", "protection"):
        return current_mode
    return "none"


def get_target_allocation_for_override(
    portfolio_regime: str,
    override_mode: str,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Weights for rebalance: regime default or upside/protection panel.

    Override panels MUST already sum to 1.0 within ``_PANEL_SUM_TOL`` — call
    ``validate_panel_sums(config)`` once at config-load time. A panel summing
    to 0 (e.g. all-zero defaults of a disabled override that fired anyway)
    falls back to the regime base allocation.
    """
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
    if abs(s - 1.0) > _PANEL_SUM_TOL:
        raise ValueError(
            f"Override panel for {portfolio_regime}/{override_mode} sums to "
            f"{s:.4f}, expected 1.0 ± {_PANEL_SUM_TOL}. Weights over "
            f"{alloc_tickers}: {raw}. Call validate_panel_sums(config) at "
            f"config load to surface this earlier."
        )
    return dict(raw)


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
