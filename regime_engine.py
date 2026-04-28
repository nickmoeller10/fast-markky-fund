import math

from utils import log


def compute_drawdown_from_ath(series):
    """Drawdown vs running all-time high. Returns (dd, ath) — dd in [0, 1]."""
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


def _coerce_dd_value(dd_value, date):
    """Return float dd or None (with one log line) for NaN/invalid inputs."""
    if dd_value is None:
        log(f"!! Drawdown is None{f' on {date}' if date else ''} — cannot determine regime.")
        return None
    try:
        if math.isnan(dd_value):
            log(f"!! Drawdown is NaN{f' on {date}' if date else ''} — cannot determine regime.")
            return None
    except (TypeError, ValueError):
        pass
    try:
        if hasattr(dd_value, "item"):
            dd_value = dd_value.item()
        return float(dd_value)
    except (ValueError, TypeError):
        log(f"!! Drawdown is not a valid number{f' on {date}' if date else ''}: {dd_value}")
        return None


def determine_regime(dd_value, config, date=None):
    """Map a drawdown value to a regime label using config['regimes'] bands.

    Non-last regimes use [low, high); the last regime uses [low, high].
    NaN/None/non-numeric inputs return None (logged once).
    """
    dd_value = _coerce_dd_value(dd_value, date)
    if dd_value is None:
        return None

    regime_keys = list(config["regimes"].keys())
    for i, regime in enumerate(regime_keys):
        params = config["regimes"][regime]
        low, high = params["dd_low"], params["dd_high"]
        is_last = i == len(regime_keys) - 1
        in_band = low <= dd_value <= high if is_last else low <= dd_value < high
        if in_band:
            return regime

    log(f"!! No regime matched for dd = {dd_value:.4f}{f' on {date}' if date else ''}")
    return None
