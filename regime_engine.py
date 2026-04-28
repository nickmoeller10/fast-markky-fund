
from utils import log
import math

def compute_drawdown_from_ath(series):
    """Drawdown vs running all-time high. Returns (dd, ath) — dd in [0, 1]."""
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


def determine_regime(dd_value, config, date=None):

    # Handle NaN drawdowns explicitly (check before conversion)
    if dd_value is None:
        if date:
            log(f"!! Drawdown is None on {date} — cannot determine regime.")
        else:
            log("!! Drawdown is None — cannot determine regime.")
        return None
    
    # Check for NaN (works with both numpy and Python types)
    try:
        if math.isnan(dd_value):
            if date:
                log(f"!! Drawdown is NaN on {date} — cannot determine regime.")
            else:
                log("!! Drawdown is NaN — cannot determine regime.")
            return None
    except (TypeError, ValueError):
        pass  # Not a numeric type that can be NaN
    
    # Convert to Python float if it's a numpy type to avoid formatting issues
    try:
        if hasattr(dd_value, 'item'):
            dd_value = dd_value.item()
        dd_value = float(dd_value)
    except (ValueError, TypeError):
        if date:
            log(f"!! Drawdown is not a valid number on {date}: {dd_value}")
        else:
            log(f"!! Drawdown is not a valid number: {dd_value}")
        return None

    # Extract regimes in fixed order
    regime_keys = list(config["regimes"].keys())

    for i, regime in enumerate(regime_keys):

        params = config["regimes"][regime]

        low = params["dd_low"]
        high = params["dd_high"]
        log(f"Checking {regime}: low={low}, high={high}, dd={dd_value:.4f}")

        is_last = (i == len(regime_keys) - 1)

        # Last regime uses inclusive high
        if is_last:
            if low <= dd_value <= high:
                log(f"Regime determined: {regime} (dd = {dd_value:.4f})")
                return regime
        else:
            if low <= dd_value < high:
                log(f"Regime determined: {regime} (dd = {dd_value:.4f})")
                return regime

    # If no regime found
    if date:
        log(f"!! No regime matched for dd = {dd_value:.4f} on {date}")
    else:
        log(f"!! No regime matched for dd = {dd_value:.4f}")

    return None
