
from utils import log
import math

def compute_drawdown_from_ath(series):
    log("Calculating ATH & drawdown series...")
    ath = series.cummax()
    dd = (ath - series) / ath
    log("ATH & drawdown calculation complete.")
    return dd, ath


def determine_regime(dd_value, config, date=None):

    # Handle NaN drawdowns explicitly
    if dd_value is None or (isinstance(dd_value, float) and math.isnan(dd_value)):
        if date:
            log(f"!! Drawdown is NaN on {date} — cannot determine regime.")
        else:
            log("!! Drawdown is NaN — cannot determine regime.")
        return None

    # Extract regimes in fixed order
    regime_keys = list(config["regimes"].keys())

    for i, regime in enumerate(regime_keys):
        params = config["regimes"][regime]

        low = params["dd_low"]
        high = params["dd_high"]
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
