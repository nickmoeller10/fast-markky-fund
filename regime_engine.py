# regime_engine.py
from utils import log

def compute_drawdown_from_ath(series):
    log("Calculating ATH & drawdown series...")
    ath = series.cummax()
    dd = (ath - series) / ath
    log("ATH & drawdown calculation complete.")
    return dd, ath

def determine_regime(dd_value, config):
    for regime, params in config["regimes"].items():
        if params["dd_low"] <= dd_value < params["dd_high"]:
            log(f"Regime determined: {regime} (dd = {dd_value:.4f})")
            return regime
    log(f"!! No regime matched for dd = {dd_value:.4f}")
    return None
