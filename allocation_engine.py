# allocation_engine.py
from utils import log

def get_allocation_for_regime(regime, config):
    params = config["regimes"][regime]
    invest_tickers = config["allocation_tickers"]

    alloc = {t: params[t] for t in invest_tickers}
    return alloc

