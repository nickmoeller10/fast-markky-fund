# allocation_engine.py
from utils import log
import pandas as pd


def get_allocation_for_regime(regime, config):
    params = config["regimes"][regime]
    invest_tickers = config["allocation_tickers"]

    alloc = {t: params[t] for t in invest_tickers}
    return alloc


def _px(prices, ticker):
    if isinstance(prices, pd.Series):
        if ticker not in prices.index:
            return None
        v = prices[ticker]
    else:
        v = prices.get(ticker)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x <= 0:
        return None
    return x


def tradable_allocation(alloc, prices, config):
    """
    Map regime target weights to weights we can execute on this bar.

    - Keeps only allocation_tickers with positive target weight and a valid price; renormalizes to 1.
    - If that set is empty (e.g. 100% TQQQ before TQQQ lists): use 100% QQQ when TQQQ was targeted and QQQ prices.
    - Else 100% drawdown_ticker if it is in allocation_tickers and priced.
    - Else equal weight over any allocation ticker with a valid price.
    Returns {} only if no allocation ticker has a price (stay in cash for that rebalance).
    """
    allocation_tickers = config.get("allocation_tickers", [])
    drawdown_ticker = config.get("drawdown_ticker")

    positive = {}
    for t in allocation_tickers:
        w = float(alloc.get(t, 0) or 0)
        if w > 0 and _px(prices, t) is not None:
            positive[t] = w
    s = sum(positive.values())
    if s > 0:
        return {t: positive[t] / s for t in positive}

    # Proxy: leveraged Nasdaq sleeve -> QQQ when TQQQ not yet trading
    for t in allocation_tickers:
        w = float(alloc.get(t, 0) or 0)
        if w <= 0:
            continue
        if t == "TQQQ" and _px(prices, "TQQQ") is None:
            if "QQQ" in allocation_tickers and _px(prices, "QQQ") is not None:
                log("Tradable allocation: TQQQ not priced — using QQQ as proxy for this rebalance.")
                return {"QQQ": 1.0}

    if drawdown_ticker in allocation_tickers and _px(prices, drawdown_ticker) is not None:
        log(f"Tradable allocation: using 100% {drawdown_ticker} (only priced risk sleeve available).")
        return {drawdown_ticker: 1.0}

    priced = [t for t in allocation_tickers if _px(prices, t) is not None]
    if not priced:
        log("Tradable allocation: no priced allocation tickers — skipping deployable weights.")
        return {}
    eq = 1.0 / len(priced)
    return {t: eq for t in priced}

