# rebalance_engine.py
from utils import log
import pandas as pd

def rebalance_portfolio(portfolio_value, allocations, prices):
    log(f"Rebalancing portfolio. Value = {portfolio_value:.2f}")
    log(f"Prices: {prices.to_dict()}")
    log(f"Target allocations: {allocations}")

    shares = {}
    for ticker, weight in allocations.items():
        alloc_value = portfolio_value * weight
        price = prices[ticker]

        if pd.isna(price):
            log(f"!!! ERROR: Price for {ticker} is NaN — skipping rebalance")
            return None

        shares[ticker] = alloc_value / price
        log(f"  {ticker}: alloc={alloc_value:.2f}, price={price:.2f}, shares={shares[ticker]:.4f}")

    log(f"Rebalance complete. Shares: {shares}")
    return shares
