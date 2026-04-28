"""
Per-day row-recording helpers used by `run_backtest`.

`record_daily_row` and `record_missing_row` build the rows that ultimately
become `equity_df`. `update_portfolio_value` marks shares to a price row.
"""
from __future__ import annotations

from signal_override_engine import describe_signal_override_row
from utils import log


def _price(row, ticker):
    """Read a ticker price from either a dict or a Series; 0 if missing."""
    if isinstance(row, dict):
        return row.get(ticker, 0)
    if hasattr(row, "get"):
        return row.get(ticker, 0)
    return row[ticker] if ticker in row.index else 0


def _norm(row, ticker):
    if isinstance(row, dict):
        return row.get(f"{ticker}_norm", 0)
    if hasattr(row, "get"):
        return row.get(f"{ticker}_norm", 0)
    return row[f"{ticker}_norm"] if f"{ticker}_norm" in row.index else 0


def _shares(shares, ticker):
    return shares.get(ticker, 0) if isinstance(shares, dict) else 0


def record_missing_row(equity_rows, date, tickers, row_prices, row_norm,
                       shares, portfolio_value, prev_regime, starting_val, dd_cols,
                       regime_trajectory="", prev_market_regime=None,
                       config=None, signal_override_mode="none"):
    if shares is None:
        shares = {}

    if config is not None:
        oa, ol, oalloc = describe_signal_override_row(prev_regime, signal_override_mode, config)
    else:
        oa, ol, oalloc = "none", "", ""

    rec = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": None,
        "Portfolio_Regime": prev_regime,
        "Regime_Trajectory": regime_trajectory,
        "Prev_Market_Regime": prev_market_regime,
        "Rebalanced": "",
        "Pct_Growth": portfolio_value / starting_val - 1,
        "Signal_override_active": oa,
        "Signal_override_label": ol,
        "Signal_override_allocation": oalloc,
        **dd_cols,
    }

    for t in tickers:
        price_val = _price(row_prices, t)
        norm_val = _norm(row_norm, t)
        shares_val = _shares(shares, t)
        rec[f"{t}_price"] = price_val
        rec[f"{t}_norm"] = norm_val
        rec[f"{t}_shares"] = shares_val
        rec[f"{t}_value"] = shares_val * (price_val if price_val else 0)

    equity_rows.append(rec)


def update_portfolio_value(shares, row_prices, prev_value, quarter_returns):
    """Mark shares to today's prices, append daily return, return (new_value, new_value)."""
    if shares is None:
        shares = {}

    new_value = 0.0
    for t in shares:
        if t in row_prices:
            if isinstance(row_prices, dict):
                price = row_prices[t]
            else:
                price = row_prices.loc[t] if t in row_prices.index else 0
            new_value += shares[t] * price

    daily_ret = (new_value / prev_value - 1) if prev_value != 0 else 0
    quarter_returns.append(daily_ret)
    return new_value, new_value


def record_daily_row(equity_rows, date, tickers, row_prices, row_norm,
                     shares, portfolio_value, market_regime,
                     portfolio_regime, starting_val, rebalanced_flag,
                     dd_cols, cash_balance=0.0, regime_trajectory="",
                     prev_market_regime=None, config=None,
                     signal_override_mode="none"):
    if shares is None:
        log(f"⚠️ WARNING: shares is None in record_daily_row on {date}. Initializing as empty dict.")
        shares = {}

    total_value = portfolio_value + cash_balance

    if config is not None:
        oa, ol, oalloc = describe_signal_override_row(portfolio_regime, signal_override_mode, config)
    else:
        oa, ol, oalloc = "none", "", ""

    rec = {
        "Date": date,
        "Value": total_value,
        "Cash": cash_balance,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Regime_Trajectory": regime_trajectory,
        "Prev_Market_Regime": prev_market_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": total_value / starting_val - 1,
        "Signal_override_active": oa,
        "Signal_override_label": ol,
        "Signal_override_allocation": oalloc,
        **dd_cols,
    }

    for t in tickers:
        rec[f"{t}_price"] = row_prices[t]
        rec[f"{t}_norm"] = row_norm[f"{t}_norm"]
        rec[f"{t}_shares"] = float(_shares(shares, t))
        rec[f"{t}_value"] = float(_shares(shares, t) * row_prices[t])

    equity_rows.append(rec)
