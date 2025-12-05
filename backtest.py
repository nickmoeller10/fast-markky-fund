import pandas as pd
import numpy as np
import math
from allocation_engine import get_allocation_for_regime
from utils import log


# ------------------------------------------------------------
# Helpers to convert regimes like "R3" <-> 3
# ------------------------------------------------------------
def regime_number(label: str) -> int:
    return int(label.replace("R", ""))


def regime_label(num: int) -> str:
    return f"R{num}"


# ------------------------------------------------------------
# Rebalance frequency helper
# ------------------------------------------------------------
def get_rebalance_dates(index, frequency):
    if frequency == "none":
        return pd.DatetimeIndex([])
    if frequency == "daily":
        return index
    if frequency == "weekly":
        return index.to_series().resample("W-FRI").last().index
    if frequency == "monthly":
        return index.to_series().resample("M").last().index
    if frequency == "quarterly":
        return index.to_series().resample("Q").last().index
    if frequency == "semiannual":
        return index.to_series().resample("2Q").last().index
    if frequency == "annual":
        return index.to_series().resample("A").last().index
    raise ValueError(f"Invalid rebalance frequency: {frequency}")


# ------------------------------------------------------------
# Compute raw drawdown using true prices
# ------------------------------------------------------------
def compute_drawdown_from_ath(series):
    log("Computing raw ATH + drawdown...")
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


# ------------------------------------------------------------
# Normalized values
# ------------------------------------------------------------
def calculate_normalized_values(price_df, tickers, starting_val, start_date):
    norm_df = pd.DataFrame(index=price_df.index)

    for t in tickers:
        base_price = price_df.loc[start_date, t]
        norm = (price_df[t] / base_price) * starting_val
        norm.loc[price_df.index < start_date] = np.nan
        norm_df[f"{t}_norm"] = norm

    return norm_df


# ------------------------------------------------------------
# Initial allocation
# ------------------------------------------------------------
def get_initial_allocation(start_date, price_df, first_dd, config, regime_detector, rebalance_fn):
    regime = regime_detector(first_dd, config)
    alloc = get_allocation_for_regime(regime, config)
    shares = rebalance_fn(config["starting_balance"], alloc, price_df.loc[start_date])
    return regime, regime, shares


# ------------------------------------------------------------
# Apply asymmetric regime rules
# ------------------------------------------------------------
def apply_portfolio_regime_logic(prev_regime, market_regime):
    prev_num = regime_number(prev_regime)
    market_num = regime_number(market_regime)

    # Follow DOWN moves immediately
    if market_num > prev_num:
        return regime_label(market_num)

    # Follow UP only back to R1
    if market_num == 1:
        return "R1"

    return prev_regime


# ------------------------------------------------------------
# Missing row recorder
# ------------------------------------------------------------
def record_missing_row(equity_rows, date, tickers, row_prices, row_norm, shares, portfolio_value,
                       prev_regime, starting_val, drawdown_cols):

    record = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": None,
        "Portfolio_Regime": prev_regime,
        "Rebalanced": "",
        "Pct_Growth": portfolio_value / starting_val - 1,
        **drawdown_cols
    }

    for t in tickers:
        record[f"{t}_price"] = row_prices.get(t)
        record[f"{t}_norm"] = row_norm.get(f"{t}_norm")
        record[f"{t}_shares"] = shares.get(t, 0)
        record[f"{t}_value"] = shares.get(t, 0) * row_prices.get(t, 0)

    equity_rows.append(record)


# ------------------------------------------------------------
# Update portfolio value
# ------------------------------------------------------------
def update_portfolio_value(shares, row_prices, prev_value, quarter_returns):
    new_value = sum(shares[t] * row_prices[t] for t in shares)

    if prev_value != 0:
        daily_ret = (new_value / prev_value) - 1
    else:
        daily_ret = 0

    quarter_returns.append(daily_ret)
    return new_value, new_value


# ------------------------------------------------------------
# Record daily row
# ------------------------------------------------------------
def record_daily_row(equity_rows, date, tickers, row_prices, row_norm, shares,
                     portfolio_value, market_regime, portfolio_regime,
                     starting_val, rebalanced_flag, drawdown_cols):

    record = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": portfolio_value / starting_val - 1,
        **drawdown_cols
    }

    for t in tickers:
        record[f"{t}_price"] = row_prices[t]
        record[f"{t}_norm"] = row_norm[f"{t}_norm"]
        record[f"{t}_shares"] = float(shares.get(t, 0))
        record[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    equity_rows.append(record)


# ------------------------------------------------------------
# Record quarterly row
# ------------------------------------------------------------
def record_quarterly_row(quarterly_rows, date, tickers, row_prices, shares,
                         portfolio_value, market_regime, portfolio_regime,
                         last_rebalance_value, quarter_returns, drawdown_cols):

    qoq_ret = (portfolio_value / last_rebalance_value) - 1
    qoq_vol = pd.Series(quarter_returns).std() if len(quarter_returns) else 0

    row = {
        "Date": date,
        "Portfolio_Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "QoQ_Return": qoq_ret,
        "QoQ_Volatility": qoq_vol,
        **drawdown_cols
    }

    for t in tickers:
        row[f"{t}_shares"] = float(shares.get(t, 0))
        row[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    quarterly_rows.append(row)
    return portfolio_value, []


# ------------------------------------------------------------
# MAIN BACKTEST ENGINE
# ------------------------------------------------------------
def run_backtest(price_data, config, regime_fn, regime_detector, rebalance_fn):

    tickers = config["tickers"]
    starting_val = config["starting_balance"]
    drawdown_ticker = config["drawdown_ticker"]

    # Price data
    price_df = price_data[tickers].copy()
    start_date = price_df.index[price_df.notna().any(axis=1)][0]

    # Normalized benchmarks
    norm_df = calculate_normalized_values(price_df, tickers, starting_val, start_date)

    # Raw drawdown for regime detection
    dd_raw, ath_raw = compute_drawdown_from_ath(price_data[drawdown_ticker])
    dd_raw = dd_raw.fillna(0).clip(lower=0, upper=1)

    # Rebalance schedule
    rebalance_freq = config.get("rebalance_frequency", "monthly")
    rebalance_dates = get_rebalance_dates(price_df.index, rebalance_freq)
    rebalance_enabled = len(tickers) > 1 and rebalance_freq != "none"

    # Initial allocation
    first_dd = dd_raw.loc[start_date]
    market_regime, portfolio_regime, shares = get_initial_allocation(
        start_date, price_df, first_dd, config, regime_detector, rebalance_fn
    )

    portfolio_value = starting_val
    prev_value = starting_val

    equity_rows = []
    quarterly_rows = []

    last_rebalance_value = portfolio_value
    quarter_returns = []

    # ============================================
    # MAIN LOOP
    # ============================================
    for date in price_df.index:

        row_prices = price_df.loc[date]
        row_norm = norm_df.loc[date]

        # Compute DD columns for this date
        raw_price = price_data.loc[date, drawdown_ticker]
        raw_ath = ath_raw.loc[date]
        raw_dd = dd_raw.loc[date]

        portfolio_ath = np.nan
        portfolio_dd = np.nan

        if date == start_date:
            portfolio_ath = portfolio_value
            portfolio_dd = 0
        else:
            # Will fill after computing new value
            pass

        dd_cols = {
            f"{drawdown_ticker}_ATH_raw": raw_ath,
            f"{drawdown_ticker}_DD_raw": raw_dd,
            "Portfolio_ATH": None,  # filled later
            "Portfolio_DD": None
        }

        # Missing rows
        if row_prices.isna().any():
            record_missing_row(
                equity_rows, date, tickers, row_prices, row_norm,
                shares, portfolio_value, portfolio_regime,
                starting_val, dd_cols
            )
            continue

        # Determine market regime
        market_regime = regime_detector(raw_dd, config)
        rebalanced_flag = ""

        # Rebalance
        if rebalance_enabled and date in rebalance_dates:

            # Compute portfolio ATH + DD BEFORE writing quarterly row
            if len(equity_rows) == 0:
                portfolio_ath = portfolio_value
            else:
                portfolio_ath = max(equity_rows[-1]["Portfolio_ATH"], portfolio_value)

            portfolio_dd = (portfolio_value / portfolio_ath) - 1

            # Update dd_cols with correct values
            dd_cols["Portfolio_ATH"] = portfolio_ath
            dd_cols["Portfolio_DD"] = portfolio_dd

            # Apply asymmetric regime rules
            new_regime = apply_portfolio_regime_logic(portfolio_regime, market_regime)
            portfolio_regime = new_regime

            # Apply new allocation
            alloc = get_allocation_for_regime(portfolio_regime, config)
            shares = rebalance_fn(portfolio_value, alloc, row_prices)

            rebalanced_flag = "Rebalanced"

            # Record quarterly row with correct Portfolio_ATH/DD
            record_quarterly_row(
                quarterly_rows, date, tickers, row_prices, shares,
                portfolio_value, market_regime, portfolio_regime,
                last_rebalance_value, quarter_returns, dd_cols
            )

            last_rebalance_value = portfolio_value
            quarter_returns = []

        # Update portfolio value
        portfolio_value, prev_value = update_portfolio_value(
            shares, row_prices, prev_value, quarter_returns
        )

        # Compute portfolio DD
        if len(equity_rows) == 0:
            portfolio_ath = portfolio_value
        else:
            portfolio_ath = max(equity_rows[-1]["Portfolio_ATH"], portfolio_value)

        portfolio_dd = (portfolio_value / portfolio_ath) - 1

        dd_cols["Portfolio_ATH"] = portfolio_ath
        dd_cols["Portfolio_DD"] = portfolio_dd

        # Record daily row
        record_daily_row(
            equity_rows, date, tickers, row_prices, row_norm, shares,
            portfolio_value, market_regime, portfolio_regime,
            starting_val, rebalanced_flag, dd_cols
        )

    # Convert DFs
    equity_df = pd.DataFrame(equity_rows)
    quarterly_df = pd.DataFrame(quarterly_rows)

    return equity_df, quarterly_df
