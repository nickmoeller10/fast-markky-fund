
import pandas as pd
import numpy as np
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
# MODULE 1 — Normalized Values
# ------------------------------------------------------------
# def calculate_normalized_values(price_df, tickers, starting_val, start_date):
#     norm_df = pd.DataFrame(index=price_df.index)
#     for t in tickers:
#         base_price = price_df.loc[start_date, t]
#         norm_series = (price_df[t] / base_price) * starting_val
#         norm_series[:start_date] = np.nan
#         norm_df[f"{t}_norm"] = norm_series
#     return norm_df
def calculate_normalized_values(price_df, tickers, starting_val, start_date):
    norm_df = pd.DataFrame(index=price_df.index)

    for t in tickers:
        base_price = price_df.loc[start_date, t]

        norm_series = (price_df[t] / base_price) * starting_val

        # Ensure only pre-start rows are blank
        norm_series.loc[price_df.index < start_date] = np.nan

        norm_df[f"{t}_norm"] = norm_series

    return norm_df


# ------------------------------------------------------------
# MODULE 2 — Initial Allocation
# ------------------------------------------------------------
def get_initial_allocation(start_date, price_df, first_dd, config, regime_detector, rebalance_fn):
    market_regime = regime_detector(first_dd, config)
    portfolio_regime = market_regime
    alloc = get_allocation_for_regime(portfolio_regime, config)
    shares = rebalance_fn(config["starting_balance"], alloc, price_df.loc[start_date])

    return market_regime, portfolio_regime, shares


# ------------------------------------------------------------
# MODULE 3 — Apply asymmetric regime rules
# ------------------------------------------------------------
def apply_portfolio_regime_logic(prev_regime, market_regime):
    prev_num = regime_number(prev_regime)
    market_num = regime_number(market_regime)

    # RULE 1 — Follow market DOWN
    if market_num > prev_num:
        return regime_label(market_num)

    # RULE 2 — Follow up ONLY when back to R1
    if market_num == 1:
        return "R1"

    # RULE 3 — Otherwise hold
    return prev_regime


# ------------------------------------------------------------
# MODULE 4 — Missing data row recorder
# ------------------------------------------------------------
def record_missing_row(equity_rows, date, tickers, row_prices, row_norm, shares, portfolio_value, prev_regime, starting_val):
    record = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": None,
        "Portfolio_Regime": prev_regime,
        "Rebalanced": "",
        "Pct_Growth": portfolio_value / starting_val - 1
    }
    for t in tickers:
        record[f"{t}_price"] = row_prices.get(t)
        record[f"{t}_norm"] = row_norm.get(f"{t}_norm")
        record[f"{t}_shares"] = shares.get(t, 0)
        record[f"{t}_value"] = shares.get(t, 0) * row_prices.get(t, 0)

    equity_rows.append(record)


# ------------------------------------------------------------
# MODULE 5 — Portfolio value update
# ------------------------------------------------------------
def update_portfolio_value(shares, row_prices, prev_value, current_quarter_returns):
    new_value = sum(shares[t] * row_prices[t] for t in shares)

    if prev_value != 0:
        daily_ret = (new_value / prev_value) - 1
    else:
        daily_ret = 0

    current_quarter_returns.append(daily_ret)
    return new_value, new_value  # (portfolio_value, prev_value)


# ------------------------------------------------------------
# MODULE 6 — Record daily row
# ------------------------------------------------------------
def record_daily_row(equity_rows, date, tickers, row_prices, row_norm, shares,
                     portfolio_value, market_regime, portfolio_regime, starting_val, rebalanced_flag):

    record = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": portfolio_value / starting_val - 1
    }

    for t in tickers:
        record[f"{t}_price"] = row_prices[t]
        record[f"{t}_norm"] = row_norm[f"{t}_norm"]
        record[f"{t}_shares"] = float(shares.get(t, 0))
        record[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    equity_rows.append(record)


# ------------------------------------------------------------
# MODULE 7 — Record quarterly row
# ------------------------------------------------------------
def record_quarterly_row(quarterly_rows, date, tickers, row_prices, shares,
                         portfolio_value, market_regime, portfolio_regime,
                         last_rebalance_value, current_quarter_returns):

    qoq_return = (portfolio_value / last_rebalance_value) - 1
    qoq_volatility = pd.Series(current_quarter_returns).std() if len(current_quarter_returns) else 0

    quarter_rec = {
        "Date": date,
        "Portfolio_Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "QoQ_Return": qoq_return,
        "QoQ_Volatility": qoq_volatility,
    }

    for t in tickers:
        quarter_rec[f"{t}_shares"] = float(shares.get(t, 0))
        quarter_rec[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    quarterly_rows.append(quarter_rec)
    return portfolio_value, []  # reset jitter trackers


def run_backtest(price_data, config, regime_fn, regime_detector, rebalance_fn):

    log("Starting backtest...")

    tickers = config["tickers"]
    starting_val = config["starting_balance"]

    # ============================================================
    # PRICE DATA SETUP
    # ============================================================
    price_df = price_data[tickers].copy()
    start_date = price_df.dropna().index.min()
    log(f"[NORMALIZATION] First valid date: {start_date.date()}")

    # ------------------------------------------------------------
    # Normalized values
    # ------------------------------------------------------------
    norm_df = calculate_normalized_values(price_df, tickers, starting_val, start_date)

    # ------------------------------------------------------------
    # DRAWNDOWN SERIES
    # ------------------------------------------------------------
    primary = tickers[0]
    dd_series, ath_series = regime_fn(price_df[primary])
    log("Drawdown series calculated.")

    # Fix NaN dd values
    dd_series = dd_series.fillna(0).clip(lower=0, upper=1)

    # ------------------------------------------------------------
    # REBALANCE SCHEDULE
    # ------------------------------------------------------------
    rebalance_freq = config.get("rebalance_frequency", "monthly")
    rebalance_dates = get_rebalance_dates(price_df.index, rebalance_freq)
    rebalance_enabled = len(tickers) > 1 and rebalance_freq != "none"

    log(f"Rebalance frequency: {rebalance_freq}")
    log(f"Rebalance dates: {list(rebalance_dates)}")

    # ------------------------------------------------------------
    # INITIAL ALLOCATION
    # ------------------------------------------------------------
    first_dd = dd_series.loc[start_date]
    market_regime, portfolio_regime, shares = get_initial_allocation(
        start_date, price_df, first_dd, config, regime_detector, rebalance_fn
    )

    previous_regime = portfolio_regime
    portfolio_value = starting_val

    log(f"Initial market regime: {market_regime}")
    log(f"Initial portfolio regime: {portfolio_regime}")
    log(f"Initial shares: {shares}")

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------
    equity_rows = []
    quarterly_rows = []

    # QoQ tracking
    last_rebalance_value = portfolio_value
    current_quarter_returns = []
    prev_value = portfolio_value

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for date in price_df.index:

        row_prices = price_df.loc[date]
        row_norm = norm_df.loc[date]

        # ------------------------------------------------------------
        # HANDLE MISSING ROWS
        # ------------------------------------------------------------
        if row_prices.isna().any():
            record_missing_row(
                equity_rows, date, tickers, row_prices, row_norm,
                shares, portfolio_value, previous_regime, starting_val
            )
            continue

        # ------------------------------------------------------------
        # MARKET REGIME TODAY
        # ------------------------------------------------------------
        dd_val = dd_series.loc[date]
        market_regime = regime_detector(dd_val, config)

        rebalanced_flag = ""

        # ------------------------------------------------------------
        # REBALANCE LOGIC
        # ------------------------------------------------------------
        if rebalance_enabled and date in rebalance_dates:

            # Apply asymmetric regime rules
            new_regime = apply_portfolio_regime_logic(portfolio_regime, market_regime)
            portfolio_regime = new_regime

            # Apply new allocations
            alloc = get_allocation_for_regime(portfolio_regime, config)
            shares = rebalance_fn(portfolio_value, alloc, row_prices)

            rebalanced_flag = "Rebalanced"

            # QoQ Performance
            qoq_return = (portfolio_value / last_rebalance_value) - 1
            qoq_volatility = (
                pd.Series(current_quarter_returns).std()
                if len(current_quarter_returns)
                else 0
            )

            record_quarterly_row(
                quarterly_rows, date, tickers, row_prices, shares,
                portfolio_value, market_regime, portfolio_regime,
                last_rebalance_value, current_quarter_returns
            )

            last_rebalance_value = portfolio_value
            current_quarter_returns = []  # reset

        # ------------------------------------------------------------
        # UPDATE PORTFOLIO VALUE
        # ------------------------------------------------------------
        portfolio_value, prev_value = update_portfolio_value(
            shares, row_prices, prev_value, current_quarter_returns
        )

        # ------------------------------------------------------------
        # DAILY RECORD
        # ------------------------------------------------------------
        record_daily_row(
            equity_rows, date, tickers, row_prices, row_norm, shares,
            portfolio_value, market_regime, portfolio_regime,
            starting_val, rebalanced_flag
        )

        previous_regime = portfolio_regime

    log("Backtest completed.")

    equity_df = pd.DataFrame(equity_rows)
    quarterly_df = pd.DataFrame(quarterly_rows)

    return equity_df, quarterly_df
