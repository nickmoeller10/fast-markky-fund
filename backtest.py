import pandas as pd
import numpy as np
from allocation_engine import get_allocation_for_regime
from utils import log


# ------------------------------------------------------------
# Regime helpers
# ------------------------------------------------------------
def regime_number(label: str) -> int:
    return int(label.replace("R", ""))


def regime_label(num: int) -> str:
    return f"R{num}"


# ------------------------------------------------------------
# Rebalance date utilities
# ------------------------------------------------------------
def get_rebalance_dates(index, frequency):
    if frequency in ["none", "instant"]:
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
# Normalized value calculation
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
# Drawdown from ATH
# ------------------------------------------------------------
def compute_drawdown_from_ath(series):
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


# ------------------------------------------------------------
# Initial allocation
# ------------------------------------------------------------
def get_initial_allocation(start_date, price_df, first_dd, config, regime_detector, rebalance_fn):
    market_regime = regime_detector(first_dd, config)
    portfolio_regime = market_regime
    alloc = get_allocation_for_regime(portfolio_regime, config)
    shares = rebalance_fn(config["starting_balance"], alloc, price_df.loc[start_date])
    return market_regime, portfolio_regime, shares


# ------------------------------------------------------------
# Asymmetric rules (Option A)
# ------------------------------------------------------------
def apply_asymmetric_rules(prev_regime, market_regime):
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    # If market goes DOWN → follow it immediately
    if mkt_n > prev_n:
        return regime_label(mkt_n)

    # Only move back UP if market is fully safe (R1)
    if mkt_n == 1:
        return "R1"

    # Otherwise hold previous regime
    return prev_regime


# ------------------------------------------------------------
# Recording helpers
# ------------------------------------------------------------
def record_missing_row(equity_rows, date, tickers, row_prices, row_norm,
                       shares, portfolio_value, prev_regime, starting_val, dd_cols):

    rec = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": None,
        "Portfolio_Regime": prev_regime,
        "Rebalanced": "",
        "Pct_Growth": portfolio_value / starting_val - 1,
        **dd_cols
    }

    for t in tickers:
        rec[f"{t}_price"] = row_prices.get(t)
        rec[f"{t}_norm"] = row_norm.get(f"{t}_norm")
        rec[f"{t}_shares"] = shares.get(t, 0)
        rec[f"{t}_value"] = shares.get(t, 0) * row_prices.get(t, 0)

    equity_rows.append(rec)


def update_portfolio_value(shares, row_prices, prev_value, quarter_returns):
    new_value = sum(shares[t] * row_prices[t] for t in shares)
    daily_ret = (new_value / prev_value - 1) if prev_value != 0 else 0
    quarter_returns.append(daily_ret)
    return new_value, new_value


def record_daily_row(equity_rows, date, tickers, row_prices, row_norm,
                     shares, portfolio_value, market_regime,
                     portfolio_regime, starting_val, rebalanced_flag,
                     dd_cols):

    rec = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": portfolio_value / starting_val - 1,
        **dd_cols
    }

    for t in tickers:
        rec[f"{t}_price"] = row_prices[t]
        rec[f"{t}_norm"] = row_norm[f"{t}_norm"]
        rec[f"{t}_shares"] = float(shares.get(t, 0))
        rec[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    equity_rows.append(rec)


def record_quarterly_row(quarterly_rows, date, tickers, row_prices, shares,
                         portfolio_value, market_regime, portfolio_regime,
                         last_rebalance_value, quarter_returns, dd_cols):

    qoq_ret = (portfolio_value / last_rebalance_value) - 1
    qoq_vol = pd.Series(quarter_returns).std() if len(quarter_returns) else 0

    row = {
        "Date": date,
        "Portfolio_Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "QoQ_Return": qoq_ret,
        "QoQ_Volatility": qoq_vol,
        **dd_cols
    }

    for t in tickers:
        row[f"{t}_shares"] = float(shares.get(t, 0))
        row[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    quarterly_rows.append(row)


# ------------------------------------------------------------
# MAIN BACKTEST ENGINE (Corrected)
# ------------------------------------------------------------
def run_backtest(price_data, config, dd_fn, regime_detector, rebalance_fn):
    log("ENTERED run_backtest()")

    tickers = config["tickers"]
    starting_val = config["starting_balance"]
    drawdown_ticker = config["drawdown_ticker"]

    # -------------------------
    # Load prices & setup
    # -------------------------
    price_df = price_data[tickers].copy()
    start_date = price_df.index[price_df.notna().any(axis=1)][0]

    norm_df = calculate_normalized_values(price_df, tickers, starting_val, start_date)

    dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
    dd_raw = dd_raw.fillna(0).clip(0, 1)

    rebalance_freq = config.get("rebalance_frequency", "monthly")
    rebalance_dates = get_rebalance_dates(price_df.index, rebalance_freq)
    instant_mode = rebalance_freq == "instant"

    # -------------------------
    # INITIAL ALLOCATION
    # -------------------------
    first_dd = dd_raw.loc[start_date]
    market_regime, portfolio_regime, shares = get_initial_allocation(
        start_date, price_df, first_dd, config, regime_detector, rebalance_fn
    )

    portfolio_value = starting_val
    prev_value = starting_val
    portfolio_ath = starting_val

    # -------------------------
    # STORAGE
    # -------------------------
    equity_rows = []
    rebalance_rows = []

    last_rebalance_value = portfolio_value
    quarter_returns = []   # reused as volatility measure for rebalance summary

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for date in price_df.index:

        row_prices = price_df.loc[date]
        row_norm = norm_df.loc[date]
        raw_dd = dd_raw.loc[date]
        raw_ath = ath_raw.loc[date]

        # Build DD columns
        dd_cols = {
            f"{drawdown_ticker}_ATH_raw": raw_ath,
            f"{drawdown_ticker}_DD_raw": raw_dd,
            "Portfolio_ATH": None,
            "Portfolio_DD": None
        }

        # Missing prices → record structure but skip logic
        if row_prices.isna().any():
            record_missing_row(
                equity_rows, date, tickers, row_prices, row_norm,
                shares, portfolio_value, portfolio_regime,
                starting_val, dd_cols
            )
            continue

        # -------------------------
        # Regime detection
        # -------------------------
        market_regime = regime_detector(raw_dd, config)
        new_regime = portfolio_regime
        do_rebalance = False
        rebalanced_flag = ""

        # -------------------------
        # INSTANT MODE (rebalance whenever regime changes)
        # -------------------------
        if instant_mode:
            new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
            do_rebalance = new_regime != portfolio_regime

        # -------------------------
        # PERIODIC MODE
        # -------------------------
        else:
            if date in rebalance_dates:
                new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
                do_rebalance = True

        # -------------------------
        # EXECUTE REBALANCE IF TRIGGERED
        # -------------------------
        if do_rebalance:

            portfolio_regime = new_regime
            alloc = get_allocation_for_regime(new_regime, config)
            shares = rebalance_fn(portfolio_value, alloc, row_prices)
            rebalanced_flag = "Rebalanced"

            # update value immediately post-rebalance
            portfolio_value, prev_value = update_portfolio_value(
                shares, row_prices, prev_value, quarter_returns
            )

            portfolio_ath = max(portfolio_ath, portfolio_value)
            portfolio_dd = (portfolio_value / portfolio_ath) - 1

            # ADD REBALANCE LOG ENTRY
            rebalance_rows.append({
                "Date": date,
                "Portfolio_Value": portfolio_value,
                "Market_Regime": market_regime,
                "Portfolio_Regime": portfolio_regime,
                "QoQ_Return": (portfolio_value / last_rebalance_value) - 1,
                "QoQ_Volatility": pd.Series(quarter_returns).std() if len(quarter_returns) else 0,
                f"{drawdown_ticker}_ATH_raw": raw_ath,
                f"{drawdown_ticker}_DD_raw": raw_dd,
                "Portfolio_ATH": portfolio_ath,
                "Portfolio_DD": portfolio_dd,
                **{f"{t}_shares": float(shares.get(t, 0)) for t in tickers},
                **{f"{t}_value": float(shares.get(t, 0) * row_prices[t]) for t in tickers}
            })

            last_rebalance_value = portfolio_value
            quarter_returns = []

        # -------------------------
        # UPDATE VALUE FOR REGULAR DAYS
        # -------------------------
        else:
            portfolio_value, prev_value = update_portfolio_value(
                shares, row_prices, prev_value, quarter_returns
            )

            portfolio_ath = max(portfolio_ath, portfolio_value)
            portfolio_dd = (portfolio_value / portfolio_ath) - 1

        # update DD columns
        dd_cols["Portfolio_ATH"] = portfolio_ath
        dd_cols["Portfolio_DD"] = portfolio_dd

        # -------------------------
        # RECORD DAILY ROW
        # -------------------------
        record_daily_row(
            equity_rows, date, tickers, row_prices, row_norm, shares,
            portfolio_value, market_regime, portfolio_regime,
            starting_val, rebalanced_flag, dd_cols
        )

    # -------------------------
    # BUILD DATAFRAMES
    # -------------------------
    equity_df = pd.DataFrame(equity_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)

    log("LEAVING run_backtest(), returning DataFrames")
    return equity_df, rebalance_df
