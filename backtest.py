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
# Rebalancing Strategy Functions
# ------------------------------------------------------------
def apply_asymmetric_rules_down_only(prev_regime, market_regime):
    """
    Regime Shift Down Only: Rebalance immediately when market goes DOWN,
    but only move back UP when market fully recovers to R1.
    """
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


def apply_asymmetric_rules_up_only(prev_regime, market_regime):
    """
    Regime Shift Up Only: Rebalance immediately when market goes UP,
    but hold position when market goes DOWN (except when reaching bottom).
    When in bottom regime (R3), rebalance on every way up.
    """
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    # If market goes UP → follow it immediately
    if mkt_n < prev_n:
        return regime_label(mkt_n)

    # If currently in bottom regime (R3) and market goes up, follow it
    if prev_n == 3 and mkt_n < 3:
        return regime_label(mkt_n)

    # If market goes to bottom regime (R3), follow it
    if mkt_n == 3:
        return "R3"

    # Otherwise hold previous regime (don't follow down)
    return prev_regime


def apply_always_rebalance(prev_regime, market_regime):
    """
    Always: Rebalance whenever market regime changes, regardless of direction.
    """
    return market_regime


def apply_rebalancing_strategy(prev_regime, market_regime, strategy):
    """
    Apply the specified rebalancing strategy.
    
    Args:
        prev_regime: Current portfolio regime
        market_regime: Current market regime
        strategy: "down_only", "up_only", or "always"
    
    Returns:
        New portfolio regime based on strategy
    """
    if strategy == "down_only":
        return apply_asymmetric_rules_down_only(prev_regime, market_regime)
    elif strategy == "up_only":
        return apply_asymmetric_rules_up_only(prev_regime, market_regime)
    elif strategy == "always":
        return apply_always_rebalance(prev_regime, market_regime)
    else:
        # Default to down_only for backward compatibility
        return apply_asymmetric_rules_down_only(prev_regime, market_regime)


# Backward compatibility alias
apply_asymmetric_rules = apply_asymmetric_rules_down_only


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
                     dd_cols, cash_balance=0.0):

    # Include cash in total value
    total_value = portfolio_value + cash_balance
    
    rec = {
        "Date": date,
        "Value": total_value,
        "Cash": cash_balance,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": total_value / starting_val - 1,
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
def run_backtest(price_data, config, dd_fn, regime_detector, rebalance_fn, dividend_data=None):
    log("ENTERED run_backtest()")

    tickers = config["tickers"]
    starting_val = config["starting_balance"]
    drawdown_ticker = config["drawdown_ticker"]
    
    # Dividend reinvestment settings
    dividend_reinvestment = config.get("dividend_reinvestment", False)
    dividend_target = config.get("dividend_reinvestment_target", "cash")
    allocation_tickers = config.get("allocation_tickers", tickers)

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
    cash_balance = 0.0  # Track cash from dividends if not reinvested immediately

    # -------------------------
    # STORAGE
    # -------------------------
    equity_rows = []
    rebalance_rows = []
    dividend_rows = []  # Track dividend events

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
        # Dividend handling
        # -------------------------
        if dividend_reinvestment and dividend_data is not None and not dividend_data.empty:
            daily_dividends = 0.0
            for ticker in allocation_tickers:
                # Check if ticker exists in dividend_data columns and we have shares
                if ticker in dividend_data.columns and ticker in shares and shares.get(ticker, 0) > 0:
                    # Get dividend for this date
                    if date in dividend_data.index:
                        dividend_per_share = dividend_data.loc[date, ticker]
                        if pd.notna(dividend_per_share) and dividend_per_share > 0:
                            num_shares = shares.get(ticker, 0)
                            dividend_amount = dividend_per_share * num_shares
                            daily_dividends += dividend_amount
                            
                            # Calculate yield (dividend per share / price per share)
                            ticker_price = row_prices.get(ticker, 0)
                            dividend_yield = (dividend_per_share / ticker_price * 100) if ticker_price > 0 else 0
                            
                            # Calculate percentage of portfolio value (before dividend is applied)
                            total_portfolio_value_before = portfolio_value + cash_balance
                            dividend_pct = (dividend_amount / total_portfolio_value_before * 100) if total_portfolio_value_before > 0 else 0
                            
                            # Record dividend event
                            dividend_rows.append({
                                "Date": date,
                                "Ticker": ticker,
                                "Dividend_Per_Share": dividend_per_share,
                                "Shares": num_shares,
                                "Dividend_Amount": dividend_amount,
                                "Dividend_Yield": dividend_yield,
                                "Portfolio_Pct": dividend_pct,
                                "Reinvestment_Target": dividend_target,
                                "Portfolio_Value": total_portfolio_value_before
                            })
                            
                            if dividend_target == "cash":
                                # Add to cash balance
                                cash_balance += dividend_amount
                                # Cash is included in total value calculation below
                            elif dividend_target in allocation_tickers and dividend_target in row_prices:
                                # Reinvest immediately into target ticker
                                target_price = row_prices[dividend_target]
                                if target_price > 0 and pd.notna(target_price):
                                    additional_shares = dividend_amount / target_price
                                    shares[dividend_target] = shares.get(dividend_target, 0) + additional_shares
                                    # Portfolio value will be recalculated below via update_portfolio_value()
                                    # which includes the new shares
                            
                            # Log dividend for debugging
                            log(f"Dividend: {ticker} paid ${dividend_amount:.2f} ({dividend_per_share:.4f} per share, {num_shares:.4f} shares)")
            
            # After processing all dividends, update portfolio value if dividends were reinvested into shares
            # (cash dividends are already in cash_balance and will be included in total value)
            if daily_dividends > 0 and dividend_target != "cash":
                # Recalculate portfolio value to include new shares from dividend reinvestment
                portfolio_value = sum(shares[t] * row_prices.get(t, 0) for t in shares if t in row_prices)
                log(f"Portfolio value updated after dividend reinvestment: ${portfolio_value:,.2f}")

        # -------------------------
        # Regime detection
        # -------------------------
        market_regime = regime_detector(raw_dd, config)
        new_regime = portfolio_regime
        do_rebalance = False
        rebalanced_flag = ""

        # -------------------------
        # Get rebalancing strategy from config
        # -------------------------
        rebalance_strategy = config.get("rebalance_strategy", "down_only")
        
        # -------------------------
        # INSTANT MODE (rebalance whenever regime changes)
        # -------------------------
        if instant_mode:
            new_regime = apply_rebalancing_strategy(portfolio_regime, market_regime, rebalance_strategy)
            do_rebalance = new_regime != portfolio_regime

        # -------------------------
        # PERIODIC MODE
        # -------------------------
        else:
            if date in rebalance_dates:
                new_regime = apply_rebalancing_strategy(portfolio_regime, market_regime, rebalance_strategy)
                do_rebalance = True

        # -------------------------
        # EXECUTE REBALANCE IF TRIGGERED
        # -------------------------
        if do_rebalance:
            # Include cash balance in portfolio value for rebalancing
            portfolio_value_with_cash = portfolio_value + cash_balance
            
            portfolio_regime = new_regime
            alloc = get_allocation_for_regime(new_regime, config)
            shares = rebalance_fn(portfolio_value_with_cash, alloc, row_prices)
            
            # Reset cash balance after rebalancing (it's been allocated)
            cash_balance = 0.0
            portfolio_value = portfolio_value_with_cash
            rebalanced_flag = "Rebalanced"

            # update value immediately post-rebalance
            portfolio_value, prev_value = update_portfolio_value(
                shares, row_prices, prev_value, quarter_returns
            )
            
            # Cash has been allocated, so total value is just portfolio_value now
            # (cash_balance was reset to 0.0 above)
            total_value = portfolio_value
            portfolio_ath = max(portfolio_ath, total_value)
            portfolio_dd = (total_value / portfolio_ath) - 1 if portfolio_ath > 0 else 0

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
            # Update portfolio value (this will include any new shares from dividend reinvestment)
            portfolio_value, prev_value = update_portfolio_value(
                shares, row_prices, prev_value, quarter_returns
            )
            # Include cash in total value for ATH tracking (cash includes dividends held as cash)
            total_value = portfolio_value + cash_balance
            portfolio_ath = max(portfolio_ath, total_value)
            portfolio_dd = (total_value / portfolio_ath) - 1 if portfolio_ath > 0 else 0

        # update DD columns
        dd_cols["Portfolio_ATH"] = portfolio_ath
        dd_cols["Portfolio_DD"] = portfolio_dd

        # -------------------------
        # RECORD DAILY ROW
        # -------------------------
        record_daily_row(
            equity_rows, date, tickers, row_prices, row_norm, shares,
            portfolio_value, market_regime, portfolio_regime,
            starting_val, rebalanced_flag, dd_cols, cash_balance
        )

    # -------------------------
    # BUILD DATAFRAMES
    # -------------------------
    equity_df = pd.DataFrame(equity_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)
    dividend_df = pd.DataFrame(dividend_rows) if dividend_rows else pd.DataFrame()

    log("LEAVING run_backtest(), returning DataFrames")
    return equity_df, rebalance_df, dividend_df
