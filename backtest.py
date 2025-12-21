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
    """
    Calculate drawdown from all-time high.
    Uses cummax() which calculates ATH from the beginning of the series.
    """
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


def compute_drawdown_from_historical_ath(series, historical_ath_value=None):
    """
    Calculate drawdown using a historical ATH value.
    If historical_ath_value is provided, use it; otherwise use cummax().
    
    Args:
        series: Price series for the portfolio period
        historical_ath_value: Optional ATH value from full historical data
    
    Returns:
        Tuple of (drawdown_series, ath_series)
    """
    if historical_ath_value is not None:
        # Use the historical ATH value
        # ATH series is the max of historical ATH and current series cummax
        ath = series.cummax().clip(lower=historical_ath_value)
        # But we want to use the historical ATH as the baseline
        # So if current price is below historical ATH, use historical ATH
        ath = pd.Series(index=series.index, data=historical_ath_value)
        # Update ATH if current series exceeds historical ATH
        current_max = series.cummax()
        ath = pd.concat([ath, current_max], axis=1).max(axis=1)
    else:
        # Fallback to standard calculation
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

    # Handle None shares (safety check)
    if shares is None:
        shares = {}

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
        # Handle both dict and Series/DataFrame access
        if isinstance(row_prices, dict):
            price_val = row_prices.get(t, 0)
        else:
            price_val = row_prices.get(t, 0) if hasattr(row_prices, 'get') else (row_prices[t] if t in row_prices.index else 0)
        
        if isinstance(row_norm, dict):
            norm_val = row_norm.get(f"{t}_norm", 0)
        else:
            norm_val = row_norm.get(f"{t}_norm", 0) if hasattr(row_norm, 'get') else (row_norm[f"{t}_norm"] if f"{t}_norm" in row_norm.index else 0)
        
        shares_val = shares.get(t, 0) if isinstance(shares, dict) else 0
        
        rec[f"{t}_price"] = price_val
        rec[f"{t}_norm"] = norm_val
        rec[f"{t}_shares"] = shares_val
        rec[f"{t}_value"] = shares_val * (price_val if price_val else 0)

    equity_rows.append(rec)


def update_portfolio_value(shares, row_prices, prev_value, quarter_returns):
    # Handle None shares
    if shares is None:
        shares = {}
    
    # Calculate portfolio value
    new_value = 0.0
    for t in shares:
        if t in row_prices:
            price = row_prices[t] if isinstance(row_prices, dict) else (row_prices.loc[t] if t in row_prices.index else 0)
            new_value += shares[t] * price
    
    daily_ret = (new_value / prev_value - 1) if prev_value != 0 else 0
    quarter_returns.append(daily_ret)
    return new_value, new_value


def record_daily_row(equity_rows, date, tickers, row_prices, row_norm,
                     shares, portfolio_value, market_regime,
                     portfolio_regime, starting_val, rebalanced_flag,
                     dd_cols, cash_balance=0.0):
    from utils import log
    
    # Handle None shares
    if shares is None:
        log(f"⚠️ WARNING: shares is None in record_daily_row on {date}. Initializing as empty dict.")
        shares = {}

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
        rec[f"{t}_shares"] = float(shares.get(t, 0) if isinstance(shares, dict) else 0)
        rec[f"{t}_value"] = float((shares.get(t, 0) if isinstance(shares, dict) else 0) * row_prices[t])

    equity_rows.append(rec)


def record_quarterly_row(quarterly_rows, date, tickers, row_prices, shares,
                         portfolio_value, market_regime, portfolio_regime,
                         last_rebalance_value, quarter_returns, dd_cols):
    from utils import log
    
    # Handle None shares
    if shares is None:
        log(f"⚠️ WARNING: shares is None in record_quarterly_row on {date}. Initializing as empty dict.")
        shares = {}

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

    # -------------------------
    # PRE-PORTFOLIO SIMULATION: Track regimes from historical ATH
    # -------------------------
    # Download full historical data for drawdown_ticker from its inception
    log(f"Downloading full historical data for {drawdown_ticker} from inception for pre-portfolio simulation...")
    import yfinance as yf
    
    # Download from ticker inception (use 1980 as safe early date - yfinance returns from actual inception)
    historical_series_full = None
    historical_dd_full = None
    historical_ath_full = None
    pre_portfolio_start_date = None
    
    try:
        historical_data = yf.download(
            drawdown_ticker, 
            start="1980-01-01",  # Very early date - yfinance will return from ticker's actual inception
            end=None,  # Get all available data up to current
            auto_adjust=True,
            progress=False
        )
        
        if not historical_data.empty and "Close" in historical_data.columns:
            historical_series_full = historical_data["Close"].dropna()
            
            if len(historical_series_full) > 0:
                log(f"Downloaded {len(historical_series_full)} days of historical data for {drawdown_ticker}")
                log(f"Historical data range: {historical_series_full.index.min()} to {historical_series_full.index.max()}")
                
                # Calculate ATH and drawdown from full historical series (from ticker inception)
                historical_ath_full = historical_series_full.cummax()
                historical_dd_full = (historical_ath_full - historical_series_full) / historical_ath_full
                
                # Get the earliest date (ticker inception)
                pre_portfolio_start_date = historical_series_full.index.min()
                log(f"Pre-portfolio simulation will start from {pre_portfolio_start_date} (ticker inception)")
                
                # Get price and drawdown at portfolio start date
                portfolio_start_ts = pd.to_datetime(start_date)
                if portfolio_start_ts in historical_series_full.index:
                    start_price = float(historical_series_full.loc[portfolio_start_ts])
                    start_ath = float(historical_ath_full.loc[portfolio_start_ts])
                    start_dd = float(historical_dd_full.loc[portfolio_start_ts])
                    log(f"At portfolio start ({start_date}): Price=${start_price:.2f}, ATH=${start_ath:.2f}, DD={start_dd:.2%}")
                else:
                    # Find closest date before portfolio start
                    dates_before = historical_series_full.index[historical_series_full.index < portfolio_start_ts]
                    if len(dates_before) > 0:
                        closest_date = dates_before[-1]
                        start_price = float(historical_series_full.loc[closest_date])
                        start_ath = float(historical_ath_full.loc[closest_date])
                        start_dd = float(historical_dd_full.loc[closest_date])
                        log(f"At portfolio start ({start_date}): Using closest historical date {closest_date}, Price=${start_price:.2f}, ATH=${start_ath:.2f}, DD={start_dd:.2%}")
    except Exception as e:
        log(f"Warning: Error downloading historical data for {drawdown_ticker}: {e}")
    
    # -------------------------
    # PRE-PORTFOLIO SIMULATION: Track regimes while holding cash
    # -------------------------
    portfolio_regime_at_start = None
    market_regime_at_start = None
    
    if historical_dd_full is not None and pre_portfolio_start_date is not None:
        log(f"Running pre-portfolio simulation from {pre_portfolio_start_date} to {start_date} (holding cash, tracking regimes)...")
        
        portfolio_start_ts = pd.to_datetime(start_date)
        # Get all dates from historical data up to (and including if available) portfolio start
        pre_portfolio_dates = historical_dd_full.index[
            (historical_dd_full.index >= pre_portfolio_start_date) & 
            (historical_dd_full.index <= portfolio_start_ts)
        ]
        
        # If portfolio start date is not in historical index, get dates up to but not including it
        if len(pre_portfolio_dates) == 0 or (portfolio_start_ts not in pre_portfolio_dates and len(pre_portfolio_dates) > 0):
            pre_portfolio_dates = historical_dd_full.index[
                (historical_dd_full.index >= pre_portfolio_start_date) & 
                (historical_dd_full.index < portfolio_start_ts)
            ]
        
        # Track regime through pre-portfolio period
        current_portfolio_regime = None
        rebalance_strategy = config.get("rebalance_strategy", "down_only")
        
        for pre_date in pre_portfolio_dates:
            pre_dd = historical_dd_full.loc[pre_date]
            market_regime = regime_detector(pre_dd, config)
            
            if market_regime is not None:
                # Apply rebalancing strategy to determine portfolio regime
                if current_portfolio_regime is None:
                    current_portfolio_regime = market_regime
                else:
                    # Apply rebalancing strategy
                    if rebalance_strategy == "down_only":
                        current_portfolio_regime = apply_asymmetric_rules_down_only(current_portfolio_regime, market_regime)
                    elif rebalance_strategy == "up_only":
                        current_portfolio_regime = apply_asymmetric_rules_up_only(current_portfolio_regime, market_regime)
                    elif rebalance_strategy == "always":
                        current_portfolio_regime = apply_always_rebalance(current_portfolio_regime, market_regime)
                    else:
                        current_portfolio_regime = market_regime
        
        # Get final regime at portfolio start
        # Try to get drawdown at portfolio start date, or use the last pre-portfolio date
        if portfolio_start_ts in historical_dd_full.index:
            portfolio_start_dd = historical_dd_full.loc[portfolio_start_ts]
            market_regime_at_start = regime_detector(portfolio_start_dd, config)
        elif len(pre_portfolio_dates) > 0:
            # Use the last available date before portfolio start
            final_pre_date = pre_portfolio_dates[-1]
            portfolio_start_dd = historical_dd_full.loc[final_pre_date]
            market_regime_at_start = regime_detector(portfolio_start_dd, config)
        else:
            # Fallback: calculate drawdown from price_data for start date
            if start_date in price_data.index and drawdown_ticker in price_data.columns:
                start_price = price_data[drawdown_ticker].loc[start_date]
                if pd.notna(start_price) and historical_ath_full is not None:
                    # Use historical ATH to calculate drawdown
                    portfolio_start_ts = pd.to_datetime(start_date)
                    historical_ath_before = historical_ath_full[historical_ath_full.index < portfolio_start_ts]
                    if len(historical_ath_before) > 0:
                        max_ath = float(historical_ath_before.max())
                        first_dd = (max_ath - float(start_price)) / max_ath if max_ath > 0 else 0.0
                    else:
                        first_dd = 0.0
                else:
                    first_dd = 0.0
            else:
                first_dd = 0.0
            market_regime_at_start = regime_detector(first_dd, config)
        
        # Determine portfolio regime at start
        if current_portfolio_regime is None:
            portfolio_regime_at_start = market_regime_at_start
        else:
            # Apply rebalancing strategy one more time for the start date
            if rebalance_strategy == "down_only":
                portfolio_regime_at_start = apply_asymmetric_rules_down_only(current_portfolio_regime, market_regime_at_start)
            elif rebalance_strategy == "up_only":
                portfolio_regime_at_start = apply_asymmetric_rules_up_only(current_portfolio_regime, market_regime_at_start)
            elif rebalance_strategy == "always":
                portfolio_regime_at_start = apply_always_rebalance(current_portfolio_regime, market_regime_at_start)
            else:
                portfolio_regime_at_start = market_regime_at_start
        
        log(f"Pre-portfolio simulation complete. Regime at portfolio start: Market={market_regime_at_start}, Portfolio={portfolio_regime_at_start}")
    
    # -------------------------
    # Calculate drawdown for portfolio period using historical ATH
    # -------------------------
    if historical_ath_full is not None and historical_series_full is not None:
        # Get historical ATH value before portfolio start
        portfolio_start_ts = pd.to_datetime(start_date)
        historical_ath_before_start = historical_ath_full[historical_ath_full.index < portfolio_start_ts]
        
        if len(historical_ath_before_start) > 0:
            max_historical_ath = float(historical_ath_before_start.max())
            log(f"Historical ATH for {drawdown_ticker} before portfolio start: ${max_historical_ath:.2f}")
            
            # Calculate drawdown for portfolio period
            portfolio_prices = price_data[drawdown_ticker].reindex(price_data.index)
            
            if len(portfolio_prices.dropna()) > 0:
                # Calculate ATH for portfolio period
                ath_raw = pd.Series(index=portfolio_prices.index, dtype=float)
                running_max = max_historical_ath
                
                for date in portfolio_prices.index:
                    current_price = portfolio_prices.loc[date]
                    if pd.notna(current_price):
                        running_max = max(running_max, current_price)
                        ath_raw.loc[date] = running_max
                    else:
                        ath_raw.loc[date] = running_max
                
                # Calculate drawdown
                dd_raw = (ath_raw - portfolio_prices) / ath_raw
            else:
                log(f"Warning: No portfolio period data for {drawdown_ticker}, using standard calculation")
                dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
        else:
            log(f"Warning: No historical ATH before {start_date}, using standard calculation")
            dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
    else:
        # Fallback to standard calculation
        log(f"Warning: Using standard ATH calculation (no historical data available)")
        dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
    
    # Ensure dd_raw and ath_raw are aligned with price_data index
    dd_raw = dd_raw.reindex(price_data.index).fillna(0).clip(0, 1)
    ath_raw = ath_raw.reindex(price_data.index)

    rebalance_freq = config.get("rebalance_frequency", "monthly")
    rebalance_dates = get_rebalance_dates(price_df.index, rebalance_freq)
    instant_mode = rebalance_freq == "instant"

    # -------------------------
    # INITIAL ALLOCATION
    # -------------------------
    # Use the regime determined from pre-portfolio simulation, or calculate if not available
    if portfolio_regime_at_start is not None and market_regime_at_start is not None:
        log(f"Using pre-portfolio simulation regime: Market={market_regime_at_start}, Portfolio={portfolio_regime_at_start}")
        market_regime = market_regime_at_start
        portfolio_regime = portfolio_regime_at_start
        # Get allocation for the determined regime and rebalance
        alloc = get_allocation_for_regime(portfolio_regime, config)
        shares = rebalance_fn(config["starting_balance"], alloc, price_df.loc[start_date])
        log(f"Initial rebalance on {start_date} into {portfolio_regime} with allocation: {alloc}")
    else:
        # Fallback: calculate regime from first day's drawdown
        first_dd = dd_raw.loc[start_date]
        market_regime, portfolio_regime, shares = get_initial_allocation(
            start_date, price_df, first_dd, config, regime_detector, rebalance_fn
        )
        log(f"Using standard initial allocation (no pre-portfolio simulation): Market={market_regime}, Portfolio={portfolio_regime}")
    
    # Ensure shares is initialized (should never be None, but handle edge cases)
    if shares is None:
        log(f"⚠️ ERROR: get_initial_allocation returned None for shares on {start_date}. Initializing as empty dict.")
        shares = {}
    elif not isinstance(shares, dict):
        log(f"⚠️ WARNING: shares is not a dict on {start_date} (type: {type(shares)}). Converting to dict.")
        shares = {} if shares is None else dict(shares) if hasattr(shares, 'items') else {}

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
            # Ensure shares is a dict
            if shares is None:
                log(f"⚠️ WARNING: shares is None during dividend handling on {date}. Initializing as empty dict.")
                shares = {}
            
            daily_dividends = 0.0
            for ticker in allocation_tickers:
                # Check if ticker exists in dividend_data columns and we have shares
                if ticker in dividend_data.columns and isinstance(shares, dict) and ticker in shares and shares.get(ticker, 0) > 0:
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
