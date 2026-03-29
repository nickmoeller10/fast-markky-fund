# data_loader.py
from utils import log
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd


def normalize_close_columns(closes):
    """
    yfinance often returns a Close block with MultiIndex columns ('Close', 'QQQ').
    Flatten to ticker-only column names for downstream code.
    """
    if closes is None:
        return closes
    if isinstance(closes, pd.Series):
        return closes
    if not isinstance(closes, pd.DataFrame):
        return closes
    out = closes.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    return out


def yf_close_to_series(close_block, ticker_hint=None):
    """
    Build a single 1D price Series from yfinance 'Close' output (Series or DataFrame).
    Required for correct rolling / cummax drawdown math.
    """
    if close_block is None:
        return pd.Series(dtype=float)
    if isinstance(close_block, pd.Series):
        return close_block.dropna().astype(float).sort_index()
    if not isinstance(close_block, pd.DataFrame):
        return pd.Series(dtype=float)
    df = normalize_close_columns(close_block).dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float)
    if ticker_hint is not None and ticker_hint in df.columns:
        s = df[ticker_hint]
    elif df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        s = df.iloc[:, 0]
    return s.astype(float).dropna().sort_index()


def load_price_data(tickers, start_date, end_date=None, include_dividends=False):
    """
    Load price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string or None for current date
        include_dividends: If True, also download and return dividend data
    
    Returns:
        If include_dividends=False: DataFrame with Close prices
        If include_dividends=True: Tuple of (price_df, dividend_df)
    """
    if end_date is None:
        log(f"Downloading price data for: {tickers}, starting {start_date} (end: current date)")
        data = yf.download(tickers, start=start_date, auto_adjust=True)
    else:
        log(f"Downloading price data for: {tickers}, from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    log(f"Download complete. Shape: {data.shape}")
    closes = data["Close"].dropna(how="all")
    closes = normalize_close_columns(closes)
    log(f"Cleaned price data shape: {closes.shape}")

    if not include_dividends:
        return closes
    
    # Download dividend data
    log("Downloading dividend data...")
    dividend_data = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=start_date, end=end_date if end_date else None)
            if not hist.empty and "Dividends" in hist.columns:
                dividends = hist["Dividends"]
                dividend_data[ticker] = dividends
                log(f"  {ticker}: {len(dividends[dividends > 0])} dividend payments")
            else:
                dividend_data[ticker] = pd.Series(dtype=float)
                log(f"  {ticker}: No dividend data available")
        except Exception as e:
            log(f"  Warning: Could not download dividends for {ticker}: {e}")
            dividend_data[ticker] = pd.Series(dtype=float)
    
    # Create dividend DataFrame aligned with price data
    dividend_df = pd.DataFrame(index=closes.index)
    for ticker in tickers:
        if ticker in dividend_data:
            # Get the raw dividend series
            raw_dividends = dividend_data[ticker]
            
            # Align dividends with price index
            # Use forward fill to preserve dividend values on the next trading day if dividend date doesn't match
            # First, create a series with dividend dates, then align to price index
            dividend_series = pd.Series(index=closes.index, dtype=float)
            dividend_series[:] = 0.0  # Initialize with zeros
            
            # For each dividend payment date, find the corresponding or next trading day
            aligned_count = 0
            # Normalize price index to naive datetime for comparison
            price_index = closes.index
            if hasattr(price_index, 'tz') and price_index.tz is not None:
                price_index_naive = price_index.tz_localize(None)
            else:
                price_index_naive = price_index
            
            for div_date, div_amount in raw_dividends[raw_dividends > 0].items():
                # Normalize dividend date to naive datetime for comparison
                if hasattr(div_date, 'tz') and div_date.tz is not None:
                    div_date_naive = div_date.tz_localize(None)
                else:
                    div_date_naive = pd.Timestamp(div_date).normalize()
                
                # Normalize to date only (remove time component) for comparison
                div_date_normalized = pd.Timestamp(div_date_naive).normalize()
                
                # Find the date in the price index (exact match or next trading day)
                # Compare normalized dates
                matching_dates = price_index_naive[price_index_naive.normalize() == div_date_normalized]
                if len(matching_dates) > 0:
                    # Use the original index value for assignment
                    dividend_series.loc[matching_dates[0]] = div_amount
                    aligned_count += 1
                else:
                    # Find the next trading day after the dividend date
                    next_trading_days = price_index_naive[price_index_naive.normalize() >= div_date_normalized]
                    if len(next_trading_days) > 0:
                        # Use the original index value for assignment
                        dividend_series.loc[next_trading_days[0]] = div_amount
                        aligned_count += 1
            
            if aligned_count > 0:
                log(f"  {ticker}: Aligned {aligned_count} dividend payments to trading days")
            
            dividend_df[ticker] = dividend_series
        else:
            dividend_df[ticker] = 0.0
    
    log(f"Dividend data shape: {dividend_df.shape}")
    return closes, dividend_df


# Yahoo symbol for the CBOE Volatility Index (index level, not a tradable ETF)
VIX_YAHOO_SYMBOL = "^VIX"


def load_vix_series(start_date, end_date=None):
    """
    Daily VIX index close (context / display only — not used for allocation).

    Returns a float Series indexed by date (timezone-naive), or empty Series on failure.
    """
    if end_date is None:
        log(f"Downloading {VIX_YAHOO_SYMBOL} from {start_date} (end: current date)")
        data = yf.download(VIX_YAHOO_SYMBOL, start=start_date, auto_adjust=True)
    else:
        log(f"Downloading {VIX_YAHOO_SYMBOL} from {start_date} to {end_date}")
        data = yf.download(VIX_YAHOO_SYMBOL, start=start_date, end=end_date, auto_adjust=True)

    if data is None or data.empty:
        log(f"Warning: no VIX data returned for {VIX_YAHOO_SYMBOL}")
        return pd.Series(dtype=float)

    closes = data["Close"].dropna(how="all")
    closes = normalize_close_columns(closes)
    s = yf_close_to_series(closes, ticker_hint=VIX_YAHOO_SYMBOL)
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    return s.sort_index()


def attach_vix_to_equity_df(equity_df, vix_series):
    """
    Align VIX closes to equity_df rows by calendar Date (left join on backtest dates).
    """
    if equity_df is None or equity_df.empty:
        return equity_df
    out = equity_df.copy()
    if vix_series is None or len(vix_series) == 0:
        out["VIX"] = np.nan
        return out

    vx = vix_series.copy()
    vx.index = pd.to_datetime(vx.index).normalize()
    # One value per calendar day (last observation if duplicates)
    by_day = vx.groupby(vx.index).last()

    d = pd.to_datetime(out["Date"])
    if getattr(d.dtype, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    d = d.dt.normalize()
    out["VIX"] = d.map(by_day).astype(float)
    return out
