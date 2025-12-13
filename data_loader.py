# data_loader.py
from utils import log
import yfinance as yf
from datetime import datetime
import pandas as pd

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
