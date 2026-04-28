# worst_case_simulator.py
# =====================================================================
# FAST MARKKY FUND — Worst-Case Simulation using NASDAQ Composite
# =====================================================================
# Simulates QQQ and TQQQ back to February 1985 using ^IXIC (NASDAQ Composite)
# as the base index. QQQ tracks NASDAQ-100, which is highly correlated with
# NASDAQ Composite. TQQQ is simulated as 3x leveraged QQQ.
#
# Logic:
#   • Download ^IXIC from 1985-02-01 onwards
#   • Simulate QQQ returns based on ^IXIC returns (with correlation adjustment)
#   • Simulate TQQQ as 3x QQQ returns
#   • Splice real data when available (QQQ from 1999, TQQQ from 2010)
#   • Handle other funds (XLU, etc.) that don't date back to 1985
# =====================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from utils import log
from data_loader import (
    SYNTHETIC_TICKERS,
    _attach_synthetic_columns,
    _split_real_and_synthetic,
)
from worst_case_synthetic import (
    build_synth_qqq_tqqq,
    clamp_return,
    compute_qqq_ixic_beta,
    initial_qqq_price,
)


# Sentinel "always available" date for synthetic tickers — used so the
# simulator never sends "$"/"CASH" through yfinance. Older than ^IXIC's
# 1985-02-01 inception, so synthetic tickers don't gate the start date.
_SYNTHETIC_EARLIEST = pd.Timestamp("1980-01-01")


# =====================================================================
# Helper → Get earliest available date for a ticker
# =====================================================================
def get_earliest_date(ticker, start_date="1980-01-01"):
    """Get the earliest available date for a ticker.

    Synthetic tickers (CASH/$) never hit yfinance — they return a sentinel
    pre-1985 date so they never gate the simulation start.
    """
    if ticker in SYNTHETIC_TICKERS:
        return _SYNTHETIC_EARLIEST
    try:
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if not data.empty and "Close" in data.columns:
            closes = data["Close"].dropna()
            if len(closes) > 0:
                earliest = closes.index.min()
                # Return as Timestamp for consistent handling
                return pd.Timestamp(earliest)
    except Exception as e:
        log(f"  Warning: Could not get earliest date for {ticker}: {e}")
    return None


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def generate_worst_case_prices(config, requested_tickers, start_date=None, end_date=None):
    """
    Creates synthetic QQQ and TQQQ price history using ^IXIC (NASDAQ Composite).
    QQQ and TQQQ are simulated from 1985, but the returned data respects the requested date range.
    
    Args:
        config: Configuration dictionary
        requested_tickers: List of tickers requested by user
        start_date: Optional start date (string or datetime). If None, uses earliest available date for other tickers.
        end_date: Optional end date (string or datetime). If None, uses latest available date.
    
    Returns:
        Tuple of (price_df, earliest_dates_dict)
        - price_df: DataFrame with simulated/real prices for the requested date range
        - earliest_dates_dict: Dict mapping ticker -> earliest available date
    """
    
    # ------------------------------------------------------------
    # 0. Determine earliest date for OTHER tickers (excluding QQQ/TQQQ)
    # ------------------------------------------------------------
    log("[SIMULATOR] Determining earliest available dates for tickers...")
    earliest_dates = {}

    # QQQ and TQQQ are simulated from 1985, so exclude them from earliest date check.
    # Synthetic tickers (CASH/$) never gate start date — get_earliest_date short-circuits.
    other_tickers = [t for t in requested_tickers if t not in ["QQQ", "TQQQ"]]

    for ticker in other_tickers:
        earliest_date = get_earliest_date(ticker, start_date="1980-01-01")
        if earliest_date:
            earliest_dates[ticker] = earliest_date
            log(f"[SIMULATOR] {ticker} earliest date: {earliest_date.date()}")
        else:
            log(f"[SIMULATOR] Warning: Could not determine earliest date for {ticker}")
    
    # QQQ and TQQQ are always simulated from 1985 using NASDAQ Composite
    log(f"[SIMULATOR] QQQ: Simulated from 1985-02-01 using NASDAQ Composite (^IXIC)")
    log(f"[SIMULATOR] TQQQ: Simulated from 1985-02-01 as 3x leveraged QQQ")
    
    # Determine the actual simulation start date
    # If user provided start_date, use it (but ensure it's not before earliest dates for other tickers)
    # If no start_date provided, use the latest earliest date for other tickers
    if start_date:
        user_start = pd.to_datetime(start_date)
        log(f"[SIMULATOR] User requested start date: {user_start.date()}")
        
        # Check if user's start date is before earliest dates for other tickers
        if earliest_dates:
            earliest_other = max(earliest_dates.values())
            if user_start < earliest_other:
                log(f"[SIMULATOR] Warning: User start date ({user_start.date()}) is before earliest date for other tickers ({earliest_other.date()})")
                log(f"[SIMULATOR] Will use {earliest_other.date()} as the effective start date")
                simulation_start_date = earliest_other
            else:
                simulation_start_date = user_start
        else:
            # No other tickers, can use user's start date (but not before 1985)
            if user_start < pd.Timestamp("1985-02-01"):
                log(f"[SIMULATOR] User start date is before 1985, using 1985-02-01")
                simulation_start_date = pd.Timestamp("1985-02-01")
            else:
                simulation_start_date = user_start
    else:
        # No user start date, use earliest date for other tickers
        if earliest_dates:
            simulation_start_date = max(earliest_dates.values())
            log(f"[SIMULATOR] No user start date, using earliest date for other tickers: {simulation_start_date.date()}")
        else:
            # No other tickers, so we can start from 1985
            simulation_start_date = pd.Timestamp("1985-02-01")
            log(f"[SIMULATOR] No other tickers and no user start date, using 1985-02-01")
    
    # Determine end date
    if end_date:
        user_end = pd.to_datetime(end_date)
        log(f"[SIMULATOR] User requested end date: {user_end.date()}")
        simulation_end_date = user_end
    else:
        # Will use the latest available date from the data
        simulation_end_date = None
        log(f"[SIMULATOR] No user end date, will use latest available data")
    
    # Always use 1985-02-01 as the base for ^IXIC (NASDAQ Composite inception)
    ixic_base_start = "1985-02-01"
    
    log(f"[SIMULATOR] Using ^IXIC (NASDAQ Composite) as base index from {ixic_base_start}")
    
    # ------------------------------------------------------------
    # 1. Load ^IXIC (NASDAQ Composite) from 1985 (for simulation)
    # ------------------------------------------------------------
    log("[SIMULATOR] Downloading ^IXIC data...")
    ixic_raw = yf.download("^IXIC", start=ixic_base_start, auto_adjust=True, progress=False)
    
    if ixic_raw.empty:
        raise RuntimeError("[SIMULATOR] ERROR: Could not download ^IXIC (NASDAQ Composite).")
    
    # `.squeeze("columns")` collapses the single-ticker DataFrame to a Series so
    # downstream `float(ixic_close.iloc[0])` doesn't raise the pandas FutureWarning
    # about implicit float() on a 1-element Series.
    ixic_close = ixic_raw["Close"].squeeze("columns").dropna()
    ixic_ret = ixic_close.pct_change().dropna()
    
    log(f"[SIMULATOR] Loaded ^IXIC history: {len(ixic_close)} rows")
    log(f"[SIMULATOR] ^IXIC date range: {ixic_close.index.min()} to {ixic_close.index.max()}")
    
    # ------------------------------------------------------------
    # 2. Load real ETF data to get inception dates and correlation
    # ------------------------------------------------------------
    log("[SIMULATOR] Loading real ETF data for correlation and inception dates...")
    # Filter synthetic tickers (CASH/$) BEFORE the yfinance call — sending
    # them to yf.download returns the real "CASH" ticker (Pathward Financial
    # Inc.) instead of the synthetic MMF series.
    real_tickers = list(set(
        ["QQQ", "TQQQ"]
        + [t for t in requested_tickers if t not in ["QQQ", "TQQQ"] and t not in SYNTHETIC_TICKERS]
    ))
    
    # Download real data - handle single vs multiple tickers
    if len(real_tickers) == 1:
        # Single ticker returns different structure
        ticker_data = yf.download(real_tickers[0], start="1990-01-01", auto_adjust=True, progress=False)
        if ticker_data.empty or "Close" not in ticker_data.columns:
            raise RuntimeError(f"[SIMULATOR] ERROR: No real data found for {real_tickers[0]}.")
        real_close = pd.DataFrame({real_tickers[0]: ticker_data["Close"]})
    else:
        real_data = yf.download(real_tickers, start="1990-01-01", auto_adjust=True, progress=False)
        
        if real_data.empty:
            raise RuntimeError("[SIMULATOR] ERROR: No real ETF data found.")
        
        # Handle MultiIndex columns (when downloading multiple tickers)
        if isinstance(real_data.columns, pd.MultiIndex):
            if "Close" in real_data.columns.levels[0]:
                real_close = real_data["Close"]
            else:
                raise RuntimeError("[SIMULATOR] ERROR: 'Close' not found in downloaded data.")
        elif "Close" in real_data.columns:
            # Single level columns - might be single ticker or different structure
            real_close = real_data["Close"]
            # If it's a Series, convert to DataFrame
            if isinstance(real_close, pd.Series):
                real_close = pd.DataFrame({real_tickers[0]: real_close})
        else:
            raise RuntimeError("[SIMULATOR] ERROR: Could not extract Close prices from downloaded data.")
    
    # Ensure real_close is a DataFrame
    if isinstance(real_close, pd.Series):
        real_close = pd.DataFrame({real_close.name: real_close})
    
    real_close = real_close.dropna(how="all", axis=1)
    
    # Get inception dates
    def inception(symbol):
        if symbol in real_close.columns:
            s = real_close[symbol].dropna()
            return s.index.min() if len(s) > 0 else None
        return None
    
    qqq_start = inception("QQQ")
    tqqq_start = inception("TQQQ")
    
    log(f"[SIMULATOR] Real QQQ inception: {qqq_start}")
    log(f"[SIMULATOR] Real TQQQ inception: {tqqq_start}")
    
    # ------------------------------------------------------------
    # 3. Compute QQQ vs ^IXIC beta for realistic synthetic returns
    # ------------------------------------------------------------
    correlation, beta = compute_qqq_ixic_beta(ixic_close, real_close, qqq_start)

    # ------------------------------------------------------------
    # 4. Build synthetic QQQ and TQQQ price paths from ^IXIC
    # ------------------------------------------------------------
    initial_qqq = initial_qqq_price(ixic_close, real_close, qqq_start)
    synth_qqq, synth_tqqq = build_synth_qqq_tqqq(ixic_close, ixic_ret, beta, initial_qqq)
    full_index = ixic_close.index
    
    # ------------------------------------------------------------
    # 5. QQQ and TQQQ are entirely synthetic (no real data splicing)
    # ------------------------------------------------------------
    # QQQ is entirely based on NASDAQ Composite, not real QQQ data
    # TQQQ is 3x leveraged QQQ
    log(f"[SIMULATOR] QQQ: Using synthetic data from 1985-02-01 (based entirely on ^IXIC)")
    log(f"[SIMULATOR] TQQQ: Using synthetic data from 1985-02-01 (3x leveraged QQQ)")
    
    final_qqq = synth_qqq
    final_tqqq = synth_tqqq
    
    # ------------------------------------------------------------
    # 6. Build final DataFrame with all requested tickers
    # ------------------------------------------------------------
    price_df = pd.DataFrame(index=full_index)
    price_df["QQQ"] = final_qqq
    price_df["TQQQ"] = final_tqqq
    
    # Add other tickers (use real data only, no simulation).
    # Synthetic tickers (CASH/$) are appended in a single _attach_synthetic_columns
    # pass after this loop — never sent to yfinance.
    synthetic_requested = [t for t in requested_tickers if t in SYNTHETIC_TICKERS]
    for ticker in requested_tickers:
        if ticker in ["QQQ", "TQQQ"]:
            continue
        if ticker in SYNTHETIC_TICKERS:
            continue
        if ticker in real_close.columns:
            real_series = real_close[ticker].dropna()
            if len(real_series) > 0:
                # Align to full index, forward fill from first available date
                aligned_series = real_series.reindex(full_index, method='ffill')
                price_df[ticker] = aligned_series
                log(f"[SIMULATOR] {ticker}: Using real data from {real_series.index.min().date()}")
            else:
                log(f"[SIMULATOR] Warning: {ticker} has no data, skipping")
        else:
            # Try downloading this ticker separately
            try:
                ticker_data = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
                if not ticker_data.empty and "Close" in ticker_data.columns:
                    real_series = ticker_data["Close"].dropna()
                    if len(real_series) > 0:
                        aligned_series = real_series.reindex(full_index, method='ffill')
                        price_df[ticker] = aligned_series
                        log(f"[SIMULATOR] {ticker}: Downloaded and using real data from {real_series.index.min().date()}")
                    else:
                        log(f"[SIMULATOR] Warning: {ticker} has no data after download, skipping")
                else:
                    log(f"[SIMULATOR] Warning: {ticker} not found in real data, skipping")
            except Exception as e:
                log(f"[SIMULATOR] Warning: Could not download {ticker}: {e}, skipping")

    # Attach synthetic ticker columns (CASH/$) — generated locally, never fetched.
    if synthetic_requested:
        price_df = _attach_synthetic_columns(price_df, synthetic_requested)
        for ticker in synthetic_requested:
            earliest_dates[ticker] = _SYNTHETIC_EARLIEST
            log(f"[SIMULATOR] {ticker}: Synthetic series attached (zero drawdown, daily-compounded MMF)")
    
    # Add QQQ and TQQQ to earliest_dates (they're simulated from 1985)
    if "QQQ" in requested_tickers:
        earliest_dates["QQQ"] = pd.Timestamp("1985-02-01")
    if "TQQQ" in requested_tickers:
        earliest_dates["TQQQ"] = pd.Timestamp("1985-02-01")
    
    # Filter to the requested date range
    # QQQ and TQQQ are simulated from 1985, so they'll have data from simulation_start_date
    price_df = price_df.loc[simulation_start_date:]
    
    # Apply end date if specified
    if simulation_end_date:
        price_df = price_df.loc[:simulation_end_date]
    
    # Drop rows where all values are NaN
    price_df = price_df.dropna(how="all")
    
    log(f"[SIMULATOR] Final price DataFrame shape: {price_df.shape}")
    log(f"[SIMULATOR] Date range: {price_df.index.min().date()} to {price_df.index.max().date()}")
    log(f"[SIMULATOR] Using simulated QQQ/TQQQ data from 1985, filtered to requested date range")
    
    return price_df, earliest_dates
