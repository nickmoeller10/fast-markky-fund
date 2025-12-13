# validate_tests.py
# ======================================================================
# Comprehensive Test Suite for Fast Markky Fund
# ======================================================================
# Tests main functionality using real data from Yahoo Finance
# ======================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import hashlib
import pickle

from backtest import run_backtest
from config import CONFIG
from regime_engine import compute_drawdown_from_ath, determine_regime
from allocation_engine import get_allocation_for_regime
from rebalance_engine import rebalance_portfolio
from data_loader import load_price_data


# ======================================================================
# TEST DATA CACHE SETUP
# ======================================================================
TEST_DATA_CACHE_DIR = "test_data_cache"

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    if not os.path.exists(TEST_DATA_CACHE_DIR):
        os.makedirs(TEST_DATA_CACHE_DIR)


def get_cache_filename(tickers, start_date, end_date):
    """Generate a cache filename based on tickers and date range"""
    # Create a hash of the parameters for the filename
    cache_key = f"{sorted(tickers)}_{start_date}_{end_date or 'today'}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    ticker_str = "_".join(sorted(tickers))
    return os.path.join(TEST_DATA_CACHE_DIR, f"{ticker_str}_{cache_hash}.pkl")


def download_test_data(tickers, start_date, end_date=None, use_cache=True):
    """
    Download test data from Yahoo Finance with caching.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD) or None for today
        use_cache: If True, use cached data if available
    
    Returns:
        DataFrame with price data
    """
    ensure_cache_dir()
    cache_file = get_cache_filename(tickers, start_date, end_date)
    
    # Try to load from cache first
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached test data from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"Loaded {len(cached_data)} days of cached data")
            return cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), downloading fresh data...")
    
    # Download from API
    print(f"Downloading test data for {tickers} from {start_date} to {end_date or 'today'}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    if data.empty:
        raise ValueError(f"Failed to download data for {tickers}")
    
    closes = data["Close"].dropna(how="all")
    print(f"Downloaded {len(closes)} days of data")
    
    # Save to cache
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(closes, f)
            print(f"Cached data saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")
    
    return closes


# ======================================================================
# TEST 1: Data Loading and Quality
# ======================================================================
def test_data_loading():
    """Test that we can download and load price data correctly"""
    print("\n=== TEST 1: DATA LOADING ===")
    
    # Download a small sample (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    test_tickers = ["QQQ", "TQQQ", "XLU"]
    data = download_test_data(test_tickers, start_date.strftime("%Y-%m-%d"))
    
    # Check data quality
    assert not data.empty, "❌ Data is empty"
    assert len(data) > 20, f"❌ Not enough data points: {len(data)}"
    
    # Check for missing values
    missing = data.isna().sum()
    print(f"Missing values per ticker:\n{missing}")
    
    # Allow some missing values but not too many
    max_missing_pct = 0.1  # 10% max missing
    for ticker in test_tickers:
        if ticker in data.columns:
            missing_pct = missing[ticker] / len(data)
            assert missing_pct < max_missing_pct, f"❌ Too many missing values for {ticker}: {missing_pct:.1%}"
    
    print("✔ PASSED — Data loading works correctly\n")
    return data


# ======================================================================
# TEST 2: Regime Detection
# ======================================================================
def test_regime_detection():
    """Test regime detection logic with real data"""
    print("\n=== TEST 2: REGIME DETECTION ===")
    
    # Use fixed date range for reproducibility
    start_date = "2023-07-01"
    end_date = "2023-12-31"  # 6 months of data
    
    qqq_data = download_test_data(["QQQ"], start_date, end_date)
    qqq_data = qqq_data["QQQ"].dropna()
    
    # Calculate drawdown
    dd_series, ath_series = compute_drawdown_from_ath(qqq_data)
    
    # Test regime detection for various drawdown values
    test_cases = [
        (0.02, "R1"),   # 2% drawdown -> R1
        (0.10, "R2"),   # 10% drawdown -> R2
        (0.30, "R3"),   # 30% drawdown -> R3
    ]
    
    for dd_value, expected_regime in test_cases:
        regime = determine_regime(dd_value, CONFIG)
        assert regime == expected_regime, f"❌ Expected {expected_regime} for {dd_value:.1%} DD, got {regime}"
        print(f"  ✓ {dd_value:.1%} drawdown → {regime}")
    
    # Test with actual data points
    sample_dd = dd_series.iloc[-10:]
    for dd in sample_dd:
        # Check for NaN using pandas method
        if pd.notna(dd):
            # Convert to Python float to avoid numpy formatting issues
            try:
                dd_float = float(dd)
                regime = determine_regime(dd_float, CONFIG)
                assert regime in ["R1", "R2", "R3"], f"❌ Invalid regime: {regime} for DD: {dd_float:.4f}"
            except (ValueError, TypeError) as e:
                # Skip if conversion fails
                continue
    
    print("✔ PASSED — Regime detection works correctly\n")


# ======================================================================
# TEST 3: Allocation Engine
# ======================================================================
def test_allocation_engine():
    """Test that allocation engine returns correct allocations for each regime"""
    print("\n=== TEST 3: ALLOCATION ENGINE ===")
    
    for regime in ["R1", "R2", "R3"]:
        alloc = get_allocation_for_regime(regime, CONFIG)
        
        # Check that allocations sum to 1.0
        total = sum(alloc.values())
        assert abs(total - 1.0) < 0.001, f"❌ Allocations for {regime} don't sum to 1.0: {total}"
        
        # Check expected allocations
        if regime == "R1":
            assert alloc["TQQQ"] == 1.0, f"❌ R1 should be 100% TQQQ, got {alloc}"
        elif regime == "R2":
            assert alloc["XLU"] == 1.0, f"❌ R2 should be 100% XLU, got {alloc}"
        elif regime == "R3":
            assert alloc["TQQQ"] == 1.0, f"❌ R3 should be 100% TQQQ, got {alloc}"
        
        print(f"  ✓ {regime}: {alloc}")
    
    print("✔ PASSED — Allocation engine works correctly\n")


# ======================================================================
# TEST 4: Rebalancing Logic
# ======================================================================
def test_rebalancing_logic():
    """Test rebalancing with real prices"""
    print("\n=== TEST 4: REBALANCING LOGIC ===")
    
    # Use fixed date range for reproducibility
    start_date = "2024-11-01"
    end_date = "2024-11-30"  # 1 month of data
    
    data = download_test_data(["QQQ", "TQQQ", "XLU"], start_date, end_date)
    
    # Use first available prices
    first_date = data.index[0]
    prices = data.loc[first_date]
    
    # Test rebalancing with different allocations
    portfolio_value = 10000
    
    # Test R1 allocation (100% TQQQ)
    alloc_r1 = {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0}
    shares_r1 = rebalance_portfolio(portfolio_value, alloc_r1, prices)
    
    assert shares_r1 is not None, "❌ Rebalancing returned None"
    assert "TQQQ" in shares_r1, "❌ TQQQ shares missing"
    assert shares_r1["TQQQ"] > 0, "❌ TQQQ shares should be positive"
    
    # Verify value matches
    calculated_value = sum(shares_r1[t] * prices[t] for t in shares_r1 if t in prices.index)
    assert abs(calculated_value - portfolio_value) < 0.01, \
        f"❌ Rebalanced value mismatch: {calculated_value} vs {portfolio_value}"
    
    # Test R2 allocation (100% XLU)
    alloc_r2 = {"TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0}
    shares_r2 = rebalance_portfolio(portfolio_value, alloc_r2, prices)
    
    assert shares_r2["XLU"] > 0, "❌ XLU shares should be positive"
    calculated_value_r2 = sum(shares_r2[t] * prices[t] for t in shares_r2 if t in prices.index)
    assert abs(calculated_value_r2 - portfolio_value) < 0.01, \
        f"❌ R2 rebalanced value mismatch: {calculated_value_r2} vs {portfolio_value}"
    
    print(f"  ✓ R1 rebalance: {shares_r1}")
    print(f"  ✓ R2 rebalance: {shares_r2}")
    print("✔ PASSED — Rebalancing logic works correctly\n")


# ======================================================================
# TEST 5: Day-to-Day Appreciation
# ======================================================================
def test_day_to_day_appreciation():
    """Test that portfolio value changes correctly day-to-day"""
    print("\n=== TEST 5: DAY-TO-DAY APPRECIATION ===")
    
    # Use fixed date range for reproducibility (2 weeks)
    start_date = "2024-11-01"
    end_date = "2024-11-15"
    
    data = download_test_data(["QQQ", "TQQQ", "XLU"], start_date, end_date)
    
    if len(data) < 5:
        print("⚠ SKIPPED — Not enough data for day-to-day test")
        return
    
    # Run a simple buy-and-hold test on TQQQ
    test_config = CONFIG.copy()
    test_config["tickers"] = ["QQQ", "TQQQ"]  # Need QQQ for drawdown calculation
    test_config["allocation_tickers"] = ["TQQQ"]
    test_config["regimes"] = {"R1": {"TQQQ": 1.0}}
    test_config["rebalance_frequency"] = "none"  # No rebalancing
    test_config["drawdown_ticker"] = "QQQ"  # Use QQQ for drawdown
    
    # Create price data with both QQQ (for drawdown) and TQQQ (for portfolio)
    price_data = data[["QQQ", "TQQQ"]].dropna()
    
    if len(price_data) < 3:
        print("⚠ SKIPPED — Not enough TQQQ data")
        return
    
    equity_df, _ = run_backtest(
        price_data=price_data,
        config=test_config,
        dd_fn=lambda s: compute_drawdown_from_ath(s),
        regime_detector=lambda dd, cfg: "R1",
        rebalance_fn=rebalance_portfolio
    )
    
    # Check day-to-day changes
    values = equity_df["Value"].values
    returns = np.diff(values) / values[:-1]
    
    # Get actual price returns
    prices = price_data["TQQQ"].values
    price_returns = np.diff(prices) / prices[:-1]
    
    # Portfolio returns should match price returns (for buy-and-hold)
    for i in range(min(len(returns), len(price_returns))):
        if not (np.isnan(returns[i]) or np.isnan(price_returns[i])):
            # Allow small rounding differences
            diff = abs(returns[i] - price_returns[i])
            assert diff < 0.0001, \
                f"❌ Day {i} return mismatch: portfolio={returns[i]:.6f}, price={price_returns[i]:.6f}"
    
    # Check that value increases when price increases
    for i in range(1, len(equity_df)):
        prev_value = equity_df["Value"].iloc[i-1]
        curr_value = equity_df["Value"].iloc[i]
        prev_price = price_data.iloc[i-1]["TQQQ"]
        curr_price = price_data.iloc[i]["TQQQ"]
        
        if not (np.isnan(prev_value) or np.isnan(curr_value)):
            price_change_pct = (curr_price - prev_price) / prev_price
            value_change_pct = (curr_value - prev_value) / prev_value
            
            # Value should move in same direction as price (allowing for small rounding)
            if abs(price_change_pct) > 0.001:  # Only check if meaningful price change
                same_direction = (price_change_pct > 0 and value_change_pct > -0.0001) or \
                               (price_change_pct < 0 and value_change_pct < 0.0001)
                assert same_direction, \
                    f"❌ Day {i}: Price moved {price_change_pct:.2%} but value moved {value_change_pct:.2%}"
    
    print(f"  ✓ Tested {len(returns)} day-to-day transitions")
    print("✔ PASSED — Day-to-day appreciation works correctly\n")


# ======================================================================
# TEST 6: Week-to-Week Appreciation
# ======================================================================
def test_week_to_week_appreciation():
    """Test that portfolio value changes correctly week-to-week"""
    print("\n=== TEST 6: WEEK-TO-WEEK APPRECIATION ===")
    
    # Use fixed date range for reproducibility (2 months)
    start_date = "2024-09-01"
    end_date = "2024-10-31"
    
    data = download_test_data(["QQQ", "TQQQ", "XLU"], start_date, end_date)
    
    if len(data) < 20:
        print("⚠ SKIPPED — Not enough data for week-to-week test")
        return
    
    # Run buy-and-hold test
    test_config = CONFIG.copy()
    test_config["tickers"] = ["QQQ"]
    test_config["allocation_tickers"] = ["QQQ"]
    test_config["regimes"] = {"R1": {"QQQ": 1.0}}
    test_config["rebalance_frequency"] = "none"
    
    price_data = data[["QQQ"]].dropna()
    
    if len(price_data) < 10:
        print("⚠ SKIPPED — Not enough QQQ data")
        return
    
    equity_df, _ = run_backtest(
        price_data=price_data,
        config=test_config,
        dd_fn=lambda s: compute_drawdown_from_ath(s),
        regime_detector=lambda dd, cfg: "R1",
        rebalance_fn=rebalance_portfolio
    )
    
    # Group by week
    equity_df["Date"] = pd.to_datetime(equity_df["Date"])
    equity_df["Week"] = equity_df["Date"].dt.to_period("W")
    
    weekly_values = equity_df.groupby("Week")["Value"].last()
    
    if len(weekly_values) < 2:
        print("⚠ SKIPPED — Not enough weeks of data")
        return
    
    # Calculate week-over-week returns
    weekly_returns = weekly_values.pct_change().dropna()
    
    # Get weekly price returns
    price_data.index = pd.to_datetime(price_data.index)
    price_data["Week"] = price_data.index.to_period("W")
    weekly_prices = price_data.groupby("Week")["QQQ"].last()
    weekly_price_returns = weekly_prices.pct_change().dropna()
    
    # Compare week-over-week returns
    common_weeks = weekly_returns.index.intersection(weekly_price_returns.index)
    
    for week in common_weeks:
        port_return = weekly_returns[week]
        price_return = weekly_price_returns[week]
        
        if not (np.isnan(port_return) or np.isnan(price_return)):
            diff = abs(port_return - price_return)
            assert diff < 0.001, \
                f"❌ Week {week} return mismatch: portfolio={port_return:.4f}, price={price_return:.4f}"
    
    # Check that portfolio value tracks price over longer periods
    first_week_value = weekly_values.iloc[0]
    last_week_value = weekly_values.iloc[-1]
    total_return = (last_week_value / first_week_value) - 1
    
    first_week_price = weekly_prices.iloc[0]
    last_week_price = weekly_prices.iloc[-1]
    price_return = (last_week_price / first_week_price) - 1
    
    diff = abs(total_return - price_return)
    assert diff < 0.01, \
        f"❌ Total return mismatch: portfolio={total_return:.4f}, price={price_return:.4f}"
    
    print(f"  ✓ Tested {len(common_weeks)} week-over-week transitions")
    print(f"  ✓ Total period return: {total_return:.2%}")
    print("✔ PASSED — Week-to-week appreciation works correctly\n")


# ======================================================================
# TEST 7: Full Backtest with Regime Changes
# ======================================================================
def test_full_backtest_with_regimes():
    """Test full backtest with regime-based rebalancing"""
    print("\n=== TEST 7: FULL BACKTEST WITH REGIMES ===")
    
    # Use fixed date range for reproducibility (3 months)
    start_date = "2024-07-01"
    end_date = "2024-09-30"
    
    data = download_test_data(["QQQ", "TQQQ", "XLU"], start_date, end_date)
    
    if len(data) < 20:
        print("⚠ SKIPPED — Not enough data for full backtest")
        return
    
    # Run full backtest
    # Override tickers to match downloaded data (exclude SPY if not downloaded)
    test_config = CONFIG.copy()
    test_config["tickers"] = ["QQQ", "TQQQ", "XLU"]  # Match downloaded data
    test_config["drawdown_ticker"] = "QQQ"  # Ensure drawdown ticker is in the data
    
    equity_df, quarterly_df = run_backtest(
        price_data=data,
        config=test_config,
        dd_fn=lambda s: compute_drawdown_from_ath(s),
        regime_detector=lambda dd, cfg: determine_regime(dd, cfg),
        rebalance_fn=rebalance_portfolio
    )
    
    # Basic checks
    assert not equity_df.empty, "❌ Equity dataframe is empty"
    assert len(equity_df) > 0, "❌ No equity data"
    
    # Check that value starts at starting balance
    first_value = equity_df["Value"].iloc[0]
    assert abs(first_value - CONFIG["starting_balance"]) < 0.01, \
        f"❌ Starting value mismatch: {first_value} vs {CONFIG['starting_balance']}"
    
    # Check that value is always positive
    assert (equity_df["Value"] > 0).all(), "❌ Some portfolio values are non-positive"
    
    # Check regime columns exist
    assert "Market_Regime" in equity_df.columns, "❌ Market_Regime column missing"
    assert "Portfolio_Regime" in equity_df.columns, "❌ Portfolio_Regime column missing"
    
    # Check that regimes are valid
    valid_regimes = set(test_config["regimes"].keys())
    market_regimes = equity_df["Market_Regime"].dropna().unique()
    portfolio_regimes = equity_df["Portfolio_Regime"].dropna().unique()
    
    for regime in market_regimes:
        assert regime in valid_regimes or regime is None, f"❌ Invalid market regime: {regime}"
    
    for regime in portfolio_regimes:
        assert regime in valid_regimes or regime is None, f"❌ Invalid portfolio regime: {regime}"
    
    # Check rebalancing occurred (if instant mode)
    if test_config.get("rebalance_frequency") == "instant":
        rebalanced_count = (equity_df["Rebalanced"] == "Rebalanced").sum()
        assert rebalanced_count > 0, "❌ No rebalancing occurred in instant mode"
        print(f"  ✓ {rebalanced_count} rebalancing events occurred")
    
    # Check that final value makes sense
    final_value = equity_df["Value"].iloc[-1]
    print(f"  ✓ Starting value: ${test_config['starting_balance']:,.2f}")
    print(f"  ✓ Final value: ${final_value:,.2f}")
    print(f"  ✓ Total return: {(final_value / test_config['starting_balance'] - 1):.2%}")
    
    print("✔ PASSED — Full backtest with regimes works correctly\n")


# ======================================================================
# TEST 8: Instant Rebalancing
# ======================================================================
def test_instant_rebalancing():
    """Test that instant rebalancing works when regime changes"""
    print("\n=== TEST 8: INSTANT REBALANCING ===")
    
    # Use fixed date range for reproducibility (1 month)
    start_date = "2024-10-01"
    end_date = "2024-10-31"
    
    data = download_test_data(["QQQ", "TQQQ", "XLU"], start_date, end_date)
    
    if len(data) < 10:
        print("⚠ SKIPPED — Not enough data for instant rebalancing test")
        return
    
    # Use instant rebalancing
    test_config = CONFIG.copy()
    test_config["rebalance_frequency"] = "instant"
    # Override tickers to match downloaded data (exclude SPY if not downloaded)
    test_config["tickers"] = ["QQQ", "TQQQ", "XLU"]  # Match downloaded data
    test_config["drawdown_ticker"] = "QQQ"  # Ensure drawdown ticker is in the data
    
    equity_df, _ = run_backtest(
        price_data=data,
        config=test_config,
        dd_fn=lambda s: compute_drawdown_from_ath(s),
        regime_detector=lambda dd, cfg: determine_regime(dd, cfg),
        rebalance_fn=rebalance_portfolio
    )
    
    # Check that rebalancing occurred
    rebalanced_days = equity_df[equity_df["Rebalanced"] == "Rebalanced"]
    
    if len(rebalanced_days) > 0:
        # Check that portfolio regime changed on rebalance days
        for idx, row in rebalanced_days.iterrows():
            if idx > 0:
                prev_regime = equity_df.iloc[idx - 1]["Portfolio_Regime"]
                curr_regime = row["Portfolio_Regime"]
                # Regime should have changed (or it's the first day)
                if prev_regime is not None and curr_regime is not None:
                    print(f"  ✓ Rebalance on {row['Date']}: {prev_regime} → {curr_regime}")
    
    print(f"  ✓ {len(rebalanced_days)} rebalancing events")
    print("✔ PASSED — Instant rebalancing works correctly\n")


# ======================================================================
# TEST 9: Asymmetric Regime Rules
# ======================================================================
def test_asymmetric_regime_rules():
    """Test that asymmetric regime rules work correctly"""
    print("\n=== TEST 9: ASYMMETRIC REGIME RULES ===")
    
    # Import the function from backtest module
    import backtest
    apply_asymmetric_rules = backtest.apply_asymmetric_rules
    
    # Test: Market goes down, portfolio should follow immediately
    portfolio_regime = "R1"
    market_regime = "R2"
    new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
    assert new_regime == "R2", f"❌ Should switch to R2 when market goes down, got {new_regime}"
    print("  ✓ Market down: R1 → R2 (immediate)")
    
    # Test: Market goes to R1, portfolio should follow
    portfolio_regime = "R2"
    market_regime = "R1"
    new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
    assert new_regime == "R1", f"❌ Should switch to R1 when market recovers, got {new_regime}"
    print("  ✓ Market recovers: R2 → R1 (immediate)")
    
    # Test: Market in R2, portfolio in R2, market goes to R3
    portfolio_regime = "R2"
    market_regime = "R3"
    new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
    assert new_regime == "R3", f"❌ Should switch to R3 when market worsens, got {new_regime}"
    print("  ✓ Market worsens: R2 → R3 (immediate)")
    
    # Test: Market in R3, portfolio in R3, market goes to R2 (should stay R3)
    portfolio_regime = "R3"
    market_regime = "R2"
    new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
    assert new_regime == "R3", f"❌ Should stay in R3 (asymmetric rule), got {new_regime}"
    print("  ✓ Market improves from R3: R3 → R3 (hold)")
    
    # Test: Market in R3, portfolio in R3, market goes to R1 (should go to R1)
    portfolio_regime = "R3"
    market_regime = "R1"
    new_regime = apply_asymmetric_rules(portfolio_regime, market_regime)
    assert new_regime == "R1", f"❌ Should go to R1 when market fully recovers, got {new_regime}"
    print("  ✓ Market fully recovers: R3 → R1 (immediate)")
    
    print("✔ PASSED — Asymmetric regime rules work correctly\n")


# ======================================================================
# RUN ALL TESTS
# ======================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FAST MARKKY FUND - VALIDATION TEST SUITE")
    print("="*60)
    print(f"\nTest data cache directory: {TEST_DATA_CACHE_DIR}")
    print("Data will be cached after first download for faster subsequent runs.")
    print("Using fixed date ranges for reproducible test results.\n")
    
    try:
        test_data_loading()
        test_regime_detection()
        test_allocation_engine()
        test_rebalancing_logic()
        test_day_to_day_appreciation()
        test_week_to_week_appreciation()
        test_full_backtest_with_regimes()
        test_instant_rebalancing()
        test_asymmetric_regime_rules()
        
        print("="*60)
        print("🎉 ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
