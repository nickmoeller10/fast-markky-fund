# app.py
# ======================================================================
# Fast Markky Fund - Streamlit Application
# ======================================================================
# Main entry point - fully UI-driven configuration and backtesting
# ======================================================================

import copy
import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional
import pickle
import os

# Import our modules
from config import CONFIG
from data_loader import (
    load_price_data,
    attach_vix_to_equity_df,
    fetch_vix_series_for_equity_dates,
)
from regime_engine import compute_drawdown_from_ath, determine_regime
from allocation_engine import get_allocation_for_regime
from rebalance_engine import rebalance_portfolio
from backtest import run_backtest
from dashboard import render_dashboard
from utils import log

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="Fast Markky Fund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# SESSION STATE INITIALIZATION
# ======================================================================
if 'config' not in st.session_state:
    # Defaults match config.CONFIG (edit config.py as single source of truth)
    st.session_state.config = copy.deepcopy(CONFIG)

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Configuration"

# Streamlit >=1.33: fragments limit reruns to this block when widgets here change.
# Older versions: decorator is a no-op (full-script rerun, same as before).
_st_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _config_fragment_fallback(f):
    return f


if _st_fragment is None:
    _st_fragment = _config_fragment_fallback


@st.cache_data(ttl=86_400, show_spinner=False)
def _cached_ticker_earliest_date(ticker: str) -> Optional[str]:
    """One row per ticker per day TTL — avoids Yahoo round-trips on every UI tick."""
    from worst_case_simulator import get_earliest_date

    ts = get_earliest_date(ticker, start_date="1980-01-01")
    if ts is None:
        return None
    return ts.strftime("%Y-%m-%d")


# ======================================================================
# CONFIGURATION PAGE
# ======================================================================
@_st_fragment
def render_configuration_editor():
    """All config widgets + validation; reruns as a unit when any widget here changes."""
    config = st.session_state.config

    # Portfolio Parameters Section
    with st.expander("📊 Portfolio Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            config["starting_balance"] = st.number_input(
                "Starting Balance ($)",
                min_value=1.0,
                value=float(config["starting_balance"]),
                step=1000.0,
                help="Initial portfolio value"
            )
        
        with col2:
            # Set very wide date range to allow any reasonable date
            min_date = date(1900, 1, 1)
            max_date = date(2100, 12, 31)
            
            start_date_input = st.date_input(
                "Start Date",
                value=pd.to_datetime(config["start_date"]).date() if isinstance(config["start_date"], str) else config["start_date"],
                help="Backtest start date",
                min_value=min_date,
                max_value=max_date
            )
            config["start_date"] = start_date_input.strftime("%Y-%m-%d")
        
        with col3:
            use_end_date = st.checkbox("Set End Date", value=config["end_date"] is not None)
            if use_end_date:
                if config["end_date"]:
                    try:
                        end_date_val = pd.to_datetime(config["end_date"]).date() if isinstance(config["end_date"], str) else config["end_date"]
                    except:
                        end_date_val = date.today()
                else:
                    end_date_val = date.today()
                
                # Set very wide date range, but ensure end date is after start date
                min_date = date(1900, 1, 1)
                max_date = date(2100, 12, 31)
                
                # Ensure end date is at least the start date
                try:
                    start_date_val = pd.to_datetime(config["start_date"]).date() if isinstance(config["start_date"], str) else config["start_date"]
                    if isinstance(start_date_val, str):
                        start_date_val = pd.to_datetime(start_date_val).date()
                    if end_date_val < start_date_val:
                        end_date_val = start_date_val
                except:
                    pass
                
                end_date_input = st.date_input(
                    "End Date",
                    value=end_date_val,
                    help="Backtest end date (leave unchecked for current date). Must be after start date.",
                    key="end_date_input",
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Validate that end date is after start date
                try:
                    start_date_val = pd.to_datetime(config["start_date"]).date() if isinstance(config["start_date"], str) else config["start_date"]
                    if isinstance(start_date_val, str):
                        start_date_val = pd.to_datetime(start_date_val).date()
                    if end_date_input < start_date_val:
                        st.error(f"⚠️ End date must be after start date ({start_date_val}).")
                        config["end_date"] = None
                    else:
                        config["end_date"] = end_date_input.strftime("%Y-%m-%d")
                except Exception as e:
                    config["end_date"] = end_date_input.strftime("%Y-%m-%d")
            else:
                config["end_date"] = None
    
    # Simulated Scenario (pre QQQ)
    with st.expander("🔮 Simulated scenario (QQQ & TQQQ from ^IXIC)", expanded=True):
        use_worst_case = st.checkbox(
            "Enable simulated QQQ & TQQQ (NASDAQ Composite)",
            value=bool(config.get("use_worst_case_simulation", True)),
            help=(
                "When on: **QQQ** and **TQQQ** price series are not downloaded from Yahoo for the backtest. "
                "Instead they are built from **NASDAQ Composite (^IXIC)**: a synthetic QQQ track from daily index returns, "
                "and **TQQQ** as approximately **3× daily leveraged** QQQ (same methodology as the simulator module). "
                "That lets you start in 1999 (or back to 1985) with both names populated. "
                "**XLU, SPY, and other tickers stay real Yahoo data** from their actual listing dates. "
                "Turn off to use live **QQQ** and **TQQQ** only (TQQQ missing before ~2010 → QQQ proxy in code)."
            ),
        )
        config["use_worst_case_simulation"] = use_worst_case
        
        if use_worst_case:
            # Check earliest dates for OTHER tickers (excluding QQQ/TQQQ) for informational purposes
            allocation_tickers = config.get("allocation_tickers", config["tickers"])
            all_tickers = list(set(allocation_tickers + config.get("tickers", [])))
            
            # Exclude QQQ and TQQQ from earliest date check (they're simulated from 1985)
            other_tickers = [t for t in all_tickers if t not in ["QQQ", "TQQQ"]]
            
            # Get earliest dates for OTHER tickers only (for info display)
            ticker_earliest_dates = {}
            for ticker in other_tickers:
                iso = _cached_ticker_earliest_date(ticker)
                if iso:
                    ticker_earliest_dates[ticker] = pd.Timestamp(iso)
            
            st.info("""
            **What is simulated (when this is enabled)**

            - **QQQ** — Synthetic daily series from **^IXIC** (NASDAQ Composite) back to **1985-02-01**, scaled to behave like a Nasdaq-100–style equity index (not official QQQ history).
            - **TQQQ** — **3× daily leveraged** version of that synthetic QQQ path (ProShares-style daily reset leverage, not a guarantee of matching live TQQQ after 2010).
            - **Start/end dates** you set above filter the combined panel; the simulator fills QQQ/TQQQ for the whole range.
            - **All other tickers** (e.g. **XLU**, **SPY**) are **real Yahoo Finance** closes and only exist from each ticker’s actual inception.

            **Why use it:** Real **TQQQ** did not trade until **~2010**; without simulation, pre-2010 R1 targets **TQQQ** but the engine holds **QQQ** as a proxy. With simulation, both symbols have prices so allocations can follow the regime weights.
            """)
            
            st.markdown("**📅 Fund Availability Dates:**")
            earliest_info = []
            
            # Show QQQ and TQQQ (always simulated from 1985)
            earliest_info.append("• **QQQ**: Simulated from 1985-02-01 (based entirely on NASDAQ Composite ^IXIC)")
            earliest_info.append("• **TQQQ**: Simulated from 1985-02-01 (3x leveraged QQQ)")
            
            # Check other tickers
            for ticker in other_tickers:
                iso = _cached_ticker_earliest_date(ticker)
                if iso:
                    earliest_info.append(f"• **{ticker}**: Earliest date is {iso}")
                else:
                    earliest_info.append(f"• **{ticker}**: Could not determine earliest date")
            
            for info in earliest_info:
                st.caption(info)
            
            # Show information about date limitations
            if ticker_earliest_dates:
                earliest_other = max(ticker_earliest_dates.values())
                st.info(f"ℹ️ **Note**: Other tickers (like XLU) have real data starting from **{earliest_other.strftime('%Y-%m-%d')}**. If your start date is before this, those tickers won't have data until their inception dates. QQQ and TQQQ are simulated from 1985, so they'll have data for your entire date range.")
            else:
                st.success(f"✅ **Note**: No other tickers in your portfolio. QQQ and TQQQ are simulated from 1985, so you can use any date range starting from 1985-02-01.")
        else:
            st.caption(
                "Using **live Yahoo** closes for QQQ and TQQQ. Before TQQQ existed (~2010), targets that call for TQQQ are implemented with **QQQ** as a proxy."
            )

    # Rebalancing Settings
    with st.expander("🔄 Rebalancing Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rebalance_options = ["instant", "daily", "weekly", "monthly", "quarterly", "semiannual", "annual", "none"]
            config["rebalance_frequency"] = st.selectbox(
                "Rebalance Frequency",
                options=rebalance_options,
                index=rebalance_options.index(config["rebalance_frequency"]),
                help="How often to rebalance the portfolio"
            )
        
        with col2:
            config["drawdown_ticker"] = st.selectbox(
                "Drawdown Ticker",
                options=config["tickers"],
                index=config["tickers"].index(config["drawdown_ticker"]) if config["drawdown_ticker"] in config["tickers"] else 0,
                help="Ticker whose price path drives regime drawdown (standard ATH or rolling window)"
            )
        
        with col3:
            strategy_options = ["down_only", "up_only", "always", "per_regime"]
            strategy_labels = {
                "down_only": "Regime Shift Down Only",
                "up_only": "Regime Shift Up Only",
                "always": "Always (follow market regime)",
                "per_regime": "Per-regime direction (see each regime below)",
            }
            strategy_descriptions = {
                "down_only": "Rebalance when market goes DOWN, hold on partial recoveries",
                "up_only": "Rebalance when market goes UP, hold on declines",
                "always": "Rebalance whenever regime changes",
                "per_regime": "When the market enters a regime, match or hold based on that regime's Up/Down setting.",
            }
            
            current_strategy = config.get("rebalance_strategy", "down_only")
            strategy_index = strategy_options.index(current_strategy) if current_strategy in strategy_options else 0
            
            selected_strategy = st.selectbox(
                "Rebalance Strategy",
                options=strategy_options,
                index=strategy_index,
                format_func=lambda x: strategy_labels[x],
                help=strategy_descriptions.get(current_strategy, "")
            )
            config["rebalance_strategy"] = selected_strategy
            
            # Show description
            st.caption(strategy_descriptions.get(selected_strategy, ""))
        
        st.markdown("**Drawdown reference**")
        dw_a, dw_b = st.columns(2)
        with dw_a:
            config["drawdown_window_enabled"] = st.checkbox(
                "Rolling drawdown window (vs. standard ATH)",
                value=bool(config.get("drawdown_window_enabled", True)),
                help="When enabled, the reference peak is the highest close over the trailing N calendar years. "
                     "Until N full years of history exist for the drawdown ticker, standard ATH (cummax) is used. "
                     "Full ticker history is still downloaded for pre-portfolio simulation and future use.",
            )
        with dw_b:
            if config["drawdown_window_enabled"]:
                config["drawdown_window_years"] = int(
                    st.number_input(
                        "Window length (calendar years)",
                        min_value=1,
                        value=max(1, int(config.get("drawdown_window_years", 2))),
                        step=1,
                        help="Integer number of calendar years in the trailing peak window.",
                    )
                )
    
    # Dividend Reinvestment Settings
    with st.expander("💰 Dividend Reinvestment", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            config["dividend_reinvestment"] = st.checkbox(
                "Enable dividend handling",
                value=config.get("dividend_reinvestment", True),
                help="Accrue dividends per target. 'cash' sweeps into the next rebalance with portfolio value.",
            )
        
        with col2:
            if config["dividend_reinvestment"]:
                # Build options: cash + allocation tickers
                dividend_options = ["cash"] + config.get("allocation_tickers", config["tickers"])
                current_target = config.get("dividend_reinvestment_target", "cash")
                
                # Find index, default to 0 (cash) if not found
                try:
                    target_index = dividend_options.index(current_target)
                except ValueError:
                    target_index = 0
                
                selected_target = st.selectbox(
                    "Reinvestment Target",
                    options=dividend_options,
                    index=target_index,
                    help="Cash: dividends sit in cash_balance until the next rebalance, then deploy with full notional into regime weights."
                )
                config["dividend_reinvestment_target"] = selected_target
                
                if selected_target == "cash":
                    st.caption("💵 Dividends will be held as cash and redistributed on next rebalance")
                else:
                    st.caption(f"📈 Dividends will be immediately reinvested into {selected_target}")
            else:
                st.caption("Dividend reinvestment is disabled")
    
    # Ticker Configuration
    with st.expander("📈 Ticker Configuration", expanded=True):
        st.markdown("**Available Tickers** (for data and benchmarks)")
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value=", ".join(config["tickers"]),
            help="Tickers to download and display (e.g., QQQ, TQQQ, XLU, SPY)"
        )
        config["tickers"] = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        st.markdown("**Allocation Tickers** (used in portfolio)")
        alloc_input = st.text_input(
            "Allocation Tickers (comma-separated)",
            value=", ".join(config["allocation_tickers"]),
            help="Tickers that can be held in the portfolio (must be subset of Available Tickers)"
        )
        config["allocation_tickers"] = [t.strip().upper() for t in alloc_input.split(",") if t.strip()]
        
        # Validate allocation tickers are in main tickers
        invalid = [t for t in config["allocation_tickers"] if t not in config["tickers"]]
        if invalid:
            st.warning(f"⚠️ These allocation tickers are not in available tickers: {invalid}")
    
    # Regime Configuration
    with st.expander("🎯 Regime Definitions", expanded=True):
        st.markdown("Configure market regimes based on drawdown thresholds")
        
        regimes = config["regimes"]
        regime_names = sorted(list(regimes.keys()))  # Sort for consistent display
        
        # Add/Remove regime controls
        col_add, col_info = st.columns([1, 3])
        with col_add:
            if st.button("➕ Add Regime", key="add_regime_btn"):
                # Generate next regime name (R1, R2, R3, etc.)
                existing_numbers = []
                for name in regime_names:
                    if name.startswith("R") and name[1:].isdigit():
                        existing_numbers.append(int(name[1:]))
                
                if existing_numbers:
                    next_num = max(existing_numbers) + 1
                else:
                    next_num = 1
                
                new_regime_name = f"R{next_num}"
                
                # Create new regime with default values
                new_regime = {
                    "dd_low": 0.0,
                    "dd_high": 1.0,
                    "rebalance_on_downward": "match",
                    "rebalance_on_upward": "match",
                }
                
                # Initialize allocations for all allocation tickers
                for ticker in config["allocation_tickers"]:
                    new_regime[ticker] = 0.0
                
                # If there are allocation tickers, distribute evenly
                if len(config["allocation_tickers"]) > 0:
                    equal_alloc = 1.0 / len(config["allocation_tickers"])
                    for ticker in config["allocation_tickers"]:
                        new_regime[ticker] = equal_alloc
                
                regimes[new_regime_name] = new_regime
                st.rerun()
        
        with col_info:
            st.caption("💡 Add regimes to define different market conditions. Each regime needs drawdown thresholds and allocations that sum to 100%.")
        
        st.markdown("---")
        
        # Display existing regimes with remove buttons
        for i, regime_name in enumerate(regime_names):
            # Create columns for regime header with remove button
            header_col1, header_col2 = st.columns([10, 1])
            
            with header_col1:
                st.markdown(f"#### {regime_name}")
            
            with header_col2:
                # Only allow removal if there's more than one regime
                if len(regime_names) > 1:
                    if st.button("🗑️", key=f"remove_{regime_name}", help=f"Remove {regime_name}"):
                        del regimes[regime_name]
                        st.rerun()
                else:
                    st.caption("(min 1)")
            
            regime = regimes[regime_name]
            
            # Ensure all allocation tickers exist in regime
            for ticker in config["allocation_tickers"]:
                if ticker not in regime:
                    regime[ticker] = 0.0
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Drawdown Range:**")
                regime["dd_low"] = st.number_input(
                    "DD Low (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(regime["dd_low"]) * 100,
                    step=0.5,
                    format="%.1f",
                    key=f"{regime_name}_dd_low"
                ) / 100.0
                regime["dd_high"] = st.number_input(
                    "DD High (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(regime["dd_high"]) * 100,
                    step=0.5,
                    format="%.1f",
                    key=f"{regime_name}_dd_high"
                ) / 100.0
            
            with col2:
                st.markdown("**Allocations:**")
                
                # Ensure all allocation tickers exist in regime
                for ticker in config["allocation_tickers"]:
                    if ticker not in regime:
                        regime[ticker] = 0.0
                
                _regime_meta = frozenset(
                    {"dd_low", "dd_high", "rebalance_on_downward", "rebalance_on_upward"}
                )
                # Remove keys that are not allocation tickers or regime metadata
                tickers_to_remove = [
                    t
                    for t in regime.keys()
                    if t not in config["allocation_tickers"] and t not in _regime_meta
                ]
                for t in tickers_to_remove:
                    del regime[t]
                if "rebalance_on_downward" not in regime:
                    regime["rebalance_on_downward"] = "match"
                if "rebalance_on_upward" not in regime:
                    regime["rebalance_on_upward"] = "match"
                
                if len(config["allocation_tickers"]) > 0:
                    alloc_cols = st.columns(len(config["allocation_tickers"]))
                    total_alloc = 0.0
                    
                    for j, ticker in enumerate(config["allocation_tickers"]):
                        with alloc_cols[j]:
                            alloc_value = st.number_input(
                                ticker,
                                min_value=0.0,
                                max_value=100.0,
                                value=float(regime.get(ticker, 0.0)) * 100,
                                step=1.0,
                                format="%.0f",
                                key=f"{regime_name}_{ticker}"
                            ) / 100.0
                            regime[ticker] = alloc_value
                            total_alloc += alloc_value
                    
                    # Display total with color coding
                    if abs(total_alloc - 1.0) > 0.001:
                        st.error(f"⚠️ {regime_name} allocations sum to {total_alloc:.1%}, should be 100%")
                    else:
                        st.success(f"✓ Total: {total_alloc:.0%}")
                else:
                    st.warning("⚠️ No allocation tickers configured")

                st.markdown("**Rebalance when market enters this regime** (used if strategy is *Per-regime direction*)")
                rc1, rc2 = st.columns(2)
                dir_opts = ["match", "hold"]
                dir_labels = {
                    "match": "Match — adopt this regime's allocation",
                    "hold": "Hold — keep prior portfolio regime",
                }
                with rc1:
                    regime["rebalance_on_downward"] = st.selectbox(
                        "From worse stress (downward move into this regime)",
                        options=dir_opts,
                        index=dir_opts.index(regime.get("rebalance_on_downward", "match"))
                        if regime.get("rebalance_on_downward", "match") in dir_opts
                        else 0,
                        format_func=lambda x: dir_labels[x],
                        key=f"{regime_name}_reb_down",
                    )
                with rc2:
                    regime["rebalance_on_upward"] = st.selectbox(
                        "From better conditions (upward move into this regime)",
                        options=dir_opts,
                        index=dir_opts.index(regime.get("rebalance_on_upward", "match"))
                        if regime.get("rebalance_on_upward", "match") in dir_opts
                        else 0,
                        format_func=lambda x: dir_labels[x],
                        key=f"{regime_name}_reb_up",
                    )
                st.caption(
                    "R1 only receives upward entries from R2+; deepest regime only receives downward entries. "
                    "Equity curve adds **Regime_Trajectory** (Upward/Downward/Flat) vs prior day's **market** regime."
                )
            
            if i < len(regime_names) - 1:  # Don't add separator after last regime
                st.markdown("---")
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            config["minimum_allocation"] = st.number_input(
                "Minimum Allocation",
                min_value=0.0,
                max_value=1.0,
                value=float(config["minimum_allocation"]),
                step=0.01,
                format="%.2f",
                help="Minimum allocation percentage guardrail"
            )
        
        with col2:
            config["rebalance_holiday_rule"] = st.selectbox(
                "Holiday Rule",
                options=["next_trading_day", "previous_trading_day"],
                index=0 if config["rebalance_holiday_rule"] == "next_trading_day" else 1,
                help="How to handle rebalancing on holidays"
            )
        
        # Note: Rebalance Strategy is now in main Rebalancing Settings section above
    
    # Configuration validation warnings
    warnings = []
    
    # Check allocation tickers
    invalid = [t for t in config["allocation_tickers"] if t not in config["tickers"]]
    if invalid:
        warnings.append(f"Allocation tickers {invalid} not in available tickers")
    
    # Check regime allocations
    for regime_name, regime in config["regimes"].items():
        total = sum(regime.get(t, 0) for t in config["allocation_tickers"])
        if abs(total - 1.0) > 0.001:
            warnings.append(f"{regime_name} allocations don't sum to 100%")
    
    if warnings:
        st.warning("⚠️ **Configuration Issues:**")
        for warning in warnings:
            st.warning(f"  • {warning}")
    
    # Display current config summary
    with st.expander("📋 Configuration Summary (JSON)", expanded=False):
        st.json(config)


def render_configuration_page():
    """Page chrome and run button (full rerun). Config fields live in a fragment."""
    st.title("⚙️ Configuration")
    st.markdown(
        "Adjust settings below — only the configuration panel updates while you edit. "
        "**Run Backtest** loads data, simulates, and opens results (full run)."
    )
    st.markdown("---")
    render_configuration_editor()
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest_from_ui()


# ======================================================================
# RUN BACKTEST FUNCTION
# ======================================================================
def run_backtest_from_ui():
    """Execute backtest with current configuration"""
    
    config = st.session_state.config
    
    # Validate configuration
    errors = []
    
    # Check we have tickers
    if not config["tickers"]:
        errors.append("No tickers specified")
    
    if not config["allocation_tickers"]:
        errors.append("No allocation tickers specified")
    
    # Check allocation tickers
    invalid = [t for t in config["allocation_tickers"] if t not in config["tickers"]]
    if invalid:
        errors.append(f"Allocation tickers {invalid} not in available tickers")
    
    # Check drawdown ticker is in tickers
    if config["drawdown_ticker"] not in config["tickers"]:
        errors.append(f"Drawdown ticker '{config['drawdown_ticker']}' not in available tickers")
    
    # Check regime allocations sum to 1.0
    for regime_name, regime in config["regimes"].items():
        total = sum(regime.get(t, 0) for t in config["allocation_tickers"])
        if abs(total - 1.0) > 0.001:
            errors.append(f"{regime_name} allocations don't sum to 100% (currently {total:.1%})")
    
    if errors:
        st.error("❌ **Configuration Errors:**")
        for error in errors:
            st.error(f"  • {error}")
        return
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load data (or generate worst-case simulation)
        use_worst_case = config.get("use_worst_case_simulation", False)
        
        if use_worst_case:
            status_text.text("🔮 Generating simulated scenario from NASDAQ Composite...")
            progress_bar.progress(10)
            
            from worst_case_simulator import generate_worst_case_prices
            
            # Generate simulated prices using user's start/end dates
            price_data, earliest_dates = generate_worst_case_prices(
                config, 
                config["tickers"],
                start_date=config.get("start_date"),
                end_date=config.get("end_date")
            )
            
            # Update config with actual dates used (may be adjusted if user's dates are invalid)
            config["start_date"] = price_data.index.min().strftime("%Y-%m-%d")
            config["end_date"] = price_data.index.max().strftime("%Y-%m-%d")
            
            log(f"[APP] Simulated scenario: {config['start_date']} to {config['end_date']}")
            
            # For worst-case, we don't support dividends (simulated data)
            dividend_data = None
            dividend_reinvestment = False
            config["dividend_reinvestment"] = False
            
            progress_bar.progress(30)
        else:
            status_text.text("📥 Loading price data from Yahoo Finance...")
            progress_bar.progress(10)
            
            # Check if dividend reinvestment is enabled
            dividend_reinvestment = config.get("dividend_reinvestment", False)
            
            if dividend_reinvestment:
                price_data, dividend_data = load_price_data(
                    config["tickers"],
                    config["start_date"],
                    config.get("end_date"),
                    include_dividends=True
                )
            else:
                price_data = load_price_data(
                    config["tickers"],
                    config["start_date"],
                    config.get("end_date"),
                    include_dividends=False
                )
                dividend_data = None
        
        progress_bar.progress(30)
        
        # Step 2: Run backtest
        status_text.text("🔄 Running backtest simulation...")
        progress_bar.progress(50)
        
        equity_df, quarterly_df, dividend_df = run_backtest(
            price_data,
            config,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, cfg: determine_regime(dd, cfg),
            rebalance_portfolio,
            dividend_data=dividend_data
        )

        # ^VIX from Yahoo — always load for Performance Summary (even when QQQ/TQQQ are simulated)
        try:
            vix_s = fetch_vix_series_for_equity_dates(equity_df)
            equity_df = attach_vix_to_equity_df(equity_df, vix_s)
        except Exception as ex:
            log(f"Warning: could not load ^VIX for summary column: {ex}")
            equity_df = attach_vix_to_equity_df(equity_df, pd.Series(dtype=float))
        
        progress_bar.progress(90)
        
        # Step 3: Store results
        status_text.text("💾 Saving results...")
        st.session_state.backtest_results = {
            'equity_df': equity_df,
            'quarterly_df': quarterly_df,
            'dividend_df': dividend_df,
            'config': config.copy()
        }
        
        progress_bar.progress(100)
        status_text.success("✅ Backtest complete! Switching to results...")
        
        # Switch to results page
        st.session_state.current_page = "Results"
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ **Error running backtest:** {str(e)}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

# ======================================================================
# RESULTS PAGE
# ======================================================================
@_st_fragment
def render_results_dashboard():
    """Charts, tables, and downloads — isolated reruns from sidebar/header."""
    results = st.session_state.backtest_results
    if results is None:
        return
    render_dashboard(
        results["equity_df"],
        results["quarterly_df"],
        results["config"],
        dividend_df=results.get("dividend_df", pd.DataFrame()),
    )


def render_results_page():
    """Render the results/dashboard page"""
    
    if st.session_state.backtest_results is None:
        st.warning("⚠️ No backtest results available. Please run a backtest from the Configuration page.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("⚙️ Go to Configuration", type="primary", use_container_width=True):
                st.session_state.current_page = "Configuration"
                st.rerun()
        return
    
    # Navigation header (full rerun on click — not inside fragment)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("📊 Backtest Results")
    with col2:
        if st.button("⚙️ Edit Config", use_container_width=True):
            st.session_state.current_page = "Configuration"
            st.rerun()
    with col3:
        if st.button("🔄 Re-run", use_container_width=True):
            run_backtest_from_ui()
    
    st.markdown("---")
    render_results_dashboard()

# ======================================================================
# MAIN APP
# ======================================================================
def main():
    """Main application entry point"""
    
    # Header at top of page
    st.header("Back Test Simulation")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📈 Fast Markky Fund")
        st.markdown("*Tactical Portfolio Backtesting*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            options=["⚙️ Configuration", "📊 Results"],
            index=0 if st.session_state.current_page == "Configuration" else 1,
            label_visibility="collapsed"
        )
        
        # Extract page name (remove emoji)
        st.session_state.current_page = page.split(" ", 1)[1] if " " in page else page
        
        st.markdown("---")
        
        # Quick stats if results exist
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            equity_df = results['equity_df']
            
            st.markdown("### 📈 Quick Stats")
            start_val = equity_df["Value"].iloc[0]
            end_val = equity_df["Value"].iloc[-1]
            total_return = (end_val / start_val - 1) * 100
            
            st.metric("Final Value", f"${end_val:,.2f}")
            st.metric("Total Return", f"{total_return:.2f}%")
            
            # Calculate years
            start_date = pd.to_datetime(equity_df["Date"].iloc[0])
            end_date = pd.to_datetime(equity_df["Date"].iloc[-1])
            years = (end_date - start_date).days / 365.25
            if years > 0:
                cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
                st.metric("CAGR", f"{cagr:.2f}%")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        Configure your backtest parameters, 
        then run to see results.
        
        All settings are stored in 
        session state.
        """)
    
    # Render current page
    if st.session_state.current_page == "Configuration":
        render_configuration_page()
    elif st.session_state.current_page == "Results":
        render_results_page()

if __name__ == "__main__":
    main()

