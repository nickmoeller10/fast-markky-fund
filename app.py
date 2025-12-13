# app.py
# ======================================================================
# Fast Markky Fund - Streamlit Application
# ======================================================================
# Main entry point - fully UI-driven configuration and backtesting
# ======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime, date
import pickle
import os

# Import our modules
from data_loader import load_price_data
from regime_engine import compute_drawdown_from_ath, determine_regime
from allocation_engine import get_allocation_for_regime
from rebalance_engine import rebalance_portfolio
from backtest import run_backtest
from dashboard import render_dashboard

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
    # Default configuration
    st.session_state.config = {
        "starting_balance": 10000,
        "start_date": "2010-12-02",
        "end_date": None,  # None = current date
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_holiday_rule": "next_trading_day",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "minimum_allocation": 0.00,
        "regimes": {
            "R1": {
                "dd_low": 0.00,
                "dd_high": 0.06,
                "TQQQ": 1.00,
                "QQQ": 0.00,
                "XLU": 0.00,
            },
            "R2": {
                "dd_low": 0.06,
                "dd_high": 0.28,
                "TQQQ": 0.00,
                "QQQ": 0.00,
                "XLU": 1.00,
            },
            "R3": {
                "dd_low": 0.28,
                "dd_high": 1.00,
                "TQQQ": 1.00,
                "QQQ": 0.00,
                "XLU": 0.00,
            },
        },
        "use_worst_case_simulation": False,
        "benchmark_ticker": "QQQ",
        "worst_case_start_date": "1950-01-01",
        "worst_case_output_dir": "simulation_outputs",
    }

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Configuration"

# ======================================================================
# CONFIGURATION PAGE
# ======================================================================
def render_configuration_page():
    """Render the configuration/setup page"""
    
    st.title("⚙️ Configuration")
    st.markdown("Configure your backtest parameters below, then click **Run Backtest** to execute.")
    st.markdown("---")
    
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
            start_date_input = st.date_input(
                "Start Date",
                value=pd.to_datetime(config["start_date"]).date() if isinstance(config["start_date"], str) else config["start_date"],
                help="Backtest start date",
                min_value=None,  # Remove minimum restriction
                max_value=None   # Remove maximum restriction
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
                end_date_input = st.date_input(
                    "End Date",
                    value=end_date_val,
                    help="Backtest end date (leave unchecked for current date)",
                    key="end_date_input",
                    min_value=None,  # Remove minimum restriction
                    max_value=None   # Remove maximum restriction
                )
                config["end_date"] = end_date_input.strftime("%Y-%m-%d")
            else:
                config["end_date"] = None
    
    # Rebalancing Settings
    with st.expander("🔄 Rebalancing Settings", expanded=True):
        col1, col2 = st.columns(2)
        
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
                help="Ticker used to measure drawdown from all-time high"
            )
    
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
        
        for regime_name in regime_names:
            st.markdown(f"#### {regime_name}")
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
                
                # Remove tickers that are no longer in allocation_tickers
                tickers_to_remove = [t for t in regime.keys() if t not in config["allocation_tickers"] and t not in ["dd_low", "dd_high"]]
                for t in tickers_to_remove:
                    del regime[t]
                
                if len(config["allocation_tickers"]) > 0:
                    alloc_cols = st.columns(len(config["allocation_tickers"]))
                    total_alloc = 0.0
                    
                    for i, ticker in enumerate(config["allocation_tickers"]):
                        with alloc_cols[i]:
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
            
            if regime_name != regime_names[-1]:  # Don't add separator after last regime
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
    
    st.markdown("---")
    
    # Run Backtest Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        if run_button:
            run_backtest_from_ui()
    
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
        # Step 1: Load data
        status_text.text("📥 Loading price data from Yahoo Finance...")
        progress_bar.progress(10)
        
        price_data = load_price_data(
            config["tickers"],
            config["start_date"],
            config.get("end_date")
        )
        
        progress_bar.progress(30)
        
        # Step 2: Run backtest
        status_text.text("🔄 Running backtest simulation...")
        progress_bar.progress(50)
        
        equity_df, quarterly_df = run_backtest(
            price_data,
            config,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, cfg: determine_regime(dd, cfg),
            rebalance_portfolio
        )
        
        progress_bar.progress(90)
        
        # Step 3: Store results
        status_text.text("💾 Saving results...")
        st.session_state.backtest_results = {
            'equity_df': equity_df,
            'quarterly_df': quarterly_df,
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
    
    # Navigation header
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
    
    # Render dashboard
    results = st.session_state.backtest_results
    render_dashboard(
        results['equity_df'],
        results['quarterly_df'],
        results['config']
    )

# ======================================================================
# MAIN APP
# ======================================================================
def main():
    """Main application entry point"""
    
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

