# dashboard.py
# ======================================================================
# FAST MARKKY FUND — Interactive Web Dashboard
# ======================================================================
# Modern web-based visualization replacing Excel export
# Built with Streamlit and Plotly for interactive charts
# ======================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io
from exporter import export_to_excel
from utils import max_drawdown_from_equity_curve
from metrics import calculate_metrics
from data_loader import VIX_YAHOO_SYMBOL
from dashboard_data import (
    PERF_SIGNAL_COLUMNS_ORDER,
    PERF_SIGNAL_COLUMN_DISPLAY_NAMES,
    PERFORMANCE_SUMMARY_GUIDE,
    REGIME_DESCRIPTIONS,
    perf_cell_empty as _perf_cell_empty,
    todays_regime_status,
)
from dashboard_charts import (
    create_allocation_chart,
    create_dividend_chart,
    create_drawdown_chart,
    create_equity_curve_chart,
    create_performance_summary_chart,
    create_regime_timeline,
)
from allocation_engine import get_allocation_for_regime
from signal_override_engine import allocation_human_readable


def _set_page_config_if_standalone():
    """Set Streamlit page config; no-op if a parent app already set it."""
    try:
        st.set_page_config(
            page_title="Fast Markky Fund - Backtest Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass



def render_todays_regime_status(equity_df, config):
    """Render the Today's Status panel above the historical metrics."""
    status = todays_regime_status(equity_df, config)

    as_of_str = status["as_of"].strftime("%Y-%m-%d") if status["as_of"] is not None else "—"
    st.subheader(f"📍 Today's Status (as of {as_of_str})")

    col1, col2, col3 = st.columns(3)

    with col1:
        market = status["market_regime"] or "—"
        portfolio = status["portfolio_regime"] or "—"
        market_desc = REGIME_DESCRIPTIONS.get(market, "")
        portfolio_desc = REGIME_DESCRIPTIONS.get(portfolio, "")
        st.metric("Market Regime", market)
        if market_desc:
            st.caption(market_desc)
        st.metric("Portfolio Regime", portfolio)
        if portfolio_desc and portfolio != market:
            st.caption(portfolio_desc)

    with col2:
        if status["override_active"] == "none":
            st.metric("Signal Override", "None")
            st.caption("Using base regime allocation")
        else:
            st.metric("Signal Override", status["override_active"].title())
            st.caption(f'"{status["override_label"]}"')

    with col3:
        st.metric("Recommended Allocation", " ")
        st.markdown(f"**{status['recommended_allocation'] or '—'}**")

    st.markdown("---")


# ======================================================================
# MAIN DASHBOARD FUNCTION
# ======================================================================
def render_dashboard(equity_df, quarterly_df, config, dividend_df=None):
    """Render the complete dashboard"""
    _set_page_config_if_standalone()

    if dividend_df is None:
        dividend_df = pd.DataFrame()
    
    # Title and header
    st.title("📈 Fast Markky Fund - Backtest Dashboard")
    st.markdown("---")

    # Today's Status panel — pulled from the last row of equity_df so the
    # widget cannot disagree with the backtest result.
    render_todays_regime_status(equity_df, config)

    # Calculate metrics
    metrics = calculate_metrics(equity_df, config)

    # Key Performance Metrics - Enhanced Display
    st.header("📊 Key Performance Metrics")
    
    # Primary metrics (larger, more prominent)
    st.subheader("💰 Portfolio Performance")
    primary_col1, primary_col2, primary_col3 = st.columns(3)
    
    with primary_col1:
        final_value_delta = metrics['end_value'] - metrics['start_value']
        st.metric(
            "Final Portfolio Value",
            f"${metrics['end_value']:,.2f}",
            f"${final_value_delta:,.2f}",
            delta_color="normal"
        )
        st.caption(f"Starting: ${metrics['start_value']:,.2f}")
    
    with primary_col2:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2%}",
            None,
            delta_color="normal"
        )
        st.caption(f"Absolute gain: ${final_value_delta:,.2f}")
    
    with primary_col3:
        st.metric(
            "CAGR",
            f"{metrics['cagr']:.2%}",
            None,
            delta_color="normal"
        )
        st.caption(f"Over {metrics['years']:.1f} years")
    
    st.markdown("---")
    
    # Secondary metrics (risk and ratios)
    st.subheader("📈 Risk & Performance Ratios")
    secondary_col1, secondary_col2, secondary_col3, secondary_col4 = st.columns(4)
    
    with secondary_col1:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2%}",
            None,
            delta_color="inverse"
        )
        st.caption("Largest % fall from any prior peak (full backtest)")
    
    with secondary_col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            None,
            delta_color="normal"
        )
        st.caption("Risk-adjusted return")
    
    with secondary_col3:
        sortino_display = "∞" if metrics['sortino_ratio'] == float('inf') else f"{metrics['sortino_ratio']:.2f}"
        st.metric(
            "Sortino Ratio",
            sortino_display,
            None,
            delta_color="normal"
        )
        st.caption("Downside risk-adjusted")
    
    with secondary_col4:
        beta_label = f"Beta vs {metrics['beta_benchmark']}" if metrics['beta_benchmark'] != "N/A" else "Beta"
        st.metric(
            beta_label,
            f"{metrics['beta']:.2f}",
            None,
            delta_color="normal"
        )
        st.caption("Market sensitivity")
    
    st.markdown("---")
    
    # Equity Curve Chart
    st.header("📈 Equity Curve")
    equity_chart = create_equity_curve_chart(equity_df, config)
    st.plotly_chart(equity_chart, use_container_width=True)
    
    # Drawdown Charts
    st.header("📉 Drawdown Analysis")
    drawdown_ticker = config.get("drawdown_ticker", "QQQ")
    drawdown_chart = create_drawdown_chart(equity_df, drawdown_ticker)
    st.plotly_chart(drawdown_chart, use_container_width=True)
    
    # Regime Timeline and Allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🔄 Regime Timeline")
        regime_chart = create_regime_timeline(equity_df)
        if regime_chart:
            st.plotly_chart(regime_chart, use_container_width=True)
        else:
            st.info("No regime data available")
    
    with col2:
        st.header("💼 Portfolio Allocation")
        alloc_chart = create_allocation_chart(equity_df, config)
        if alloc_chart:
            st.plotly_chart(alloc_chart, use_container_width=True)
        else:
            st.info("No allocation data available")
    
    st.markdown("---")
    
    # Dividend Distribution Chart
    # Always show this section if dividend reinvestment is enabled
    dividend_reinvestment_enabled = config.get("dividend_reinvestment", False)
    
    if dividend_reinvestment_enabled:
        st.header("💰 Dividend Distribution")
        
        # Check if dividend_df exists and has data
        has_dividend_data = (
            dividend_df is not None 
            and isinstance(dividend_df, pd.DataFrame)
            and not dividend_df.empty 
            and len(dividend_df) > 0
        )
        
        if has_dividend_data:
            dividend_chart = create_dividend_chart(dividend_df, equity_df)
            if dividend_chart:
                st.plotly_chart(dividend_chart, use_container_width=True)
            
            # Dividend Summary Table
            with st.expander("📋 Dividend Summary", expanded=True):
                summary_df = dividend_df.copy()
                summary_df["Date"] = pd.to_datetime(summary_df["Date"])
                summary_df = summary_df.sort_values("Date", ascending=False)
                
                # Format columns for display
                display_cols = ["Date", "Ticker", "Dividend_Amount", "Dividend_Yield", "Portfolio_Pct", "Reinvestment_Target"]
                display_cols = [c for c in display_cols if c in summary_df.columns]
                
                formatted_df = summary_df[display_cols].copy()
                if "Dividend_Amount" in formatted_df.columns:
                    formatted_df["Dividend_Amount"] = formatted_df["Dividend_Amount"].apply(lambda x: f"${x:,.2f}")
                if "Dividend_Yield" in formatted_df.columns:
                    formatted_df["Dividend_Yield"] = formatted_df["Dividend_Yield"].apply(lambda x: f"{x:.2f}%")
                if "Portfolio_Pct" in formatted_df.columns:
                    formatted_df["Portfolio_Pct"] = formatted_df["Portfolio_Pct"].apply(lambda x: f"{x:.3f}%")
                
                st.dataframe(formatted_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Dividends", f"${dividend_df['Dividend_Amount'].sum():,.2f}")
                with col2:
                    st.metric("Dividend Events", len(dividend_df))
                with col3:
                    avg_yield = dividend_df['Dividend_Yield'].mean() if 'Dividend_Yield' in dividend_df.columns else 0
                    st.metric("Avg Yield", f"{avg_yield:.2f}%")
                with col4:
                    total_pct = dividend_df['Portfolio_Pct'].sum() if 'Portfolio_Pct' in dividend_df.columns else 0
                    st.metric("Total % of Portfolio", f"{total_pct:.2f}%")
        else:
            st.info("💰 Dividend reinvestment is enabled but no dividends were received during this period.")
            st.caption("This could mean:")
            st.caption("• The tickers in your portfolio don't pay dividends")
            st.caption("• The date range selected had no dividend payments")
            st.caption("• Dividend data was not available for the selected period")
    
    st.markdown("---")
    
    # Performance Summary Table
    st.header("📊 Performance Summary")
    st.caption(
        f"VIX column (when present): **close** of the CBOE Volatility Index from Yahoo Finance "
        f"(`{VIX_YAHOO_SYMBOL}`) on each backtest date — context only; not a holding and not used for regime/allocation."
    )
    perf_summary_chart = create_performance_summary_chart(equity_df)
    if perf_summary_chart is not None:
        st.plotly_chart(perf_summary_chart, use_container_width=True)
        st.caption(
            "Chart: VIX uses the same dates as the table below. "
            "Very short gaps (≤5 rows) are forward-filled **only in the chart** for readability. "
            "Sanity check: VIX typically spikes in **Mar 2020**, **2008–09**, and **early 2022** when those ranges are included."
        )
    elif "VIX" in equity_df.columns and equity_df["VIX"].notna().sum() == 0:
        st.warning(
            "**VIX column is empty** for this run (download failed, worst-case simulation path, or date misalignment). "
            "The chart requires non-null VIX closes aligned to backtest dates."
        )

    drawdown_ticker = config.get("drawdown_ticker", "QQQ") if config else "QQQ"
    ath_col = f"{drawdown_ticker}_ATH_raw"
    try:
        _nw = int(config.get("drawdown_window_years", 5)) if config else 5
    except (TypeError, ValueError):
        _nw = 5
    _window_on = bool(config.get("drawdown_window_enabled")) if config else False
    if _window_on and _nw > 0:
        _window_desc = f"{_nw}-year window"
    else:
        _window_desc = "full history"
    ath_label = f"{drawdown_ticker} - {_window_desc} ATH ($)"
    if ath_col in equity_df.columns:
        st.caption(
            f"{ath_label} is the reference peak used for regime drawdown that day. "
            f"{drawdown_ticker} close ($) is the adjusted close from your backtest download. "
            "Drawdown is (ATH minus close) divided by ATH. "
            "ATH should always be greater than or equal to close on the same row."
        )
    
    # Create a formatted display of key columns
    display_df = equity_df.copy()
    
    # Select key columns for display
    key_cols = ["Date", "Value", "Pct_Growth"]
    if ath_col in display_df.columns:
        key_cols.append(ath_col)
    px_col = f"{drawdown_ticker}_price"
    if px_col in display_df.columns and px_col not in key_cols:
        key_cols.append(px_col)
    
    # Add normalized benchmark columns
    norm_cols = [c for c in equity_df.columns if c.endswith("_norm")]
    key_cols.extend(norm_cols)
    
    # Add regime columns
    if "Market_Regime" in equity_df.columns:
        key_cols.append("Market_Regime")
    if "Portfolio_Regime" in equity_df.columns:
        key_cols.append("Portfolio_Regime")
    for _sov in (
        "Signal_override_active",
        "Signal_override_label",
        "Signal_override_allocation",
    ):
        if _sov in equity_df.columns and _sov not in key_cols:
            key_cols.append(_sov)
    if "Regime_Trajectory" in equity_df.columns:
        key_cols.append("Regime_Trajectory")
    if "Prev_Market_Regime" in equity_df.columns:
        key_cols.append("Prev_Market_Regime")
    
    # Add shares and values for allocation tickers
    if config and "allocation_tickers" in config:
        for ticker in config["allocation_tickers"]:
            if f"{ticker}_shares" in equity_df.columns:
                key_cols.append(f"{ticker}_shares")
            if f"{ticker}_value" in equity_df.columns:
                key_cols.append(f"{ticker}_value")

    # Signal layers + composite (same order as daily export columns at end)
    for c in PERF_SIGNAL_COLUMNS_ORDER:
        if c in equity_df.columns and c not in key_cols:
            key_cols.append(c)
    
    # Filter to only columns that exist
    display_cols = [c for c in key_cols if c in display_df.columns]
    display_df = display_df[display_cols].copy()
    
    # Format the dataframe for display
    def format_equity_curve(df):
        """Format equity curve dataframe for better display"""
        formatted_df = df.copy()
        
        # Format percentage columns
        if "Pct_Growth" in formatted_df.columns:
            formatted_df["Pct_Growth"] = formatted_df["Pct_Growth"].apply(
                lambda x: f"{x:.2%}" if not _perf_cell_empty(x) else ""
            )
        
        # Format dollar columns (incl. drawdown reference ATH and spot prices)
        dollar_cols = ["Value"] + [
            c for c in formatted_df.columns
            if c.endswith("_value")
            or c.endswith("_norm")
            or c.endswith("_ATH_raw")
            or c.endswith("_price")
        ]
        for col in dollar_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"${x:,.2f}" if not _perf_cell_empty(x) else ""
                )
        
        # Format share columns
        share_cols = [c for c in formatted_df.columns if c.endswith("_shares")]
        for col in share_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:,.4f}" if not _perf_cell_empty(x) and x != 0 else ""
                )

        if "VIX" in formatted_df.columns:
            formatted_df["VIX"] = formatted_df["VIX"].apply(
                lambda x: f"{x:.2f}" if not _perf_cell_empty(x) else ""
            )
        for c in ("VIX_252d_mean", "VIX_252d_stdev"):
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: f"{float(x):.3f}" if not _perf_cell_empty(x) else ""
                )
        for c in ("VIX_zscore", "VIX_zscore_direction"):
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: f"{float(x):.3f}" if not _perf_cell_empty(x) else ""
                )
        for c in list(formatted_df.columns):
            if c.startswith("MACD_"):
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: f"{float(x):.4f}" if not _perf_cell_empty(x) else ""
                )
        for c in ("MA_50", "MA_200"):
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: f"{float(x):.2f}" if not _perf_cell_empty(x) else ""
                )
        for c in ("Signal_L1", "Signal_L2", "Signal_L3", "Signal_total"):
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: f"{int(round(float(x)))}" if not _perf_cell_empty(x) else ""
                )
        for c in (
            "VIX_regime_label",
            "MA_regime_label",
            "Signal_label",
            "Signal_override_active",
            "Signal_override_label",
            "Signal_override_allocation",
        ):
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(
                    lambda x: "" if _perf_cell_empty(x) else str(x)
                )
        
        return formatted_df
    
    # Add date range filter
    if len(display_df) > 0:
        min_date = pd.to_datetime(display_df["Date"]).min()
        max_date = pd.to_datetime(display_df["Date"]).max()
        
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            start_filter = st.date_input(
                "Start Date",
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        with col_filter2:
            end_filter = st.date_input(
                "End Date",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        
        # Filter by date range
        display_df["Date"] = pd.to_datetime(display_df["Date"])
        filtered_df = display_df[
            (display_df["Date"] >= pd.Timestamp(start_filter)) &
            (display_df["Date"] <= pd.Timestamp(end_filter))
        ].copy()
        
        # Format for display
        formatted_df = format_equity_curve(filtered_df)
        rename_perf = {}
        if ath_col in formatted_df.columns:
            rename_perf[ath_col] = ath_label
        if px_col in formatted_df.columns:
            rename_perf[px_col] = f"{drawdown_ticker} close ($)"
        for col, title in PERF_SIGNAL_COLUMN_DISPLAY_NAMES.items():
            if col in formatted_df.columns:
                rename_perf[col] = title
        if rename_perf:
            formatted_df = formatted_df.rename(columns=rename_perf)
        
        with st.expander("📖 Performance column guide (VIX / MACD / MA / composite)", expanded=False):
            st.markdown(PERFORMANCE_SUMMARY_GUIDE)
        
        # Display table with pagination
        st.dataframe(
            formatted_df,
            use_container_width=True,
            height=520,
            hide_index=True
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(display_df)} rows")
    else:
        st.info("No performance data available")
    
    st.markdown("---")
    
    # Rebalance Events
    if quarterly_df is not None and not quarterly_df.empty:
        st.header("🔄 Rebalance Events")
        
        # Filter to rebalance events if column exists
        if "Rebalanced" in quarterly_df.columns:
            rebalance_df = quarterly_df[quarterly_df["Rebalanced"] == "Rebalanced"].copy()
        else:
            rebalance_df = quarterly_df.copy()
        
        if not rebalance_df.empty:
            # Format for display
            display_cols = ["Date", "Portfolio_Value", "Market_Regime", "Portfolio_Regime"]
            if "QoQ_Return" in rebalance_df.columns:
                display_cols.append("QoQ_Return")
            if "QoQ_Volatility" in rebalance_df.columns:
                display_cols.append("QoQ_Volatility")
            
            available_cols = [c for c in display_cols if c in rebalance_df.columns]
            st.dataframe(
                rebalance_df[available_cols].style.format({
                    "Portfolio_Value": "${:,.2f}",
                    "QoQ_Return": "{:.2%}",
                    "QoQ_Volatility": "{:.2%}"
                }),
                use_container_width=True,
                height=300
            )
        else:
            st.info("No rebalance events recorded")
    
    st.markdown("---")
    
    # Configuration Summary
    with st.expander("⚙️ Configuration Summary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Parameters")
            st.write(f"**Starting Balance:** ${config.get('starting_balance', 0):,.2f}")
            st.write(f"**Start Date:** {config.get('start_date', 'N/A')}")
            st.write(f"**End Date:** {config.get('end_date', 'Current')}")
            st.write(f"**Rebalance Frequency:** {config.get('rebalance_frequency', 'N/A')}")
            st.write(f"**Drawdown Ticker:** {config.get('drawdown_ticker', 'N/A')}")
            if config.get("drawdown_window_enabled"):
                st.write(
                    f"**Drawdown reference:** Rolling {int(config.get('drawdown_window_years', 5))}-year window"
                )
            else:
                st.write("**Drawdown reference:** Standard ATH")
        
        with col2:
            st.subheader("Regimes")
            alloc_tickers = config.get("allocation_tickers", [])
            for regime, params in config.get("regimes", {}).items():
                st.write(f"**{regime}:** {params.get('dd_low', 0):.0%} - {params.get('dd_high', 0):.0%} DD")
                alloc_str = ", ".join(
                    f"{t}: {float(params.get(t, 0)):.0%}"
                    for t in alloc_tickers
                    if t in params
                )
                reb_parts = []
                if params.get("rebalance_on_downward") is not None:
                    reb_parts.append(f"↓ {params['rebalance_on_downward']}")
                if params.get("rebalance_on_upward") is not None:
                    reb_parts.append(f"↑ {params['rebalance_on_upward']}")
                reb_suffix = f"  |  {', '.join(reb_parts)}" if reb_parts else ""
                st.write(f"  Allocation: {alloc_str}{reb_suffix}")
    
    # Data Export
    st.markdown("---")
    st.header("💾 Data Export")
    
    # Excel Export Button
    st.subheader("📊 Excel Export")
    
    def generate_excel_file():
        """Generate Excel file in memory"""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Use the exporter function to create Excel file
            from exporter import (
                create_formats, write_equity_sheet, write_chart_sheet,
                write_quarterly_sheet, write_parameters_sheet,
                write_regimes_sheet, write_results_sheet
            )
            
            # Create Excel writer
            writer = pd.ExcelWriter(
                tmp_filename,
                engine="xlsxwriter",
                engine_kwargs={"options": {"nan_inf_to_errors": True}},
            )
            workbook = writer.book
            
            formats = create_formats(workbook)
            
            # Write all sheets
            write_equity_sheet(writer, equity_df, formats, config["tickers"])
            write_chart_sheet(workbook, equity_df)
            write_quarterly_sheet(writer, quarterly_df, formats, config)
            write_parameters_sheet(writer, config)
            write_regimes_sheet(writer, config, formats)
            write_results_sheet(writer, equity_df, formats)
            
            writer.close()
            
            # Read file into memory
            with open(tmp_filename, 'rb') as f:
                file_data = f.read()
            
            return file_data
        finally:
            # Clean up temp file
            if os.path.exists(tmp_filename):
                try:
                    os.remove(tmp_filename)
                except:
                    pass
    
    # Generate and download Excel file
    if st.button("📊 Generate Excel Report", help="Click to generate the full Excel report (may take a few seconds)"):
        with st.spinner("Generating Excel report... This may take a few seconds."):
            try:
                excel_data = generate_excel_file()
                st.session_state['excel_data'] = excel_data
                st.session_state['excel_generated'] = True
                st.success("Excel report generated successfully!")
            except Exception as e:
                st.error(f"Error generating Excel file: {str(e)}")
                st.session_state['excel_generated'] = False
    
    # Show download button if Excel has been generated
    if st.session_state.get('excel_generated', False) and 'excel_data' in st.session_state:
        st.download_button(
            label="📥 Download Full Excel Report (backtest_results.xlsx)",
            data=st.session_state['excel_data'],
            file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads the complete Excel report with all sheets: Equity Curve, Chart, Rebalance Summary, Parameters, Regimes, and Results"
        )
    elif not st.session_state.get('excel_generated', False):
        st.info("💡 Click 'Generate Excel Report' above to create the full Excel file with all sheets.")
    
    st.markdown("---")
    st.subheader("📄 CSV Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Equity Data (CSV)",
            data=equity_df.to_csv(index=False).encode('utf-8'),
            file_name=f"equity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if quarterly_df is not None and not quarterly_df.empty:
            st.download_button(
                label="📥 Download Rebalance Data (CSV)",
                data=quarterly_df.to_csv(index=False).encode('utf-8'),
                file_name=f"rebalance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


# ======================================================================
# STANDALONE MODE (for backward compatibility)
# ======================================================================
if __name__ == "__main__":
    st.info("""
    This dashboard is now integrated into the main app.
    
    To use:
    1. Run: `streamlit run app.py`
    2. Configure parameters in the Configuration page
    3. Click "Run Backtest"
    4. View results in the Results page
    """)

