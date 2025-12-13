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
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
from exporter import export_to_excel


# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="Fast Markky Fund - Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def calculate_metrics(equity_df, config=None):
    """Calculate key performance metrics"""
    start_val = equity_df["Value"].iloc[0]
    end_val = equity_df["Value"].iloc[-1]
    start_date = pd.to_datetime(equity_df["Date"].iloc[0])
    end_date = pd.to_datetime(equity_df["Date"].iloc[-1])
    
    years = (end_date - start_date).days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
    
    total_return = (end_val / start_val) - 1
    
    # Calculate max drawdown
    equity_series = equity_df["Value"]
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate volatility (annualized)
    returns = equity_df["Value"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe = (cagr / volatility) if volatility > 0 else 0
    
    # Calculate Sortino ratio (downside deviation only)
    # Downside deviation: square root of mean of squared negative returns
    # Target return is 0 (no risk-free rate assumption)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        # Calculate downside deviation: sqrt(mean(squared negative returns))
        downside_variance = (downside_returns ** 2).mean()
        downside_deviation = np.sqrt(downside_variance) * np.sqrt(252)  # Annualized
        sortino = (cagr / downside_deviation) if downside_deviation > 0 else 0
    else:
        # No negative returns - Sortino is undefined (infinite or very high)
        sortino = float('inf') if cagr > 0 else 0
    
    # Calculate Beta (portfolio sensitivity to market)
    # Beta = Covariance(portfolio returns, benchmark returns) / Variance(benchmark returns)
    # Try SPY first, then QQQ, then first available normalized benchmark
    beta = None
    beta_benchmark = None
    
    # Priority: SPY > QQQ > first available normalized benchmark
    benchmark_candidates = ["SPY_norm", "QQQ_norm"]
    if config and "tickers" in config:
        # Add other tickers as potential benchmarks
        for ticker in config["tickers"]:
            if ticker not in ["SPY", "QQQ"]:
                benchmark_candidates.append(f"{ticker}_norm")
    
    # Calculate portfolio returns once
    portfolio_returns = equity_df["Value"].pct_change().dropna()
    
    for bench_col in benchmark_candidates:
        if bench_col in equity_df.columns:
            benchmark_values = equity_df[bench_col].dropna()
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_values.pct_change().dropna()
            
            # Align returns by index (only use dates where both have data)
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(common_index) > 30:  # Need at least 30 data points for meaningful beta
                portfolio_ret_aligned = portfolio_returns.loc[common_index]
                benchmark_ret_aligned = benchmark_returns.loc[common_index]
                
                # Remove any remaining NaN values
                valid_mask = portfolio_ret_aligned.notna() & benchmark_ret_aligned.notna()
                portfolio_ret_clean = portfolio_ret_aligned[valid_mask]
                benchmark_ret_clean = benchmark_ret_aligned[valid_mask]
                
                if len(portfolio_ret_clean) > 30:
                    # Calculate beta: Cov(portfolio, market) / Var(market)
                    covariance = np.cov(portfolio_ret_clean, benchmark_ret_clean)[0, 1]
                    benchmark_variance = np.var(benchmark_ret_clean)
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        beta_benchmark = bench_col.replace("_norm", "")
                        break
    
    # If no beta calculated, set to 0
    if beta is None:
        beta = 0.0
        beta_benchmark = "N/A"
    
    return {
        "start_value": start_val,
        "end_value": end_val,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "beta": beta,
        "beta_benchmark": beta_benchmark,
        "years": years,
        "start_date": start_date,
        "end_date": end_date
    }


def create_equity_curve_chart(equity_df, config):
    """Create interactive equity curve with benchmarks"""
    fig = go.Figure()
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Value"],
        name="Portfolio Value",
        line=dict(color="#003366", width=2),
        hovertemplate="<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
    ))
    
    # Normalized benchmarks
    norm_cols = [c for c in equity_df.columns if c.endswith("_norm")]
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(norm_cols):
        ticker = col.replace("_norm", "")
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=equity_df[col],
            name=f"{ticker} (Normalized)",
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Portfolio Value vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    return fig


def create_drawdown_chart(equity_df, drawdown_ticker):
    """Create drawdown visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Portfolio Drawdown", f"{drawdown_ticker} Drawdown"),
        row_heights=[0.5, 0.5]
    )
    
    # Portfolio drawdown
    if "Portfolio_DD" in equity_df.columns:
        portfolio_dd = equity_df["Portfolio_DD"] * 100
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=portfolio_dd,
            name="Portfolio DD",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red", width=1),
            hovertemplate="Portfolio DD: %{y:.2f}%<extra></extra>"
        ), row=1, col=1)
    
    # Market drawdown
    dd_col = f"{drawdown_ticker}_DD_raw"
    if dd_col in equity_df.columns:
        market_dd = equity_df[dd_col] * 100
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=market_dd,
            name=f"{drawdown_ticker} DD",
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.3)",
            line=dict(color="orange", width=1),
            hovertemplate=f"{drawdown_ticker} DD: %{{y:.2f}}%<extra></extra>"
        ), row=2, col=1)
    
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode="x unified",
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def create_regime_timeline(equity_df):
    """Create regime timeline visualization"""
    if "Portfolio_Regime" not in equity_df.columns:
        return None
    
    # Create regime color mapping
    regime_colors = {
        "R1": "#00CC00",  # Green
        "R2": "#FF9900",  # Orange
        "R3": "#CC0000",  # Red
    }
    
    # Create regime segments
    equity_df = equity_df.copy()
    equity_df["Regime_Color"] = equity_df["Portfolio_Regime"].map(regime_colors)
    
    fig = go.Figure()
    
    # Plot portfolio value with regime-colored background
    for regime in equity_df["Portfolio_Regime"].unique():
        if pd.isna(regime):
            continue
        regime_data = equity_df[equity_df["Portfolio_Regime"] == regime]
        fig.add_trace(go.Scatter(
            x=regime_data["Date"],
            y=regime_data["Value"],
            name=f"Regime {regime}",
            line=dict(color=regime_colors.get(regime, "#666"), width=2),
            mode="lines",
            hovertemplate=f"<b>{regime}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>"
        ))
    
    # Add rebalance markers
    rebalance_dates = equity_df[equity_df["Rebalanced"] == "Rebalanced"]
    if not rebalance_dates.empty:
        fig.add_trace(go.Scatter(
            x=rebalance_dates["Date"],
            y=rebalance_dates["Value"],
            mode="markers",
            name="Rebalance",
            marker=dict(symbol="diamond", size=10, color="black", line=dict(width=1, color="white")),
            hovertemplate="<b>Rebalance</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Portfolio Value with Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_allocation_chart(equity_df, config):
    """Create portfolio allocation over time"""
    tickers = config["allocation_tickers"]
    
    # Calculate allocation percentages
    allocation_data = []
    for ticker in tickers:
        value_col = f"{ticker}_value"
        if value_col in equity_df.columns:
            allocation_data.append({
                "Date": equity_df["Date"],
                "Ticker": ticker,
                "Allocation": equity_df[value_col] / equity_df["Value"] * 100
            })
    
    if not allocation_data:
        return None
    
    df_alloc = pd.concat([pd.DataFrame(d) for d in allocation_data])
    
    fig = go.Figure()
    
    colors = {"QQQ": "#1f77b4", "TQQQ": "#ff7f0e", "XLU": "#2ca02c"}
    
    for ticker in tickers:
        ticker_data = df_alloc[df_alloc["Ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_data["Date"],
            y=ticker_data["Allocation"],
            name=ticker,
            stackgroup="one",
            fillcolor=colors.get(ticker, "#999"),
            line=dict(width=0),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Allocation: %{{y:.1f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="Portfolio Allocation Over Time",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        yaxis=dict(range=[0, 100])
    )
    
    return fig


# ======================================================================
# MAIN DASHBOARD FUNCTION
# ======================================================================
def render_dashboard(equity_df, quarterly_df, config):
    """Render the complete dashboard"""
    
    # Title and header
    st.title("📈 Fast Markky Fund - Backtest Dashboard")
    st.markdown("---")
    
    # Calculate metrics
    metrics = calculate_metrics(equity_df, config)
    
    # Key Metrics Row
    st.header("📊 Key Performance Metrics")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        st.metric(
            "Final Value",
            f"${metrics['end_value']:,.2f}",
            f"${metrics['end_value'] - metrics['start_value']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2%}",
        )
    
    with col3:
        st.metric(
            "CAGR",
            f"{metrics['cagr']:.2%}",
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2%}",
        )
    
    with col5:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
        )
    
    with col6:
        sortino_display = "∞" if metrics['sortino_ratio'] == float('inf') else f"{metrics['sortino_ratio']:.2f}"
        st.metric(
            "Sortino Ratio",
            sortino_display,
        )
    
    with col7:
        beta_label = f"Beta vs {metrics['beta_benchmark']}" if metrics['beta_benchmark'] != "N/A" else "Beta"
        st.metric(
            beta_label,
            f"{metrics['beta']:.2f}",
        )
    
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
    
    # Equity Curve Table
    st.header("📊 Equity Curve Data")
    
    # Create a formatted display of key columns
    display_df = equity_df.copy()
    
    # Select key columns for display
    key_cols = ["Date", "Value", "Pct_Growth"]
    
    # Add normalized benchmark columns
    norm_cols = [c for c in equity_df.columns if c.endswith("_norm")]
    key_cols.extend(norm_cols)
    
    # Add regime columns
    if "Market_Regime" in equity_df.columns:
        key_cols.append("Market_Regime")
    if "Portfolio_Regime" in equity_df.columns:
        key_cols.append("Portfolio_Regime")
    
    # Add shares and values for allocation tickers
    if config and "allocation_tickers" in config:
        for ticker in config["allocation_tickers"]:
            if f"{ticker}_shares" in equity_df.columns:
                key_cols.append(f"{ticker}_shares")
            if f"{ticker}_value" in equity_df.columns:
                key_cols.append(f"{ticker}_value")
    
    # Filter to only columns that exist
    display_cols = [c for c in key_cols if c in display_df.columns]
    display_df = display_df[display_cols].copy()
    
    # Format the dataframe for display
    def format_equity_curve(df):
        """Format equity curve dataframe for better display"""
        formatted_df = df.copy()
        
        # Format percentage columns
        if "Pct_Growth" in formatted_df.columns:
            formatted_df["Pct_Growth"] = formatted_df["Pct_Growth"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        
        # Format dollar columns
        dollar_cols = ["Value"] + [c for c in formatted_df.columns if c.endswith("_value") or c.endswith("_norm")]
        for col in dollar_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else ""
                )
        
        # Format share columns
        share_cols = [c for c in formatted_df.columns if c.endswith("_shares")]
        for col in share_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:,.4f}" if pd.notna(x) and x != 0 else ""
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
        
        # Display table with pagination
        st.dataframe(
            formatted_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(display_df)} rows")
    else:
        st.info("No equity curve data available")
    
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
        
        with col2:
            st.subheader("Regimes")
            for regime, params in config.get("regimes", {}).items():
                st.write(f"**{regime}:** {params.get('dd_low', 0):.0%} - {params.get('dd_high', 0):.0%} DD")
                alloc_str = ", ".join([f"{k}: {v:.0%}" for k, v in params.items() if k not in ['dd_low', 'dd_high']])
                st.write(f"  Allocation: {alloc_str}")
    
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
# STANDALONE MODE
# ======================================================================
if __name__ == "__main__":
    st.error("""
    This dashboard is designed to be called from the main application.
    
    To use the dashboard:
    1. Run your backtest as normal
    2. Select 'View Dashboard' when prompted
    3. Or call render_dashboard(equity_df, quarterly_df, config) from your code
    """)

