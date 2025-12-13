# Fast Markky Fund - Application Guide

## Overview

The Fast Markky Fund now features a fully UI-driven interface built with Streamlit. All configuration is done through a clean web interface - no need to edit config files!

## Running the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## Application Structure

### Two Main Pages

1. **⚙️ Configuration Page** - Set up your backtest parameters
2. **📊 Results Page** - View interactive charts and analysis

### Navigation

- Use the sidebar radio buttons to switch between pages
- Quick stats appear in the sidebar when results are available

## Configuration Page

### 📊 Portfolio Parameters
- **Starting Balance**: Initial portfolio value
- **Start Date**: Backtest start date
- **End Date**: Optional end date (uncheck for current date)

### 🔄 Rebalancing Settings
- **Rebalance Frequency**: Choose from instant, daily, weekly, monthly, quarterly, etc.
- **Drawdown Ticker**: Ticker used to measure market drawdown

### 📈 Ticker Configuration
- **Available Tickers**: Tickers to download and display (comma-separated)
- **Allocation Tickers**: Tickers that can be held in portfolio (must be subset of available)

### 🎯 Regime Definitions
Configure each regime (R1, R2, R3):
- **Drawdown Range**: Low and high thresholds (as percentages)
- **Allocations**: Percentage allocation for each ticker (must sum to 100%)

### 🔧 Advanced Settings
- Minimum allocation guardrail
- Holiday handling rules

### Running the Backtest

1. Configure all parameters
2. Click **🚀 Run Backtest** button
3. Watch progress bar as data loads and backtest runs
4. Automatically switches to Results page when complete

## Results Page

### Features

- **Key Performance Metrics**: Final value, CAGR, Sharpe, Sortino, Beta
- **Interactive Charts**: 
  - Equity curve vs benchmarks
  - Drawdown analysis
  - Regime timeline
  - Portfolio allocation over time
- **Equity Curve Data Table**: Formatted table with date filtering
- **Rebalance Events**: Table of all rebalancing transactions
- **Excel Export**: Generate full Excel report with all sheets
- **CSV Downloads**: Download equity and rebalance data

### Navigation

- **⚙️ Edit Config**: Return to configuration page
- **🔄 Re-run**: Run backtest again with current settings

## Key Features

### Session State
- All configuration is stored in Streamlit's session state
- Changes persist as you navigate between pages
- No need to save/load config files

### Real-time Validation
- Configuration errors are shown immediately
- Allocation totals are validated (must sum to 100%)
- Ticker validation ensures consistency

### Progress Tracking
- Progress bar shows backtest execution steps
- Status messages indicate current operation
- Automatic page switching on completion

## Workflow

1. **Launch**: `streamlit run app.py`
2. **Configure**: Set parameters in Configuration page
3. **Run**: Click "Run Backtest"
4. **Analyze**: View results in Results page
5. **Iterate**: Change parameters and re-run anytime

## Tips

- **Quick Iteration**: Change one parameter and click "Re-run" to see new results instantly
- **Date Ranges**: Use fixed end dates for reproducible results
- **Regime Testing**: Adjust drawdown thresholds to see how strategy performs
- **Allocation Testing**: Try different allocation percentages to optimize returns

## Troubleshooting

**App won't launch:**
- Ensure Streamlit is installed: `pip install streamlit`
- Check port 8501 is available

**Backtest fails:**
- Check all regime allocations sum to 100%
- Ensure allocation tickers are in available tickers list
- Verify date range is valid

**Charts not displaying:**
- Ensure Plotly is installed: `pip install plotly`
- Check browser console for errors

---

**Enjoy the new UI-driven experience!** 🚀

