# Interactive Dashboard Guide

## Overview

The Fast Markky Fund now includes a modern, interactive web-based dashboard built with **Streamlit** and **Plotly** that replaces (or complements) the Excel export with beautiful, interactive visualizations.

## Features

The dashboard includes:

1. **Key Performance Metrics** - Final value, total return, CAGR, max drawdown, Sharpe ratio
2. **Interactive Equity Curve** - Portfolio value vs normalized benchmarks with hover tooltips
3. **Drawdown Analysis** - Dual charts showing portfolio and market drawdowns
4. **Regime Timeline** - Visual representation of regime changes with rebalance markers
5. **Portfolio Allocation** - Stacked area chart showing allocation percentages over time
6. **Rebalance Events Table** - Detailed table of all rebalance events
7. **Configuration Summary** - Expandable section showing all parameters and regime definitions
8. **Data Export** - Download buttons for CSV exports

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install Streamlit and Plotly directly:

```bash
pip install streamlit plotly
```

## Usage

### Option 1: Launch from Main Script

1. Run your backtest as normal:
   ```bash
   python main.py
   ```

2. When prompted "View interactive dashboard? (Y/N):", answer **Y**

3. Choose to launch immediately or run manually later

### Option 2: Manual Launch

1. Run your backtest first (this creates `dashboard_data.pkl`)

2. Launch the dashboard:
   ```bash
   streamlit run dashboard_runner.py
   ```

3. The dashboard will open automatically in your default web browser

4. Press `Ctrl+C` in the terminal to stop the server

## Dashboard Sections

### Key Metrics
- **Final Value**: Ending portfolio value with change from start
- **Total Return**: Overall percentage return
- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Worst peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric

### Charts
All charts are interactive:
- **Hover** over data points to see detailed values
- **Zoom** by clicking and dragging
- **Pan** by clicking and dragging when zoomed
- **Reset** by double-clicking
- **Toggle** series by clicking legend items

### Data Tables
- Rebalance events are displayed in an interactive table
- Data can be sorted and filtered
- CSV downloads available for all data

## Advantages Over Excel

1. **Interactive Charts** - Zoom, pan, hover for detailed exploration
2. **Real-time Updates** - No need to regenerate files
3. **Better Visualizations** - Modern, clean design with Plotly
4. **Accessibility** - View from any device with a web browser
5. **No File Management** - No need to manage multiple Excel files
6. **Shareable** - Easy to share via Streamlit Cloud or local network

## Technical Details

- Built with **Streamlit** for the web framework
- **Plotly** for interactive charts
- Data is temporarily stored in `dashboard_data.pkl` (auto-cleaned)
- Dashboard runner is a standalone script for easy execution

## Troubleshooting

**Dashboard won't launch:**
- Ensure Streamlit is installed: `pip install streamlit`
- Check that `dashboard_data.pkl` exists (run backtest first)

**Charts not displaying:**
- Verify Plotly is installed: `pip install plotly`
- Check browser console for errors

**Data not loading:**
- Run the backtest first to generate `dashboard_data.pkl`
- Ensure you're in the project root directory

## Future Enhancements

Potential additions:
- Comparison mode (multiple backtests side-by-side)
- Custom date range filtering
- Export charts as images
- Performance attribution analysis
- Monte Carlo simulation results

