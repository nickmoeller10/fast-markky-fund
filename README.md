# Fast Markky Fund

A tactical portfolio backtesting engine for a QQQ–TQQQ–XLU strategy that dynamically adjusts allocations based on market drawdown regimes.

## 📊 Overview

Fast Markky Fund implements a regime-based tactical asset allocation strategy that switches between aggressive (TQQQ) and defensive (XLU) positions based on market drawdowns from all-time highs. The system includes comprehensive backtesting capabilities, worst-case scenario simulation, and an interactive web dashboard for results visualization.

## 🎯 Strategy

### Regime-Based Allocation

The strategy uses three market regimes based on QQQ drawdown from a rolling 1-year all-time high.

> **Note:** the allocations and thresholds below are the current **defaults**. They are sensible starting values, not tuned production weights, and are subject to change as the strategy is iterated. See `config.py` to override.

- **R1 — Ride High** (0–6% drawdown): **100% TQQQ**
  - Aggressive leveraged position during market strength
  - Signal overrides:
    - *Max Bull* (signal > +2): hold 100% TQQQ
    - *Bull Fading* (signal < −2): de-leverage to 100% QQQ

- **R2 — Cautious Defense** (6–20% drawdown): **70% XLU + 30% QQQ**
  - Mostly defensive but retains some equity exposure
  - Signal overrides:
    - *Recovery Confirmed* (signal > +2): rotate to 70% QQQ + 30% XLU
    - *Deteriorating Fast* (signal < −3): full defense — 100% XLU

- **R3 — Contrarian Buyback** (20%+ drawdown): **50% TQQQ + 50% QQQ**
  - Aggressive re-entry — deep drawdowns historically precede strong recoveries
  - Signal overrides:
    - *Capitulation Reversal* (signal > +3): lean harder — 80% TQQQ + 20% QQQ
    - *Crisis Deepening* (signal < −4): pull back to 100% XLU

### Composite Signal

The signal total ranges over `[−6, +6]` and combines:
- **L1** — VIX z-score (rolling 252-day mean ± 1σ buckets)
- **L2** — SPY MACD (12/26/9)
- **L3** — SPY MA50 vs MA200 crossover

### Asymmetric Regime Rules

The default `per_regime` rebalance strategy lets each regime declare:
- `rebalance_on_downward` — should the portfolio follow when the market gets worse and arrives here?
- `rebalance_on_upward` — should the portfolio follow when the market improves and arrives here?

Both default to `match` (always follow). Set either to `hold` to delay rebalancing into a regime — useful when you want the portfolio to skip a partial recovery and only rebalance on a full one.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fast-markky-fund
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

**New UI-Driven Approach (Recommended):**

1. Launch the Streamlit application:
```bash
streamlit run app.py
```

2. The app will open in your browser automatically

3. Configure your parameters in the **Configuration** page:
   - Set starting balance, start/end dates
   - Choose rebalance frequency
   - Configure tickers and regimes
   - Adjust allocation percentages

4. Click **🚀 Run Backtest** to execute

5. View results in the **Results** page with interactive charts

6. Modify parameters and re-run anytime without restarting

**Legacy Console Mode (Optional):**

If you prefer the console interface:
```bash
python main.py
```

## 📈 Features

### Core Capabilities

- **Regime Detection**: Automatic market regime classification based on drawdown
- **Flexible Rebalancing**: Instant, weekly, monthly, quarterly, or custom frequencies
- **Worst-Case Simulation**: Extended historical testing using synthetic data (back to 1950)
- **Excel Export**: Comprehensive multi-sheet reports with charts
- **Interactive Dashboard**: Modern web-based visualization (see below)

### Interactive Dashboard

The dashboard provides:

- **Key Metrics**: Final value, CAGR, Sharpe ratio, max drawdown, volatility
- **Equity Curve**: Interactive chart comparing portfolio vs normalized benchmarks
- **Drawdown Analysis**: Dual charts for portfolio and market drawdowns
- **Regime Timeline**: Visual representation of regime changes with rebalance markers
- **Allocation Chart**: Stacked area chart showing portfolio allocation over time
- **Rebalance Events**: Detailed table of all rebalance transactions
- **Configuration Summary**: Complete parameter and regime documentation
- **Data Export**: CSV download buttons

**Launch Dashboard:**
```bash
# After running a backtest
streamlit run dashboard_runner.py
```

Or select "View interactive dashboard" when prompted after running `python main.py`.

## ⚙️ Configuration

Edit `config.py` to customize:

### Portfolio Parameters
```python
"starting_balance": 10000,
"start_date": "2010-12-02",
"end_date": None,  # None = current date
"drawdown_ticker": "QQQ",
```

### Rebalancing
```python
"rebalance_frequency": "instant",  # Options: instant, weekly, monthly, quarterly, etc.
```

### Regimes
```python
"regimes": {
    "R1": {
        "dd_low": 0.00,
        "dd_high": 0.06,
        "TQQQ": 1.00,
        "QQQ": 0.00,
        "XLU": 0.00,
    },
    # ... more regimes
}
```

## 📁 Project Structure

```
fast-markky-fund/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── backtest.py             # Core backtesting engine
├── regime_engine.py        # Regime detection logic
├── allocation_engine.py    # Allocation calculations
├── rebalance_engine.py    # Rebalancing execution
├── data_loader.py          # Price data loading (yfinance)
├── dashboard.py            # Interactive dashboard
├── dashboard_runner.py     # Dashboard launcher
├── exporter.py             # Excel export functionality
├── worst_case_simulator.py # Synthetic data generation
├── worst_case_runner.py    # Worst-case simulation runner
├── console_ui.py           # CLI user interface
├── utils.py                # Utility functions
└── requirements.txt        # Python dependencies
```

## 🔧 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **yfinance**: Financial data retrieval
- **streamlit**: Web dashboard framework
- **plotly**: Interactive charts
- **xlsxwriter**: Excel export (optional)

See `requirements.txt` for complete list and versions.

## 📊 Output Formats

### Excel Export
Multi-sheet workbook with:
- Equity Curve (daily data)
- Chart (portfolio vs benchmarks)
- Rebalance Summary
- Parameters
- Regimes
- Results (final metrics)

### Interactive Dashboard
Web-based visualization with:
- Interactive charts (zoom, pan, hover)
- Real-time data exploration
- CSV export capabilities
- Modern, responsive design

## 🧪 Worst-Case Simulation

Test your strategy against extended historical periods:

1. Enable in config: `"use_worst_case_simulation": True`
2. Run backtest and select worst-case simulation option
3. System generates synthetic QQQ/TQQQ/XLU data back to 1950
4. Uses S&P 500 volatility-adjusted returns
5. Splices real ETF data after inception dates

## 📝 Example Workflow

```bash
# 1. Configure strategy in config.py
# 2. Run backtest
python main.py

# 3. Review results in terminal
# 4. Export to Excel (optional)
# 5. Launch dashboard
streamlit run dashboard_runner.py

# 6. Explore interactive charts
# 7. Download CSV data if needed
```

## 🎓 Understanding the Results

### Key Metrics Explained

- **CAGR**: Compound Annual Growth Rate - annualized return
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Volatility**: Annualized standard deviation of returns

### Regime Behavior

- **R1 → R2**: Immediate defensive switch (protect capital)
- **R2 → R3**: Aggressive re-entry (buy the dip)
- **R3 → R1**: Only after full recovery (avoid false signals)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional regime strategies
- More sophisticated rebalancing logic
- Enhanced visualization features
- Performance optimizations

## 📄 License

[Add your license here]

## ⚠️ Disclaimer

This is a backtesting tool for educational and research purposes. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

---

**Built with Python, Streamlit, and Plotly** 📈
