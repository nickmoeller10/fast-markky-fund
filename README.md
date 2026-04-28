# Fast Markky Fund

A tactical portfolio backtesting engine for a QQQ–TQQQ–XLU strategy that dynamically adjusts allocations based on market drawdown regimes.

## 📊 Overview

Fast Markky Fund implements a regime-based tactical asset allocation strategy that switches between aggressive (TQQQ) and defensive (XLU) positions based on market drawdowns from all-time highs. The system includes comprehensive backtesting capabilities, worst-case scenario simulation, and an interactive web dashboard for results visualization.

## 🎯 Strategy

### Regime-Based Allocation

The strategy uses three market regimes based on QQQ drawdown from a rolling 3-year all-time high.

> **Note:** the allocations and thresholds below come from **iter 25** of the iterative config search — the current all-time score champion (0.88) with median CAGR 28.61% and worst max DD −29.80% across 11 Monte Carlo entry points (1999-04 through 2020-07). They are the current best-known baseline, not arbitrary defaults — but iteration continues. See `config.py` to override and `docs/superpowers/methodologies/iterative-config-search.md` for the active hypothesis ladder.

- **R1 — Ride High** (0–11.24% drawdown): **81% TQQQ + 7% QQQ + 12% XLU**
  - TQQQ-heavy leveraged position during calm market — max compounding
  - Signal overrides:
    - *Strong Bull* (signal ≥ +1): rotate to 8% TQQQ + 47% QQQ + 45% XLU
    - *Bull Fading* (signal ≤ −2): de-leverage to 19% QQQ + 12% XLU (cash-heavy)

- **R2 — Passthrough Cash Buffer** (11.24–19.61% drawdown): **89% CASH + 6% QQQ + 5% XLU**
  - Cash-dominant transition band; `hold/hold` rebalance behavior means the strategy doesn't churn while crossing this band — once entered from above (R1 → R2) it's effectively neutral until R3 fires
  - Signal overrides:
    - *Recovery Confirmed* (signal ≥ +4): rotate to 30% TQQQ + 27% QQQ + 16% XLU
    - *Deteriorating* (signal ≤ −3): rotate to 11% QQQ + 48% XLU

- **R3 — Absolute Defense** (19.61%+ drawdown): **100% CASH**
  - Deep crisis stops the bleed — the structural breakthrough that prevents XLU's own −39% 2008 drawdown from leaking through
  - Signal overrides:
    - *Capitulation Reversal* (signal ≥ +1): rotate to 40% TQQQ + 30% QQQ + 19% XLU
    - *Crisis Deepening* (signal ≤ −4): rotate to 24% QQQ + 75% XLU

**CASH** is a synthetic risk-free sleeve (zero drawdown, ~4% APY proxy) — not the yfinance "CASH" ticker (Pathward Financial). Both `"CASH"` and `"$"` are recognized aliases. See `data_loader.SYNTHETIC_TICKERS`.

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
"start_date": "1999-01-04",
"end_date": "2026-03-27",
"drawdown_ticker": "QQQ",
"drawdown_window_enabled": True,
"drawdown_window_years": 3,  # rolling 3y peak (iter 25 finding: dominates 1y/2y for DD control)
```

### Rebalancing
```python
"rebalance_frequency": "instant",  # Options: instant, weekly, monthly, quarterly, etc.
"rebalance_strategy": "per_regime",  # each regime declares hold/match per direction
```

### Regimes (iter 25 champion)
```python
"regimes": {
    "R1": {
        "dd_low": 0.00,
        "dd_high": 0.1124,
        "TQQQ": 0.8112,
        "QQQ": 0.0657,
        "XLU": 0.1231,
        "CASH": 0.0,
        "rebalance_on_downward": "match",
        "rebalance_on_upward": "match",
    },
    "R2": {
        "dd_low": 0.1124,
        "dd_high": 0.1961,
        "TQQQ": 0.0,
        "QQQ": 0.0645,
        "XLU": 0.0462,
        "CASH": 0.8894,
        "rebalance_on_downward": "hold",  # passthrough — don't churn entering R2
        "rebalance_on_upward": "hold",
    },
    "R3": {
        "dd_low": 0.1961,
        "dd_high": 1.00,
        "TQQQ": 0.0,
        "QQQ": 0.0,
        "XLU": 0.0,
        "CASH": 1.0,
        "rebalance_on_downward": "match",
        "rebalance_on_upward": "hold",
    },
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

### Regime Behavior (iter 25 champion)

- **R1 → R2** (downward): R2 is `hold` on downward — portfolio stays in R1 weights as drawdown crosses 11.24%. Avoids churning the cash-buffer transition band.
- **R2 → R3** (downward): R3 is `match` on downward — strategy snaps to 100% CASH. Stops the bleed before crisis deepens.
- **R3 → R2** (upward): R2 is `hold` on upward — portfolio stays in CASH through the partial recovery, doesn't re-risk too early.
- **R2 → R1** (upward): R1 is `match` on upward — full re-entry into TQQQ-heavy weights once a full recovery is confirmed.

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
