# Quick Start Guide

## Step 1: Install Dependencies

First, make sure you have Python 3.7+ installed, then install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy yfinance streamlit plotly xlsxwriter openpyxl
```

## Step 2: Configure (Optional)

Edit `config.py` if you want to customize:
- Start/end dates
- Starting balance
- Regime thresholds
- Rebalance frequency

**Default settings work fine for a first run!**

## Step 3: Run the Program

Simply execute:

```bash
python main.py
```

## What Happens Next

The program will guide you through an interactive session:

1. **Configuration Review**
   - Displays your current settings
   - Asks: "Do these parameters look good? (Y/N):"
   - Answer `Y` to proceed or `N` to exit and edit config

2. **Backtest Confirmation**
   - Asks: "Run backtest? (Y/N):"
   - Answer `Y` to start

3. **Data Loading**
   - Downloads price data from Yahoo Finance
   - This may take 10-30 seconds depending on date range

4. **Backtest Execution**
   - Runs the strategy simulation
   - Shows progress in terminal

5. **Results Display**
   - Shows final portfolio value
   - Displays suggested current allocation

6. **Excel Export (Optional)**
   - Asks: "Export results to Excel? (Y/N):"
   - Answer `Y` to create Excel file

7. **Interactive Dashboard (Recommended)**
   - Asks: "View interactive dashboard? (Y/N):"
   - Answer `Y` to launch web dashboard
   - Then choose to launch now or later

8. **Worst-Case Simulation (Optional)**
   - Asks: "Run worst-case historical simulation? (Y/N):"
   - Answer `Y` for extended historical testing

## Example Session

```
$ python main.py

Welcome to the Tactical QQQ–TQQQ–XLU Portfolio Engine.

Here are your current parameters:
starting_balance: 10000
start_date: 2010-12-02
end_date: 2021-12-31
...

Do these parameters look good? (Y/N): Y
Run backtest? (Y/N): Y

Loading price data...
Downloading price data for: ['QQQ', 'TQQQ', 'XLU', 'SPY'], starting 2010-12-02
Download complete. Shape: (2785, 4)

Running backtest...
Backtest complete!
Final portfolio value: 125432.50

Suggested current allocation based on regime:
{'TQQQ': 1.0, 'QQQ': 0.0, 'XLU': 0.0}

Export results to Excel? (Y/N): Y
Exporting results...
Excel export complete!

View interactive dashboard? (Y/N): Y
Preparing dashboard data...
Data saved to dashboard_data.pkl

To view the dashboard, run:
  streamlit run dashboard_runner.py

Or launch it now? (Y/N): Y

Launching dashboard in your browser...
```

## Viewing the Dashboard

### Option A: Launch from Main Program
- Answer `Y` when asked "View interactive dashboard?"
- Then answer `Y` to "Or launch it now?"
- Dashboard opens automatically in your browser

### Option B: Launch Manually
After running the backtest, the program creates `dashboard_data.pkl`. Then run:

```bash
streamlit run dashboard_runner.py
```

The dashboard will open at `http://localhost:8501`

**To stop the dashboard:** Press `Ctrl+C` in the terminal

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "No data available" in dashboard
- Make sure you ran the backtest first
- Check that `dashboard_data.pkl` exists in the project directory

### yfinance download fails
- Check your internet connection
- Try again (Yahoo Finance can be intermittent)
- Verify the ticker symbols in `config.py` are correct

### Dashboard won't launch
```bash
# Make sure Streamlit is installed
pip install streamlit

# Try running directly
streamlit run dashboard_runner.py
```

## Quick Commands Reference

```bash
# Run backtest
python main.py

# View dashboard (after running backtest)
streamlit run dashboard_runner.py

# Install dependencies
pip install -r requirements.txt

# Check Python version (need 3.7+)
python --version
```

## Next Steps

- Experiment with different regime thresholds in `config.py`
- Try different rebalance frequencies
- Run worst-case simulation for extended testing
- Explore the interactive dashboard charts
- Export data to CSV from the dashboard

---

**That's it! You're ready to backtest your strategy.** 🚀

