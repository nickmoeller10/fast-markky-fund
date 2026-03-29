# main.py

from config import CONFIG
from console_ui import *
from data_loader import (
    load_price_data,
    attach_vix_to_equity_df,
    fetch_vix_series_for_equity_dates,
)
from signal_layers import compute_signal_layer_columns, reorder_signal_override_columns_after_signals
from regime_engine import compute_drawdown_from_ath, determine_regime
from allocation_engine import get_allocation_for_regime
from rebalance_engine import rebalance_portfolio
from backtest import run_backtest
from exporter import export_to_excel
from worst_case_runner import run_worst_case_simulation
import config
import pandas as pd


def run():
    print("\nWelcome to the Tactical QQQ–TQQQ–XLU Portfolio Engine.\n")

    print_config()

    if not confirm_parameters():
        print("No worries! Please update your config and run again.")
        return

    if not confirm_backtest():
        print("Exiting program. Run again anytime.")
        return

    print("\nLoading price data...\n")
    price_data = load_price_data(
        CONFIG["tickers"], 
        CONFIG["start_date"],
        CONFIG.get("end_date")  # Gets end_date if set, otherwise None (defaults to current)
    )

    print("Running backtest...\n")
    equity_df, quarterly_df, _ = run_backtest(
        price_data,
        CONFIG,
        lambda s: compute_drawdown_from_ath(s),
        lambda dd, cfg: determine_regime(dd, cfg),
        rebalance_portfolio
    )

    try:
        vix_s = fetch_vix_series_for_equity_dates(equity_df)
        equity_df = attach_vix_to_equity_df(equity_df, vix_s)
    except Exception as e:
        print(f"[LOG] Warning: could not attach ^VIX column: {e}")
        equity_df = attach_vix_to_equity_df(equity_df, pd.Series(dtype=float))

    equity_df = compute_signal_layer_columns(equity_df)
    equity_df = reorder_signal_override_columns_after_signals(equity_df)

    print("Backtest complete!")
    print("Final portfolio value:", equity_df["Value"].iloc[-1])

    print("\nSuggested current allocation based on regime:")
    drawdown_ticker = config.CONFIG.get("drawdown_ticker")

    if drawdown_ticker is None or drawdown_ticker not in price_data.columns:
        raise ValueError(f"drawdown-ticker '{drawdown_ticker}' not found in price data.")

    dd_col = f"{drawdown_ticker}_DD_raw"
    if dd_col in equity_df.columns:
        latest_dd = float(equity_df[dd_col].iloc[-1])
    else:
        dd_series, _ = compute_drawdown_from_ath(price_data[drawdown_ticker])
        latest_dd = float(dd_series.iloc[-1])
    regime = determine_regime(latest_dd, CONFIG)
    print(get_allocation_for_regime(regime, CONFIG))

    if confirm_export():
        print("Exporting results...\n")
        export_to_excel(equity_df, quarterly_df, CONFIG)
        print("Excel export complete!\n")

    # ----------------------------------------
    # OPTIONAL: VIEW INTERACTIVE DASHBOARD
    # ----------------------------------------
    if confirm_dashboard():
        print("\nPreparing dashboard data...\n")
        
        # Save data for dashboard
        import pickle
        data_file = "dashboard_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'equity_df': equity_df,
                'quarterly_df': quarterly_df,
                'config': CONFIG
            }, f)
        
        print(f"Data saved to {data_file}")
        print("\nTo view the dashboard, run:")
        print("  streamlit run dashboard_runner.py")
        print("\nOr launch it now? (Y/N): ", end="")
        launch_now = input().strip().upper()
        
        if launch_now == "Y":
            import subprocess
            import sys
            import os
            
            print("\nLaunching dashboard in your browser...")
            print("Press Ctrl+C in this terminal to stop the dashboard server.\n")
            
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_runner.py"])
            except KeyboardInterrupt:
                print("\nDashboard server stopped.")
            finally:
                # Cleanup
                if os.path.exists(data_file):
                    os.remove(data_file)

    # ----------------------------------------
    # OPTIONAL: RUN WORST-CASE SIMULATION
    # ----------------------------------------
    print("\nRun worst-case historical simulation? (Y/N): ", end="")
    run_worst = input().strip().upper()

    if run_worst == "Y":
        print("\nRunning worst-case simulator...\n")

        w_equity_df, w_quarterly_df = run_worst_case_simulation()

        print("Worst-case simulation complete!")
        print("Final simulated portfolio value:", w_equity_df['Value'].iloc[-1])

        print("\nExporting worst-case results...\n")
        export_to_excel(
            w_equity_df,
            w_quarterly_df,
            CONFIG,
            is_worst_case=True
        )
        print("Worst-case Excel export complete!\n")

    if confirm_exit():
        print("Thank you for using the Tactical Portfolio Engine!")



if __name__ == "__main__":
    run()
