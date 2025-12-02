# main.py

from config import CONFIG
from console_ui import *
from data_loader import load_price_data
from regime_engine import compute_drawdown_from_ath, determine_regime
from allocation_engine import get_allocation_for_regime
from rebalance_engine import rebalance_portfolio
from backtest import run_backtest
from exporter import export_to_excel


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
    price_data = load_price_data(CONFIG["tickers"], CONFIG["start_date"])

    print("Running backtest...\n")
    equity_df, quarterly_df = run_backtest(
        price_data,
        CONFIG,
        lambda s: compute_drawdown_from_ath(s),
        lambda dd, cfg: determine_regime(dd, cfg),
        rebalance_portfolio
    )

    print("Backtest complete!")
    print("Final portfolio value:", equity_df["Value"].iloc[-1])

    print("\nSuggested current allocation based on regime:")
    latest_dd, _ = compute_drawdown_from_ath(price_data["QQQ"])
    regime = determine_regime(latest_dd.iloc[-1], CONFIG)
    print(get_allocation_for_regime(regime, CONFIG))

    if confirm_export():
        print("Exporting results...\n")
        export_to_excel(equity_df, quarterly_df, CONFIG)
        print("Excel export complete!\n")

    if confirm_exit():
        print("Thank you for using the Tactical Portfolio Engine!")


if __name__ == "__main__":
    run()
