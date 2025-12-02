# console_ui.py

from config import CONFIG

def print_config():
    print("\nHere are your current parameters:\n")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print()

def confirm_parameters():
    choice = input("Do these parameters look good? (Y/N): ").strip().upper()
    return choice == "Y"

def confirm_backtest():
    return input("Run backtest? (Y/N): ").strip().upper() == "Y"

def confirm_export():
    return input("Export results to Excel? (Y/N): ").strip().upper() == "Y"

def confirm_exit():
    return input("Exit program? (Y/N): ").strip().upper() == "Y"
