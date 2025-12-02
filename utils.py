# utils.py

def next_trading_day(date, available_dates):
    available = [d for d in available_dates if d >= date]
    return available[0] if available else None

# utils.py
def log(msg):
    print(f"[LOG] {msg}")
