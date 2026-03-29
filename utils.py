# utils.py

import pandas as pd


def max_drawdown_from_equity_curve(values) -> float:
    """
    Worst peak-to-trough decline over the entire series (running peak from day 0).

    At each date t: drawdown(t) = W_t / max_{u<=t} W_u - 1.
    Returns min_t drawdown(t), a non-positive float (e.g. -0.35 means -35% from a prior peak).
    NaNs are dropped; rows with non-positive running peak are skipped for the ratio.
    """
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    valid = peak > 0
    if not valid.any():
        return 0.0
    dd = (s / peak) - 1.0
    dd = dd.where(valid)
    return float(dd.min())


def next_trading_day(date, available_dates):
    available = [d for d in available_dates if d >= date]
    return available[0] if available else None

# utils.py
def log(msg):
    print(f"[LOG] {msg}")
