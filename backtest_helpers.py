"""
Small pure helpers used by `run_backtest`.

Extracted from backtest.py so the orchestrator stays focused on the per-day loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scalar_drawdown_for_regime(val):
    """Coerce yfinance/indexing quirks to a float for regime_detector; NaN/inf → 0.0 (treat as R1 band)."""
    if val is None:
        return 0.0
    if isinstance(val, pd.Series):
        val = val.dropna()
        val = val.iloc[0] if len(val) else np.nan
    try:
        x = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


def regime_number(label: str) -> int:
    return int(label.replace("R", ""))


def regime_label(num: int) -> str:
    return f"R{num}"


def bottom_regime_number(config) -> int:
    """Largest regime index (e.g. 4 for R4) for asymmetric rules that reference the deepest regime."""
    keys = list((config or {}).get("regimes") or [])
    if not keys:
        return 3
    return max(regime_number(k) for k in keys)


def get_rebalance_dates(index, frequency):
    if frequency in ("none", "instant"):
        return pd.DatetimeIndex([])

    resample_rules = {
        "daily": None,
        "weekly": "W-FRI",
        "monthly": "M",
        "quarterly": "Q",
        "semiannual": "2Q",
        "annual": "A",
    }
    if frequency not in resample_rules:
        raise ValueError(f"Invalid rebalance frequency: {frequency}")
    rule = resample_rules[frequency]
    if rule is None:
        return index
    return index.to_series().resample(rule).last().index


def calculate_normalized_values(price_df, tickers, starting_val, start_date):
    """
    Per-ticker normalized path from starting_val. If a ticker has no price on start_date
    (e.g. not listed yet), use the first valid price on or after start_date as the base.
    """
    norm_df = pd.DataFrame(index=price_df.index)
    for t in tickers:
        s = price_df[t]
        base_price = s.loc[start_date] if start_date in s.index else np.nan
        if pd.isna(base_price):
            after = s.loc[s.index >= start_date].dropna()
            if after.empty:
                norm_df[f"{t}_norm"] = np.nan
                continue
            base_price = float(after.iloc[0])
            first_valid_idx = after.index[0]
        else:
            first_valid_idx = start_date
        norm = (s / base_price) * starting_val
        norm.loc[price_df.index < first_valid_idx] = np.nan
        norm_df[f"{t}_norm"] = norm
    return norm_df
