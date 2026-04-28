"""
Drawdown computations used by the backtest engine.

`compute_rolling_ath_and_dd` is the production reference-high implementation
(rolling N-calendar-year window, with cummax fallback during bootstrap).
`build_regime_signal_drawdown` enforces the T+1 lag for regime detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_ath_and_dd(series: pd.Series, n_calendar_years: int):
    """
    Reference high = max(close) over the trailing N calendar years (inclusive of t).
    Until the calendar span from the first bar to t reaches N years, use standard
    ATH (cummax from inception) for that date.

    Returns:
        (ref_high_series, dd_series), same index as sorted non-NaN input.
    """
    s = series.sort_index().dropna()
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if n_calendar_years <= 0:
        ath = s.cummax()
        dd = (ath - s) / ath
        return ath, dd

    idx = s.index
    values = s.to_numpy(dtype=float, copy=False)
    n = len(s)
    ref = np.empty(n, dtype=float)
    cummax_vals = np.maximum.accumulate(values)

    first_ts = idx[0]
    need_until = first_ts + pd.DateOffset(years=n_calendar_years)

    for i in range(n):
        ts = idx[i]
        if ts < need_until:
            ref[i] = cummax_vals[i]
            continue
        win_start = ts - pd.DateOffset(years=n_calendar_years)
        j = int(idx.searchsorted(win_start, side="left"))
        j = min(max(j, 0), i)
        ref[i] = float(values[j : i + 1].max())

    ath_series = pd.Series(ref, index=idx, dtype=float)
    dd_series = (ath_series - s) / ath_series
    return ath_series, dd_series


def build_regime_signal_drawdown(
    dd_raw: pd.Series,
    exec_index: pd.DatetimeIndex,
    full_index: pd.DatetimeIndex,
    historical_dd_full: pd.Series | None = None,
) -> pd.Series:
    """
    Drawdown used for regime detection on execution date D: prior trading day's close
    on full_index (already encoded in dd_raw vs ATH as of that day). First bar in
    exec_index with no prior row on full_index uses historical_dd_full (last bar
    strictly before D) if provided, else same-day dd on D. Portfolio trades still
    size at row_prices on D (daily close as proxy for next-session execution).
    """
    s = (
        dd_raw.reindex(full_index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # Hot path: for each exec date d, the value at "last full_index entry strictly
    # before d" is `searchsorted(d, side='left') - 1`. Vectorized; loops only run
    # in the rare fallback case where exec_index begins before full_index.
    s_vals = s.to_numpy(dtype=float)
    pos = full_index.searchsorted(exec_index.to_numpy(), side="left") - 1
    out = np.empty(len(exec_index), dtype=float)
    has_prev = pos >= 0
    out[has_prev] = s_vals[pos[has_prev]]

    for i in np.where(~has_prev)[0]:
        d = exec_index[i]
        if historical_dd_full is not None:
            prev_h = historical_dd_full.index[historical_dd_full.index < d]
            if len(prev_h):
                out[i] = float(historical_dd_full.loc[prev_h[-1]])
                continue
        pv = s.loc[d] if d in s.index else np.nan
        out[i] = float(pv) if pd.notna(pv) else 0.0

    return pd.Series(out, index=exec_index, dtype=float)
