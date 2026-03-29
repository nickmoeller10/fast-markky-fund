# signal_layers.py
# ======================================================================
# Layer 1 (VIX), Layer 2 (MACD / SPY), Layer 3 (50/200 MA / SPY), composite.
# Appended to equity_df after backtest + VIX attach. Display/regime TBD.
# ======================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_spy_series, load_vix_series

SPY_PRICE_COL = "SPY_price"
VIX_COL = "VIX"

# Calendar lookback before first portfolio session so 252d VIX z-score and 200d MA
# have full windows on the first backtest bar (~252 trading days ≈ 15 months; use 800d buffer).
SIGNAL_HISTORY_LOOKBACK_CALENDAR_DAYS = 800


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _vix_regime_from_z(z: pd.Series) -> pd.Series:
    """Buckets aligned with Signal_L1 cutoffs (high z = fear)."""
    out = pd.Series(pd.NA, index=z.index, dtype=object)
    m = z.notna()
    # Mutually exclusive masks (same as if / elif chain on z)
    out.loc[m & (z > 2.0)] = "Extreme Fear"
    out.loc[m & ~(z > 2.0) & (z > 1.0)] = "Elevated"
    out.loc[m & ~(z > 1.0) & (z > -1.0)] = "Normal"
    out.loc[m & ~(z > -1.0) & (z > -2.0)] = "Complacent"
    out.loc[m & ~(z > -2.0)] = "Extreme Complacency"
    return out


def _signal_l1(vix_z: pd.Series) -> pd.Series:
    """Matches spec: if z>2 → +2; elif z>1 → +1; elif z>-1 → 0; elif z>-2 → -1; else -2."""
    s = pd.Series(np.nan, index=vix_z.index, dtype=float)
    m = vix_z.notna()
    zz = vix_z[m].to_numpy(dtype=float)
    val = np.select(
        [zz > 2.0, zz > 1.0, zz > -1.0, zz > -2.0],
        [2.0, 1.0, 0.0, -1.0],
        default=-2.0,
    )
    s.loc[m] = val
    return s


def _signal_l2(
    macd_line: pd.Series,
    macd_signal: pd.Series,
    macd_hist: pd.Series,
    hist_delta: pd.Series,
    spy_close: pd.Series,
) -> pd.Series:
    """MACD-based discrete signal; first matching rule wins."""
    n = len(spy_close)
    out = pd.Series(np.nan, index=spy_close.index, dtype=float)

    prev_line = macd_line.shift(1)
    prev_sig = macd_signal.shift(1)
    bull_cross = (prev_line <= prev_sig) & (macd_line > macd_signal)
    bear_cross = (prev_line >= prev_sig) & (macd_line < macd_signal)

    # Bearish divergence: SPY at rolling 55d high while MACD below its prior-window strength
    # (heuristic — classic divergence needs swing peaks; this approximates "price high, MACD not").
    r = 55
    spy_roll_max = spy_close.rolling(r, min_periods=max(25, r // 2)).max()
    at_spy_high = spy_close >= (spy_roll_max * 0.9999)
    macd_past_peak = macd_line.shift(1).rolling(r - 1, min_periods=10).max()
    bearish_div = at_spy_high & macd_line.notna() & (macd_line < macd_past_peak)

    bc = bull_cross.fillna(False).to_numpy()
    bec = bear_cross.fillna(False).to_numpy()
    bd = bearish_div.fillna(False).to_numpy()

    for i in range(n):
        if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
            continue
        hd = hist_delta.iloc[i]
        mh = macd_hist.iloc[i]
        if bc[i] and pd.notna(hd) and hd > 0:
            out.iloc[i] = 2.0
        elif macd_line.iloc[i] > 0 and pd.notna(mh) and mh > 0:
            out.iloc[i] = 1.0
        elif pd.notna(mh) and mh > 0 and pd.notna(hd) and hd < 0:
            out.iloc[i] = 0.0
        elif bec[i] and pd.notna(mh) and mh < 0:
            out.iloc[i] = -2.0
        elif bd[i]:
            out.iloc[i] = -1.0
        else:
            out.iloc[i] = 0.0

    return out


def _signal_l3(spy: pd.Series, ma50: pd.Series, ma200: pd.Series) -> pd.Series:
    spread = ma50 - ma200
    spread_prev = spread.shift(1)
    out = pd.Series(np.nan, index=spy.index, dtype=float)

    c2 = (ma50 > ma200) & (spread > spread_prev)
    c1 = (spy > ma200) & (ma50 < ma200)
    c0 = (spy > ma50) & (spy < ma200)
    cm2 = (ma50 < ma200) & (spread < spread_prev)

    for i in range(len(spy)):
        if pd.isna(ma50.iloc[i]) or pd.isna(ma200.iloc[i]) or pd.isna(spy.iloc[i]):
            continue
        if bool(c2.iloc[i]):
            out.iloc[i] = 2.0
        elif bool(c1.iloc[i]):
            out.iloc[i] = 1.0
        elif bool(c0.iloc[i]):
            out.iloc[i] = 0.0
        elif bool(cm2.iloc[i]):
            out.iloc[i] = -2.0
        else:
            out.iloc[i] = 0.0

    return out


def _ma_regime_label(ma50: pd.Series, ma200: pd.Series) -> pd.Series:
    lab = pd.Series(pd.NA, index=ma50.index, dtype=object)
    m = ma50.notna() & ma200.notna()
    eq = np.isclose(ma50, ma200, rtol=0.0, atol=1e-8)
    lab.loc[m & (ma50 > ma200)] = "Golden Cross"
    lab.loc[m & (ma50 < ma200)] = "Death Cross"
    lab.loc[m & eq] = "Neutral"
    return lab


def _signal_label_from_total(total: pd.Series) -> pd.Series:
    lab = pd.Series(pd.NA, index=total.index, dtype=object)
    m = total.notna()
    t = total[m]
    r = pd.Series(pd.NA, index=t.index, dtype=object)
    r[t >= 5] = "Strong Buy"
    r[(t >= 3) & (t < 5)] = "Buy"
    r[(t >= 1) & (t < 3)] = "Lean Long"
    r[(t >= -1) & (t < 1)] = "Neutral"
    r[(t >= -3) & (t < -1)] = "Reduce"
    r[t < -3] = "Strong Sell"
    lab.loc[m] = r
    return lab


def _normalize_dt_index(idx: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    x = pd.DatetimeIndex(pd.to_datetime(idx))
    if x.tz is not None:
        x = x.tz_convert("UTC").tz_localize(None)
    return x.normalize()


def build_combined_spy_series(
    trading_index: pd.DatetimeIndex,
    spy_panel: pd.Series,
    lookback_calendar_days: int = SIGNAL_HISTORY_LOOKBACK_CALENDAR_DAYS,
) -> pd.Series:
    """
    Merge Yahoo SPY history (from before ``trading_index[0]``) with panel closes;
    panel wins on overlapping dates (same convention as drawdown merge).
    """
    ti = _normalize_dt_index(trading_index).sort_values()
    if len(ti) == 0:
        return pd.to_numeric(spy_panel, errors="coerce").sort_index()

    t0, t1 = ti[0], ti[-1]
    start = (pd.Timestamp(t0) - pd.Timedelta(days=int(lookback_calendar_days))).strftime("%Y-%m-%d")
    end_excl = (pd.Timestamp(t1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = load_spy_series(start, end_excl)

    panel = pd.to_numeric(spy_panel, errors="coerce").copy()
    panel.index = _normalize_dt_index(panel.index)

    if hist is None or len(hist) == 0:
        return panel.sort_index()

    hist = hist.copy()
    hist.index = _normalize_dt_index(hist.index)
    return panel.combine_first(hist).sort_index()


def build_combined_vix_series(
    trading_index: pd.DatetimeIndex,
    vix_panel: pd.Series | None,
    lookback_calendar_days: int = SIGNAL_HISTORY_LOOKBACK_CALENDAR_DAYS,
) -> pd.Series:
    """Merge extended ^VIX history with optional panel VIX; panel wins on overlap."""
    ti = _normalize_dt_index(trading_index).sort_values()
    if len(ti) == 0:
        return pd.Series(dtype=float)

    t0, t1 = ti[0], ti[-1]
    start = (pd.Timestamp(t0) - pd.Timedelta(days=int(lookback_calendar_days))).strftime("%Y-%m-%d")
    end_excl = (pd.Timestamp(t1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = load_vix_series(start, end_excl)

    if vix_panel is None or len(vix_panel) == 0:
        return hist.sort_index() if hist is not None else pd.Series(dtype=float)

    panel = pd.to_numeric(vix_panel, errors="coerce").copy()
    panel.index = _normalize_dt_index(panel.index)

    if hist is None or len(hist) == 0:
        return panel.sort_index()

    hist = hist.copy()
    hist.index = _normalize_dt_index(hist.index)
    return panel.combine_first(hist).sort_index()


def _signal_layers_dataframe(spy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    Build all signal columns on a shared index (extended trading timeline).
    ``spy`` and ``vix`` must use the same DatetimeIndex.
    """
    idx = spy.index
    vix = pd.to_numeric(vix.reindex(idx), errors="coerce").ffill().bfill()
    spy = pd.to_numeric(spy.reindex(idx), errors="coerce")

    vix_mean = vix.rolling(252, min_periods=252).mean()
    vix_std = vix.rolling(252, min_periods=252).std(ddof=0)
    z = (vix - vix_mean) / vix_std.replace(0, np.nan)
    z_dir = z - z.shift(1)

    l1 = _signal_l1(z)

    ema12 = _ema(spy, 12)
    ema26 = _ema(spy, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    hist_delta = macd_hist - macd_hist.shift(1)
    l2 = _signal_l2(macd_line, macd_signal, macd_hist, hist_delta, spy)

    ma50 = spy.rolling(50, min_periods=50).mean()
    ma200 = spy.rolling(200, min_periods=200).mean()
    l3 = _signal_l3(spy, ma50, ma200)

    valid = l1.notna() & l2.notna() & l3.notna()
    total = pd.Series(np.nan, index=idx, dtype=float)
    total.loc[valid] = l1.loc[valid] + l2.loc[valid] + l3.loc[valid]

    return pd.DataFrame(
        {
            "VIX": vix,
            "VIX_252d_mean": vix_mean,
            "VIX_252d_stdev": vix_std,
            "VIX_zscore": z,
            "VIX_zscore_direction": z_dir,
            "VIX_regime_label": _vix_regime_from_z(z),
            "Signal_L1": l1,
            "MACD_12ema": ema12,
            "MACD_26ema": ema26,
            "MACD_line": macd_line,
            "MACD_signal": macd_signal,
            "MACD_histogram": macd_hist,
            "MACD_histogram_delta": hist_delta,
            "Signal_L2": l2,
            "MA_50": ma50,
            "MA_200": ma200,
            "MA_regime_label": _ma_regime_label(ma50, ma200),
            "Signal_L3": l3,
            "Signal_total": total,
            "Signal_label": _signal_label_from_total(total),
        },
        index=idx,
    )


def _signal_layers_for_trading_dates(
    trading_index: pd.DatetimeIndex,
    spy_combined: pd.Series,
    vix_combined: pd.Series,
) -> pd.DataFrame:
    """
    Run rollings on the union of extended SPY/VIX history and portfolio dates,
    then return rows aligned to ``trading_index`` only.
    """
    ti = _normalize_dt_index(trading_index).sort_values()
    if len(ti) == 0:
        return pd.DataFrame()

    t1 = ti[-1]
    work = spy_combined.index.union(vix_combined.index).union(ti)
    work = work[work <= t1]
    work = pd.DatetimeIndex(sorted(pd.to_datetime(work.unique())))

    spy_w = pd.to_numeric(spy_combined.reindex(work), errors="coerce")
    vix_w = pd.to_numeric(vix_combined.reindex(work), errors="coerce").ffill().bfill()

    full = _signal_layers_dataframe(spy_w, vix_w)
    return full.reindex(ti)


def build_signal_total_series(
    trading_index: pd.DatetimeIndex,
    spy_combined: pd.Series,
    vix_combined: pd.Series,
) -> pd.Series:
    """
    Signal_total (−6…+6) aligned to ``trading_index``, using **extended** SPY/VIX
    series (see ``build_combined_spy_series`` / ``build_combined_vix_series``).
    """
    block = _signal_layers_for_trading_dates(trading_index, spy_combined, vix_combined)
    if block.empty or "Signal_total" not in block.columns:
        return pd.Series(np.nan, index=_normalize_dt_index(trading_index).sort_values())
    return block["Signal_total"]


def compute_signal_layer_columns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Layer 1–3 and composite columns to a copy of ``equity_df``.

    Downloads extended SPY and ^VIX history before the first equity row so rolling
    windows (252d VIX, 50/200d MA, MACD) are valid from the first session. Row ``VIX``
    is updated from the extended path when available so the spot column matches the
    series used for z-score.
    """
    if equity_df is None or equity_df.empty:
        return equity_df

    out = equity_df.copy()
    if "Date" not in out.columns:
        return out

    dates = _normalize_dt_index(pd.DatetimeIndex(pd.to_datetime(out["Date"])))
    ti = pd.DatetimeIndex(sorted(dates.unique()))

    if SPY_PRICE_COL not in out.columns:
        spy_vals = np.full(len(out), np.nan, dtype=float)
    else:
        spy_vals = pd.to_numeric(out[SPY_PRICE_COL], errors="coerce").to_numpy(dtype=float, copy=False)

    spy_panel = (
        pd.Series(spy_vals, index=dates).groupby(level=0).last().sort_index()
    )

    vix_panel_series = None
    if VIX_COL in out.columns:
        vx = pd.to_numeric(out[VIX_COL], errors="coerce")
        vix_panel_series = (
            pd.Series(vx.values, index=dates).groupby(level=0).last().sort_index()
        )

    spy_c = build_combined_spy_series(ti, spy_panel)
    vix_c = build_combined_vix_series(ti, vix_panel_series)

    block = _signal_layers_for_trading_dates(ti, spy_c, vix_c)
    row_dates = dates
    aligned = block.reindex(row_dates)

    for col in block.columns:
        out[col] = aligned[col].values

    return out


def reorder_signal_override_columns_after_signals(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Place Signal_override_* immediately after Signal_label (Performance Summary order).
    """
    if equity_df is None or equity_df.empty:
        return equity_df
    extra = [
        c
        for c in (
            "Signal_override_active",
            "Signal_override_label",
            "Signal_override_allocation",
        )
        if c in equity_df.columns
    ]
    if not extra:
        return equity_df
    others = [c for c in equity_df.columns if c not in extra]
    anchor = "Signal_label"
    if anchor in others:
        i = others.index(anchor) + 1
        order = others[:i] + extra + others[i:]
    else:
        order = others + extra
    return equity_df[order]
