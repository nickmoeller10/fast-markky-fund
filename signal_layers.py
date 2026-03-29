# signal_layers.py
# ======================================================================
# Layer 1 (VIX), Layer 2 (MACD / SPY), Layer 3 (50/200 MA / SPY), composite.
# Appended to equity_df after backtest + VIX attach. Display/regime TBD.
# ======================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

SPY_PRICE_COL = "SPY_price"
VIX_COL = "VIX"


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


def build_signal_total_series(
    trading_index: pd.DatetimeIndex,
    spy_close: pd.Series,
    vix_close: pd.Series,
) -> pd.Series:
    """
    Same Signal_total (−6…+6) as ``compute_signal_layer_columns``, aligned to ``trading_index``.
    Used inside ``run_backtest`` before equity rows exist. Extended ``vix_close`` history
    (pre-panel) improves early z-score stability.
    """
    if trading_index is None or len(trading_index) == 0:
        return pd.Series(dtype=float)

    idx = trading_index
    spy = pd.to_numeric(spy_close.reindex(idx), errors="coerce")
    vix = pd.to_numeric(vix_close.reindex(idx).ffill().bfill(), errors="coerce")

    vix_mean = vix.rolling(252, min_periods=252).mean()
    vix_std = vix.rolling(252, min_periods=252).std(ddof=0)
    z = (vix - vix_mean) / vix_std.replace(0, np.nan)
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

    total = pd.Series(np.nan, index=idx, dtype=float)
    valid = l1.notna() & l2.notna() & l3.notna()
    total.loc[valid] = l1.loc[valid] + l2.loc[valid] + l3.loc[valid]
    return total


def compute_signal_layer_columns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Layer 1–3 and composite columns to a copy of ``equity_df``.
    Expects ``VIX`` and ``SPY_price`` when available; missing inputs → NaN / NA for that layer.
    New columns are appended at the end (stable order for exports).
    """
    if equity_df is None or equity_df.empty:
        return equity_df

    out = equity_df.copy()
    idx = out.index

    # --- Layer 1: VIX ---
    vix = pd.to_numeric(out.get(VIX_COL), errors="coerce")
    vix_mean = vix.rolling(252, min_periods=252).mean()
    vix_std = vix.rolling(252, min_periods=252).std(ddof=0)
    z = (vix - vix_mean) / vix_std.replace(0, np.nan)
    z_dir = z - z.shift(1)

    out["VIX_252d_mean"] = vix_mean
    out["VIX_252d_stdev"] = vix_std
    out["VIX_zscore"] = z
    out["VIX_zscore_direction"] = z_dir
    out["VIX_regime_label"] = _vix_regime_from_z(z)
    out["Signal_L1"] = _signal_l1(z)

    # --- Layer 2 & 3: SPY ---
    if SPY_PRICE_COL not in out.columns:
        spy = pd.Series(np.nan, index=idx, dtype=float)
    else:
        spy = pd.to_numeric(out[SPY_PRICE_COL], errors="coerce")

    ema12 = _ema(spy, 12)
    ema26 = _ema(spy, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    hist_delta = macd_hist - macd_hist.shift(1)

    out["MACD_12ema"] = ema12
    out["MACD_26ema"] = ema26
    out["MACD_line"] = macd_line
    out["MACD_signal"] = macd_signal
    out["MACD_histogram"] = macd_hist
    out["MACD_histogram_delta"] = hist_delta
    out["Signal_L2"] = _signal_l2(macd_line, macd_signal, macd_hist, hist_delta, spy)

    ma50 = spy.rolling(50, min_periods=50).mean()
    ma200 = spy.rolling(200, min_periods=200).mean()
    out["MA_50"] = ma50
    out["MA_200"] = ma200
    out["MA_regime_label"] = _ma_regime_label(ma50, ma200)
    out["Signal_L3"] = _signal_l3(spy, ma50, ma200)

    l1 = out["Signal_L1"]
    l2 = out["Signal_L2"]
    l3 = out["Signal_L3"]
    valid = l1.notna() & l2.notna() & l3.notna()
    total = pd.Series(np.nan, index=idx, dtype=float)
    total.loc[valid] = l1.loc[valid] + l2.loc[valid] + l3.loc[valid]
    out["Signal_total"] = total
    out["Signal_label"] = _signal_label_from_total(total)

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
