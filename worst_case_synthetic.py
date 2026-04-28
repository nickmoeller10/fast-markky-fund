"""
Pure synthetic-price builders used by `worst_case_simulator`.

`compute_qqq_ixic_beta` derives the beta to use when stretching ^IXIC returns
into synthetic QQQ. `build_synth_qqq_tqqq` then walks the price path
(vectorized via cumprod). No yfinance / network dependency in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import log


_DEFAULT_QQQ_IXIC_CORR = 0.95
_DEFAULT_QQQ_IXIC_BETA = 1.0
_BETA_MIN_OVERLAP = 30


def clamp_return(r, min_r: float = -0.99, max_r: float = 3.00):
    """Clamp extreme returns to prevent synthetic blow-ups."""
    return np.clip(r, min_r, max_r)


def compute_qqq_ixic_beta(ixic_close: pd.Series, real_close: pd.DataFrame, qqq_start):
    """
    Estimate beta of QQQ vs ^IXIC over the overlap window.

    Returns (correlation, beta) — falls back to documented defaults
    (corr=0.95, beta=1.0) when data is insufficient.
    """
    if "QQQ" not in real_close.columns or qqq_start is None:
        return _fallback_beta()

    qqq_data = real_close["QQQ"].dropna()
    overlap_start = max(qqq_data.index.min(), ixic_close.index.min())
    overlap_end = min(qqq_data.index.max(), ixic_close.index.max())
    if overlap_start >= overlap_end:
        return _fallback_beta()

    overlap_ixic = ixic_close.loc[overlap_start:overlap_end]
    overlap_qqq = qqq_data.loc[overlap_start:overlap_end]
    common_dates = overlap_ixic.index.intersection(overlap_qqq.index)
    if len(common_dates) <= _BETA_MIN_OVERLAP:
        return _fallback_beta()

    ixic_ret = overlap_ixic.loc[common_dates].pct_change().dropna()
    qqq_ret = overlap_qqq.loc[common_dates].pct_change().dropna()
    common_ret_dates = ixic_ret.index.intersection(qqq_ret.index)
    if len(common_ret_dates) <= _BETA_MIN_OVERLAP:
        return _fallback_beta()

    ixic_values = np.atleast_1d(ixic_ret.loc[common_ret_dates].values).flatten()
    qqq_values = np.atleast_1d(qqq_ret.loc[common_ret_dates].values).flatten()
    n = min(len(ixic_values), len(qqq_values))
    if n <= _BETA_MIN_OVERLAP:
        return _fallback_beta()

    ixic_values = ixic_values[:n]
    qqq_values = qqq_values[:n]
    correlation = float(np.corrcoef(ixic_values, qqq_values)[0, 1])
    ixic_variance = float(np.var(ixic_values))
    if ixic_variance <= 0:
        return _fallback_beta()
    covariance = float(np.cov(ixic_values, qqq_values)[0, 1])
    beta = covariance / ixic_variance
    log(f"[SIMULATOR] QQQ/^IXIC correlation: {correlation:.4f}")
    log(f"[SIMULATOR] QQQ/^IXIC beta: {beta:.4f}")
    return correlation, beta


def _fallback_beta():
    log(f"[SIMULATOR] Using default QQQ/^IXIC correlation: {_DEFAULT_QQQ_IXIC_CORR:.4f}")
    return _DEFAULT_QQQ_IXIC_CORR, _DEFAULT_QQQ_IXIC_BETA


def initial_qqq_price(ixic_close: pd.Series, real_close: pd.DataFrame, qqq_start) -> float:
    """Calibrate the synthetic QQQ starting price from real-QQQ inception, falling back to ^IXIC/10."""
    initial_ixic = float(ixic_close.iloc[0])
    if qqq_start and "QQQ" in real_close.columns:
        first_qqq = float(real_close["QQQ"].dropna().iloc[0])
        ixic_at_start = float(ixic_close.loc[qqq_start]) if qqq_start in ixic_close.index else initial_ixic
        price = initial_ixic * (first_qqq / ixic_at_start)
        log(f"[SIMULATOR] Calibrated QQQ initial price from real QQQ data: ${price:.2f}")
        return price
    price = initial_ixic / 10.0
    log(f"[SIMULATOR] Using default QQQ initial price: ${price:.2f}")
    return price


def build_synth_qqq_tqqq(ixic_close: pd.Series, ixic_ret: pd.Series, beta: float,
                         initial_qqq: float) -> tuple[pd.Series, pd.Series]:
    """
    Walk synthetic QQQ + TQQQ price paths vectorized.

    QQQ return = clamp(^IXIC return * beta)
    TQQQ return = clamp(QQQ return * 3) — leveraged proxy
    Synth path = initial * cumprod(1 + r). Day 0 contributes no growth.
    """
    full_index = ixic_close.index
    initial_tqqq = initial_qqq / 3.0
    log(f"[SIMULATOR] Initial QQQ price (from ^IXIC): ${initial_qqq:.2f}")
    log(f"[SIMULATOR] Initial TQQQ price (3x leveraged): ${initial_tqqq:.2f}")

    r_ixic = ixic_ret.reindex(full_index).fillna(0.0).to_numpy(dtype=float)
    r_qqq = clamp_return(r_ixic * beta)
    r_tqqq = clamp_return(r_qqq * 3.0)
    # Day 0 has no prior bar — pct_change is NaN there; cumprod base must be 1.
    r_qqq[0] = 0.0
    r_tqqq[0] = 0.0

    synth_qqq = pd.Series(initial_qqq * np.cumprod(1.0 + r_qqq), index=full_index, dtype=float)
    synth_tqqq = pd.Series(initial_tqqq * np.cumprod(1.0 + r_tqqq), index=full_index, dtype=float)
    return synth_qqq, synth_tqqq
