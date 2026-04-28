"""
Pure-data helpers and constants used by the dashboard.

Kept free of streamlit / plotly so widget-data unit tests
(`tests/test_dashboard_widgets.py`, `tests/test_financial_metrics.py` via
`metrics.py`) can import without requiring UI dependencies.

dashboard.py re-exports these symbols for backward compat.
"""
from __future__ import annotations

import pandas as pd

from allocation_engine import get_allocation_for_regime
from data_loader import VIX_YAHOO_SYMBOL
from signal_override_engine import allocation_human_readable


# Ordered columns appended after core snapshot fields in Performance Summary
PERF_SIGNAL_COLUMNS_ORDER = [
    "VIX",
    "VIX_252d_mean",
    "VIX_252d_stdev",
    "VIX_zscore",
    "VIX_zscore_direction",
    "VIX_regime_label",
    "Signal_L1",
    "MACD_12ema",
    "MACD_26ema",
    "MACD_line",
    "MACD_signal",
    "MACD_histogram",
    "MACD_histogram_delta",
    "Signal_L2",
    "MA_50",
    "MA_200",
    "MA_regime_label",
    "Signal_L3",
    "Signal_total",
    "Signal_label",
]


PERF_SIGNAL_COLUMN_DISPLAY_NAMES = {
    "VIX": f"VIX close ({VIX_YAHOO_SYMBOL})",
    "VIX_252d_mean": "VIX 252d mean",
    "VIX_252d_stdev": "VIX 252d stdev",
    "VIX_zscore": "VIX z-score",
    "VIX_zscore_direction": "VIX z Δ (1d)",
    "VIX_regime_label": "VIX regime",
    "Signal_L1": "Signal L1",
    "MACD_12ema": "MACD 12 EMA",
    "MACD_26ema": "MACD 26 EMA",
    "MACD_line": "MACD line",
    "MACD_signal": "MACD signal",
    "MACD_histogram": "MACD histogram",
    "MACD_histogram_delta": "MACD hist Δ",
    "Signal_L2": "Signal L2",
    "MA_50": "MA 50",
    "MA_200": "MA 200",
    "MA_regime_label": "MA regime",
    "Signal_L3": "Signal L3",
    "Signal_total": "Signal total",
    "Signal_label": "Signal label",
    "Signal_override_active": "Signal override (active)",
    "Signal_override_label": "Signal override label",
    "Signal_override_allocation": "Signal override allocation",
}


PERFORMANCE_SUMMARY_GUIDE = """
**VIX / Layer 1**
| Column | Description |
|--------|-------------|
| VIX close | Raw **^VIX** close that day |
| VIX 252d mean | Rolling 252-day average of VIX — “normal” for that era |
| VIX 252d stdev | Rolling 252-day stdev — typical VIX variability |
| VIX z-score | (VIX − mean) / stdev — extremity vs recent history |
| VIX z Δ (1d) | Today’s z minus yesterday’s — fear rising or falling |
| VIX regime | Extreme Fear / Elevated / Normal / Complacent / Extreme Complacency |
| Signal L1 | Score −2…+2 from z-score buckets |

**MACD / Layer 2 (SPY)**
| Column | Description |
|--------|-------------|
| MACD 12 / 26 EMA | Fast / slow EMA of SPY close |
| MACD line | 12 EMA − 26 EMA |
| MACD signal | 9-day EMA of MACD line |
| MACD histogram | Line − signal |
| MACD hist Δ | Today’s histogram minus yesterday’s |
| Signal L2 | Score −2…+2 from crossovers, histogram, divergence heuristic |

**50/200 MA / Layer 3 (SPY)**
| Column | Description |
|--------|-------------|
| MA 50 / MA 200 | Simple moving averages of SPY close |
| MA regime | Golden Cross / Death Cross / Neutral |
| Signal L3 | Score −2…+2 from MA spread and SPY position |

**Composite**
| Column | Description |
|--------|-------------|
| Signal total | L1 + L2 + L3 (−6…+6) |
| Signal label | Strong Buy → Strong Sell buckets |

**Signal overrides (allocation overlay)**
| Column | Description |
|--------|-------------|
| Signal override (active) | none / upside / protection — which overlay is active from **today’s** Signal_total vs thresholds (above = score ≥ threshold, below = score ≤ threshold; protection wins if both) |
| Signal override label | Config label for the active panel |
| Signal override allocation | Human-readable target weights when an override is active |
"""


REGIME_DESCRIPTIONS = {
    "R1": "Ride High",
    "R2": "Cautious Defense",
    "R3": "Contrarian Buyback",
}


def perf_cell_empty(x) -> bool:
    """True if a Performance Summary cell value should render as blank."""
    if x is None or x is pd.NA:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def todays_regime_status(equity_df, config):
    """
    Pull the most recent row from equity_df and resolve the recommended allocation.
    Returns a dict; render_todays_regime_status() consumes it for display.
    """
    last = equity_df.iloc[-1]
    market_regime = last.get("Market_Regime")
    portfolio_regime = last.get("Portfolio_Regime")
    override_active = str(last.get("Signal_override_active") or "none")
    override_label = str(last.get("Signal_override_label") or "")
    override_alloc_str = str(last.get("Signal_override_allocation") or "")

    if override_active != "none" and override_alloc_str:
        recommended_alloc_str = override_alloc_str
    elif portfolio_regime and config and portfolio_regime in config.get("regimes", {}):
        base_alloc = get_allocation_for_regime(portfolio_regime, config)
        recommended_alloc_str = allocation_human_readable(
            base_alloc, config.get("allocation_tickers", [])
        )
    else:
        recommended_alloc_str = ""

    as_of = pd.to_datetime(last.get("Date")) if last.get("Date") is not None else None

    return {
        "as_of": as_of,
        "market_regime": market_regime,
        "portfolio_regime": portfolio_regime,
        "override_active": override_active,
        "override_label": override_label,
        "recommended_allocation": recommended_alloc_str,
    }
