"""
Freeze a canonical historical-data snapshot into ``data_cache/``.

Run after a config change or when adding new tickers to the optimizer pool.
After freezing, all backtests can run with ``FMF_DATA_MODE=frozen`` and produce
deterministic results regardless of yfinance data revisions.

Usage:
    python3 scripts/freeze_data.py
"""

import os
import sys
from pathlib import Path

# Allow running from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force refresh mode for the entire freeze run
os.environ["FMF_DATA_MODE"] = "refresh"

from config import CONFIG   # noqa: E402
from data_cache import cached_yf_download, cache_status   # noqa: E402
from data_loader import load_price_data, load_spy_series, load_vix_series   # noqa: E402

# Production tickers (from config.py)
PRODUCTION_TICKERS = ["QQQ", "TQQQ", "XLU", "SPY"]
# Alternative tickers the optimizer may include in candidate configs
OPTIMIZER_ALTERNATIVES = ["TLT", "GLD", "SPLV", "BIL"]
# Drawdown ticker (its full history is downloaded separately by run_backtest)
DRAWDOWN_TICKER = "QQQ"

# Date ranges
PANEL_START = "1999-01-04"
# Capture BOTH the open-ended fetch (for app.py runtime "data through today")
# AND a fixed-end fetch matching CONFIG["end_date"] (for optimizer reproducibility).
PANEL_END_FIXED = CONFIG.get("end_date", "2026-03-27")
SIGNAL_HISTORY_START = "1996-01-01"  # ~3y buffer before panel for 252d/200d warmups
HISTORICAL_INCEPTION_START = "1980-01-01"   # for QQQ pre-portfolio history


def main() -> None:
    print("=" * 70)
    print("FREEZING DATA CACHE")
    print(f"Mode: {os.environ['FMF_DATA_MODE']}")
    print("=" * 70)

    # 1. Production panel + alternative tickers (open-ended → snapshot to today)
    all_panel_tickers = sorted(set(PRODUCTION_TICKERS) | set(OPTIMIZER_ALTERNATIVES))
    print(f"\n[1a/5] Panel data (open-ended) — {all_panel_tickers}, {PANEL_START} → today")
    load_price_data(all_panel_tickers, PANEL_START, end_date=None)

    # 2. Same panel, fixed end_date (matches CONFIG["end_date"]) → optimizer reproducibility
    print(f"\n[1b/5] Panel data (fixed end) — {all_panel_tickers}, {PANEL_START} → {PANEL_END_FIXED}")
    load_price_data(all_panel_tickers, PANEL_START, end_date=PANEL_END_FIXED)

    # 3. Drawdown-ticker historical (1980→today) used by backtest.py:600
    print(f"\n[2/5] Historical drawdown data — {DRAWDOWN_TICKER} from {HISTORICAL_INCEPTION_START}")
    cached_yf_download(
        DRAWDOWN_TICKER,
        start=HISTORICAL_INCEPTION_START,
        end=None,
        auto_adjust=True,
        progress=False,
    )

    # 4. Extended SPY for signal layer warmup
    print(f"\n[3/5] SPY signal history — from {SIGNAL_HISTORY_START}")
    load_spy_series(SIGNAL_HISTORY_START, end_date=None)

    # 5. Extended VIX for signal layer warmup
    print(f"\n[4/5] VIX signal history — from {SIGNAL_HISTORY_START}")
    load_vix_series(SIGNAL_HISTORY_START, end_date=None)

    print("\n" + "=" * 70)
    print("CACHE STATUS")
    print("=" * 70)
    status = cache_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    print("\nFreeze complete. Cache files are in", status["cache_dir"])
    print("To use the frozen snapshot: FMF_DATA_MODE=frozen python3 main.py\n")


if __name__ == "__main__":
    main()
