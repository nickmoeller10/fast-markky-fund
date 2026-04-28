# ======================================================================
# FAST MARKKY FUND — Core Configuration File
# ======================================================================
# User-supplied baseline config (set 2026-04-28). Three-regime structure
# with R1 conservative-leverage TQQQ, R2 100% XLU defensive holding,
# R3 100% $ absolute defense. Only R1 has signal overrides enabled
# (Strong Bull → identical to base; Bull Fading → 25/0/75/0 protective).
#
# This is the new starting point for iter 27+ — replaces the iter-25
# raw-weight reconstruction. Reasons:
#   - Overrides enabled-vs-disabled were producing different equity curves
#     even when their weights matched the base, due to a daily-loop
#     compounding bug (rebalance used yesterday's NAV instead of today's
#     mark). Fixed in backtest.py 2026-04-28; all prior iteration scores
#     are now stale and need re-baselining.
#   - The iter-25 raw weights conflated multiple effects under that bug.
#     Starting from a clean simple config is the cleaner re-baseline.
#
# Synthetic cash sleeve uses the "$" alias (data_loader.CASH_ALIASES).
# Every override panel must sum to 1.0 — enforced at backtest start by
# signal_override_engine.validate_panel_sums().
# ======================================================================

CONFIG = {

    # =============================================================
    # PORTFOLIO STARTING PARAMETERS
    # =============================================================
    "starting_balance": 10000,

    "start_date": "1999-01-04",  # Arbitrary US trading day (Monday) in 1999

    "end_date": "2026-03-27",

    "drawdown_ticker": "QQQ",  # The ticker we measure drawdown from

    # False: standard ATH (cummax from inception + pre-start history download).
    # True: rolling peak over the last N calendar years (fallback to standard ATH until N years exist).
    # iter 20/25 finding: 3-yr window dominates 1y/2y for DD control — smoother
    # rolling peak gives more decisive regime transitions.
    "drawdown_window_enabled": True,
    "drawdown_window_years": 1,

    # =============================================================
    # REBALANCE SETTINGS
    # =============================================================
    "rebalance_frequency": "instant",
    "rebalance_holiday_rule": "next_trading_day",
    "rebalance_strategy": "per_regime",
    "dividend_reinvestment": False,
    "dividend_reinvestment_target": "cash",

    # =============================================================
    # TICKER CONFIGURATION
    # =============================================================
    # "$" is the synthetic risk-free sleeve (zero drawdown, ~4% APY) — see
    # data_loader.SYNTHETIC_TICKERS / CASH_ALIASES. Both "$" and "CASH" are
    # recognized; we use "$" in production to make it visually clear it isn't
    # the real yfinance "CASH" ticker (Pathward Financial Inc.).
    "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "$"],
    "allocation_tickers": ["TQQQ", "QQQ", "XLU", "$"],
    "minimum_allocation": 0.0,

    # =============================================================
    # REGIME DEFINITIONS — clean baseline (set 2026-04-28)
    # =============================================================
    # Drawdown bands measured against a 1y rolling QQQ peak.
    #
    # R1 — Ride High (0–12% dd): 75% TQQQ + 25% XLU. Conservative leverage.
    # R2 — XLU Defensive (12–19% dd): 100% XLU.
    # R3 — Absolute Defense (19%+ dd): 100% $ (synthetic cash, zero DD).
    #
    # R1 overrides are the only enabled signal panels:
    #   - Strong Bull (signal > +3): same 75/0/25/0 as base (no-op trigger).
    #   - Bull Fading (signal < -2): defensive 25/0/75/0 (lower TQQQ, more XLU).
    #
    # R2 / R3 override panels are disabled and zero-weight; they exist only
    # to satisfy the schema and validate_panel_sums (each disabled panel
    # mirrors the regime base so it sums to 1.0).
    "regimes": {
        "R1": {
            "dd_low": 0.0,
            "dd_high": 0.12,
            "TQQQ": 0.75,
            "QQQ": 0.0,
            "XLU": 0.25,
            "$": 0.0,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": True,
                    "label": "R1 Strong Bull",
                    "direction": "above",
                    "threshold": 3,
                    "TQQQ": 0.75,
                    "QQQ": 0.0,
                    "XLU": 0.25,
                    "$": 0.0,
                },
                "protection": {
                    "enabled": True,
                    "label": "R1 Bull Fading",
                    "direction": "below",
                    "threshold": -2,
                    "TQQQ": 0.25,
                    "QQQ": 0.0,
                    "XLU": 0.75,
                    "$": 0.0,
                },
            },
        },
        "R2": {
            "dd_low": 0.12,
            "dd_high": 0.19,
            "TQQQ": 0.0,
            "QQQ": 0.0,
            "XLU": 1.0,
            "$": 0.0,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": False,
                    "label": "R2 Recovery Confirmed",
                    "direction": "above",
                    "threshold": 4,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 1.0,
                    "$": 0.0,
                },
                "protection": {
                    "enabled": False,
                    "label": "R2 Deteriorating",
                    "direction": "below",
                    "threshold": -3,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 1.0,
                    "$": 0.0,
                },
            },
        },
        "R3": {
            "dd_low": 0.19,
            "dd_high": 1.0,
            "TQQQ": 0.0,
            "QQQ": 0.0,
            "XLU": 0.0,
            "$": 1.0,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": False,
                    "label": "R3 Capitulation Reversal",
                    "direction": "above",
                    "threshold": 1,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 0.0,
                    "$": 1.0,
                },
                "protection": {
                    "enabled": False,
                    "label": "R3 Crisis Deepening",
                    "direction": "below",
                    "threshold": -4,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 0.0,
                    "$": 1.0,
                },
            },
        },
    },

    # ======================================================================
    # WORST-CASE SIMULATION SETTINGS
    # ======================================================================
    "use_worst_case_simulation": True,
    "benchmark_ticker": "QQQ",
    "worst_case_start_date": "1950-01-01",
    "worst_case_output_dir": "simulation_outputs",
}
