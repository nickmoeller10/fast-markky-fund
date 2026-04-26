# ======================================================================
# FAST MARKKY FUND — Core Configuration File
# ======================================================================
# Declarative settings for:
#   • Backtest parameters
#   • Regime definitions
#   • Rebalancing rules
#   • Simulation behaviors
#   • Worst-case synthetic data generation
#
# NOTE: These are sensible defaults — not tuned production values.
# Thresholds, allocations, and signal-override settings are subject to
# change as the strategy is iterated. Treat as a starting point.
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
    "drawdown_window_enabled": True,
    "drawdown_window_years": 1,

    # =============================================================
    # REBALANCE SETTINGS
    # =============================================================
    # Options: "instant", "weekly", "monthly", "quarterly",
    #          "semiannual", "annual", "none" → buy & hold
    "rebalance_frequency": "instant",  # Instant typically yields best performance
    "rebalance_holiday_rule": "next_trading_day",
    # Global: "down_only", "up_only", "always", or "per_regime" (uses each regime's
    # rebalance_on_downward / rebalance_on_upward: "match" | "hold").
    "rebalance_strategy": "per_regime",
    "dividend_reinvestment": False,  # True: accrue to cash (or ticker) per target; sweeps on rebalance if cash
    "dividend_reinvestment_target": "cash",

    # =============================================================
    # TICKER CONFIGURATION
    # =============================================================
    "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],  # Used for pulling data and displaying normalized values
    "allocation_tickers": ["QQQ", "TQQQ", "XLU"],  # Tickers used inside portfolio regimes
    "minimum_allocation": 0.0,

    # =============================================================
    # REGIME DEFINITIONS — DEFAULTS (subject to change)
    # =============================================================
    # Drawdown bands measured against a 1y rolling QQQ peak.
    #
    # R1 — Ride High (0–6% dd): 100% TQQQ — leveraged equity in calm markets.
    # R2 — Cautious Defense (6–20% dd): mostly XLU, retains some QQQ exposure.
    # R3 — Contrarian Buyback (20%+ dd): TQQQ/QQQ mix — deep drawdowns historically
    #      precede strong recoveries; this is intentional offensive positioning.
    #
    # Signal overrides (per regime):
    #   • upside fires when composite signal_total > threshold (+6 max)
    #   • protection fires when composite signal_total < threshold (–6 min)
    #   • Both can be enabled; protection wins if both qualify.
    "regimes": {
        "R1": {
            "dd_low": 0.0,
            "dd_high": 0.06,
            "TQQQ": 1.0,
            "QQQ": 0.0,
            "XLU": 0.0,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": True,
                    "label": "Max Bull",
                    "direction": "above",
                    "threshold": 2,
                    "TQQQ": 1.0,
                    "QQQ": 0.0,
                    "XLU": 0.0,
                },
                "protection": {
                    "enabled": True,
                    "label": "Bull Fading",
                    "direction": "below",
                    "threshold": -2,
                    "TQQQ": 0.0,
                    "QQQ": 1.0,
                    "XLU": 0.0,
                },
            },
        },
        "R2": {
            "dd_low": 0.06,
            "dd_high": 0.20,
            "TQQQ": 0.0,
            "QQQ": 0.30,
            "XLU": 0.70,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": True,
                    "label": "Recovery Confirmed",
                    "direction": "above",
                    "threshold": 2,
                    "TQQQ": 0.0,
                    "QQQ": 0.70,
                    "XLU": 0.30,
                },
                "protection": {
                    "enabled": True,
                    "label": "Deteriorating Fast",
                    "direction": "below",
                    "threshold": -3,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 1.0,
                },
            },
        },
        "R3": {
            "dd_low": 0.20,
            "dd_high": 1.0,
            "TQQQ": 0.50,
            "QQQ": 0.50,
            "XLU": 0.0,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
            "signal_overrides": {
                "upside": {
                    "enabled": True,
                    "label": "Capitulation Reversal",
                    "direction": "above",
                    "threshold": 3,
                    "TQQQ": 0.80,
                    "QQQ": 0.20,
                    "XLU": 0.0,
                },
                "protection": {
                    "enabled": True,
                    "label": "Crisis Deepening",
                    "direction": "below",
                    "threshold": -4,
                    "TQQQ": 0.0,
                    "QQQ": 0.0,
                    "XLU": 1.0,
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
