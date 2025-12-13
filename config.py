# ======================================================================
# FAST MARKKY FUND — Core Configuration File
# ======================================================================
# Declarative settings for:
#   • Backtest parameters
#   • Regime definitions
#   • Rebalancing rules
#   • Simulation behaviors
#   • Worst-case synthetic data generation
# ======================================================================

CONFIG = {

    # =============================================================
    # PORTFOLIO STARTING PARAMETERS
    # =============================================================
    "starting_balance": 10000,

    # "start_date": "2021-11-18",   # Worst-case 2021
    "start_date": "2010-12-02",     # Near TQQQ start

    # "end_date": None,              # Explicitly use current date (default behavior)
    "end_date": "2021-12-31",   

    "drawdown_ticker": "QQQ", # The Ticker we want to measure drawdown from

    # =============================================================
    # REBALANCE SETTINGS
    # =============================================================
    # Options:
    #   "instant", "weekly", "monthly", "quarterly",
    #   "semiannual", "annual", "none" → buy & hold
    #
    # "rebalance_frequency": "weekly",
    "rebalance_frequency": "instant", # Instant yields best performance typically
    "rebalance_holiday_rule": "next_trading_day",

    # =============================================================
    # TICKER CONFIGURATION
    # =============================================================
    "tickers": ["QQQ", "TQQQ", "XLU", "SPY"], # Used for pulling data and displaying normalized values in data and charts
    "allocation_tickers": ["QQQ", "TQQQ", "XLU"], # The actual tickers used in the portfolio regimes
    "minimum_allocation": 0.00,  # Guardrail

    # =============================================================
    # REGIME DEFINITIONS
    # =============================================================
    "regimes": {

        # R1 — Ride High
        "R1": {
            "dd_low": 0.00,
            "dd_high": 0.06,
            "TQQQ": 1.00,
            "QQQ": 0.00,
            "XLU": 0.00,
        },

        # R2 — Safeguard
        "R2": {
            "dd_low": 0.06,
            "dd_high": 0.28,
            "TQQQ": 0.00,
            "QQQ": 0.00,
            "XLU": 1.00,
        },

        # R3 — Safe Buyback
        "R3": {
            "dd_low": 0.28,
            "dd_high": 1.00,
            "TQQQ": 1.00,
            "QQQ": 0.00,
            "XLU": 0.00,
        },

        # "R4": {
        #     "dd_low": 0.30,
        #     "dd_high": 1.00,
        #     "TQQQ": 0.50,
        #     "QQQ": 0.00,
        #     "XLU": 0.50,
        # },
    },

    # ======================================================================
    # WORST-CASE SIMULATION SETTINGS
    # ======================================================================
    "use_worst_case_simulation": False,
    "benchmark_ticker": "QQQ",
    "worst_case_start_date": "1950-01-01",
    "worst_case_output_dir": "simulation_outputs",
}
