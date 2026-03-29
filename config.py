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

    "start_date": "1999-01-04",  # Arbitrary US trading day (Monday) in 1999

    "end_date": "2026-03-27",

    "drawdown_ticker": "QQQ", # The Ticker we want to measure drawdown from

    # False: standard ATH (cummax from inception + pre-start history download).
    # True: rolling peak over the last N calendar years (fallback to standard ATH until N years exist).
    "drawdown_window_enabled": True,
    "drawdown_window_years": 2,

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
    # Global: "down_only", "up_only", "always", or "per_regime" (uses each regime's
    # rebalance_on_downward / rebalance_on_upward: "match" | "hold").
    "rebalance_strategy": "per_regime",
    "dividend_reinvestment": False,  # True: accrue to cash (or ticker) per target; sweeps on rebalance if cash
    "dividend_reinvestment_target": "cash",

    # =============================================================
    # TICKER CONFIGURATION
    # =============================================================
    "tickers": ["QQQ", "TQQQ", "XLU", "SPY"], # Used for pulling data and displaying normalized values in data and charts
    "allocation_tickers": ["QQQ", "TQQQ", "XLU"], # The actual tickers used in the portfolio regimes
    "minimum_allocation": 0.0,

    # =============================================================
    # REGIME DEFINITIONS
    # =============================================================
    # 2y rolling QQQ peak. Per-regime rebalance_on_*: match | hold.
    "regimes": {
        "R1": {
            "dd_low": 0.00,
            "dd_high": 0.04,
            "TQQQ": 1.00,
            "QQQ": 0.00,
            "XLU": 0.00,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
        },
        "R2": {
            "dd_low": 0.04,
            "dd_high": 0.28,
            "TQQQ": 0.00,
            "QQQ": 0.00,
            "XLU": 1.00,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "hold",
        },
        "R3": {
            "dd_low": 0.25,
            "dd_high": 0.30,
            "TQQQ": 1.00,
            "QQQ": 0.00,
            "XLU": 0.00,
            "rebalance_on_downward": "hold",
            "rebalance_on_upward": "match",
        },
        "R4": {
            "dd_low": 0.30,
            "dd_high": 1.00,
            "TQQQ": 0.00,
            "QQQ": 0.00,
            "XLU": 1.00,
            "rebalance_on_downward": "match",
            "rebalance_on_upward": "match",
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
