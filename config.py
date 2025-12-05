
# config.py
# ======================================================================
# FAST MARKKY FUND — Core Configuration File
# ======================================================================
# This file controls:
#   • Backtest parameters
#   • Regime definitions
#   • Rebalancing rules
#   • Simulation behaviors
#   • Worst-case synthetic data generation
#
# NOTE:
#   Nothing in here performs computation. It is purely declarative.
# ======================================================================


CONFIG = {

    # =============================================================
    # PORTFOLIO STARTING PARAMETERS
    # =============================================================
    "starting_balance": 10000,
    # "start_date": "2023-01-02",

    # Worst case 2021
    "start_date": "2021-11-18",

    # Going way back
    # "start_date": "2010-12-02",

    #BTC Test
    # "start_date": "2017-01-04",
    "drawdown_ticker": "QQQ",

    # =============================================================
    # REBALANCE SETTINGS
    # =============================================================
    # Options:
    #   "daily", "weekly", "monthly", "quarterly", "semiannual", "annual"
    #   "none" → buy & hold
    #
    "rebalance_frequency": "weekly",
    "rebalance_holiday_rule": "next_trading_day",

    # Restricts how fast the portfolio can move between regimes.
    "max_regime_step": 6,

    # =============================================================
    # TICKER CONFIG
    # =============================================================
    # Normal backtests use these tickers.
    # Worst-case simulator will output synthetic price series
    # matching these names.
    #
    "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
    "allocation_tickers": ["QQQ", "TQQQ", "XLU"],

    # ETFs should not fall below 5% allocation (just a guardrail)
    "minimum_allocation": 0.00,

    # =============================================================
    # REGIME DEFINITIONS
    # =============================================================
    # Drawdown bands with allocations.

"regimes": {
# RIDE HIGH
    "R1": {
        "dd_low": 0.00,
        "dd_high": 0.03,
        "TQQQ": 0.70,
        "QQQ": 0.30,
        "XLU": 0.00
    },
# SAFE GUARD
    "R2": {
        "dd_low": 0.03,
        "dd_high": 0.10,
        "TQQQ": 0.20,
        "QQQ": 0.80,
        "XLU": 0.00
    },
# GET OUT OF POSITION
    "R3": {
        "dd_low": 0.10,
        "dd_high": 0.25,
        "TQQQ": 0.00,
        "QQQ": 1.00,
        "XLU": 0.00
    },
# SAFE BUY BACK
    "R4": {
        "dd_low": 0.25,
        "dd_high": 0.35,
        "TQQQ": 0.50,
        "QQQ": 0.50,
        "XLU": 0.00
    },
# BE AGGRESSIVE
    "R5": {
        "dd_low": 0.35,
        "dd_high": 1.00,
        "TQQQ": 0.70,
        "QQQ": 0.30,
        "XLU": 0.00
    }
},



    # =============================================================
    # DIVIDENDS
    # =============================================================
    "dividend_reinvest_into": "TQQQ",

    # =============================================================
    # DRAWDOWN CALCULATION
    # =============================================================
    "drawdown_method": "ATH",

    # =============================================================
    # TRADING SETTINGS
    # =============================================================
    "transaction_cost": 0.0,
    "allow_cash": False,

    # ======================================================================
    # WORST-CASE SIMULATION SETTINGS
    # ======================================================================
    # Whether to optionally run the synthetic 1950→present stress test.
    #
    # If True, worst_case_runner.py will ask to run a synthetic simulation
    # after the normal backtest.
    #
    "use_worst_case_simulation": False,

    # Benchmark ticker for synthetic modeling.
    # ^GSPC = S&P 500 index (goes back to ~1950)
    "benchmark_ticker": "QQQ",

    # Earliest date for worst-case synthetic data.
    # This may be changed by the user as needed.
    "worst_case_start_date": "1950-01-01",

    # Directory where synthetic CSVs will be versioned.
    "worst_case_output_dir": "simulation_outputs",

}

