# config.py

CONFIG = {
    "starting_balance": 10000,
    "start_date": "2010-12-01",

    # ============================================================
    # REBALANCE SETTINGS
    # ============================================================
    # Supported rebalance frequencies:
    #   "none"        → never rebalance (buy & hold)
    #   "daily"       → rebalance every trading day
    #   "weekly"      → rebalance every Friday
    #   "monthly"     → rebalance at end of each month
    #   "quarterly"   → rebalance at end of each quarter  (DEFAULT)
    #   "semiannual"  → rebalance every 6 months
    #   "annual"      → rebalance at end of the year
    #
    # NOTE: Rebalancing is automatically disabled if only one ticker is present.
    #
    "rebalance_frequency": "quarterly",

    # ============================================================
    # TICKERS
    # ============================================================
    "tickers": ["TQQQ", "QQQ", "XLU", "VOO"],
    "allocation_tickers": ["TQQQ", "QQQ", "XLU"],
    "max_regime_step": 6,

    # ============================================================
    # ALLOCATION RULES by REGIME
    # ============================================================
    "minimum_allocation": 0.05,

"regimes": {

    "R1": {
        "dd_low": 0.00,
        "dd_high": 0.03,
        "TQQQ": 0.05,
        "QQQ": 0.50,
        "XLU": 0.45
    },

    "R2": {
        "dd_low": 0.03,
        "dd_high": 0.10,
        "TQQQ": 0.10,
        "QQQ": 0.50,
        "XLU": 0.40
    },

    "R3": {
        "dd_low": 0.10,
        "dd_high": 0.20,
        "TQQQ": 0.15,
        "QQQ": 0.50,
        "XLU": 0.35
    },

    "R4": {
        "dd_low": 0.20,
        "dd_high": 0.30,
        "TQQQ": 0.25,
        "QQQ": 0.50,
        "XLU": 0.25
    },

    "R5": {
        "dd_low": 0.30,
        "dd_high": 0.40,
        "TQQQ": 0.45,
        "QQQ": 0.50,
        "XLU": 0.05
    },

    "R6": {
        "dd_low": 0.40,
        "dd_high": 1.00,
        "TQQQ": 0.60,
        "QQQ": 0.40,
        "XLU": 0.00
    }
},


    # ============================================================
    # DIVIDENDS / CASH
    # ============================================================
    "dividend_reinvest_into": "TQQQ",

    # ============================================================
    # DRAWDOWN CALCULATION
    # ============================================================
    # Options:
    #   "ATH" → drawdown from all-time high (default)
    #   (future: rolling window, moving high, etc.)
    #
    "drawdown_method": "ATH",

    # ============================================================
    # MISC
    # ============================================================
    "rebalance_holiday_rule": "next_trading_day",
    "transaction_cost": 0.0,
    "allow_cash": False
}
