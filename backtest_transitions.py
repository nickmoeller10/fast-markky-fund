"""
Regime-transition strategies + initial allocation.

Pure functions (no I/O, no state). Tested directly by tests/test_regime_transitions.py.

The trickiest entry here is `apply_per_regime_direction_strategy`: it consults
the *target* regime's `rebalance_on_upward` / `rebalance_on_downward` keys,
not the source's. See CLAUDE.md "Per-regime hold/match — THE TRICKY ONE".
"""
from __future__ import annotations

from allocation_engine import get_allocation_for_regime, tradable_allocation

from backtest_helpers import bottom_regime_number, regime_label, regime_number


def get_initial_allocation(start_date, price_df, first_dd, config, regime_detector, rebalance_fn):
    market_regime = regime_detector(first_dd, config)
    portfolio_regime = market_regime
    alloc = get_allocation_for_regime(portfolio_regime, config)
    alloc_t = tradable_allocation(alloc, price_df.loc[start_date], config)
    if not alloc_t:
        return market_regime, portfolio_regime, None
    shares = rebalance_fn(config["starting_balance"], alloc_t, price_df.loc[start_date])
    return market_regime, portfolio_regime, shares


def apply_asymmetric_rules_down_only(prev_regime, market_regime):
    """
    Regime Shift Down Only: Rebalance immediately when market goes DOWN,
    but only move back UP when market fully recovers to R1.
    """
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    if mkt_n > prev_n:
        return regime_label(mkt_n)
    if mkt_n == 1:
        return "R1"
    return prev_regime


def apply_asymmetric_rules_up_only(prev_regime, market_regime, bottom_regime_num: int = 3):
    """
    Regime Shift Up Only: Rebalance immediately when market goes UP,
    but hold position when market goes DOWN (except when reaching bottom).
    When in the deepest regime (e.g. R4), rebalance on every way up.
    """
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    if mkt_n < prev_n:
        return regime_label(mkt_n)
    if prev_n == bottom_regime_num and mkt_n < bottom_regime_num:
        return regime_label(mkt_n)
    if mkt_n == bottom_regime_num:
        return regime_label(bottom_regime_num)
    return prev_regime


def apply_always_rebalance(prev_regime, market_regime):
    """Rebalance whenever market regime changes, regardless of direction."""
    return market_regime


def regime_trajectory_label(prev_market_regime, market_regime):
    """
    R1 = lowest drawdown stress (rank 1); higher Rn = deeper drawdown bands.
    Downward = market worsened (higher R#). Upward = improved (lower R#).
    """
    if prev_market_regime is None or market_regime is None:
        return ""
    try:
        pn = regime_number(prev_market_regime)
        cn = regime_number(market_regime)
    except (ValueError, TypeError):
        return ""
    if cn > pn:
        return "Downward"
    if cn < pn:
        return "Upward"
    return "Flat"


def _regime_rebalance_mode(regime_params, direction: str) -> str:
    """direction: 'downward' | 'upward'. Returns 'match' or 'hold'."""
    key = "rebalance_on_downward" if direction == "downward" else "rebalance_on_upward"
    raw = regime_params.get(key, "match")
    if raw is None:
        return "match"
    s = str(raw).strip().lower()
    if s in ("hold", "ignore", "no", "false"):
        return "hold"
    return "match"


def apply_per_regime_direction_strategy(
    portfolio_regime, prev_market_regime, market_regime, config
):
    """
    When the market regime *changes* from prev_market_regime to market_regime,
    decide whether portfolio_regime updates to market_regime using the *target*
    regime's rebalance_on_downward / rebalance_on_upward ('match' | 'hold').
    Missing keys default to 'match'. First day (prev None): align with market.
    """
    if market_regime is None:
        return portfolio_regime
    if prev_market_regime is None:
        return market_regime
    if prev_market_regime == market_regime:
        return portfolio_regime

    try:
        pn = regime_number(prev_market_regime)
        cn = regime_number(market_regime)
    except (ValueError, TypeError):
        return market_regime

    regimes = config.get("regimes") or {}
    target_params = regimes.get(market_regime, {})
    direction = "downward" if cn > pn else "upward"
    if _regime_rebalance_mode(target_params, direction) == "match":
        return market_regime
    return portfolio_regime


def apply_rebalancing_strategy(prev_regime, market_regime, strategy, config=None):
    """Dispatch to the named strategy. Defaults to down_only for unknown values."""
    bottom_n = bottom_regime_number(config) if config is not None else 3
    if strategy == "per_regime":
        # Caller should use apply_per_regime_direction_strategy with prev_market_regime;
        # this branch is a fallback if mis-invoked.
        return apply_always_rebalance(prev_regime, market_regime)
    if strategy == "down_only":
        return apply_asymmetric_rules_down_only(prev_regime, market_regime)
    if strategy == "up_only":
        return apply_asymmetric_rules_up_only(prev_regime, market_regime, bottom_n)
    if strategy == "always":
        return apply_always_rebalance(prev_regime, market_regime)
    return apply_asymmetric_rules_down_only(prev_regime, market_regime)


# Backward-compat alias used by validate_tests.py
apply_asymmetric_rules = apply_asymmetric_rules_down_only
