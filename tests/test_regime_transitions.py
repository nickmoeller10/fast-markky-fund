"""
Asymmetric regime transition logic.
Protects: backtest.apply_per_regime_direction_strategy,
          backtest.apply_asymmetric_rules_down_only,
          backtest.apply_asymmetric_rules_up_only
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtest import (
    apply_asymmetric_rules_down_only,
    apply_asymmetric_rules_up_only,
    apply_per_regime_direction_strategy,
)


@pytest.fixture
def per_regime_match_all():
    return {
        "regimes": {
            "R1": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R2": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R3": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
        }
    }


@pytest.fixture
def production_per_regime_config():
    return {
        "regimes": {
            "R1": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R2": {"rebalance_on_downward": "match", "rebalance_on_upward": "hold"},
            "R3": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
        }
    }


@pytest.mark.unit
def test_per_regime_first_day_aligns_with_market(per_regime_match_all):
    result = apply_per_regime_direction_strategy("R1", None, "R2", per_regime_match_all)
    assert result == "R2"


@pytest.mark.unit
def test_per_regime_no_change_returns_portfolio(per_regime_match_all):
    result = apply_per_regime_direction_strategy("R1", "R1", "R1", per_regime_match_all)
    assert result == "R1"


@pytest.mark.unit
def test_per_regime_downward_match(per_regime_match_all):
    result = apply_per_regime_direction_strategy("R1", "R1", "R2", per_regime_match_all)
    assert result == "R2"


@pytest.mark.unit
def test_per_regime_upward_match(per_regime_match_all):
    result = apply_per_regime_direction_strategy("R2", "R2", "R1", per_regime_match_all)
    assert result == "R1"


@pytest.mark.unit
def test_per_regime_partial_recovery_hold(production_per_regime_config):
    result = apply_per_regime_direction_strategy("R3", "R3", "R2", production_per_regime_config)
    assert result == "R3", "Portfolio must hold at R3 when market only partially recovers to R2"


@pytest.mark.unit
def test_per_regime_full_recovery_matches(production_per_regime_config):
    result = apply_per_regime_direction_strategy("R3", "R2", "R1", production_per_regime_config)
    assert result == "R1", "Portfolio must follow to R1 when market fully recovers"


@pytest.mark.unit
def test_per_regime_sequence_full_roundtrip(production_per_regime_config):
    """R1→R2→R3→R2→R1 sequence with production config: R1 → R2 → R3 → R3 (hold) → R1."""
    cfg = production_per_regime_config
    steps = [
        ("R1", None, "R1", "R1"),
        ("R1", "R1", "R2", "R2"),
        ("R2", "R2", "R3", "R3"),
        ("R3", "R3", "R2", "R3"),
        ("R3", "R2", "R1", "R1"),
    ]
    for portfolio, prev_mkt, new_mkt, expected in steps:
        result = apply_per_regime_direction_strategy(portfolio, prev_mkt, new_mkt, cfg)
        assert result == expected, (
            f"Step {prev_mkt}→{new_mkt} (portfolio={portfolio}): expected {expected}, got {result}"
        )


@pytest.mark.unit
def test_per_regime_none_market_regime_returns_portfolio(per_regime_match_all):
    result = apply_per_regime_direction_strategy("R2", "R1", None, per_regime_match_all)
    assert result == "R2"


@pytest.mark.unit
def test_down_only_follows_down():
    assert apply_asymmetric_rules_down_only("R1", "R2") == "R2"


@pytest.mark.unit
def test_down_only_holds_on_partial_recovery():
    assert apply_asymmetric_rules_down_only("R2", "R2") == "R2"


@pytest.mark.unit
def test_down_only_only_recovers_at_r1():
    assert apply_asymmetric_rules_down_only("R2", "R2") == "R2"


@pytest.mark.unit
def test_down_only_r3_to_r2_holds():
    assert apply_asymmetric_rules_down_only("R3", "R2") == "R3"


@pytest.mark.unit
def test_down_only_r2_to_r1_recovers():
    assert apply_asymmetric_rules_down_only("R2", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r3_to_r1_recovers():
    assert apply_asymmetric_rules_down_only("R3", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r1_flat_stays_r1():
    assert apply_asymmetric_rules_down_only("R1", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r1_to_r3_follows():
    assert apply_asymmetric_rules_down_only("R1", "R3") == "R3"


@pytest.mark.unit
def test_up_only_follows_up():
    assert apply_asymmetric_rules_up_only("R2", "R1", 3) == "R1"


@pytest.mark.unit
def test_up_only_holds_on_down():
    assert apply_asymmetric_rules_up_only("R1", "R2", 3) == "R1"


@pytest.mark.unit
def test_up_only_bottom_regime_follows_up():
    assert apply_asymmetric_rules_up_only("R3", "R2", 3) == "R2"


@pytest.mark.unit
def test_up_only_bottom_regime_always_catches_r3():
    assert apply_asymmetric_rules_up_only("R1", "R3", 3) == "R3"


@pytest.mark.unit
def test_up_only_r1_flat_stays_r1():
    assert apply_asymmetric_rules_up_only("R1", "R1", 3) == "R1"


_REGIMES = ["R1", "R2", "R3"]
_REGIME_STRAT = st.sampled_from(_REGIMES)


@given(
    portfolio=_REGIME_STRAT,
    prev_mkt=_REGIME_STRAT,
    mkt=_REGIME_STRAT,
)
@settings(max_examples=300)
def test_per_regime_always_returns_valid_label(portfolio, prev_mkt, mkt):
    cfg = {
        "regimes": {
            "R1": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R2": {"rebalance_on_downward": "match", "rebalance_on_upward": "hold"},
            "R3": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
        }
    }
    result = apply_per_regime_direction_strategy(portfolio, prev_mkt, mkt, cfg)
    assert result in _REGIMES, f"Got invalid regime label: {result!r}"


@given(
    portfolio=_REGIME_STRAT,
    market=_REGIME_STRAT,
)
@settings(max_examples=200)
def test_down_only_always_valid(portfolio, market):
    result = apply_asymmetric_rules_down_only(portfolio, market)
    assert result in _REGIMES
