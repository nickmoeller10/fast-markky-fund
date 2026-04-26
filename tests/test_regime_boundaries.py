"""
Regime detection boundary correctness.
Protects: regime_engine.determine_regime(), regime_engine.compute_drawdown_from_ath()
"""
import math
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from regime_engine import determine_regime, compute_drawdown_from_ath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def std_config():
    return {
        "regimes": {
            "R1": {"dd_low": 0.0, "dd_high": 0.08},
            "R2": {"dd_low": 0.08, "dd_high": 0.28},
            "R3": {"dd_low": 0.28, "dd_high": 1.0},
        }
    }


# ---------------------------------------------------------------------------
# Boundary correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_exact_lower_r2_boundary(std_config):
    # dd=0.08 is the exact lower bound of R2 → must land in R2 (low <= dd < high for non-last)
    assert determine_regime(0.08, std_config) == "R2"


@pytest.mark.unit
def test_exact_upper_r3_boundary(std_config):
    # dd=1.0 → R3 (last regime uses inclusive high: low <= dd <= high)
    assert determine_regime(1.0, std_config) == "R3"


@pytest.mark.unit
def test_zero_drawdown_is_r1(std_config):
    assert determine_regime(0.0, std_config) == "R1"


@pytest.mark.unit
def test_full_drawdown_is_r3(std_config):
    # Total loss: dd=1.0 must always be caught by the last regime
    assert determine_regime(1.0, std_config) == "R3"


@pytest.mark.unit
def test_nan_drawdown_returns_none(std_config):
    assert determine_regime(float("nan"), std_config) is None


@pytest.mark.unit
def test_none_drawdown_returns_none(std_config):
    assert determine_regime(None, std_config) is None


@pytest.mark.unit
def test_boundary_r1_r2_just_below(std_config):
    # 0.0799 is in R1 (< 0.08 threshold)
    assert determine_regime(0.0799, std_config) == "R1"


@pytest.mark.unit
def test_boundary_r2_r3_just_below(std_config):
    # 0.2799 is in R2 (< 0.28 threshold)
    assert determine_regime(0.2799, std_config) == "R2"


@pytest.mark.unit
def test_boundary_r2_r3_exact(std_config):
    # 0.28 is the exact lower bound of R3
    assert determine_regime(0.28, std_config) == "R3"


@pytest.mark.unit
def test_midpoint_r2(std_config):
    assert determine_regime(0.18, std_config) == "R2"


@pytest.mark.unit
def test_midpoint_r3(std_config):
    assert determine_regime(0.50, std_config) == "R3"


@pytest.mark.unit
def test_numpy_float_input(std_config):
    # np.float64 inputs must be handled (not crash on formatting)
    assert determine_regime(np.float64(0.05), std_config) == "R1"
    assert determine_regime(np.float64(0.08), std_config) == "R2"


# ---------------------------------------------------------------------------
# compute_drawdown_from_ath mechanics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_drawdown_monotonic_rising_prices_is_zero():
    prices = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
    dd, ath = compute_drawdown_from_ath(prices)
    assert (dd == 0.0).all(), "Rising prices must always have 0% drawdown"


@pytest.mark.unit
def test_drawdown_recovery_returns_to_zero():
    prices = pd.Series([100.0, 80.0, 90.0, 100.0, 105.0])
    dd, ath = compute_drawdown_from_ath(prices)
    assert dd.iloc[-1] == pytest.approx(0.0), "After full recovery ATH resets and dd=0"


@pytest.mark.unit
def test_drawdown_partial_recovery():
    # Falls 20% then recovers to -10%
    prices = pd.Series([100.0, 80.0, 90.0])
    dd, ath = compute_drawdown_from_ath(prices)
    assert dd.iloc[-1] == pytest.approx(0.10, abs=1e-9)


@pytest.mark.unit
def test_ath_tracks_correctly():
    prices = pd.Series([100.0, 120.0, 90.0, 130.0, 125.0])
    dd, ath = compute_drawdown_from_ath(prices)
    assert float(ath.iloc[2]) == pytest.approx(120.0)  # ATH at 90 is still 120
    assert float(ath.iloc[4]) == pytest.approx(130.0)  # ATH updated to 130


# ---------------------------------------------------------------------------
# Hypothesis property test
# ---------------------------------------------------------------------------

@given(dd_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_valid_drawdown_always_returns_regime(dd_value):
    """Any dd in [0, 1] with a complete regime config must return a non-None regime."""
    config = {
        "regimes": {
            "R1": {"dd_low": 0.0, "dd_high": 0.08},
            "R2": {"dd_low": 0.08, "dd_high": 0.28},
            "R3": {"dd_low": 0.28, "dd_high": 1.0},
        }
    }
    result = determine_regime(dd_value, config)
    assert result is not None, f"dd={dd_value} returned None — regime coverage gap"
    assert result in ("R1", "R2", "R3")
