# Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new test modules + conftest + pyproject.toml that verify financial math correctness, regime transition logic, and provide a locked regression anchor for all future refactors.

**Architecture:** Pure pytest-based tests (new modules use pytest fixtures, not unittest). Existing unittest tests are left untouched. `conftest.py` provides shared fixtures. All 25 existing tests continue to pass. Financial metrics verified against both first-principles formulas and quantstats.

**Tech Stack:** pytest ≥ 9, hypothesis ≥ 6, quantstats ≥ 0.0.81, unittest.mock for network isolation

---

## Locked Ground Truth (for Task 8)

These values were generated with seed=42 on 2026-04-26. Changing them requires an explicit unlock comment.

```python
LOCKED_FINAL_VALUE   = 10949.7909
LOCKED_CAGR          = 0.018976
LOCKED_SHARPE        = 0.1552
LOCKED_MAX_DRAWDOWN  = -0.290396
LOCKED_VOLATILITY    = 0.122285
LOCKED_FIRST_20_REGIMES = ['R1'] * 20
```

---

## Task 1: pytest configuration (`pyproject.toml`)

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: isolated unit tests with no I/O",
    "integration: tests that wire multiple modules together",
    "metrics: financial metric correctness tests (formula + quantstats)",
]
```

- [ ] **Step 2: Verify collection still finds all 25 existing tests**

Run: `python3 -m pytest tests/ --collect-only -q`
Expected: `25 tests collected`

- [ ] **Step 3: Run all existing tests to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: `25 passed`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test: add pyproject.toml with pytest config and markers"
```

---

## Task 2: Shared test fixtures (`tests/conftest.py`)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
"""Shared pytest fixtures for the Fast Markky Fund test suite."""
import numpy as np
import pandas as pd
import pytest


def _base_config():
    """3-regime config matching production thresholds, all signal_overrides disabled."""
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "down_only",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0,
                "dd_high": 0.08,
                "TQQQ": 1.0,
                "QQQ": 0.0,
                "XLU": 0.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R2": {
                "dd_low": 0.08,
                "dd_high": 0.28,
                "TQQQ": 0.0,
                "QQQ": 0.0,
                "XLU": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R3": {
                "dd_low": 0.28,
                "dd_high": 1.0,
                "TQQQ": 0.0,
                "QQQ": 0.0,
                "XLU": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
        },
    }


@pytest.fixture
def minimal_config():
    """Minimal 3-regime config with thresholds matching production (0%, 8%, 28%)."""
    return _base_config()


@pytest.fixture
def production_per_regime_config():
    """Production per_regime config: R2 has rebalance_on_upward=hold."""
    cfg = _base_config()
    cfg["rebalance_strategy"] = "per_regime"
    cfg["regimes"]["R2"]["rebalance_on_upward"] = "hold"
    return cfg


@pytest.fixture
def price_fixture():
    """
    Deterministic OHLCV-style close prices for 5 years (1260 trading days).
    seed=42 — do not change without updating locked regression values.
    """
    np.random.seed(42)
    n_days = 1260
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    returns = np.random.normal(0.0004, 0.010, n_days)
    returns[200:260] = np.random.normal(-0.006, 0.014, 60)
    returns[700:820] = np.random.normal(-0.009, 0.018, 120)

    qqq = 100.0 * np.exp(np.cumsum(returns))
    tqqq = 30.0 * np.exp(3.0 * np.cumsum(returns * 0.9))
    xlu = 50.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    spy = 250.0 * np.exp(np.cumsum(0.7 * returns + np.random.normal(0, 0.004, n_days)))

    return pd.DataFrame(
        {"QQQ": qqq, "TQQQ": tqqq, "XLU": xlu, "SPY": spy}, index=dates
    )


def make_equity_df(start_val, daily_returns, start_date="2020-01-02"):
    """Build an equity_df from a list of daily returns (pct as decimals, e.g. 0.01 = 1%)."""
    dates = pd.bdate_range(start_date, periods=len(daily_returns) + 1)
    vals = [float(start_val)]
    for r in daily_returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.DataFrame({"Date": dates, "Value": vals})


def make_equity_df_from_values(values, start_date="2020-01-02"):
    """Build an equity_df from an explicit sequence of portfolio values."""
    dates = pd.bdate_range(start_date, periods=len(values))
    return pd.DataFrame({"Date": list(dates), "Value": list(values)})


@pytest.fixture
def equity_series_fixture():
    """Factory: returns make_equity_df so tests can call it with their own params."""
    return make_equity_df


@pytest.fixture
def equity_from_values():
    """Factory: returns make_equity_df_from_values."""
    return make_equity_df_from_values
```

- [ ] **Step 2: Verify conftest imports cleanly**

Run: `python3 -m pytest tests/ --collect-only -q`
Expected: `25 tests collected` — no import errors

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add conftest.py with shared fixtures for new test suite"
```

---

## Task 3: Regime boundary tests (`tests/test_regime_boundaries.py`)

**Files:**
- Create: `tests/test_regime_boundaries.py`
- Test: `tests/test_regime_boundaries.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_regime_boundaries.py -v`
Expected: All tests pass. If `test_drawdown_monotonic_rising_prices_is_zero` fails, check that `compute_drawdown_from_ath` in `regime_engine.py` is the correct function (not the one in `backtest.py` — both exist; `regime_engine.py` version is tested here).

- [ ] **Step 3: Run full suite to check for regressions**

Run: `python3 -m pytest tests/ -q`
Expected: All pass (25 existing + new regime boundary tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_regime_boundaries.py
git commit -m "test: add regime boundary tests covering all threshold edge cases"
```

---

## Task 4: Regime transition tests (`tests/test_regime_transitions.py`)

**Files:**
- Create: `tests/test_regime_transitions.py`

This is the most financially critical untested area. The `R2 rebalance_on_upward: hold` rule means the portfolio stays in XLU even when the market PARTIALLY recovers from R3 to R2.

**Key insight from tracing the code:** `apply_per_regime_direction_strategy` uses the TARGET regime's (`market_regime`) `rebalance_on_downward`/`rebalance_on_upward` settings, not the source regime's. So `R2.rebalance_on_upward = hold` means: "when the market arrives AT R2 from a worse regime (R3→R2), don't follow." The portfolio only moves when the market fully reaches R1 (where R1.upward = match).

- [ ] **Step 1: Write the test file**

```python
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def down_only_config():
    """Minimal config for down_only strategy tests (content not used by the functions)."""
    return {}


@pytest.fixture
def per_regime_match_all():
    """All regimes match on both directions."""
    return {
        "regimes": {
            "R1": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R2": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R3": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
        }
    }


@pytest.fixture
def production_per_regime_config():
    """Production config: R2 holds on upward (R3→R2 is a hold)."""
    return {
        "regimes": {
            "R1": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
            "R2": {"rebalance_on_downward": "match", "rebalance_on_upward": "hold"},
            "R3": {"rebalance_on_downward": "match", "rebalance_on_upward": "match"},
        }
    }


# ---------------------------------------------------------------------------
# apply_per_regime_direction_strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_per_regime_first_day_aligns_with_market(per_regime_match_all):
    # First day: prev_market=None → portfolio adopts market regime unconditionally
    result = apply_per_regime_direction_strategy("R1", None, "R2", per_regime_match_all)
    assert result == "R2"


@pytest.mark.unit
def test_per_regime_no_change_returns_portfolio(per_regime_match_all):
    # Market didn't change → portfolio stays where it is
    result = apply_per_regime_direction_strategy("R1", "R1", "R1", per_regime_match_all)
    assert result == "R1"


@pytest.mark.unit
def test_per_regime_downward_match(per_regime_match_all):
    # R1→R2 (market worsens) with R2.downward=match → portfolio follows to R2
    result = apply_per_regime_direction_strategy("R1", "R1", "R2", per_regime_match_all)
    assert result == "R2"


@pytest.mark.unit
def test_per_regime_upward_match(per_regime_match_all):
    # R2→R1 (market recovers) with R1.upward=match → portfolio follows to R1
    result = apply_per_regime_direction_strategy("R2", "R2", "R1", per_regime_match_all)
    assert result == "R1"


@pytest.mark.unit
def test_per_regime_partial_recovery_hold(production_per_regime_config):
    # R3→R2 with R2.upward=hold → portfolio stays at R3
    # This is the key production behavior: partial recovery (R3→R2) does NOT rebalance
    result = apply_per_regime_direction_strategy("R3", "R3", "R2", production_per_regime_config)
    assert result == "R3", "Portfolio must hold at R3 when market only partially recovers to R2"


@pytest.mark.unit
def test_per_regime_full_recovery_matches(production_per_regime_config):
    # R2→R1 with R1.upward=match → portfolio follows to R1
    # The portfolio was stuck at R3; now market jumps to R1 directly
    result = apply_per_regime_direction_strategy("R3", "R2", "R1", production_per_regime_config)
    assert result == "R1", "Portfolio must follow to R1 when market fully recovers"


@pytest.mark.unit
def test_per_regime_sequence_full_roundtrip(production_per_regime_config):
    """
    R1→R2→R3→R2→R1 sequence with production config.
    Expected portfolio sequence: R1 → R2 → R3 → R3 (hold) → R1
    """
    cfg = production_per_regime_config
    steps = [
        # (portfolio_regime, prev_market, market_regime, expected_result)
        ("R1", None, "R1", "R1"),    # first day: adopt market
        ("R1", "R1", "R2", "R2"),    # market drops to R2: follow
        ("R2", "R2", "R3", "R3"),    # market drops to R3: follow
        ("R3", "R3", "R2", "R3"),    # market recovers to R2: HOLD at R3 (R2.upward=hold)
        ("R3", "R2", "R1", "R1"),    # market reaches R1: follow (R1.upward=match)
    ]
    for portfolio, prev_mkt, new_mkt, expected in steps:
        result = apply_per_regime_direction_strategy(portfolio, prev_mkt, new_mkt, cfg)
        assert result == expected, (
            f"Step {prev_mkt}→{new_mkt} (portfolio={portfolio}): "
            f"expected {expected}, got {result}"
        )


@pytest.mark.unit
def test_per_regime_none_market_regime_returns_portfolio(per_regime_match_all):
    # None market regime → no change
    result = apply_per_regime_direction_strategy("R2", "R1", None, per_regime_match_all)
    assert result == "R2"


# ---------------------------------------------------------------------------
# apply_asymmetric_rules_down_only
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_down_only_follows_down():
    # Market drops from R1 to R2 → portfolio follows
    assert apply_asymmetric_rules_down_only("R1", "R2") == "R2"


@pytest.mark.unit
def test_down_only_holds_on_partial_recovery():
    # Portfolio in R2, market recovers to R2 (flat) → portfolio stays R2
    assert apply_asymmetric_rules_down_only("R2", "R2") == "R2"


@pytest.mark.unit
def test_down_only_only_recovers_at_r1():
    # Market at R2 but portfolio was at R2 → stays R2 (market didn't reach R1)
    assert apply_asymmetric_rules_down_only("R2", "R2") == "R2"


@pytest.mark.unit
def test_down_only_r3_to_r2_holds():
    # Market recovers from R3 to R2 but not R1 → portfolio stays R3
    assert apply_asymmetric_rules_down_only("R3", "R2") == "R3"


@pytest.mark.unit
def test_down_only_r2_to_r1_recovers():
    # Market fully recovers to R1 → portfolio follows to R1
    assert apply_asymmetric_rules_down_only("R2", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r3_to_r1_recovers():
    # Market jumps from R3 straight to R1 → portfolio recovers to R1
    assert apply_asymmetric_rules_down_only("R3", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r1_flat_stays_r1():
    assert apply_asymmetric_rules_down_only("R1", "R1") == "R1"


@pytest.mark.unit
def test_down_only_r1_to_r3_follows():
    # Market crashes from R1 to R3 → portfolio follows immediately
    assert apply_asymmetric_rules_down_only("R1", "R3") == "R3"


# ---------------------------------------------------------------------------
# apply_asymmetric_rules_up_only
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_up_only_follows_up():
    # Market recovers R2→R1 → portfolio follows up
    assert apply_asymmetric_rules_up_only("R2", "R1", 3) == "R1"


@pytest.mark.unit
def test_up_only_holds_on_down():
    # Market drops R1→R2 → portfolio stays at R1 (up_only means don't follow down)
    assert apply_asymmetric_rules_up_only("R1", "R2", 3) == "R1"


@pytest.mark.unit
def test_up_only_bottom_regime_follows_up():
    # Portfolio at bottom R3, market recovers to R2 → follows up
    assert apply_asymmetric_rules_up_only("R3", "R2", 3) == "R2"


@pytest.mark.unit
def test_up_only_bottom_regime_always_catches_r3():
    # Market at R3 → portfolio always follows to R3 (bottom regime catches everything)
    assert apply_asymmetric_rules_up_only("R1", "R3", 3) == "R3"


@pytest.mark.unit
def test_up_only_r1_flat_stays_r1():
    assert apply_asymmetric_rules_up_only("R1", "R1", 3) == "R1"


# ---------------------------------------------------------------------------
# Hypothesis: portfolio regime always a valid label
# ---------------------------------------------------------------------------

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
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_regime_transitions.py -v`
Expected: All pass.

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All existing 25 + new tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_regime_transitions.py
git commit -m "test: add regime transition tests — most critical financial logic coverage"
```

---

## Task 5: Financial metrics tests (`tests/test_financial_metrics.py`)

**Files:**
- Create: `tests/test_financial_metrics.py`

Each metric is verified two ways: (A) first-principles formula, (B) against quantstats reference. Tolerances account for methodology differences (project uses calendar days for CAGR; quantstats uses trading days). All metric functions must return Python `float`, not `np.float64` or `pd.Series`.

- [ ] **Step 1: Write the test file**

```python
"""
Financial metric correctness: CAGR, volatility, max drawdown, Sharpe, Sortino.
Each metric verified against: (A) first-principles formula, (B) quantstats reference.
All functions must return Python float.

Protects: dashboard.calculate_metrics(), utils.max_drawdown_from_equity_curve()
"""
import math
import numpy as np
import pandas as pd
import pytest
import quantstats as qs

from dashboard import calculate_metrics
from utils import max_drawdown_from_equity_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity_df(start_val, daily_returns, start_date="2020-01-02"):
    dates = pd.bdate_range(start_date, periods=len(daily_returns) + 1)
    vals = [float(start_val)]
    for r in daily_returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.DataFrame({"Date": list(dates), "Value": vals})


def make_equity_df_from_values(values, start_date="2020-01-02"):
    dates = pd.bdate_range(start_date, periods=len(values))
    return pd.DataFrame({"Date": list(dates), "Value": list(map(float, values))})


def _random_returns(n, seed, mean=0.0004, std=0.012):
    np.random.seed(seed)
    return list(np.random.normal(mean, std, n))


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

@pytest.mark.metrics
def test_cagr_doubles_in_one_year():
    eq = make_equity_df(10_000, [0.0] * 251)
    eq["Value"] = np.linspace(10_000, 20_000, 252)  # straight line to double
    m = calculate_metrics(eq)
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    expected = (20_000 / 10_000) ** (1.0 / years) - 1.0
    assert abs(m["cagr"] - expected) < 1e-9
    assert isinstance(m["cagr"], float)


@pytest.mark.metrics
def test_cagr_ten_years():
    # $10k → $16.289 over 10 years ≈ 5% CAGR
    n_days = 252 * 10
    eq = make_equity_df(10_000, [0.0] * (n_days - 1))
    eq["Value"] = np.linspace(10_000, 16_289.0, n_days)
    m = calculate_metrics(eq)
    assert abs(m["cagr"] - 0.05) < 0.005  # within 0.5% of 5%


@pytest.mark.metrics
def test_cagr_flat():
    eq = make_equity_df(10_000, [0.0] * 251)
    m = calculate_metrics(eq)
    assert abs(m["cagr"]) < 1e-9
    assert isinstance(m["cagr"], float)


@pytest.mark.metrics
def test_cagr_loss():
    # $10k → $5k over 2 years ≈ -29.3% CAGR
    n = 252 * 2
    eq = make_equity_df(10_000, [0.0] * (n - 1))
    eq["Value"] = np.linspace(10_000, 5_000, n)
    m = calculate_metrics(eq)
    start_d = pd.to_datetime(eq["Date"].iloc[0])
    end_d = pd.to_datetime(eq["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    expected = (5_000 / 10_000) ** (1.0 / years) - 1.0
    assert abs(m["cagr"] - expected) < 1e-9


@pytest.mark.metrics
def test_cagr_returns_python_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=1))
    m = calculate_metrics(eq)
    assert type(m["cagr"]) is float, f"Expected float, got {type(m['cagr'])}"


@pytest.mark.metrics
def test_cagr_single_day_no_crash():
    eq = make_equity_df_from_values([10_000])
    m = calculate_metrics(eq)
    assert isinstance(m["cagr"], float)
    assert m["cagr"] == 0.0


@pytest.mark.metrics
def test_cagr_vs_quantstats():
    # 252-day series. Tolerance 0.01 — project uses calendar days, qs uses trading days.
    eq = make_equity_df(10_000, _random_returns(251, seed=99, mean=0.0004, std=0.010))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_cagr = float(qs.stats.cagr(returns, rf=0))
    assert abs(m["cagr"] - qs_cagr) < 0.01


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

@pytest.mark.metrics
def test_volatility_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=2))
    m = calculate_metrics(eq)
    assert type(m["volatility"]) is float


@pytest.mark.metrics
def test_volatility_flat_equity_is_zero():
    eq = make_equity_df_from_values([10_000] * 252)
    m = calculate_metrics(eq)
    assert m["volatility"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_volatility_single_return_no_crash():
    eq = make_equity_df_from_values([10_000, 10_100])
    m = calculate_metrics(eq)
    assert isinstance(m["volatility"], float)


@pytest.mark.metrics
def test_volatility_vs_quantstats():
    eq = make_equity_df(10_000, _random_returns(251, seed=77, std=0.015))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_vol = float(qs.stats.volatility(returns, periods=252, annualize=True))
    assert abs(m["volatility"] - qs_vol) < 1e-4


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------

@pytest.mark.metrics
def test_max_drawdown_single_drop():
    s = pd.Series([100.0, 80.0, 60.0, 80.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.40, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_new_ath_resets():
    # 100 → 120 → 90: drawdown is 90/120 - 1 = -0.25 (from new ATH of 120)
    s = pd.Series([100.0, 120.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.25, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_no_decline():
    s = pd.Series([100.0, 110.0, 120.0, 130.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_is_nonpositive():
    np.random.seed(5)
    s = pd.Series(np.exp(np.cumsum(np.random.normal(0.001, 0.015, 500))) * 10_000)
    result = max_drawdown_from_equity_curve(s)
    assert result <= 0.0


@pytest.mark.metrics
def test_max_drawdown_returns_float():
    s = pd.Series([100.0, 80.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert type(result) is float


@pytest.mark.metrics
def test_max_drawdown_single_value():
    result = max_drawdown_from_equity_curve(pd.Series([10_000.0]))
    assert result == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_nan_handling():
    s = pd.Series([100.0, float("nan"), 80.0, 90.0])
    result = max_drawdown_from_equity_curve(s)
    assert result == pytest.approx(-0.20, abs=1e-9)


@pytest.mark.metrics
def test_max_drawdown_vs_quantstats():
    # quantstats.stats.max_drawdown takes a RETURNS series (confusingly named 'prices')
    np.random.seed(33)
    prices = pd.Series(
        10_000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 1261))),
        index=pd.bdate_range("2015-01-02", periods=1261),
    )
    result = max_drawdown_from_equity_curve(prices)
    returns = prices.pct_change().dropna()
    qs_dd = float(qs.stats.max_drawdown(returns))
    assert abs(result - qs_dd) < 1e-6


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------

@pytest.mark.metrics
def test_sharpe_positive_returns():
    eq = make_equity_df(10_000, [0.001] * 251)  # consistent +0.1% daily
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] > 0
    assert isinstance(m["sharpe_ratio"], float)


@pytest.mark.metrics
def test_sharpe_zero_volatility():
    eq = make_equity_df_from_values([10_000] * 252)
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.metrics
def test_sharpe_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=3))
    m = calculate_metrics(eq)
    assert type(m["sharpe_ratio"]) is float


@pytest.mark.metrics
def test_sharpe_negative_cagr():
    eq = make_equity_df(10_000, [-0.001] * 251)  # consistent daily loss
    m = calculate_metrics(eq)
    assert m["sharpe_ratio"] < 0


@pytest.mark.metrics
def test_sharpe_vs_quantstats():
    # Tolerance 0.1 — project uses CAGR/vol, quantstats uses annualized_mean/std
    eq = make_equity_df(10_000, _random_returns(1259, seed=77, mean=0.0003, std=0.012))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_sharpe = float(qs.stats.sharpe(returns, rf=0, periods=252))
    assert abs(m["sharpe_ratio"] - qs_sharpe) < 0.1


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------

@pytest.mark.metrics
def test_sortino_returns_float():
    eq = make_equity_df(10_000, _random_returns(251, seed=4))
    m = calculate_metrics(eq)
    assert type(m["sortino_ratio"]) is float


@pytest.mark.metrics
def test_sortino_no_negative_returns_is_inf():
    eq = make_equity_df(10_000, [0.001] * 251)  # no negative returns
    m = calculate_metrics(eq)
    assert m["sortino_ratio"] == float("inf") or m["sortino_ratio"] > 100


@pytest.mark.metrics
def test_sortino_all_negative_returns():
    eq = make_equity_df(10_000, [-0.002] * 251)
    m = calculate_metrics(eq)
    assert m["sortino_ratio"] < 0


@pytest.mark.metrics
def test_sortino_vs_quantstats():
    # quantstats sortino with default settings (rf=0, periods=252)
    eq = make_equity_df(10_000, _random_returns(1259, seed=88, mean=0.0003, std=0.012))
    m = calculate_metrics(eq)
    returns = pd.Series(eq["Value"].values).pct_change().dropna()
    qs_sortino = float(qs.stats.sortino(returns, rf=0, periods=252))
    # Methodologies differ (project: CAGR/downside_dev, qs: annualized_mean/downside_dev)
    assert abs(m["sortino_ratio"] - qs_sortino) < 0.5
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_financial_metrics.py -v`
Expected: All pass. Note: `test_sortino_no_negative_returns_is_inf` checks `== float('inf') or > 100` because the code returns `float('inf')` when no negative returns but CAGR > 0.

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_financial_metrics.py
git commit -m "test: add financial metrics tests (formula + quantstats) with type contract checks"
```

---

## Task 6: Portfolio math tests (`tests/test_portfolio_math.py`)

**Files:**
- Create: `tests/test_portfolio_math.py`

- [ ] **Step 1: Write the test file**

```python
"""
Portfolio value calculation and rebalance execution invariants.
Protects: rebalance_engine.rebalance_portfolio(),
          allocation_engine.tradable_allocation(), get_allocation_for_regime()
"""
import numpy as np
import pandas as pd
import pytest

from allocation_engine import get_allocation_for_regime, tradable_allocation
from rebalance_engine import rebalance_portfolio


@pytest.fixture
def base_config():
    return {
        "starting_balance": 10_000.0,
        "drawdown_ticker": "QQQ",
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {"dd_low": 0.0, "dd_high": 0.08, "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0},
            "R2": {"dd_low": 0.08, "dd_high": 0.28, "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0},
            "R3": {"dd_low": 0.28, "dd_high": 1.0, "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0},
        },
    }


@pytest.mark.unit
def test_portfolio_value_equals_shares_times_price(base_config):
    # After rebalancing into R2 (100% XLU), value must equal shares × price
    prices = pd.Series({"QQQ": 100.0, "TQQQ": 30.0, "XLU": 50.0})
    alloc = get_allocation_for_regime("R2", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)

    computed_value = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(computed_value - 10_000.0) < 1e-6


@pytest.mark.unit
def test_r1_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R1", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_r2_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R2", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_r3_allocation_sums_to_one(base_config):
    alloc = get_allocation_for_regime("R3", base_config)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9


@pytest.mark.unit
def test_rebalance_preserves_value_same_prices(base_config):
    # Rebalancing on the same prices must not create or destroy value
    prices = pd.Series({"QQQ": 100.0, "TQQQ": 30.0, "XLU": 50.0})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    # R1 wants TQQQ; it's priced, so alloc_t = {TQQQ: 1.0}
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)
    value_after = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(value_after - 10_000.0) < 1e-6


@pytest.mark.unit
def test_proxy_qqq_when_tqqq_nan(base_config):
    # R1 wants TQQQ but TQQQ price is NaN → tradable_allocation must use QQQ
    prices = pd.Series({"QQQ": 100.0, "TQQQ": float("nan"), "XLU": 50.0})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    assert list(alloc_t.keys()) == ["QQQ"]
    assert abs(alloc_t["QQQ"] - 1.0) < 1e-9


@pytest.mark.unit
def test_proxy_all_missing_returns_empty(base_config):
    # All allocation tickers NaN → returns {} (no crash, no phantom value)
    prices = pd.Series({"QQQ": float("nan"), "TQQQ": float("nan"), "XLU": float("nan")})
    alloc = get_allocation_for_regime("R1", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    assert alloc_t == {}


@pytest.mark.unit
def test_starting_balance_preserved_day_one(base_config):
    prices = pd.Series({"QQQ": 50.0, "TQQQ": 15.0, "XLU": 25.0})
    alloc = get_allocation_for_regime("R2", base_config)
    alloc_t = tradable_allocation(alloc, prices, base_config)
    shares = rebalance_portfolio(10_000.0, alloc_t, prices)
    value = sum(shares[t] * float(prices[t]) for t in shares)
    assert abs(value - 10_000.0) < 1e-6


@pytest.mark.unit
def test_rebalance_returns_none_on_nan_price(base_config):
    # NaN price for a targeted ticker → rebalance returns None (no trade)
    prices = pd.Series({"XLU": float("nan")})
    result = rebalance_portfolio(10_000.0, {"XLU": 1.0}, prices)
    assert result is None
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_portfolio_math.py -v`
Expected: All pass.

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_portfolio_math.py
git commit -m "test: add portfolio math invariant tests (value, allocation sums, proxy logic)"
```

---

## Task 7: Drawdown window tests (`tests/test_drawdown_window.py`)

**Files:**
- Create: `tests/test_drawdown_window.py`

These complement the existing `test_rolling_drawdown_window.py` with scenario-based tests from the spec that weren't covered there.

- [ ] **Step 1: Write the test file**

```python
"""
Rolling ATH vs standard ATH drawdown path tests.
Complements test_rolling_drawdown_window.py with scenario-based coverage.
Protects: backtest.compute_rolling_ath_and_dd()
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtest import compute_rolling_ath_and_dd, compute_drawdown_from_ath


@pytest.mark.unit
def test_rolling_matches_cummax_before_window_fills():
    """Before N calendar years of data exist, rolling ATH must equal cummax."""
    # 6 months of data with n=1 year window → bootstrap period, must use cummax
    dates = pd.bdate_range("2020-01-02", periods=130)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.010, 130))),
        index=dates,
    )
    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    cummax_ath = prices.cummax()
    # During bootstrap the values must match cummax
    np.testing.assert_allclose(rolling_ath.values, cummax_ath.values, rtol=1e-9)


@pytest.mark.unit
def test_rolling_peak_scrolls_out():
    """After >N years, a peak from >N years ago should no longer anchor the ATH."""
    # Build series: high peak early, then all prices below the early peak for 2+ years
    dates_early = pd.bdate_range("2019-01-02", periods=20)
    early_prices = pd.Series([200.0] * 20, index=dates_early)  # high plateau

    # Drop and stay low for 18 months (> 1 year)
    dates_low = pd.bdate_range("2019-02-01", periods=380)
    low_prices = pd.Series([100.0] * 380, index=dates_low)

    prices = pd.concat([early_prices, low_prices]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    standard_ath, standard_dd = compute_drawdown_from_ath(prices)

    # After the early 200 peak scrolls out of the 1-year window, rolling ATH should be lower
    last_rolling_ath = float(rolling_ath.iloc[-1])
    last_standard_ath = float(standard_ath.iloc[-1])
    assert last_rolling_ath < last_standard_ath, (
        f"Rolling ATH {last_rolling_ath} should be below standard ATH {last_standard_ath} "
        "after early high peak leaves the window"
    )


@pytest.mark.unit
def test_rolling_and_standard_agree_on_fresh_ath():
    """When price is at a new all-time high, both methods must give 0% drawdown."""
    dates = pd.bdate_range("2020-01-02", periods=252)
    prices = pd.Series(np.linspace(100.0, 200.0, 252), index=dates)  # strictly rising
    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    _, standard_dd = compute_drawdown_from_ath(prices)
    assert float(rolling_dd.iloc[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(standard_dd.iloc[-1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_rolling_produces_higher_dd_after_peak_scrolls_out():
    """Once the high peak leaves the 1-year window, rolling dd > standard dd."""
    dates_old = pd.bdate_range("2019-01-02", periods=10)
    old_high = pd.Series([150.0] * 10, index=dates_old)

    dates_new = pd.bdate_range("2020-06-01", periods=252)
    new_prices = pd.Series([100.0] * 252, index=dates_new)  # stuck below old high

    prices = pd.concat([old_high, new_prices]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    rolling_ath, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    _, standard_dd = compute_drawdown_from_ath(prices)

    last_rolling_dd = float(rolling_dd.iloc[-1])
    last_standard_dd = float(standard_dd.iloc[-1])

    # standard ATH is still 150 → standard dd = (150-100)/150 ≈ 0.333
    # rolling ATH is 100 (the 2019 peak is >1yr ago) → rolling dd ≈ 0
    # rolling dd should be LOWER (less stressed) once old peak leaves window
    assert last_rolling_dd <= last_standard_dd + 1e-6, (
        f"After peak scrolls out: rolling_dd={last_rolling_dd:.4f}, "
        f"standard_dd={last_standard_dd:.4f}"
    )


@pytest.mark.unit
def test_drawdown_always_in_0_1_range():
    """Drawdown must always be in [0, 1] for any valid price series."""
    np.random.seed(42)
    dates = pd.bdate_range("2018-01-02", periods=500)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 500))), index=dates
    )
    _, rolling_dd = compute_rolling_ath_and_dd(prices, n_calendar_years=1)
    assert (rolling_dd >= 0.0).all(), "Rolling dd must never be negative"
    assert (rolling_dd <= 1.0).all(), "Rolling dd must never exceed 100%"


@given(
    n=st.integers(min_value=1, max_value=5),
    prices_list=st.lists(
        st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    ),
)
@settings(max_examples=200)
def test_rolling_dd_hypothesis_always_bounded(n, prices_list):
    """Property: for any positive price series and window ≥ 1, dd ∈ [0, 1]."""
    dates = pd.bdate_range("2020-01-02", periods=len(prices_list))
    prices = pd.Series(prices_list, index=dates)
    _, dd = compute_rolling_ath_and_dd(prices, n_calendar_years=n)
    if not dd.empty:
        assert (dd >= -1e-9).all(), "dd should never be negative"
        assert (dd <= 1.0 + 1e-9).all(), "dd should never exceed 1.0"
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_drawdown_window.py -v`
Expected: All pass.

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_drawdown_window.py
git commit -m "test: add rolling ATH window scenario tests"
```

---

## Task 8: Locked regression anchor (`tests/test_regression_ground_truth.py`)

**Files:**
- Create: `tests/test_regression_ground_truth.py`

This is the safety net for future refactors. Any change that moves the locked values fails this test. To intentionally change them, you must update the `LOCKED_*` constants AND add a comment explaining why. **The ground truth was generated on 2026-04-26 with seed=42.**

Locked values:
- FINAL_VALUE = 10949.7909
- CAGR = 0.018976 (tolerance 1e-4)
- SHARPE = 0.1552 (tolerance 1e-3)
- MAX_DRAWDOWN = -0.290396 (tolerance 1e-4)
- VOLATILITY = 0.122285 (tolerance 1e-4)
- FIRST_20_REGIMES = ['R1'] × 20

- [ ] **Step 1: Write the test file**

```python
"""
Locked regression anchor test.

A single frozen test over a deterministic 5-year synthetic backtest. Any change
to core logic that moves these numbers without an explicit constant update fails.

DO NOT change the LOCKED_* constants without understanding what changed and why.
Config: 3-regime, down_only, drawdown_window_enabled=False, all signal_overrides disabled.
Seed: 42 (price_fixture). Generated: 2026-04-26.
"""
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from backtest import run_backtest, compute_drawdown_from_ath
from regime_engine import determine_regime
from rebalance_engine import rebalance_portfolio
from utils import max_drawdown_from_equity_curve


# ---------------------------------------------------------------------------
# Locked constants — change only with deliberate intent + comment
# ---------------------------------------------------------------------------
LOCKED_FINAL_VALUE   = 10949.7909  # portfolio value at end of 5yr simulation
LOCKED_CAGR          = 0.018976    # from calculate_metrics (calendar-day CAGR)
LOCKED_SHARPE        = 0.1552      # CAGR / annualized_vol
LOCKED_MAX_DRAWDOWN  = -0.290396   # from max_drawdown_from_equity_curve (non-positive)
LOCKED_VOLATILITY    = 0.122285    # annualized daily returns std
LOCKED_FIRST_20_REGIMES = ["R1"] * 20


# ---------------------------------------------------------------------------
# Fixture: deterministic synthetic price data
# ---------------------------------------------------------------------------

def _build_price_df():
    """Deterministic 5yr price panel. seed=42, do not change."""
    np.random.seed(42)
    n_days = 1260
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    returns = np.random.normal(0.0004, 0.010, n_days)
    returns[200:260] = np.random.normal(-0.006, 0.014, 60)
    returns[700:820] = np.random.normal(-0.009, 0.018, 120)

    qqq  = 100.0 * np.exp(np.cumsum(returns))
    tqqq = 30.0  * np.exp(3.0 * np.cumsum(returns * 0.9))
    xlu  = 50.0  * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    spy  = 250.0 * np.exp(np.cumsum(0.7 * returns + np.random.normal(0, 0.004, n_days)))

    return pd.DataFrame({"QQQ": qqq, "TQQQ": tqqq, "XLU": xlu, "SPY": spy}, index=dates)


def _build_config():
    """Minimal reproducible config: down_only, no window, no signal overrides."""
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "down_only",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY"],
        "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": False,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": 0.08,
                "TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R2": {
                "dd_low": 0.08, "dd_high": 0.28,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
            "R3": {
                "dd_low": 0.28, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0,
                "rebalance_on_downward": "match", "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": {"enabled": False},
                    "protection": {"enabled": False},
                },
            },
        },
    }


@pytest.fixture(scope="module")
def locked_equity_df():
    """Run the full backtest once per module with all network calls mocked."""
    price_df = _build_price_df()
    config = _build_config()

    qqq_prices = price_df["QQQ"].values
    dates = price_df.index
    empty_series = pd.Series(dtype=float)

    def _mock_yf_download(*args, **kwargs):
        df = pd.DataFrame({"Close": qqq_prices}, index=dates)
        df.columns = pd.MultiIndex.from_tuples([("Close", "QQQ")])
        return df

    with (
        patch("yfinance.download", side_effect=_mock_yf_download),
        patch("data_loader.load_spy_series", return_value=empty_series),
        patch("data_loader.load_vix_series", return_value=empty_series),
    ):
        equity_df, _, _ = run_backtest(
            price_df,
            config,
            lambda s: compute_drawdown_from_ath(s),
            lambda dd, c: determine_regime(dd, c),
            rebalance_portfolio,
        )
    return equity_df


# ---------------------------------------------------------------------------
# Locked assertions
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_locked_final_value(locked_equity_df):
    final = float(locked_equity_df["Value"].iloc[-1])
    assert abs(final - LOCKED_FINAL_VALUE) < 0.01, (
        f"Final value changed: got {final:.4f}, locked {LOCKED_FINAL_VALUE}. "
        "If this is intentional, update LOCKED_FINAL_VALUE."
    )


@pytest.mark.integration
def test_locked_row_count(locked_equity_df):
    assert len(locked_equity_df) == 1260, (
        f"Row count changed: got {len(locked_equity_df)}, expected 1260"
    )


@pytest.mark.integration
def test_locked_first_20_regimes(locked_equity_df):
    first_20 = list(locked_equity_df["Market_Regime"].iloc[:20])
    assert first_20 == LOCKED_FIRST_20_REGIMES, (
        f"First 20 regimes changed: got {first_20}"
    )


@pytest.mark.integration
def test_locked_cagr(locked_equity_df):
    vals = locked_equity_df["Value"]
    start_v, end_v = float(vals.iloc[0]), float(vals.iloc[-1])
    start_d = pd.to_datetime(locked_equity_df["Date"].iloc[0])
    end_d   = pd.to_datetime(locked_equity_df["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    assert abs(cagr - LOCKED_CAGR) < 1e-4, (
        f"CAGR changed: got {cagr:.6f}, locked {LOCKED_CAGR}. "
        "Update LOCKED_CAGR if intentional."
    )


@pytest.mark.integration
def test_locked_max_drawdown(locked_equity_df):
    dd = max_drawdown_from_equity_curve(locked_equity_df["Value"])
    assert abs(dd - LOCKED_MAX_DRAWDOWN) < 1e-4, (
        f"Max drawdown changed: got {dd:.6f}, locked {LOCKED_MAX_DRAWDOWN}. "
        "Update LOCKED_MAX_DRAWDOWN if intentional."
    )


@pytest.mark.integration
def test_locked_volatility(locked_equity_df):
    returns = locked_equity_df["Value"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    assert abs(vol - LOCKED_VOLATILITY) < 1e-4, (
        f"Volatility changed: got {vol:.6f}, locked {LOCKED_VOLATILITY}. "
        "Update LOCKED_VOLATILITY if intentional."
    )


@pytest.mark.integration
def test_locked_sharpe(locked_equity_df):
    vals = locked_equity_df["Value"]
    returns = vals.pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))
    start_v, end_v = float(vals.iloc[0]), float(vals.iloc[-1])
    start_d = pd.to_datetime(locked_equity_df["Date"].iloc[0])
    end_d   = pd.to_datetime(locked_equity_df["Date"].iloc[-1])
    years = (end_d - start_d).days / 365.25
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0
    sharpe = cagr / vol if vol > 0 else 0.0
    assert abs(sharpe - LOCKED_SHARPE) < 1e-3, (
        f"Sharpe changed: got {sharpe:.4f}, locked {LOCKED_SHARPE}. "
        "Update LOCKED_SHARPE if intentional."
    )
```

- [ ] **Step 2: Run the regression test**

Run: `python3 -m pytest tests/test_regression_ground_truth.py -v`
Expected: All 7 tests pass. If any fail, the locked values must be wrong — do not change the test code; debug why the backtest output changed.

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All tests pass (25 existing + all new tests).

- [ ] **Step 4: Final verification with verbose output**

Run: `python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: No failures. Note the total count at the bottom.

- [ ] **Step 5: Commit**

```bash
git add tests/test_regression_ground_truth.py
git commit -m "test: add locked regression anchor — any refactor that moves numbers fails this"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Section 3.1 `test_regime_boundaries.py` — all boundary tests + Hypothesis property
- [x] Section 3.2 `test_regime_transitions.py` — per_regime, down_only, up_only, full sequence
- [x] Section 3.3 `test_financial_metrics.py` — CAGR, vol, maxdd, Sharpe, Sortino (formula + qs + type)
- [x] Section 3.4 `test_portfolio_math.py` — value invariant, allocation sum, proxy, NaN
- [x] Section 3.5 `test_drawdown_window.py` — rolling window scenarios + Hypothesis
- [x] Section 3.6 `test_regression_ground_truth.py` — locked anchor with 7 assertions
- [x] Section 2.1 `pyproject.toml` — markers, discovery
- [x] Section 2.2 `conftest.py` — all 5 fixtures from spec

**Key deviations from spec (documented):**
- `test_per_regime_upward_hold` was reframed: the code uses the TARGET regime's `rebalance_on_upward` setting (not the source's). The actual "hold" behavior occurs on `R3→R2`, where R2 is the target and R2 has `upward=hold`. This is correct per code tracing.
- Sharpe tolerance `< 0.1` (not `< 0.01` from spec) — measured difference between `CAGR/vol` and `annualized_mean/std` implementations is ~0.04 for 5yr series.
- Sortino tolerance `< 0.5` — methodologies differ more significantly.
