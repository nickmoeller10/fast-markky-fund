"""
Unit tests for dashboard widget data resolution.
Protects: dashboard.todays_regime_status — the pure-data function behind the
"Today's Status" panel. The render function itself isn't tested (requires
streamlit runtime); test coverage focuses on correct field extraction and
the override → base-allocation fallback.
"""
import pandas as pd
import pytest

from dashboard import todays_regime_status


def _row(
    date="2026-03-27",
    market_regime="R1",
    portfolio_regime="R1",
    override_active="none",
    override_label="",
    override_allocation="",
):
    return {
        "Date": pd.Timestamp(date),
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Signal_override_active": override_active,
        "Signal_override_label": override_label,
        "Signal_override_allocation": override_allocation,
        "Value": 12345.67,
    }


def _config_with_regime(portfolio_regime="R1", weights=None):
    weights = weights or {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0}
    return {
        "allocation_tickers": ["TQQQ", "QQQ", "XLU"],
        "regimes": {
            portfolio_regime: {
                "dd_low": 0.0,
                "dd_high": 0.06,
                **weights,
            }
        },
    }


@pytest.mark.unit
def test_pulls_from_last_row():
    df = pd.DataFrame([_row(date="2024-01-01"), _row(date="2026-03-27", market_regime="R2")])
    cfg = {
        "allocation_tickers": ["TQQQ", "QQQ", "XLU"],
        "regimes": {"R2": {"TQQQ": 0.0, "QQQ": 0.30, "XLU": 0.70}},
    }
    df.iloc[-1, df.columns.get_loc("Portfolio_Regime")] = "R2"

    status = todays_regime_status(df, cfg)
    assert status["market_regime"] == "R2"
    assert status["portfolio_regime"] == "R2"
    assert status["as_of"].strftime("%Y-%m-%d") == "2026-03-27"


@pytest.mark.unit
def test_override_active_uses_override_allocation_string():
    row = _row(
        override_active="upside",
        override_label="Max Bull",
        override_allocation="TQQQ 100%, QQQ 0%, XLU 0%",
    )
    df = pd.DataFrame([row])

    status = todays_regime_status(df, _config_with_regime())
    assert status["override_active"] == "upside"
    assert status["override_label"] == "Max Bull"
    assert status["recommended_allocation"] == "TQQQ 100%, QQQ 0%, XLU 0%"


@pytest.mark.unit
def test_no_override_falls_back_to_base_regime_allocation():
    row = _row(
        portfolio_regime="R2",
        override_active="none",
        override_label="",
        override_allocation="",
    )
    df = pd.DataFrame([row])
    cfg = {
        "allocation_tickers": ["TQQQ", "QQQ", "XLU"],
        "regimes": {
            "R2": {"TQQQ": 0.0, "QQQ": 0.30, "XLU": 0.70},
        },
    }

    status = todays_regime_status(df, cfg)
    assert status["override_active"] == "none"
    assert status["override_label"] == ""
    # Base R2 weights formatted via allocation_human_readable
    assert status["recommended_allocation"] == "QQQ 30%, XLU 70%"


@pytest.mark.unit
def test_protection_override_renders_protection_string():
    row = _row(
        portfolio_regime="R2",
        override_active="protection",
        override_label="Deteriorating Fast",
        override_allocation="XLU 100%",
    )
    df = pd.DataFrame([row])
    cfg = _config_with_regime("R2", {"TQQQ": 0.0, "QQQ": 0.30, "XLU": 0.70})

    status = todays_regime_status(df, cfg)
    assert status["override_active"] == "protection"
    assert status["override_label"] == "Deteriorating Fast"
    assert status["recommended_allocation"] == "XLU 100%"


@pytest.mark.unit
def test_missing_portfolio_regime_in_config_returns_empty_allocation():
    row = _row(portfolio_regime="R7", override_active="none")
    df = pd.DataFrame([row])
    cfg = {
        "allocation_tickers": ["TQQQ", "QQQ", "XLU"],
        "regimes": {"R1": {"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0}},
    }

    status = todays_regime_status(df, cfg)
    assert status["portfolio_regime"] == "R7"
    assert status["recommended_allocation"] == ""


@pytest.mark.unit
def test_market_and_portfolio_regimes_can_differ():
    row = _row(market_regime="R1", portfolio_regime="R3", override_active="none")
    df = pd.DataFrame([row])
    cfg = _config_with_regime("R3", {"TQQQ": 0.5, "QQQ": 0.5, "XLU": 0.0})

    status = todays_regime_status(df, cfg)
    assert status["market_regime"] == "R1"
    assert status["portfolio_regime"] == "R3"
    assert status["recommended_allocation"] == "TQQQ 50%, QQQ 50%"
