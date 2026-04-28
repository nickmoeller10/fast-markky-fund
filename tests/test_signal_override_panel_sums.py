"""Panel-sum invariant tests for signal_override_engine.validate_panel_sums.

The iter-25 production config silently lost CASH from override panels: panels
summed to 0.31, runtime renormalization filled the gap, and the optimizer
search space looked larger than it actually was. validate_panel_sums catches
this by raising ValueError when any enabled panel doesn't sum to 1.0.
"""

from __future__ import annotations

import copy

import pytest

from signal_override_engine import (
    ensure_regime_signal_overrides,
    validate_panel_sums,
)


def _minimal_4coord_config() -> dict:
    """3-regime config with TQQQ/QQQ/XLU/CASH allocation universe."""
    return {
        "allocation_tickers": ["TQQQ", "QQQ", "XLU", "CASH"],
        "regimes": {
            "R1": {
                "TQQQ": 0.8, "QQQ": 0.1, "XLU": 0.1, "CASH": 0.0,
                "signal_overrides": {
                    "upside": {
                        "enabled": True, "label": "R1 up", "direction": "above", "threshold": 1,
                        "TQQQ": 0.5, "QQQ": 0.3, "XLU": 0.2, "CASH": 0.0,
                    },
                    "protection": {
                        "enabled": True, "label": "R1 prot", "direction": "below", "threshold": -2,
                        "TQQQ": 0.0, "QQQ": 0.2, "XLU": 0.1, "CASH": 0.7,
                    },
                },
            },
            "R2": {
                "TQQQ": 0.0, "QQQ": 0.05, "XLU": 0.05, "CASH": 0.9,
                "signal_overrides": {
                    "upside": {"enabled": False, "TQQQ": 0, "QQQ": 0, "XLU": 0, "CASH": 0},
                    "protection": {"enabled": False, "TQQQ": 0, "QQQ": 0, "XLU": 0, "CASH": 0},
                },
            },
            "R3": {
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "CASH": 1.0,
                "signal_overrides": {
                    "upside": {
                        "enabled": True, "label": "R3 up", "direction": "above", "threshold": 1,
                        "TQQQ": 0.4, "QQQ": 0.3, "XLU": 0.2, "CASH": 0.1,
                    },
                    "protection": {
                        "enabled": True, "label": "R3 prot", "direction": "below", "threshold": -4,
                        "TQQQ": 0.0, "QQQ": 0.25, "XLU": 0.75, "CASH": 0.0,
                    },
                },
            },
        },
    }


def test_4coord_panels_sum_to_one_pass():
    cfg = _minimal_4coord_config()
    validate_panel_sums(cfg)


def test_disabled_panel_with_zero_weights_passes():
    cfg = _minimal_4coord_config()
    # R2's overrides are disabled with all-zero weights — must not raise.
    assert cfg["regimes"]["R2"]["signal_overrides"]["upside"]["enabled"] is False
    validate_panel_sums(cfg)


def test_protection_panel_31pct_sum_raises():
    """The exact iter-25 R1 protection bug: panel sums to 0.31 because CASH was dropped."""
    cfg = _minimal_4coord_config()
    cfg["regimes"]["R1"]["signal_overrides"]["protection"] = {
        "enabled": True, "label": "broken", "direction": "below", "threshold": -2,
        "TQQQ": 0.0, "QQQ": 0.2, "XLU": 0.11, "CASH": 0.0,
    }
    with pytest.raises(ValueError, match="R1.*protection.*0.3100"):
        validate_panel_sums(cfg)


def test_base_panel_off_by_one_raises():
    cfg = _minimal_4coord_config()
    cfg["regimes"]["R1"]["TQQQ"] = 0.5  # was 0.8 — total drops to 0.7
    with pytest.raises(ValueError, match="R1.*base.*0.7"):
        validate_panel_sums(cfg)


def test_3coord_panel_passes_when_summing_to_one():
    """Configs with only TQQQ/QQQ/XLU (no CASH) and sum=1 must still pass."""
    cfg = _minimal_4coord_config()
    cfg["allocation_tickers"] = ["TQQQ", "QQQ", "XLU"]
    for regime in cfg["regimes"].values():
        for k in ("TQQQ", "QQQ", "XLU"):
            pass  # already in dict
        regime.pop("CASH", None)
    cfg["regimes"]["R1"].update({"TQQQ": 1.0, "QQQ": 0.0, "XLU": 0.0})
    cfg["regimes"]["R1"]["signal_overrides"]["upside"] = {
        "enabled": True, "label": "u", "direction": "above", "threshold": 1,
        "TQQQ": 0.5, "QQQ": 0.3, "XLU": 0.2,
    }
    cfg["regimes"]["R1"]["signal_overrides"]["protection"] = {
        "enabled": True, "label": "p", "direction": "below", "threshold": -2,
        "TQQQ": 0.0, "QQQ": 0.7, "XLU": 0.3,
    }
    cfg["regimes"]["R2"].update({"TQQQ": 0.0, "QQQ": 0.0, "XLU": 1.0})
    cfg["regimes"]["R3"].update({"TQQQ": 0.0, "QQQ": 0.5, "XLU": 0.5})
    cfg["regimes"]["R3"]["signal_overrides"]["upside"] = {
        "enabled": True, "label": "u", "direction": "above", "threshold": 1,
        "TQQQ": 0.4, "QQQ": 0.3, "XLU": 0.3,
    }
    cfg["regimes"]["R3"]["signal_overrides"]["protection"] = {
        "enabled": True, "label": "p", "direction": "below", "threshold": -4,
        "TQQQ": 0.0, "QQQ": 0.25, "XLU": 0.75,
    }
    validate_panel_sums(cfg)


def test_production_config_passes():
    """The shipped CONFIG must satisfy the invariant."""
    from config import CONFIG

    cfg = copy.deepcopy(CONFIG)
    for params in cfg["regimes"].values():
        ensure_regime_signal_overrides(params)
    validate_panel_sums(cfg)


def test_dollar_sign_alias_4coord_panel_passes():
    """Configs using '$' instead of 'CASH' as the synthetic alias must validate."""
    cfg = _minimal_4coord_config()
    cfg["allocation_tickers"] = ["TQQQ", "QQQ", "XLU", "$"]
    for regime in cfg["regimes"].values():
        regime["$"] = regime.pop("CASH")
        for panel_kind in ("upside", "protection"):
            panel = regime["signal_overrides"][panel_kind]
            panel["$"] = panel.pop("CASH")
    validate_panel_sums(cfg)


def test_ensure_regime_signal_overrides_preserves_extra_ticker_keys():
    """Regression: the iter-25 bug stemmed from ensure_regime_signal_overrides
    silently dropping CASH/$ from panels because it only iterated default keys."""
    regime_params = {
        "TQQQ": 0.0, "QQQ": 0.05, "XLU": 0.05, "CASH": 0.9,
        "signal_overrides": {
            "upside": {
                "enabled": True, "label": "u", "direction": "above", "threshold": 4,
                "TQQQ": 0.3, "QQQ": 0.27, "XLU": 0.16, "CASH": 0.27,
            },
            "protection": {
                "enabled": True, "label": "p", "direction": "below", "threshold": -3,
                "TQQQ": 0.0, "QQQ": 0.11, "XLU": 0.48, "CASH": 0.41,
            },
        },
    }
    ensure_regime_signal_overrides(regime_params)
    assert regime_params["signal_overrides"]["upside"]["CASH"] == 0.27
    assert regime_params["signal_overrides"]["protection"]["CASH"] == 0.41
