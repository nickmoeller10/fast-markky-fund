"""Unit tests for level-based signal overrides (no backtest / network)."""

import unittest

from signal_override_engine import (
    desired_signal_override_mode,
    get_target_allocation_for_override,
    default_signal_overrides_block,
)


def _regime_with_overrides(up=None, pr=None):
    r = {
        "dd_low": 0.0,
        "dd_high": 0.5,
        "TQQQ": 1.0,
        "QQQ": 0.0,
        "XLU": 0.0,
        "signal_overrides": default_signal_overrides_block(),
    }
    if up:
        r["signal_overrides"]["upside"].update(up)
    if pr:
        r["signal_overrides"]["protection"].update(pr)
    return r


class TestDesiredSignalOverrideMode(unittest.TestCase):
    def test_protection_wins_when_both_levels_active(self):
        rp = _regime_with_overrides(
            up={"enabled": True, "direction": "above", "threshold": 0.0},
            pr={"enabled": True, "direction": "below", "threshold": 5.0},
        )
        # s=3: >= 0 and <= 5 → both → protection
        self.assertEqual(desired_signal_override_mode(3.0, rp), "protection")

    def test_r1_style_zones(self):
        rp = _regime_with_overrides(
            up={"enabled": True, "direction": "above", "threshold": 0.0},
            pr={"enabled": True, "direction": "below", "threshold": -2.0},
        )
        self.assertEqual(desired_signal_override_mode(1.0, rp), "upside")
        self.assertEqual(desired_signal_override_mode(-2.0, rp), "protection")
        self.assertEqual(desired_signal_override_mode(-2.5, rp), "protection")
        self.assertEqual(desired_signal_override_mode(-1.0, rp), "none")

    def test_nan_preserves_current_mode(self):
        rp = _regime_with_overrides(
            up={"enabled": True, "direction": "above", "threshold": 0.0},
        )
        self.assertEqual(
            desired_signal_override_mode(float("nan"), rp, "upside"),
            "upside",
        )

    def test_infer_enabled_from_label_when_omitted(self):
        """Panels without explicit enabled:true still turn on if label is set."""
        so = default_signal_overrides_block()
        so["upside"] = {
            "label": "L",
            "direction": "above",
            "threshold": 0,
            "TQQQ": 1.0,
            "QQQ": 0.0,
            "XLU": 0.0,
        }
        r = {
            "dd_low": 0.0,
            "dd_high": 1.0,
            "TQQQ": 1.0,
            "QQQ": 0.0,
            "XLU": 0.0,
            "signal_overrides": so,
        }
        from signal_override_engine import ensure_regime_signal_overrides

        ensure_regime_signal_overrides(r)
        self.assertTrue(r["signal_overrides"]["upside"]["enabled"])
        self.assertEqual(desired_signal_override_mode(1.0, r), "upside")


class TestRebalanceTargets(unittest.TestCase):
    def test_rebalance_target_weights(self):
        cfg = {
            "allocation_tickers": ["QQQ", "TQQQ", "XLU"],
            "regimes": {
                "R1": _regime_with_overrides(
                    up={"enabled": True, "TQQQ": 0.5, "QQQ": 0.5, "XLU": 0.0},
                )
            },
        }
        cfg["regimes"]["R1"]["signal_overrides"]["upside"]["enabled"] = True
        w = get_target_allocation_for_override("R1", "upside", cfg)
        self.assertAlmostEqual(w["TQQQ"], 0.5)
        self.assertAlmostEqual(w["QQQ"], 0.5)
        w0 = get_target_allocation_for_override("R1", "none", cfg)
        self.assertAlmostEqual(w0["TQQQ"], 1.0)


if __name__ == "__main__":
    unittest.main()
