"""End-to-end dividend reinvestment behavior through run_backtest.

Production has dividend_reinvestment=False, but the code path runs in
historical configs and in the optimizer. This file exercises both
target modes ('cash' and a ticker), and pins behavior on the
interaction days where compounding bugs hide: a dividend that lands
on a rebalance day, and a dividend that lands on a regime-change day.

The dividend block in backtest.py:976-1036 sits between today's
mark-to-market (line 964) and the regime/rebalance blocks. If the
ordering ever flips, the mark invariant or the equity curve will drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import compute_drawdown_from_ath, run_backtest
from rebalance_engine import rebalance_portfolio
from regime_engine import determine_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _disabled_panel():
    return {
        "enabled": False, "label": "", "direction": "above", "threshold": 0,
        "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0,
    }


def _div_cfg(*, target="cash", r1_alloc="XLU", r1_high=0.99):
    """2-regime config with dividend_reinvestment=True and chosen target."""
    r1 = {"TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 0.0}
    r1[r1_alloc] = 1.0
    return {
        "starting_balance": 10_000,
        "drawdown_ticker": "QQQ",
        "rebalance_frequency": "instant",
        "rebalance_strategy": "per_regime",
        "tickers": ["QQQ", "TQQQ", "XLU", "SPY", "$"],
        "allocation_tickers": ["TQQQ", "QQQ", "XLU", "$"],
        "drawdown_window_enabled": False,
        "drawdown_window_years": 1,
        "dividend_reinvestment": True,
        "dividend_reinvestment_target": target,
        "minimum_allocation": 0.0,
        "regimes": {
            "R1": {
                "dd_low": 0.0, "dd_high": r1_high,
                **r1,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": _disabled_panel(),
                    "protection": _disabled_panel(),
                },
            },
            "R2": {
                "dd_low": r1_high, "dd_high": 1.0,
                "TQQQ": 0.0, "QQQ": 0.0, "XLU": 0.0, "$": 1.0,
                "rebalance_on_downward": "match",
                "rebalance_on_upward": "match",
                "signal_overrides": {
                    "upside": _disabled_panel(),
                    "protection": _disabled_panel(),
                },
            },
        },
    }


def _add_cash(panel):
    from data_loader import _build_cash_series
    out = panel.copy()
    out["$"] = _build_cash_series(out.index)
    return out


@pytest.fixture
def isolate_yf(monkeypatch):
    import data_cache
    import data_loader
    import backtest as backtest_mod

    def empty(*a, **kw):
        return pd.DataFrame()

    monkeypatch.setattr(data_cache, "cached_yf_download", empty)
    monkeypatch.setattr(data_loader, "cached_yf_download", empty)
    if hasattr(backtest_mod, "cached_yf_download"):
        monkeypatch.setattr(backtest_mod, "cached_yf_download", empty)
    yield


@pytest.fixture
def flat_xlu_panel():
    """20 trading days, XLU = $50 flat. Dividends are easy to verify against."""
    dates = pd.bdate_range("1995-01-02", periods=20)
    df = pd.DataFrame({
        "QQQ": np.full(20, 100.0),
        "TQQQ": np.full(20, 100.0),
        "XLU": np.full(20, 50.0),
        "SPY": np.full(20, 200.0),
    }, index=dates)
    return _add_cash(df)


@pytest.fixture
def deep_dd_xlu_panel():
    """30 days, QQQ 100→60→90 (forces R1→R2 transition); XLU stays flat."""
    dates = pd.bdate_range("1995-01-02", periods=30)
    qqq = np.concatenate([
        np.linspace(100, 60, 18, endpoint=True)[:-1],
        np.linspace(62, 90, 13),
    ])
    df = pd.DataFrame({
        "QQQ": qqq,
        "TQQQ": 100 * (qqq / qqq[0]) ** 3.0,
        "XLU": np.full(len(qqq), 50.0),
        "SPY": np.full(len(qqq), 200.0),
    }, index=dates)
    return _add_cash(df)


def _div_df(panel, events):
    """Build a dividend DataFrame from a list of (date_str, ticker, dps) tuples."""
    df = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for date_str, ticker, dps in events:
        df.loc[pd.Timestamp(date_str), ticker] = float(dps)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_cash_dividend_increases_value_by_exact_amount(
    isolate_yf, flat_xlu_panel
):
    """100% XLU, prices flat, single $0.50/share XLU dividend on day 5,
    target='cash'. Day-5 portfolio value must increase by exactly
    (initial_xlu_shares × 0.50) — the dividend goes to cash."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [(panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    initial_shares = 10_000.0 / 50.0
    expected_dividend = initial_shares * 0.50
    day5_value = float(eq["Value"].iloc[5])
    day4_value = float(eq["Value"].iloc[4])
    actual_jump = day5_value - day4_value
    assert actual_jump == pytest.approx(expected_dividend, rel=1e-6), (
        f"Day-5 jump {actual_jump:.4f} != expected {expected_dividend:.4f}"
    )

    assert len(divs) == 1
    assert divs.iloc[0]["Ticker"] == "XLU"
    assert float(divs.iloc[0]["Dividend_Amount"]) == pytest.approx(expected_dividend, rel=1e-6)
    assert divs.iloc[0]["Reinvestment_Target"] == "cash"

    assert float(eq["Cash"].iloc[5]) == pytest.approx(expected_dividend, rel=1e-6)


@pytest.mark.integration
def test_ticker_reinvest_buys_shares_at_target_price(
    isolate_yf, flat_xlu_panel
):
    """100% XLU, $0.50/share dividend, target='XLU' → reinvested into XLU at
    today's price. Total value rises by the dividend; XLU share count
    increases by dividend / XLU_price."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="XLU", r1_alloc="XLU")
    div = _div_df(panel, [(panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    initial_shares = 10_000.0 / 50.0
    dividend_amount = initial_shares * 0.50
    bought = dividend_amount / 50.0

    day5_xlu_shares = float(eq["XLU_shares"].iloc[5])
    assert day5_xlu_shares == pytest.approx(initial_shares + bought, rel=1e-6)
    day5_value = float(eq["Value"].iloc[5])
    assert day5_value == pytest.approx(
        (initial_shares + bought) * 50.0, rel=1e-6
    )
    assert float(eq["Cash"].iloc[5]) == pytest.approx(0.0, abs=1e-9)
    assert divs.iloc[0]["Reinvestment_Target"] == "XLU"


@pytest.mark.integration
def test_mark_invariant_holds_with_dividends(isolate_yf, flat_xlu_panel):
    """Value[t] == sum(shares × price) + Cash on every row, even with
    multiple dividend events in cash mode."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [
        (panel.index[5].strftime("%Y-%m-%d"), "XLU", 0.50),
        (panel.index[10].strftime("%Y-%m-%d"), "XLU", 0.30),
        (panel.index[15].strftime("%Y-%m-%d"), "XLU", 0.40),
    ])

    eq, _, _ = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    panel_aligned = panel.reindex(pd.to_datetime(eq["Date"]))
    for i, row in eq.iterrows():
        date = pd.to_datetime(row["Date"])
        manual = 0.0
        for t in cfg["allocation_tickers"]:
            col = f"{t}_shares"
            if col in eq.columns and t in panel_aligned.columns:
                s = float(eq.loc[i, col] or 0)
                p = panel_aligned.loc[date, t]
                if pd.notna(p):
                    manual += s * float(p)
        manual += float(row.get("Cash", 0) or 0)
        v = float(row["Value"])
        denom = max(abs(v), 1.0)
        assert abs(v - manual) / denom <= 5e-6, (
            f"Mark invariant broke on {row['Date']}: V={v:.4f}, "
            f"shares×price+cash={manual:.4f}"
        )


@pytest.mark.integration
def test_dividend_on_regime_change_day_routes_to_post_change_regime(
    isolate_yf, deep_dd_xlu_panel
):
    """A dividend that lands on the same day a R1→R2 transition fires.

    Order of operations (per backtest.py): mark-to-market → dividend →
    regime-decision → rebalance. So the dividend lands as cash BEFORE the
    rebalance to R2's all-cash allocation, and the rebalance must absorb
    that cash into the new $ allocation. If a future change reorders these
    blocks (rebalance before dividend), the dividend cash would be left
    dangling and Cash[transition_day] would be non-zero — exactly the
    same shape as the 2026-04-28 mark-to-market bug.

    Because the regime decision uses yesterday's dd (T+1 lag), the actual
    rebalance fires on the day AFTER dd first crosses r1_high. We pick
    that day for the dividend to genuinely exercise the
    dividend-AND-rebalance-on-the-same-day ordering.
    """
    panel = deep_dd_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU", r1_high=0.10)

    qqq = panel["QQQ"]
    dd = (qqq.cummax() - qqq) / qqq.cummax()
    dd_cross_idx = int(np.where(dd.values > 0.10)[0][0])
    rebalance_idx = dd_cross_idx + 1
    assert 1 < rebalance_idx < len(panel) - 2, (
        f"rebalance_idx={rebalance_idx} too close to panel edge — panel "
        f"design degraded; cross at {dd_cross_idx}"
    )
    rebalance_date = panel.index[rebalance_idx]

    div = _div_df(panel, [(rebalance_date.strftime("%Y-%m-%d"), "XLU", 0.20)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    assert len(divs) == 1, f"Expected 1 dividend, got {len(divs)}"
    rebalance_row = eq.iloc[rebalance_idx]
    assert str(rebalance_row["Rebalanced"]) == "Rebalanced", (
        f"rebalance_idx={rebalance_idx} did not actually rebalance — "
        f"panel design or T+1 assumption broke. Row: {dict(rebalance_row)}"
    )
    assert str(rebalance_row["Portfolio_Regime"]) == "R2", (
        f"Portfolio not in R2 on rebalance day; got "
        f"{rebalance_row['Portfolio_Regime']!r}"
    )

    # Mark invariant on the rebalance row.
    manual = 0.0
    for t in cfg["allocation_tickers"]:
        col = f"{t}_shares"
        if col in eq.columns and t in panel.columns:
            s = float(rebalance_row.get(col, 0) or 0)
            p = panel.loc[rebalance_date, t]
            if pd.notna(p):
                manual += s * float(p)
    manual += float(rebalance_row.get("Cash", 0) or 0)
    v = float(rebalance_row["Value"])
    assert abs(v - manual) / max(abs(v), 1.0) <= 5e-6, (
        f"Mark invariant violated on dividend+rebalance day: "
        f"V={v:.4f}, manual={manual:.4f}"
    )

    # The dividend cash must have been absorbed by the rebalance into the $
    # sleeve. After R2 rebalance + dividend, Cash should be 0 (R2 is
    # 100% $, target='cash' means dividend goes to Cash, then the rebalance
    # sweeps Cash into $ shares as part of portfolio_value_with_cash).
    cash_after = float(rebalance_row.get("Cash", 0) or 0)
    assert cash_after == pytest.approx(0.0, abs=1e-6), (
        f"Cash on rebalance day should be 0 (dividend swept into $ sleeve); "
        f"got Cash={cash_after:.4f}. This is the failure shape of a "
        f"reorder-rebalance-before-dividend bug."
    )

    # And the dividend amount must be preserved in the post-rebalance
    # equity. Pre-rebalance NAV ≈ 200 XLU shares × $50 = $10,000; dividend
    # = 200 × $0.20 = $40; post-rebalance V should be ≈ $10,040 (modulo
    # tiny $ APY accrual over the few days from start).
    initial_xlu_shares = 10_000.0 / 50.0
    expected_dividend = initial_xlu_shares * 0.20
    expected_floor = 10_000.0 + expected_dividend - 1.0   # 1$ slack for $ APY
    assert v >= expected_floor, (
        f"Dividend amount lost on rebalance day. V={v:.4f}, "
        f"expected ≥ {expected_floor:.4f} (start + dividend − slack)"
    )


@pytest.mark.integration
def test_empty_dividend_data_equals_disabled(isolate_yf, flat_xlu_panel):
    """dividend_reinvestment=True but dividend_data has zero events should
    produce the same equity curve as dividend_reinvestment=False."""
    panel = flat_xlu_panel
    cfg_off = _div_cfg(target="cash", r1_alloc="XLU")
    cfg_off["dividend_reinvestment"] = False

    cfg_on = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [])

    eq_off, _, _ = run_backtest(
        panel, cfg_off, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio,
    )
    eq_on, _, _ = run_backtest(
        panel, cfg_on, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    diff = (eq_off["Value"].astype(float) - eq_on["Value"].astype(float)).abs().max()
    assert diff < 1e-9, (
        f"Empty-dividend run differs from disabled run by {diff:.2e}"
    )


@pytest.mark.integration
def test_dividend_df_schema(isolate_yf, flat_xlu_panel):
    """The third return value of run_backtest carries the dividend log.
    Pin its column set so downstream consumers (exporter, dashboard)
    don't break silently."""
    panel = flat_xlu_panel
    cfg = _div_cfg(target="cash", r1_alloc="XLU")
    div = _div_df(panel, [
        (panel.index[3].strftime("%Y-%m-%d"), "XLU", 0.10),
        (panel.index[7].strftime("%Y-%m-%d"), "XLU", 0.15),
    ])

    _, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    assert len(divs) == 2
    expected_cols = {
        "Date", "Ticker", "Dividend_Per_Share", "Shares", "Dividend_Amount",
        "Dividend_Yield", "Portfolio_Pct", "Reinvestment_Target",
        "Portfolio_Value",
    }
    missing = expected_cols - set(divs.columns)
    assert not missing, f"dividend_df missing expected columns: {missing}"


@pytest.mark.integration
def test_ticker_reinvest_with_nan_target_falls_back_to_cash(
    isolate_yf, flat_xlu_panel
):
    """When dividend_reinvestment_target is a ticker that has NaN price on
    the dividend day, the dividend must fall back to cash credit — not be
    silently dropped. Pins the safety fallback added with the dividend
    ordering comment.

    Setup: target='TQQQ' (a ticker in allocation_tickers), but force TQQQ
    price to NaN on the dividend day. Without the fallback, the dividend
    amount would vanish (no shares added, no cash credited). With the
    fallback, Cash should reflect the dividend amount."""
    panel = flat_xlu_panel.copy()
    div_day = panel.index[5]
    # Force TQQQ price to NaN on the dividend day to simulate pre-IPO /
    # halted-trading conditions for the reinvest target.
    panel.loc[div_day, "TQQQ"] = np.nan

    cfg = _div_cfg(target="TQQQ", r1_alloc="XLU")
    div = _div_df(panel, [(div_day.strftime("%Y-%m-%d"), "XLU", 0.50)])

    eq, _, divs = run_backtest(
        panel, cfg, compute_drawdown_from_ath, determine_regime,
        rebalance_portfolio, dividend_data=div,
    )

    initial_xlu_shares = 10_000.0 / 50.0
    expected_dividend = initial_xlu_shares * 0.50

    assert len(divs) == 1, f"Expected 1 dividend recorded; got {len(divs)}"

    # The dividend must have landed in Cash (fallback), not silently dropped.
    day5_cash = float(eq["Cash"].iloc[5])
    assert day5_cash == pytest.approx(expected_dividend, rel=1e-6), (
        f"Cash on dividend day should reflect the fallback credit "
        f"({expected_dividend:.4f}); got {day5_cash:.4f}. "
        f"This is the 'silently dropped on NaN target' regression shape."
    )

    # And TQQQ shares must NOT have been incremented (target was unpriced).
    day5_tqqq_shares = float(eq["TQQQ_shares"].iloc[5])
    assert day5_tqqq_shares == pytest.approx(0.0, abs=1e-9), (
        f"TQQQ shares should remain 0 — target was unpriced. Got {day5_tqqq_shares}"
    )

    # Mark invariant must still hold on this row (NaN-priced TQQQ excluded).
    panel_aligned = panel.reindex(pd.to_datetime(eq["Date"]))
    row = eq.iloc[5]
    manual = 0.0
    for t in cfg["allocation_tickers"]:
        col = f"{t}_shares"
        if col in eq.columns and t in panel_aligned.columns:
            s = float(row.get(col, 0) or 0)
            p = panel_aligned.loc[div_day, t]
            if pd.notna(p):
                manual += s * float(p)
    manual += float(row.get("Cash", 0) or 0)
    v = float(row["Value"])
    assert abs(v - manual) / max(abs(v), 1.0) <= 5e-6, (
        f"Mark invariant violated on NaN-target-fallback day: "
        f"V={v:.4f}, manual={manual:.4f}"
    )
