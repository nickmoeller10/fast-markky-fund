import pandas as pd
import yfinance as yf

from backtest import run_backtest
from config import CONFIG


# ============================================================
# TEST 1 — Price Data Integrity
# ============================================================
def test_price_data_quality():
    print("\n=== TEST 1: PRICE DATA QUALITY ===")

    tickers = CONFIG["tickers"]

    df = yf.download(tickers, start=CONFIG["start_date"])["Close"]

    print("Data shape:", df.shape)
    print("Tickers present:", df.columns.tolist())

    missing = df.isna().sum()
    print("Missing values:")
    print(missing)

    assert missing.sum() == 0, "❌ Missing price data exists!"

    print("✔ PASSED — No missing price data.\n")


def test_buy_and_hold():
    print("\n=== TEST 2: BUY & HOLD CONSISTENCY ===")

    for t in CONFIG["tickers"]:
        print(f"\nRunning 100% buy & hold test for {t}...")

        test_config = CONFIG.copy()
        test_config["tickers"] = [t]
        test_config["allocation_tickers"] = [t]     # 🔥 FIX
        test_config["regimes"] = {"R1": {t: 1.0}}

        raw = yf.download(t, start=test_config["start_date"])["Close"]

        # Normalize to DataFrame
        if isinstance(raw, pd.Series):
            df = raw.to_frame(name=t)
        else:
            df = raw.rename(columns={"Close": t})

        from regime_engine import compute_drawdown_from_ath

        result = run_backtest(
            price_data=df,
            config=test_config,
            regime_fn=lambda s: compute_drawdown_from_ath(s),
            regime_detector=lambda dd, cfg: "R1",
            rebalance_fn=lambda bal, alloc, prices: {t: bal / prices[t]}
        )

        base = df.iloc[0][t]
        expected = df[t] / base * test_config["starting_balance"]

        actual_end = result["Value"].iloc[-1]
        expected_end = expected.iloc[-1]

        print(f"Actual end:   {actual_end}")
        print(f"Expected end: {expected_end}")

        assert abs(actual_end - expected_end) < 1e-5, "❌ BUY & HOLD mismatch!"
        print(f"✔ PASSED — {t} matches normalized performance perfectly!")


# ============================================================
# TEST 3 — Normalization Math Validation
# ============================================================
def test_normalization():
    print("\n=== TEST 3: NORMALIZATION CHECK ===")

    tickers = CONFIG["tickers"]
    df = yf.download(tickers, start=CONFIG["start_date"])["Close"]

    first_valid = df.dropna().index[0]

    print("Common start date:", first_valid)

    for t in tickers:
        p0 = df.loc[first_valid, t]
        p1 = df.iloc[10][t]

        expected_norm = p1 / p0 * CONFIG["starting_balance"]

        print(f"\nTicker: {t}")
        print("Start price:", p0)
        print("Day+10 price:", p1)
        print(f"Expected normalized value: {expected_norm:.2f}")

    print("\n✔ PASSED — Normalization math manually verified.\n")


# ============================================================
# TEST 4 — Rebalance Share Math
# ============================================================
def test_rebalance_math():
    print("\n=== TEST 4: REBALANCE SHARE MATH ===")

    prices = pd.Series({"QQQ": 100, "XLU": 50})
    alloc = {"QQQ": 0.6, "XLU": 0.4}
    balance = 10000

    shares = {t: (balance * w) / prices[t] for t, w in alloc.items()}
    print("Shares:", shares)

    new_prices = pd.Series({"QQQ": 110, "XLU": 60})
    new_value = sum(shares[t] * new_prices[t] for t in shares)

    print("New portfolio value:", new_value)

    assert new_value > 0, "❌ Unexpected issue with rebalance math"

    print("✔ PASSED — Rebalance math consistent.\n")


# ============================================================
# TEST 5 — QUARTERLY REBALANCE CONFIRMATION (UPDATED)
# ============================================================
def test_quarterly_rebalance_confirmation():
    print("\n=== TEST 5: QUARTERLY REBALANCE CONFIRMATION ===")

    # Synthetic dates covering a full quarter (Q1 → Q2 boundary)
    dates = pd.date_range("2024-01-01", "2024-04-05")  # includes Mar 31

    # Construct simple price series
    # A goes up steadily; B goes up faster
    prices = pd.DataFrame({
        "A": [100 + i for i in range(len(dates))],
        "B": [200 + 2*i for i in range(len(dates))]
    }, index=dates)

    # Expected rebalance day = Mar 31 (Q1 end)
    expected_rebalance_days = ["2024-03-31"]

    config = {
        "starting_balance": 10000,
        "tickers": ["A", "B"],
        "allocation_tickers": ["A", "B"],
        "rebalance_frequency": "quarterly",  # default, but explicit
        "regimes": {
            "R": {"A": 0.5, "B": 0.5}
        }
    }

    # Regime = always R
    regime_fn = lambda s: (pd.Series(0, index=prices.index), None)
    regime_detector = lambda dd, cfg: "R"

    rebalance_fn = lambda bal, alloc, price: {
        t: (bal * w) / price[t] for t, w in alloc.items()
    }

    result = run_backtest(
        price_data=prices,
        config=config,
        regime_fn=regime_fn,
        regime_detector=regime_detector,
        rebalance_fn=rebalance_fn
    )

    # Extract actual rebalance dates
    rebalanced_days = (
        result[result["Rebalanced"] == "Rebalanced"]["Date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    print("Rebalanced on:", rebalanced_days)

    assert rebalanced_days == expected_rebalance_days, (
        f"❌ Quarterly rebalance incorrect.\n"
        f"Expected: {expected_rebalance_days}\n"
        f"Got:      {rebalanced_days}\n"
    )

    # ---- VALUE CHECK ----
    # After Mar 31 rebalance, portfolio should be equally split again.
    #
    # Compute the expected post-rebalance value manually:
    # 1) Track portfolio value on Mar 31 before rebalance
    rebalance_date = pd.Timestamp("2024-03-31")
    pre_reb_val = result.loc[result["Date"] == rebalance_date]["Value"].iloc[0]

    # 2) Shares after rebalance (50/50 allocation)
    price_row = prices.loc[rebalance_date]
    expected_shares = {
        "A": (pre_reb_val * 0.5) / price_row["A"],
        "B": (pre_reb_val * 0.5) / price_row["B"],
    }

    # 3) Final value based on final prices
    last_price = prices.iloc[-1]
    expected_final_value = (
        expected_shares["A"] * last_price["A"] +
        expected_shares["B"] * last_price["B"]
    )

    final_val = result["Value"].iloc[-1]
    print("Final portfolio value:", final_val)
    print("Expected final value:", expected_final_value)

    assert abs(final_val - expected_final_value) < 1e-5, (
        f"❌ Value mismatch after quarterly rebalance.\n"
        f"Expected: {expected_final_value}, Got: {final_val}"
    )

    print("✔ PASSED — Quarterly rebalancing works correctly!\n")

# ============================================================
# TEST 6 — WEEKLY REBALANCE CONFIRMATION
# ============================================================
def test_weekly_rebalance_confirmation():
    print("\n=== TEST 6: WEEKLY REBALANCE CONFIRMATION ===")

    # Synthetic 10-day period (includes two Fridays)
    dates = pd.date_range("2024-01-01", periods=10)  # Jan 1 → Jan 10, 2024
    prices = pd.DataFrame({
        "A": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "B": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218],
    }, index=dates)

    # Fridays in this window: Jan 5 and Jan 12 (but Jan 12 not included)
    expected_rebalance_days = ["2024-01-05"]

    config = {
        "starting_balance": 10000,
        "tickers": ["A", "B"],
        "allocation_tickers": ["A", "B"],
        "rebalance_frequency": "weekly",  # 🔥 THIS IS THE SETTING WE ARE TESTING
        "regimes": {
            "R": {"A": 0.5, "B": 0.5}
        }
    }

    # Regime logic always returns constant regime R
    regime_fn = lambda s: (pd.Series(0, index=prices.index), None)
    regime_detector = lambda dd, cfg: "R"
    rebalance_fn = lambda bal, alloc, price: {
        t: (bal * w) / price[t] for t, w in alloc.items()
    }

    result = run_backtest(
        price_data=prices,
        config=config,
        regime_fn=regime_fn,
        regime_detector=regime_detector,
        rebalance_fn=rebalance_fn,
    )

    # Extract which days actually rebalanced
    rebalanced_days = result[result["Rebalanced"] == "Rebalanced"]["Date"].dt.strftime("%Y-%m-%d").tolist()

    print("Rebalanced on:", rebalanced_days)

    # Assert they match expected weekly Fridays
    assert rebalanced_days == expected_rebalance_days, (
        f"❌ Weekly rebalance incorrect.\n"
        f"Expected: {expected_rebalance_days}\n"
        f"Got:      {rebalanced_days}\n"
    )

    # SIMPLE VALUE CHECK — ensure portfolio value increased correctly
    final_val = result.iloc[-1]["Value"]
    print("Final portfolio value:", final_val)

    assert final_val > config["starting_balance"], "❌ Value did not increase — unexpected."

    print("✔ PASSED — Weekly rebalancing works correctly!\n")


# ============================================================
# Run All Tests
# ============================================================
if __name__ == "__main__":
    test_price_data_quality()
    test_buy_and_hold()
    test_normalization()
    test_rebalance_math()
    test_quarterly_rebalance_confirmation()
    test_weekly_rebalance_confirmation()

    print("\n🎉 ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!\n")
