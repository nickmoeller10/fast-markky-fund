import pandas as pd
import numpy as np
from allocation_engine import get_allocation_for_regime, tradable_allocation
from data_loader import yf_close_to_series
from regime_engine import compute_drawdown_from_ath
from signal_layers import (
    build_combined_spy_series,
    build_combined_vix_series,
    build_signal_total_series,
)
from signal_override_engine import (
    ensure_regime_signal_overrides,
    desired_signal_override_mode,
    get_target_allocation_for_override,
    describe_signal_override_row,
    validate_panel_sums,
    any_override_enabled,
)
from utils import log

# Re-export from extracted modules so callers (tests, app.py, optimizer) keep
# importing them as `from backtest import X`. The original behavior is preserved.
from backtest_helpers import (
    bottom_regime_number,
    calculate_normalized_values,
    get_rebalance_dates,
    regime_label,
    regime_number,
    scalar_drawdown_for_regime as _scalar_drawdown_for_regime,
)
from backtest_drawdown import (
    build_regime_signal_drawdown,
    compute_rolling_ath_and_dd,
)
from backtest_transitions import (
    apply_always_rebalance,
    apply_asymmetric_rules,
    apply_asymmetric_rules_down_only,
    apply_asymmetric_rules_up_only,
    apply_per_regime_direction_strategy,
    apply_rebalancing_strategy,
    get_initial_allocation,
    regime_trajectory_label,
)
from backtest_recording import (
    record_daily_row,
    record_missing_row,
    update_portfolio_value,
)


# ------------------------------------------------------------
# MAIN BACKTEST ENGINE
# ------------------------------------------------------------
def run_backtest(price_data, config, dd_fn, regime_detector, rebalance_fn, dividend_data=None):
    log("ENTERED run_backtest()")

    tickers = config["tickers"]
    starting_val = config["starting_balance"]
    drawdown_ticker = config["drawdown_ticker"]
    drawdown_window_enabled = bool(config.get("drawdown_window_enabled", False))
    try:
        drawdown_window_years = int(config.get("drawdown_window_years", 5))
    except (TypeError, ValueError):
        drawdown_window_years = 5

    deepest_regime_n = bottom_regime_number(config)

    # Dividend reinvestment settings
    dividend_reinvestment = config.get("dividend_reinvestment", False)
    dividend_target = config.get("dividend_reinvestment_target", "cash")
    allocation_tickers = config.get("allocation_tickers", tickers)

    # -------------------------
    # Load prices & setup
    # -------------------------
    price_df = price_data[tickers].copy()
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(-1)

    # Panel from first row with any ticker price; capital deploys using tradable_allocation (e.g. QQQ proxy for TQQQ)
    any_row = price_df.notna().any(axis=1)
    if not any_row.any():
        raise ValueError("Price panel is empty — check tickers and date range.")
    earliest_any = price_df.index[any_row][0]
    price_df = price_df.loc[earliest_any:].copy()
    panel_start = price_df.index[0]

    if not price_df.loc[panel_start, allocation_tickers].notna().any():
        raise ValueError(
            f"On {panel_start.date()} no allocation ticker in {allocation_tickers} has a price. "
            "Choose a later start date."
        )

    norm_df = calculate_normalized_values(price_df, tickers, starting_val, panel_start)

    # -------------------------
    # PRE-PORTFOLIO SIMULATION: Track regimes from historical ATH
    # -------------------------
    # Download full historical data for drawdown_ticker from its inception
    log(f"Downloading full historical data for {drawdown_ticker} from inception for pre-portfolio simulation...")
    # Re-import inside the function so tests can monkeypatch `data_cache.cached_yf_download`
    # and have the patch take effect on every run (vs. capturing a module-top reference).
    from data_cache import cached_yf_download

    # Download from ticker inception (use 1980 as safe early date - yfinance returns from actual inception)
    # drawdown_price_series = Yahoo history merged with portfolio panel: on overlapping dates the portfolio
    # download wins so regime ATH matches {ticker}_price in the equity table (no ATH < close artifacts).
    drawdown_price_series = None
    historical_dd_full = None
    historical_ath_full = None
    pre_portfolio_start_date = None

    try:
        historical_data = cached_yf_download(
            drawdown_ticker,
            start="1980-01-01",  # Very early date - yfinance will return from ticker's actual inception
            end=None,  # Get all available data up to current
            auto_adjust=True,
            progress=False
        )
        
        if not historical_data.empty and "Close" in historical_data.columns:
            yf_series = yf_close_to_series(historical_data["Close"], drawdown_ticker)
            
            if len(yf_series) > 0:
                log(f"Downloaded {len(yf_series)} Yahoo rows for {drawdown_ticker} "
                    f"({yf_series.index.min()} to {yf_series.index.max()})")
                s_hist = yf_series.sort_index().astype(float)
                s_port = price_data[drawdown_ticker].dropna().sort_index().astype(float)
                drawdown_price_series = s_port.combine_first(s_hist)
                log(
                    f"Drawdown price path: {len(drawdown_price_series)} rows after merging "
                    f"(portfolio closes override Yahoo on shared dates for consistency with backtest prices)."
                )
                
                # Regime drawdown from unified series: standard ATH or rolling N-calendar-year peak
                if drawdown_window_enabled and drawdown_window_years > 0:
                    historical_ath_full, historical_dd_full = compute_rolling_ath_and_dd(
                        drawdown_price_series, drawdown_window_years
                    )
                    log(
                        f"Drawdown mode: rolling {drawdown_window_years}-year calendar window "
                        f"(fallback to standard ATH until {drawdown_window_years}y of history exists)"
                    )
                else:
                    historical_ath_full = drawdown_price_series.cummax()
                    historical_dd_full = (historical_ath_full - drawdown_price_series) / historical_ath_full
                    log("Drawdown mode: standard ATH (cummax from ticker inception)")
                
                pre_portfolio_start_date = drawdown_price_series.index.min()
                log(f"Pre-portfolio simulation will start from {pre_portfolio_start_date} (unified series)")
                
                portfolio_start_ts = pd.to_datetime(panel_start)
                if portfolio_start_ts in drawdown_price_series.index:
                    start_price = float(drawdown_price_series.loc[portfolio_start_ts])
                    start_ath = float(historical_ath_full.loc[portfolio_start_ts])
                    start_dd = float(historical_dd_full.loc[portfolio_start_ts])
                    log(f"At first trade date ({panel_start}): Price=${start_price:.2f}, ATH=${start_ath:.2f}, DD={start_dd:.2%}")
                else:
                    dates_before = drawdown_price_series.index[drawdown_price_series.index < portfolio_start_ts]
                    if len(dates_before) > 0:
                        closest_date = dates_before[-1]
                        start_price = float(drawdown_price_series.loc[closest_date])
                        start_ath = float(historical_ath_full.loc[closest_date])
                        start_dd = float(historical_dd_full.loc[closest_date])
                        log(f"At first trade date ({panel_start}): Using closest historical date {closest_date}, Price=${start_price:.2f}, ATH=${start_ath:.2f}, DD={start_dd:.2%}")
    except Exception as e:
        log(f"Warning: Error downloading historical data for {drawdown_ticker}: {e}")
    
    # -------------------------
    # PRE-PORTFOLIO SIMULATION: Track regimes while holding cash
    # -------------------------
    portfolio_regime_at_start = None
    market_regime_at_start = None
    
    if historical_dd_full is not None and pre_portfolio_start_date is not None:
        log(f"Running pre-portfolio simulation from {pre_portfolio_start_date} to {panel_start} (holding cash, tracking regimes)...")
        hist_regime_dd = build_regime_signal_drawdown(
            historical_dd_full,
            historical_dd_full.index,
            historical_dd_full.index,
            None,
        )

        portfolio_start_ts = pd.to_datetime(panel_start)
        # Get all dates from historical data up to (and including if available) portfolio start
        pre_portfolio_dates = historical_dd_full.index[
            (historical_dd_full.index >= pre_portfolio_start_date) & 
            (historical_dd_full.index <= portfolio_start_ts)
        ]
        
        # If portfolio start date is not in historical index, get dates up to but not including it
        if len(pre_portfolio_dates) == 0 or (portfolio_start_ts not in pre_portfolio_dates and len(pre_portfolio_dates) > 0):
            pre_portfolio_dates = historical_dd_full.index[
                (historical_dd_full.index >= pre_portfolio_start_date) & 
                (historical_dd_full.index < portfolio_start_ts)
            ]
        
        # Track regime through pre-portfolio period
        current_portfolio_regime = None
        rebalance_strategy = config.get("rebalance_strategy", "down_only")
        prev_mkt_pre = None

        for pre_date in pre_portfolio_dates:
            pre_dd = _scalar_drawdown_for_regime(hist_regime_dd.loc[pre_date])
            market_regime = regime_detector(pre_dd, config)
            
            if market_regime is not None:
                if current_portfolio_regime is None:
                    current_portfolio_regime = market_regime
                elif rebalance_strategy == "per_regime":
                    current_portfolio_regime = apply_per_regime_direction_strategy(
                        current_portfolio_regime, prev_mkt_pre, market_regime, config
                    )
                elif rebalance_strategy == "down_only":
                    current_portfolio_regime = apply_asymmetric_rules_down_only(
                        current_portfolio_regime, market_regime
                    )
                elif rebalance_strategy == "up_only":
                    current_portfolio_regime = apply_asymmetric_rules_up_only(
                        current_portfolio_regime, market_regime, deepest_regime_n
                    )
                elif rebalance_strategy == "always":
                    current_portfolio_regime = apply_always_rebalance(
                        current_portfolio_regime, market_regime
                    )
                else:
                    current_portfolio_regime = market_regime
                prev_mkt_pre = market_regime
        
        # Regime for first trade session: prior close vs ATH (not same-bar as open prices)
        open_signal_dd = build_regime_signal_drawdown(
            historical_dd_full,
            pd.DatetimeIndex([portfolio_start_ts]),
            historical_dd_full.index,
            None,
        ).iloc[0]
        market_regime_at_start = regime_detector(
            _scalar_drawdown_for_regime(open_signal_dd), config
        )

        # Determine portfolio regime at start
        if current_portfolio_regime is None:
            portfolio_regime_at_start = market_regime_at_start
        else:
            if rebalance_strategy == "per_regime":
                portfolio_regime_at_start = apply_per_regime_direction_strategy(
                    current_portfolio_regime, prev_mkt_pre, market_regime_at_start, config
                )
            elif rebalance_strategy == "down_only":
                portfolio_regime_at_start = apply_asymmetric_rules_down_only(
                    current_portfolio_regime, market_regime_at_start
                )
            elif rebalance_strategy == "up_only":
                portfolio_regime_at_start = apply_asymmetric_rules_up_only(
                    current_portfolio_regime, market_regime_at_start, deepest_regime_n
                )
            elif rebalance_strategy == "always":
                portfolio_regime_at_start = apply_always_rebalance(
                    current_portfolio_regime, market_regime_at_start
                )
            else:
                portfolio_regime_at_start = market_regime_at_start
        
        log(f"Pre-portfolio simulation complete. Regime at portfolio start: Market={market_regime_at_start}, Portfolio={portfolio_regime_at_start}")
    
    # -------------------------
    # Calculate drawdown for portfolio period (align to price_data index)
    # -------------------------
    if (
        drawdown_window_enabled
        and drawdown_window_years > 0
        and historical_ath_full is not None
        and historical_dd_full is not None
        and drawdown_price_series is not None
    ):
        portfolio_prices = price_data[drawdown_ticker].reindex(price_data.index)
        ath_aligned = historical_ath_full.reindex(price_data.index).ffill().bfill()
        ath_safe = ath_aligned.replace(0, np.nan)
        dd_raw = (ath_safe - portfolio_prices) / ath_safe
        ath_raw = ath_aligned
        if len(portfolio_prices.dropna()) == 0:
            log(f"Warning: No portfolio period data for {drawdown_ticker}, using standard calculation")
            dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
        else:
            log(f"Portfolio-period drawdown aligned to rolling {drawdown_window_years}y reference (full-history download)")
    elif historical_ath_full is not None and drawdown_price_series is not None:
        # Standard ATH: running peak including history before backtest start
        portfolio_start_ts = pd.to_datetime(panel_start)
        historical_ath_before_start = historical_ath_full[historical_ath_full.index < portfolio_start_ts]
        
        if len(historical_ath_before_start) > 0:
            max_historical_ath = float(historical_ath_before_start.max())
            log(f"Historical ATH for {drawdown_ticker} before portfolio start: ${max_historical_ath:.2f}")
            
            portfolio_prices = price_data[drawdown_ticker].reindex(price_data.index)
            
            if len(portfolio_prices.dropna()) > 0:
                ath_raw = pd.Series(index=portfolio_prices.index, dtype=float)
                running_max = max_historical_ath
                
                for date in portfolio_prices.index:
                    current_price = portfolio_prices.loc[date]
                    if pd.notna(current_price):
                        running_max = max(running_max, current_price)
                        ath_raw.loc[date] = running_max
                    else:
                        ath_raw.loc[date] = running_max
                
                ath_safe = ath_raw.replace(0, np.nan)
                dd_raw = (ath_safe - portfolio_prices) / ath_safe
            else:
                log(f"Warning: No portfolio period data for {drawdown_ticker}, using standard calculation")
                dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
        else:
            log(f"Warning: No historical ATH before {panel_start}, using standard calculation")
            dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
    else:
        if drawdown_window_enabled and drawdown_window_years > 0:
            log("Warning: No full-history download; rolling window drawdown from portfolio prices only")
            dd_series = price_data[drawdown_ticker].dropna()
            if len(dd_series) > 0:
                ath_roll, _ = compute_rolling_ath_and_dd(dd_series, drawdown_window_years)
                portfolio_prices = price_data[drawdown_ticker].reindex(price_data.index)
                ath_raw = ath_roll.reindex(price_data.index).ffill().bfill()
                ath_safe = ath_raw.replace(0, np.nan)
                dd_raw = (ath_safe - portfolio_prices) / ath_safe
            else:
                dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
        else:
            log(f"Warning: Using standard ATH calculation (no historical data available)")
            dd_raw, ath_raw = dd_fn(price_data[drawdown_ticker])
    
    # Ensure dd_raw and ath_raw are aligned with price_data index
    dd_raw = (
        dd_raw.reindex(price_data.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    ath_raw = ath_raw.reindex(price_data.index)

    regime_dd_signal = build_regime_signal_drawdown(
        dd_raw,
        price_df.index,
        price_data.index,
        historical_dd_full,
    )

    for _rp in (config.get("regimes") or {}).values():
        ensure_regime_signal_overrides(_rp)
    validate_panel_sums(config)

    # Building signal layers (SPY/VIX downloads, MACD/MA computation) is the most
    # expensive non-loop step in run_backtest. When no regime has an enabled
    # override panel, the result is unused — skip with a NaN series of matching shape.
    if any_override_enabled(config):
        if "SPY" in price_df.columns:
            spy_panel_sig = price_df["SPY"].astype(float)
        else:
            spy_panel_sig = pd.Series(np.nan, index=price_df.index, dtype=float)
        spy_signal_merged = build_combined_spy_series(price_df.index, spy_panel_sig)
        vix_signal_merged = build_combined_vix_series(price_df.index, None)
        signal_total_series = build_signal_total_series(
            price_df.index, spy_signal_merged, vix_signal_merged
        )
    else:
        signal_total_series = pd.Series(np.nan, index=price_df.index, dtype=float)

    rebalance_freq = config.get("rebalance_frequency", "monthly")
    rebalance_dates = get_rebalance_dates(price_df.index, rebalance_freq)
    instant_mode = rebalance_freq == "instant"

    # -------------------------
    # INITIAL ALLOCATION
    # -------------------------
    # Use the regime determined from pre-portfolio simulation, or calculate if not available
    if portfolio_regime_at_start is not None and market_regime_at_start is not None:
        log(f"Using pre-portfolio simulation regime: Market={market_regime_at_start}, Portfolio={portfolio_regime_at_start}")
        market_regime = market_regime_at_start
        portfolio_regime = portfolio_regime_at_start
        # Get allocation for the determined regime and rebalance (tradable subset / TQQQ→QQQ proxy, etc.)
        alloc = get_allocation_for_regime(portfolio_regime, config)
        alloc_t = tradable_allocation(alloc, price_df.loc[panel_start], config)
        if not alloc_t:
            raise ValueError(
                f"Cannot open portfolio on {panel_start}: no tradable weights for regime {portfolio_regime} "
                f"with prices for {allocation_tickers}."
            )
        shares = rebalance_fn(config["starting_balance"], alloc_t, price_df.loc[panel_start])
        log(f"Initial rebalance on {panel_start} into {portfolio_regime} with tradable allocation: {alloc_t}")
    else:
        # Fallback: regime from prior close (or same bar if no history on full index)
        first_dd = float(regime_dd_signal.loc[panel_start])
        market_regime, portfolio_regime, shares = get_initial_allocation(
            panel_start, price_df, first_dd, config, regime_detector, rebalance_fn
        )
        log(f"Using standard initial allocation (no pre-portfolio simulation): Market={market_regime}, Portfolio={portfolio_regime}")
    
    if shares is None:
        raise ValueError(
            f"Initial tradable allocation is empty on {panel_start} — need at least one priced allocation ticker."
        )
    if not isinstance(shares, dict):
        log(f"⚠️ WARNING: shares is not a dict on {panel_start} (type: {type(shares)}). Converting to dict.")
        shares = {} if shares is None else dict(shares) if hasattr(shares, 'items') else {}

    portfolio_value = starting_val
    prev_value = starting_val
    portfolio_ath = starting_val
    cash_balance = 0.0  # Track cash from dividends if not reinvested immediately

    # -------------------------
    # STORAGE
    # -------------------------
    equity_rows = []
    rebalance_rows = []
    dividend_rows = []  # Track dividend events

    last_rebalance_value = portfolio_value
    quarter_returns = []   # reused as volatility measure for rebalance summary
    prev_market_regime = None  # prior row's market regime (for trajectory + per_regime strategy)
    signal_override_mode = "none"

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for date in price_df.index:

        row_prices = price_df.loc[date]
        row_norm = norm_df.loc[date]
        raw_dd = dd_raw.loc[date]
        raw_ath = ath_raw.loc[date]

        # Build DD columns
        dd_cols = {
            f"{drawdown_ticker}_ATH_raw": raw_ath,
            f"{drawdown_ticker}_DD_raw": raw_dd,
            "Portfolio_ATH": None,
            "Portfolio_DD": None
        }

        # Cannot value holdings if any held ticker is missing a price; if flat cash, need some alloc prices to proceed
        held = shares if isinstance(shares, dict) else {}
        skip_day = False
        if any(abs(float(held.get(t, 0) or 0)) > 1e-12 for t in held):
            for t, q in held.items():
                if abs(float(q or 0)) <= 1e-12:
                    continue
                if t not in row_prices.index or pd.isna(row_prices[t]):
                    record_missing_row(
                        equity_rows, date, tickers, row_prices, row_norm,
                        shares, portfolio_value, portfolio_regime,
                        starting_val, dd_cols,
                        regime_trajectory="",
                        prev_market_regime=prev_market_regime,
                        config=config,
                        signal_override_mode=signal_override_mode,
                    )
                    skip_day = True
                    break
        elif not row_prices[allocation_tickers].notna().any():
            record_missing_row(
                equity_rows, date, tickers, row_prices, row_norm,
                shares, portfolio_value, portfolio_regime,
                starting_val, dd_cols,
                regime_trajectory="",
                prev_market_regime=prev_market_regime,
                config=config,
                signal_override_mode=signal_override_mode,
            )
            skip_day = True
        if skip_day:
            continue

        # Mark portfolio to TODAY's prices before any rebalance logic runs.
        # Without this, a rebalance on day N would build new shares from
        # yesterday's close NAV, destroying today's intraday return on every
        # rebalance day. Non-rebalance days were already correct because the
        # end-of-day update_portfolio_value() marked to today's prices.
        portfolio_value = sum(
            float(shares.get(t, 0) or 0) * float(row_prices[t])
            for t in shares
            if t in row_prices.index and pd.notna(row_prices[t])
        )

        raw_sig = signal_total_series.loc[date] if date in signal_total_series.index else np.nan
        s_curr = float(raw_sig) if pd.notna(raw_sig) else np.nan

        # -------------------------
        # Dividend handling
        # -------------------------
        # MUST run before the regime/rebalance blocks below. The dividend
        # logic reads pre-rebalance share counts to determine which holdings
        # generated dividends today (`shares.get(ticker, 0) > 0`). If this
        # block were moved to run after the rebalance, a regime-change-day
        # rebalance would zero out the original holdings before the dividend
        # could be recorded, silently dropping the dividend entirely.
        if dividend_reinvestment and dividend_data is not None and not dividend_data.empty:
            # Ensure shares is a dict
            if shares is None:
                log(f"⚠️ WARNING: shares is None during dividend handling on {date}. Initializing as empty dict.")
                shares = {}
            
            daily_dividends = 0.0
            reinvested_into_shares = False
            for ticker in allocation_tickers:
                # Check if ticker exists in dividend_data columns and we have shares
                if ticker in dividend_data.columns and isinstance(shares, dict) and ticker in shares and shares.get(ticker, 0) > 0:
                    # Get dividend for this date
                    if date in dividend_data.index:
                        dividend_per_share = dividend_data.loc[date, ticker]
                        if pd.notna(dividend_per_share) and dividend_per_share > 0:
                            num_shares = shares.get(ticker, 0)
                            dividend_amount = dividend_per_share * num_shares
                            daily_dividends += dividend_amount
                            
                            # Calculate yield (dividend per share / price per share)
                            ticker_price = row_prices.get(ticker, 0)
                            dividend_yield = (dividend_per_share / ticker_price * 100) if ticker_price > 0 else 0
                            
                            # Calculate percentage of portfolio value (before dividend is applied)
                            total_portfolio_value_before = portfolio_value + cash_balance
                            dividend_pct = (dividend_amount / total_portfolio_value_before * 100) if total_portfolio_value_before > 0 else 0
                            
                            # Record dividend event
                            dividend_rows.append({
                                "Date": date,
                                "Ticker": ticker,
                                "Dividend_Per_Share": dividend_per_share,
                                "Shares": num_shares,
                                "Dividend_Amount": dividend_amount,
                                "Dividend_Yield": dividend_yield,
                                "Portfolio_Pct": dividend_pct,
                                "Reinvestment_Target": dividend_target,
                                "Portfolio_Value": total_portfolio_value_before
                            })
                            
                            if dividend_target == "cash":
                                # Add to cash balance
                                cash_balance += dividend_amount
                                # Cash is included in total value calculation below
                            elif dividend_target in allocation_tickers and dividend_target in row_prices:
                                # Reinvest immediately into target ticker
                                target_price = row_prices[dividend_target]
                                if target_price > 0 and pd.notna(target_price):
                                    additional_shares = dividend_amount / target_price
                                    shares[dividend_target] = shares.get(dividend_target, 0) + additional_shares
                                    reinvested_into_shares = True
                                else:
                                    # Target unpriced today (NaN/zero) — fall back to cash
                                    # so the dividend is never silently dropped.
                                    cash_balance += dividend_amount
                                    log(f"Dividend reinvestment target {dividend_target} unpriced on {date}; "
                                        f"credited ${dividend_amount:.4f} to cash instead.")
                            else:
                                # Unknown / out-of-allocation target — fall back to cash.
                                cash_balance += dividend_amount
                                log(f"Dividend reinvestment target {dividend_target!r} unavailable on {date}; "
                                    f"credited ${dividend_amount:.4f} to cash instead.")

                            # Log dividend for debugging
                            log(f"Dividend: {ticker} paid ${dividend_amount:.2f} ({dividend_per_share:.4f} per share, {num_shares:.4f} shares)")
            
            # After processing all dividends, update portfolio value if dividends were reinvested into shares
            # (cash dividends are already in cash_balance and will be included in total value)
            if daily_dividends > 0 and reinvested_into_shares:
                # Recalculate portfolio value to include new shares from dividend reinvestment
                portfolio_value = sum(shares[t] * row_prices.get(t, 0) for t in shares if t in row_prices)
                log(f"Portfolio value updated after dividend reinvestment: ${portfolio_value:,.2f}")

        # -------------------------
        # Regime detection (prior session close; execution uses row_prices today)
        # -------------------------
        market_regime = regime_detector(float(regime_dd_signal.loc[date]), config)
        regime_trajectory = regime_trajectory_label(prev_market_regime, market_regime)
        prev_market_for_row = prev_market_regime

        new_regime = portfolio_regime
        do_rebalance = False
        rebalanced_flag = ""

        # -------------------------
        # Get rebalancing strategy from config
        # -------------------------
        rebalance_strategy = config.get("rebalance_strategy", "down_only")
        
        # -------------------------
        # INSTANT MODE (rebalance whenever regime changes)
        # -------------------------
        if instant_mode:
            if rebalance_strategy == "per_regime":
                new_regime = apply_per_regime_direction_strategy(
                    portfolio_regime, prev_market_regime, market_regime, config
                )
            else:
                new_regime = apply_rebalancing_strategy(
                    portfolio_regime, market_regime, rebalance_strategy, config
                )
            do_rebalance = new_regime != portfolio_regime

        # -------------------------
        # PERIODIC MODE
        # -------------------------
        else:
            if date in rebalance_dates:
                if rebalance_strategy == "per_regime":
                    new_regime = apply_per_regime_direction_strategy(
                        portfolio_regime, prev_market_regime, market_regime, config
                    )
                    do_rebalance = new_regime != portfolio_regime
                else:
                    new_regime = apply_rebalancing_strategy(
                        portfolio_regime, market_regime, rebalance_strategy, config
                    )
                    do_rebalance = True

        # -------------------------
        # EXECUTE REBALANCE IF TRIGGERED (tradable weights only; TQQQ→QQQ before TQQQ lists)
        # -------------------------
        regime_rebalance_executed = False
        if do_rebalance:
            portfolio_value_with_cash = portfolio_value + cash_balance
            alloc_t = tradable_allocation(
                get_allocation_for_regime(new_regime, config), row_prices, config
            )
            if not alloc_t:
                log(f"No tradable allocation on {date}; keeping positions and portfolio regime {portfolio_regime}.")
            else:
                new_shares = rebalance_fn(portfolio_value_with_cash, alloc_t, row_prices)
                if new_shares is None:
                    log(f"⚠️ Rebalance failed on {date}; keeping prior positions.")
                else:
                    portfolio_regime = new_regime
                    shares = new_shares
                    cash_balance = 0.0
                    portfolio_value = portfolio_value_with_cash
                    rebalanced_flag = "Rebalanced"
                    signal_override_mode = "none"
                    regime_rebalance_executed = True

                    portfolio_value, prev_value = update_portfolio_value(
                        shares, row_prices, prev_value, quarter_returns
                    )
                    total_value = portfolio_value
                    portfolio_ath = max(portfolio_ath, total_value)
                    portfolio_dd = (total_value / portfolio_ath) - 1 if portfolio_ath > 0 else 0

                    rebalance_rows.append({
                        "Date": date,
                        "Portfolio_Value": portfolio_value,
                        "Market_Regime": market_regime,
                        "Portfolio_Regime": portfolio_regime,
                        "Regime_Trajectory": regime_trajectory,
                        "Prev_Market_Regime": prev_market_for_row,
                        "QoQ_Return": (portfolio_value / last_rebalance_value) - 1,
                        "QoQ_Volatility": pd.Series(quarter_returns).std() if len(quarter_returns) else 0,
                        f"{drawdown_ticker}_ATH_raw": raw_ath,
                        f"{drawdown_ticker}_DD_raw": raw_dd,
                        "Portfolio_ATH": portfolio_ath,
                        "Portfolio_DD": portfolio_dd,
                        **{f"{t}_shares": float(shares.get(t, 0)) for t in tickers},
                        **{f"{t}_value": float(shares.get(t, 0) * row_prices[t]) for t in tickers}
                    })

                    last_rebalance_value = portfolio_value
                    quarter_returns = []

        # -------------------------
        # Signal override rebalance (only when regime did not rebalance today)
        # -------------------------
        if not regime_rebalance_executed:
            rp = config["regimes"][portfolio_regime]
            ensure_regime_signal_overrides(rp)
            next_ov = desired_signal_override_mode(s_curr, rp, signal_override_mode)
            if next_ov != signal_override_mode:
                portfolio_value_with_cash = portfolio_value + cash_balance
                alloc_t = tradable_allocation(
                    get_target_allocation_for_override(portfolio_regime, next_ov, config),
                    row_prices,
                    config,
                )
                if alloc_t:
                    new_shares = rebalance_fn(
                        portfolio_value_with_cash, alloc_t, row_prices
                    )
                    if new_shares is not None:
                        signal_override_mode = next_ov
                        shares = new_shares
                        cash_balance = 0.0
                        portfolio_value = portfolio_value_with_cash
                        rebalanced_flag = "SignalOverride"

                        portfolio_value, prev_value = update_portfolio_value(
                            shares, row_prices, prev_value, quarter_returns
                        )
                        total_value = portfolio_value
                        portfolio_ath = max(portfolio_ath, total_value)
                        portfolio_dd = (
                            (total_value / portfolio_ath) - 1 if portfolio_ath > 0 else 0
                        )

                        last_rebalance_value = portfolio_value
                        quarter_returns = []

        # -------------------------
        # UPDATE VALUE FOR REGULAR DAYS
        # -------------------------
        if not rebalanced_flag:
            # Update portfolio value (this will include any new shares from dividend reinvestment)
            portfolio_value, prev_value = update_portfolio_value(
                shares, row_prices, prev_value, quarter_returns
            )
            # Include cash in total value for ATH tracking (cash includes dividends held as cash)
            total_value = portfolio_value + cash_balance
            portfolio_ath = max(portfolio_ath, total_value)
            portfolio_dd = (total_value / portfolio_ath) - 1 if portfolio_ath > 0 else 0

        # update DD columns
        dd_cols["Portfolio_ATH"] = portfolio_ath
        dd_cols["Portfolio_DD"] = portfolio_dd

        # -------------------------
        # RECORD DAILY ROW
        # -------------------------
        record_daily_row(
            equity_rows, date, tickers, row_prices, row_norm, shares,
            portfolio_value, market_regime, portfolio_regime,
            starting_val, rebalanced_flag, dd_cols, cash_balance,
            regime_trajectory=regime_trajectory,
            prev_market_regime=prev_market_for_row,
            config=config,
            signal_override_mode=signal_override_mode,
        )
        prev_market_regime = market_regime

    # -------------------------
    # BUILD DATAFRAMES
    # -------------------------
    equity_df = pd.DataFrame(equity_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)
    dividend_df = pd.DataFrame(dividend_rows) if dividend_rows else pd.DataFrame()

    log("LEAVING run_backtest(), returning DataFrames")
    return equity_df, rebalance_df, dividend_df
