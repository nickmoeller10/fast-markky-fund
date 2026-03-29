import pandas as pd
import numpy as np
from allocation_engine import get_allocation_for_regime, tradable_allocation
from data_loader import yf_close_to_series
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
)
from utils import log


def _scalar_drawdown_for_regime(val):
    """Coerce yfinance/indexing quirks to a float for regime_detector; NaN/inf → 0.0 (treat as R1 band)."""
    if val is None:
        return 0.0
    if isinstance(val, pd.Series):
        val = val.dropna()
        val = val.iloc[0] if len(val) else np.nan
    try:
        x = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


# ------------------------------------------------------------
# Regime helpers
# ------------------------------------------------------------
def regime_number(label: str) -> int:
    return int(label.replace("R", ""))


def regime_label(num: int) -> str:
    return f"R{num}"


def bottom_regime_number(config) -> int:
    """Largest regime index (e.g. 4 for R4) for asymmetric rules that reference the deepest regime."""
    keys = list((config or {}).get("regimes") or [])
    if not keys:
        return 3
    return max(regime_number(k) for k in keys)


# ------------------------------------------------------------
# Rebalance date utilities
# ------------------------------------------------------------
def get_rebalance_dates(index, frequency):
    if frequency in ["none", "instant"]:
        return pd.DatetimeIndex([])

    if frequency == "daily":
        return index
    if frequency == "weekly":
        return index.to_series().resample("W-FRI").last().index
    if frequency == "monthly":
        return index.to_series().resample("M").last().index
    if frequency == "quarterly":
        return index.to_series().resample("Q").last().index
    if frequency == "semiannual":
        return index.to_series().resample("2Q").last().index
    if frequency == "annual":
        return index.to_series().resample("A").last().index

    raise ValueError(f"Invalid rebalance frequency: {frequency}")


# ------------------------------------------------------------
# Normalized value calculation
# ------------------------------------------------------------
def calculate_normalized_values(price_df, tickers, starting_val, start_date):
    """
    Per-ticker normalized path from starting_val. If a ticker has no price on start_date
    (e.g. not listed yet), use the first valid price on or after start_date as the base.
    """
    norm_df = pd.DataFrame(index=price_df.index)
    for t in tickers:
        s = price_df[t]
        base_price = s.loc[start_date] if start_date in s.index else np.nan
        if pd.isna(base_price):
            after = s.loc[s.index >= start_date].dropna()
            if after.empty:
                norm_df[f"{t}_norm"] = np.nan
                continue
            base_price = float(after.iloc[0])
            first_valid_idx = after.index[0]
        else:
            first_valid_idx = start_date
        norm = (s / base_price) * starting_val
        norm.loc[price_df.index < first_valid_idx] = np.nan
        norm_df[f"{t}_norm"] = norm
    return norm_df


# ------------------------------------------------------------
# Drawdown from ATH
# ------------------------------------------------------------
def compute_drawdown_from_ath(series):
    """
    Calculate drawdown from all-time high.
    Uses cummax() which calculates ATH from the beginning of the series.
    """
    ath = series.cummax()
    dd = (ath - series) / ath
    return dd, ath


def compute_rolling_ath_and_dd(series: pd.Series, n_calendar_years: int):
    """
    Reference high = max(close) over the trailing N calendar years (inclusive of t).
    Until the calendar span from the first bar to t reaches N years, use standard
    ATH (cummax from inception) for that date.

    Returns:
        (ref_high_series, dd_series), same index as sorted non-NaN input.
    """
    s = series.sort_index().dropna()
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if n_calendar_years <= 0:
        ath = s.cummax()
        dd = (ath - s) / ath
        return ath, dd

    idx = s.index
    values = s.to_numpy(dtype=float, copy=False)
    n = len(s)
    ref = np.empty(n, dtype=float)
    cummax_vals = np.maximum.accumulate(values)

    first_ts = idx[0]
    need_until = first_ts + pd.DateOffset(years=n_calendar_years)

    for i in range(n):
        ts = idx[i]
        if ts < need_until:
            ref[i] = cummax_vals[i]
            continue
        win_start = ts - pd.DateOffset(years=n_calendar_years)
        j = int(idx.searchsorted(win_start, side="left"))
        j = min(max(j, 0), i)
        ref[i] = float(values[j : i + 1].max())

    ath_series = pd.Series(ref, index=idx, dtype=float)
    dd_series = (ath_series - s) / ath_series
    return ath_series, dd_series


def build_regime_signal_drawdown(
    dd_raw: pd.Series,
    exec_index: pd.DatetimeIndex,
    full_index: pd.DatetimeIndex,
    historical_dd_full: pd.Series | None = None,
) -> pd.Series:
    """
    Drawdown used for regime detection on execution date D: prior trading day's close
    on full_index (already encoded in dd_raw vs ATH as of that day). First bar in
    exec_index with no prior row on full_index uses historical_dd_full (last bar
    strictly before D) if provided, else same-day dd on D. Portfolio trades still
    size at row_prices on D (daily close as proxy for next-session execution).
    """
    s = (
        dd_raw.reindex(full_index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    out = pd.Series(index=exec_index, dtype=float)
    for d in exec_index:
        prev = full_index[full_index < d]
        if len(prev):
            out.loc[d] = float(s.loc[prev[-1]])
        elif historical_dd_full is not None:
            prev_h = historical_dd_full.index[historical_dd_full.index < d]
            if len(prev_h):
                out.loc[d] = float(historical_dd_full.loc[prev_h[-1]])
            else:
                pv = s.loc[d] if d in s.index else np.nan
                out.loc[d] = float(pv) if pd.notna(pv) else 0.0
        else:
            pv = s.loc[d] if d in s.index else np.nan
            out.loc[d] = float(pv) if pd.notna(pv) else 0.0
    return out


def compute_drawdown_from_historical_ath(series, historical_ath_value=None):
    """
    Calculate drawdown using a historical ATH value.
    If historical_ath_value is provided, use it; otherwise use cummax().
    
    Args:
        series: Price series for the portfolio period
        historical_ath_value: Optional ATH value from full historical data
    
    Returns:
        Tuple of (drawdown_series, ath_series)
    """
    if historical_ath_value is not None:
        # Use the historical ATH value
        # ATH series is the max of historical ATH and current series cummax
        ath = series.cummax().clip(lower=historical_ath_value)
        # But we want to use the historical ATH as the baseline
        # So if current price is below historical ATH, use historical ATH
        ath = pd.Series(index=series.index, data=historical_ath_value)
        # Update ATH if current series exceeds historical ATH
        current_max = series.cummax()
        ath = pd.concat([ath, current_max], axis=1).max(axis=1)
    else:
        # Fallback to standard calculation
        ath = series.cummax()
    
    dd = (ath - series) / ath
    return dd, ath


# ------------------------------------------------------------
# Initial allocation
# ------------------------------------------------------------
def get_initial_allocation(start_date, price_df, first_dd, config, regime_detector, rebalance_fn):
    market_regime = regime_detector(first_dd, config)
    portfolio_regime = market_regime
    alloc = get_allocation_for_regime(portfolio_regime, config)
    alloc_t = tradable_allocation(alloc, price_df.loc[start_date], config)
    if not alloc_t:
        return market_regime, portfolio_regime, None
    shares = rebalance_fn(config["starting_balance"], alloc_t, price_df.loc[start_date])
    return market_regime, portfolio_regime, shares


# ------------------------------------------------------------
# Rebalancing Strategy Functions
# ------------------------------------------------------------
def apply_asymmetric_rules_down_only(prev_regime, market_regime):
    """
    Regime Shift Down Only: Rebalance immediately when market goes DOWN,
    but only move back UP when market fully recovers to R1.
    """
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    # If market goes DOWN → follow it immediately
    if mkt_n > prev_n:
        return regime_label(mkt_n)

    # Only move back UP if market is fully safe (R1)
    if mkt_n == 1:
        return "R1"

    # Otherwise hold previous regime
    return prev_regime


def apply_asymmetric_rules_up_only(prev_regime, market_regime, bottom_regime_num: int = 3):
    """
    Regime Shift Up Only: Rebalance immediately when market goes UP,
    but hold position when market goes DOWN (except when reaching bottom).
    When in the deepest regime (e.g. R4), rebalance on every way up.
    """
    prev_n = regime_number(prev_regime)
    mkt_n = regime_number(market_regime)

    # If market goes UP → follow it immediately
    if mkt_n < prev_n:
        return regime_label(mkt_n)

    if prev_n == bottom_regime_num and mkt_n < bottom_regime_num:
        return regime_label(mkt_n)

    if mkt_n == bottom_regime_num:
        return regime_label(bottom_regime_num)

    # Otherwise hold previous regime (don't follow down)
    return prev_regime


def apply_always_rebalance(prev_regime, market_regime):
    """
    Always: Rebalance whenever market regime changes, regardless of direction.
    """
    return market_regime


def regime_trajectory_label(prev_market_regime, market_regime):
    """
    Compare today's market regime to the previous trading day's market regime.
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
    """
    direction: 'downward' | 'upward'
    Returns 'match' (adopt market regime allocation) or 'hold' (keep prior portfolio regime).
    """
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
    if cn > pn:
        mode = _regime_rebalance_mode(target_params, "downward")
    else:
        mode = _regime_rebalance_mode(target_params, "upward")
    if mode == "match":
        return market_regime
    return portfolio_regime


def apply_rebalancing_strategy(prev_regime, market_regime, strategy, config=None):
    """
    Apply the specified rebalancing strategy.
    
    Args:
        prev_regime: Current portfolio regime
        market_regime: Current market regime
        strategy: "down_only", "up_only", or "always"
        config: Optional config dict (used for deepest regime index in up_only)
    
    Returns:
        New portfolio regime based on strategy
    """
    bottom_n = bottom_regime_number(config) if config is not None else 3
    if strategy == "per_regime":
        # Caller should use apply_per_regime_direction_strategy with prev_market_regime;
        # this branch is a fallback if mis-invoked.
        return apply_always_rebalance(prev_regime, market_regime)
    if strategy == "down_only":
        return apply_asymmetric_rules_down_only(prev_regime, market_regime)
    elif strategy == "up_only":
        return apply_asymmetric_rules_up_only(prev_regime, market_regime, bottom_n)
    elif strategy == "always":
        return apply_always_rebalance(prev_regime, market_regime)
    else:
        # Default to down_only for backward compatibility
        return apply_asymmetric_rules_down_only(prev_regime, market_regime)


# Backward compatibility alias
apply_asymmetric_rules = apply_asymmetric_rules_down_only


# ------------------------------------------------------------
# Recording helpers
# ------------------------------------------------------------
def record_missing_row(equity_rows, date, tickers, row_prices, row_norm,
                       shares, portfolio_value, prev_regime, starting_val, dd_cols,
                       regime_trajectory="", prev_market_regime=None,
                       config=None, signal_override_mode="none"):

    # Handle None shares (safety check)
    if shares is None:
        shares = {}

    if config is not None:
        oa, ol, oalloc = describe_signal_override_row(prev_regime, signal_override_mode, config)
    else:
        oa, ol, oalloc = "none", "", ""
    rec = {
        "Date": date,
        "Value": portfolio_value,
        "Market_Regime": None,
        "Portfolio_Regime": prev_regime,
        "Regime_Trajectory": regime_trajectory,
        "Prev_Market_Regime": prev_market_regime,
        "Rebalanced": "",
        "Pct_Growth": portfolio_value / starting_val - 1,
        "Signal_override_active": oa,
        "Signal_override_label": ol,
        "Signal_override_allocation": oalloc,
        **dd_cols
    }

    for t in tickers:
        # Handle both dict and Series/DataFrame access
        if isinstance(row_prices, dict):
            price_val = row_prices.get(t, 0)
        else:
            price_val = row_prices.get(t, 0) if hasattr(row_prices, 'get') else (row_prices[t] if t in row_prices.index else 0)
        
        if isinstance(row_norm, dict):
            norm_val = row_norm.get(f"{t}_norm", 0)
        else:
            norm_val = row_norm.get(f"{t}_norm", 0) if hasattr(row_norm, 'get') else (row_norm[f"{t}_norm"] if f"{t}_norm" in row_norm.index else 0)
        
        shares_val = shares.get(t, 0) if isinstance(shares, dict) else 0
        
        rec[f"{t}_price"] = price_val
        rec[f"{t}_norm"] = norm_val
        rec[f"{t}_shares"] = shares_val
        rec[f"{t}_value"] = shares_val * (price_val if price_val else 0)

    equity_rows.append(rec)


def update_portfolio_value(shares, row_prices, prev_value, quarter_returns):
    # Handle None shares
    if shares is None:
        shares = {}
    
    # Calculate portfolio value
    new_value = 0.0
    for t in shares:
        if t in row_prices:
            price = row_prices[t] if isinstance(row_prices, dict) else (row_prices.loc[t] if t in row_prices.index else 0)
            new_value += shares[t] * price
    
    daily_ret = (new_value / prev_value - 1) if prev_value != 0 else 0
    quarter_returns.append(daily_ret)
    return new_value, new_value


def record_daily_row(equity_rows, date, tickers, row_prices, row_norm,
                     shares, portfolio_value, market_regime,
                     portfolio_regime, starting_val, rebalanced_flag,
                     dd_cols, cash_balance=0.0, regime_trajectory="",
                     prev_market_regime=None, config=None,
                     signal_override_mode="none"):
    from utils import log
    
    # Handle None shares
    if shares is None:
        log(f"⚠️ WARNING: shares is None in record_daily_row on {date}. Initializing as empty dict.")
        shares = {}

    # Include cash in total value
    total_value = portfolio_value + cash_balance
    
    if config is not None:
        oa, ol, oalloc = describe_signal_override_row(portfolio_regime, signal_override_mode, config)
    else:
        oa, ol, oalloc = "none", "", ""

    rec = {
        "Date": date,
        "Value": total_value,
        "Cash": cash_balance,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "Regime_Trajectory": regime_trajectory,
        "Prev_Market_Regime": prev_market_regime,
        "Rebalanced": rebalanced_flag,
        "Pct_Growth": total_value / starting_val - 1,
        "Signal_override_active": oa,
        "Signal_override_label": ol,
        "Signal_override_allocation": oalloc,
        **dd_cols
    }

    for t in tickers:
        rec[f"{t}_price"] = row_prices[t]
        rec[f"{t}_norm"] = row_norm[f"{t}_norm"]
        rec[f"{t}_shares"] = float(shares.get(t, 0) if isinstance(shares, dict) else 0)
        rec[f"{t}_value"] = float((shares.get(t, 0) if isinstance(shares, dict) else 0) * row_prices[t])

    equity_rows.append(rec)


def record_quarterly_row(quarterly_rows, date, tickers, row_prices, shares,
                         portfolio_value, market_regime, portfolio_regime,
                         last_rebalance_value, quarter_returns, dd_cols):
    from utils import log
    
    # Handle None shares
    if shares is None:
        log(f"⚠️ WARNING: shares is None in record_quarterly_row on {date}. Initializing as empty dict.")
        shares = {}

    qoq_ret = (portfolio_value / last_rebalance_value) - 1
    qoq_vol = pd.Series(quarter_returns).std() if len(quarter_returns) else 0

    row = {
        "Date": date,
        "Portfolio_Value": portfolio_value,
        "Market_Regime": market_regime,
        "Portfolio_Regime": portfolio_regime,
        "QoQ_Return": qoq_ret,
        "QoQ_Volatility": qoq_vol,
        **dd_cols
    }

    for t in tickers:
        row[f"{t}_shares"] = float(shares.get(t, 0))
        row[f"{t}_value"] = float(shares.get(t, 0) * row_prices[t])

    quarterly_rows.append(row)


# ------------------------------------------------------------
# MAIN BACKTEST ENGINE (Corrected)
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
    import yfinance as yf
    
    # Download from ticker inception (use 1980 as safe early date - yfinance returns from actual inception)
    # drawdown_price_series = Yahoo history merged with portfolio panel: on overlapping dates the portfolio
    # download wins so regime ATH matches {ticker}_price in the equity table (no ATH < close artifacts).
    drawdown_price_series = None
    historical_dd_full = None
    historical_ath_full = None
    pre_portfolio_start_date = None
    
    try:
        historical_data = yf.download(
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

    if "SPY" in price_df.columns:
        spy_panel_sig = price_df["SPY"].astype(float)
    else:
        spy_panel_sig = pd.Series(np.nan, index=price_df.index, dtype=float)
    spy_signal_merged = build_combined_spy_series(price_df.index, spy_panel_sig)
    vix_signal_merged = build_combined_vix_series(price_df.index, None)
    signal_total_series = build_signal_total_series(
        price_df.index, spy_signal_merged, vix_signal_merged
    )

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

        raw_sig = signal_total_series.loc[date] if date in signal_total_series.index else np.nan
        s_curr = float(raw_sig) if pd.notna(raw_sig) else np.nan

        # -------------------------
        # Dividend handling
        # -------------------------
        if dividend_reinvestment and dividend_data is not None and not dividend_data.empty:
            # Ensure shares is a dict
            if shares is None:
                log(f"⚠️ WARNING: shares is None during dividend handling on {date}. Initializing as empty dict.")
                shares = {}
            
            daily_dividends = 0.0
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
                                    # Portfolio value will be recalculated below via update_portfolio_value()
                                    # which includes the new shares
                            
                            # Log dividend for debugging
                            log(f"Dividend: {ticker} paid ${dividend_amount:.2f} ({dividend_per_share:.4f} per share, {num_shares:.4f} shares)")
            
            # After processing all dividends, update portfolio value if dividends were reinvested into shares
            # (cash dividends are already in cash_balance and will be included in total value)
            if daily_dividends > 0 and dividend_target != "cash":
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
