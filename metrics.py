"""
Pure performance-metric calculations.

Kept free of streamlit / plotly so test suites and CLI runners can import
without UI dependencies. Dashboard re-exports `calculate_metrics` from here.
"""

import numpy as np
import pandas as pd

from utils import max_drawdown_from_equity_curve


def calculate_metrics(equity_df, config=None):
    """Calculate key performance metrics from a backtest equity_df."""
    start_val = equity_df["Value"].iloc[0]
    end_val = equity_df["Value"].iloc[-1]
    start_date = pd.to_datetime(equity_df["Date"].iloc[0])
    end_date = pd.to_datetime(equity_df["Date"].iloc[-1])

    years = (end_date - start_date).days / 365.25
    cagr = float((end_val / start_val) ** (1 / years) - 1) if years > 0 else 0.0
    total_return = float((end_val / start_val) - 1)
    max_drawdown = float(max_drawdown_from_equity_curve(equity_df["Value"]))

    returns = equity_df["Value"].pct_change().dropna()
    vol_raw = returns.std() * np.sqrt(252)
    volatility = float(vol_raw) if pd.notna(vol_raw) else 0.0

    sharpe = float(cagr / volatility) if volatility > 0 else 0.0

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_variance = (downside_returns ** 2).mean()
        downside_deviation = float(np.sqrt(downside_variance) * np.sqrt(252))
        sortino = float(cagr / downside_deviation) if downside_deviation > 0 else 0.0
    else:
        sortino = float('inf') if cagr > 0 else 0.0

    beta, beta_benchmark = _calculate_beta(equity_df, returns, config)

    return {
        "start_value": start_val,
        "end_value": end_val,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "beta": beta,
        "beta_benchmark": beta_benchmark,
        "years": years,
        "start_date": start_date,
        "end_date": end_date,
    }


def _calculate_beta(equity_df, portfolio_returns, config):
    """Cov(portfolio, benchmark) / Var(benchmark). Tries SPY → QQQ → other normalized cols."""
    benchmark_candidates = ["SPY_norm", "QQQ_norm"]
    if config and "tickers" in config:
        for ticker in config["tickers"]:
            if ticker not in ["SPY", "QQQ"]:
                benchmark_candidates.append(f"{ticker}_norm")

    for bench_col in benchmark_candidates:
        if bench_col not in equity_df.columns:
            continue
        benchmark_returns = equity_df[bench_col].dropna().pct_change().dropna()
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) <= 30:
            continue

        port_aligned = portfolio_returns.loc[common_index]
        bench_aligned = benchmark_returns.loc[common_index]
        valid_mask = port_aligned.notna() & bench_aligned.notna()
        port_clean = port_aligned[valid_mask]
        bench_clean = bench_aligned[valid_mask]
        if len(port_clean) <= 30:
            continue

        benchmark_variance = np.var(bench_clean)
        if benchmark_variance > 0:
            covariance = np.cov(port_clean, bench_clean)[0, 1]
            return covariance / benchmark_variance, bench_col.replace("_norm", "")

    return 0.0, "N/A"
