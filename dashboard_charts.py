"""
Plotly chart builders for the dashboard.

Each `create_*` function takes a piece of `equity_df` (and an optional config
or auxiliary DataFrame) and returns a plotly Figure. None is returned when the
required columns are missing — callers must handle that.

Pure builders: no streamlit calls, no I/O. Tested via dashboard rendering.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import VIX_YAHOO_SYMBOL


def create_equity_curve_chart(equity_df, config):
    """Interactive equity curve with normalized benchmark traces."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Value"],
        name="Portfolio Value",
        line=dict(color="#003366", width=2),
        hovertemplate="<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
    ))

    norm_cols = [c for c in equity_df.columns if c.endswith("_norm")]
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(norm_cols):
        ticker = col.replace("_norm", "")
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=equity_df[col],
            name=f"{ticker} (Normalized)",
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Portfolio Value vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def create_performance_summary_chart(equity_df):
    """
    Same-day context as the Performance Summary table: portfolio value + VIX close.
    Returns None if VIX data is unavailable.
    """
    if equity_df is None or equity_df.empty or "Date" not in equity_df.columns:
        return None
    if "VIX" not in equity_df.columns:
        return None

    dates = pd.to_datetime(equity_df["Date"])
    vix_raw = pd.to_numeric(equity_df["VIX"], errors="coerce")
    if vix_raw.notna().sum() == 0:
        return None

    # Short forward-fill only for plotting (e.g. rare holiday / index misaligns); table stays raw
    vix_plot = vix_raw.ffill(limit=5)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.52, 0.48],
        subplot_titles=(
            "Portfolio value ($)",
            f"VIX — CBOE close (Yahoo {VIX_YAHOO_SYMBOL})",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_df["Value"],
            name="Portfolio Value",
            line=dict(color="#003366", width=2),
            hovertemplate="<b>Portfolio</b><br>%{x}<br>$%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=vix_plot,
            name="VIX",
            line=dict(color="#b91c1c", width=1.5),
            connectgaps=False,
            hovertemplate="<b>VIX</b><br>%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="VIX level", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        title="Performance summary — portfolio vs VIX",
        height=560,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )
    return fig


def create_drawdown_chart(equity_df, drawdown_ticker):
    """Two-row drawdown view: portfolio + drawdown_ticker."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Portfolio Drawdown", f"{drawdown_ticker} Drawdown"),
        row_heights=[0.5, 0.5],
    )

    if "Portfolio_DD" in equity_df.columns:
        portfolio_dd = equity_df["Portfolio_DD"] * 100
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=portfolio_dd,
            name="Portfolio DD",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red", width=1),
            hovertemplate="Portfolio DD: %{y:.2f}%<extra></extra>",
        ), row=1, col=1)

    dd_col = f"{drawdown_ticker}_DD_raw"
    if dd_col in equity_df.columns:
        market_dd = equity_df[dd_col] * 100
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=market_dd,
            name=f"{drawdown_ticker} DD",
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.3)",
            line=dict(color="orange", width=1),
            hovertemplate=f"{drawdown_ticker} DD: %{{y:.2f}}%<extra></extra>",
        ), row=2, col=1)

    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        height=600,
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
    )
    return fig


_REGIME_LINE_COLORS = {
    "R1": "#00CC00",
    "R2": "#FF9900",
    "R3": "#CC0000",
    "R4": "#6600CC",
}


def create_regime_timeline(equity_df):
    """Equity curve segmented by Portfolio_Regime, with rebalance markers."""
    if "Portfolio_Regime" not in equity_df.columns:
        return None

    equity_df = equity_df.copy()
    equity_df["Regime_Color"] = equity_df["Portfolio_Regime"].map(_REGIME_LINE_COLORS)

    fig = go.Figure()

    for regime in equity_df["Portfolio_Regime"].unique():
        if pd.isna(regime):
            continue
        regime_data = equity_df[equity_df["Portfolio_Regime"] == regime]
        fig.add_trace(go.Scatter(
            x=regime_data["Date"],
            y=regime_data["Value"],
            name=f"Regime {regime}",
            line=dict(color=_REGIME_LINE_COLORS.get(regime, "#666"), width=2),
            mode="lines",
            hovertemplate=f"<b>{regime}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>",
        ))

    rebalance_dates = equity_df[equity_df["Rebalanced"] == "Rebalanced"]
    if not rebalance_dates.empty:
        fig.add_trace(go.Scatter(
            x=rebalance_dates["Date"],
            y=rebalance_dates["Value"],
            mode="markers",
            name="Rebalance",
            marker=dict(symbol="diamond", size=10, color="black", line=dict(width=1, color="white")),
            hovertemplate="<b>Rebalance</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Portfolio Value with Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


_ALLOCATION_TICKER_COLORS = {"QQQ": "#1f77b4", "TQQQ": "#ff7f0e", "XLU": "#2ca02c"}


def create_allocation_chart(equity_df, config):
    """Stacked-area chart of per-ticker allocation % over time."""
    tickers = config["allocation_tickers"]

    allocation_data = []
    for ticker in tickers:
        value_col = f"{ticker}_value"
        if value_col in equity_df.columns:
            allocation_data.append({
                "Date": equity_df["Date"],
                "Ticker": ticker,
                "Allocation": equity_df[value_col] / equity_df["Value"] * 100,
            })

    if not allocation_data:
        return None

    df_alloc = pd.concat([pd.DataFrame(d) for d in allocation_data])

    fig = go.Figure()
    for ticker in tickers:
        ticker_data = df_alloc[df_alloc["Ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_data["Date"],
            y=ticker_data["Allocation"],
            name=ticker,
            stackgroup="one",
            fillcolor=_ALLOCATION_TICKER_COLORS.get(ticker, "#999"),
            line=dict(width=0),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Allocation: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Portfolio Allocation Over Time",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        yaxis=dict(range=[0, 100]),
    )
    return fig


_DIVIDEND_TARGET_COLORS = {
    "cash": "#FF6B6B",
    "TQQQ": "#FFA500",
    "QQQ": "#4ECDC4",
    "XLU": "#95E1D3",
    "SPY": "#F38181",
}


def create_dividend_chart(dividend_df, equity_df):
    """Three-row dividend chart: amount, yield, portfolio impact."""
    if dividend_df.empty or len(dividend_df) == 0:
        return None

    dividend_df = dividend_df.copy()
    dividend_df["Date"] = pd.to_datetime(dividend_df["Date"])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Dividend Amount ($)", "Dividend Yield (%)", "Portfolio Impact (%)"),
        row_heights=[0.4, 0.3, 0.3],
    )

    targets = dividend_df["Reinvestment_Target"].unique()
    for target in targets:
        target_data = dividend_df[dividend_df["Reinvestment_Target"] == target]
        color = _DIVIDEND_TARGET_COLORS.get(target, "#999999")

        fig.add_trace(go.Bar(
            x=target_data["Date"],
            y=target_data["Dividend_Amount"],
            name=f"Reinvested in {target}",
            marker_color=color,
            hovertemplate=f"<b>{target}</b><br>Date: %{{x}}<br>Amount: $%{{y:,.2f}}<extra></extra>",
            legendgroup=target,
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=target_data["Date"],
            y=target_data["Dividend_Yield"],
            mode="markers",
            name=f"Yield ({target})",
            marker=dict(color=color, size=8, symbol="circle"),
            hovertemplate=f"<b>{target}</b><br>Date: %{{x}}<br>Yield: %{{y:.2f}}%<extra></extra>",
            legendgroup=target,
            showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=target_data["Date"],
            y=target_data["Portfolio_Pct"],
            mode="markers",
            name=f"Portfolio % ({target})",
            marker=dict(color=color, size=8, symbol="diamond"),
            hovertemplate=f"<b>{target}</b><br>Date: %{{x}}<br>Portfolio Impact: %{{y:.3f}}%<extra></extra>",
            legendgroup=target,
            showlegend=False,
        ), row=3, col=1)

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio %", row=3, col=1)

    fig.update_layout(
        title="Dividend Distribution Over Time",
        height=800,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
