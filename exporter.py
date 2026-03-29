import pandas as pd
import xlsxwriter
import os
import math

from utils import max_drawdown_from_equity_curve


# ------------------------------------------------------------
# Unique filename generator
# ------------------------------------------------------------
def get_unique_filename(base="backtest_results", ext="xlsx"):
    i = 1
    while True:
        candidate = f"{base}_{i}.{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ------------------------------------------------------------
# Safe cell writer
# ------------------------------------------------------------
def safe_write(ws, r, c, val, fmt, fmt_text):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        ws.write(r, c, "", fmt_text)
    else:
        ws.write(r, c, val, fmt)


# ------------------------------------------------------------
# Create Excel formats
# ------------------------------------------------------------
def create_formats(workbook):
    return {
        "header": workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#003366",
            "border": 1,
        }),
        "zebra_light": workbook.add_format({"bg_color": "#F2F2F2"}),
        "zebra_white": workbook.add_format({"bg_color": "white"}),
        "text": workbook.add_format({"align": "left"}),
        "dollar": workbook.add_format({"num_format": "$#,##0.00", "align": "right"}),
        "percent": workbook.add_format({"num_format": "0.00%", "align": "right"}),
        "date": workbook.add_format({"num_format": "yyyy-mm-dd", "align": "left"}),
        "number": workbook.add_format({"num_format": "#,##0.00", "align": "right"}),
        "final_value": workbook.add_format({
            "num_format": "$#,##0.00",
            "bg_color": "#D9EAD3",
            "bold": True,
            "align": "right",
        }),
    }


# ------------------------------------------------------------
# EQUITY CURVE SHEET
# ------------------------------------------------------------
def write_equity_sheet(writer, equity_df, formats, tickers):

    sheet_name = "Equity Curve"
    equity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    ws = writer.sheets[sheet_name]
    ws.freeze_panes(1, 1)

    header_fmt = formats["header"]
    fmt_text = formats["text"]

    price_cols = [c for c in equity_df.columns if c.endswith("_price") or c == "Value"]
    share_cols = [c for c in equity_df.columns if c.endswith("_shares")]
    value_cols = [c for c in equity_df.columns if c.endswith("_value")]

    percent_cols = [
        c for c in equity_df.columns
        if ("Pct" in c or "Return" in c or "Vol" in c
            or "Drawdown" in c or "DD" in c)
    ]

    # Header
    for col_num, col_name in enumerate(equity_df.columns):
        ws.write(0, col_num, col_name, header_fmt)

    # Data rows
    for r in range(1, len(equity_df) + 1):
        row = equity_df.iloc[r - 1]

        for c, col_name in enumerate(equity_df.columns):
            val = row[col_name]

            if col_name == "Date":
                fmt = formats["date"]
            elif col_name == "VIX":
                fmt = formats["number"]
            elif col_name in percent_cols:
                fmt = formats["percent"]
            elif col_name in price_cols or col_name in value_cols:
                fmt = formats["dollar"]
            elif col_name in share_cols:
                fmt = formats["number"]
            else:
                fmt = fmt_text

            safe_write(ws, r, c, val, fmt, fmt_text)

    # Final portfolio value row
    value_col = equity_df.columns.get_loc("Value")
    final_val = equity_df.iloc[-1]["Value"]
    safe_write(ws, len(equity_df), value_col, final_val, formats["final_value"], fmt_text)

    ws.set_column(0, len(equity_df.columns), 14)


# ------------------------------------------------------------
# CHART SHEET
# (exactly as original, ONLY Value + normalized benchmarks)
# ------------------------------------------------------------
def write_chart_sheet(workbook, equity_df):

    chart_sheet = workbook.add_worksheet("Chart")
    chart = workbook.add_chart({"type": "line"})

    val_idx = equity_df.columns.get_loc("Value")

    chart.add_series({
        "name": "Portfolio Value",
        "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
        "values": ["Equity Curve", 1, val_idx, len(equity_df), val_idx],
        "line": {"color": "#003366", "width": 2},
    })

    # Add only normalized indices
    for col in [c for c in equity_df.columns if c.endswith("_norm")]:
        ci = equity_df.columns.get_loc(col)
        chart.add_series({
            "name": col.replace("_norm", "") + " (Normalized)",
            "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
            "values": ["Equity Curve", 1, ci, len(equity_df), ci],
            "line": {"dash_type": "dash"},
        })

    chart.set_title({"name": "Portfolio vs Benchmarks"})
    chart.set_x_axis({"name": "Date"})
    chart.set_y_axis({"name": "Value ($)"})
    chart.set_legend({"position": "bottom"})

    chart_sheet.insert_chart("B2", chart)


# ------------------------------------------------------------
# QUARTERLY / SUMMARY SHEET
# ------------------------------------------------------------
def write_quarterly_sheet(writer, quarterly_df, formats, config):

    # Skip if no data
    if quarterly_df is None or quarterly_df.empty:
        return

    # Filter to only the rows where a rebalance occurred
    if "Rebalanced" in quarterly_df.columns:
        df = quarterly_df[quarterly_df["Rebalanced"] == "Rebalanced"].copy()
    else:
        df = quarterly_df.copy()

    if df.empty:
        return  # no rebalances, nothing to show

    sheet_name = "Rebalance Summary"
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    ws = writer.sheets[sheet_name]
    ws.freeze_panes(1, 1)

    header_fmt = formats["header"]
    fmt_text = formats["text"]

    # Identify percent columns
    percent_cols = [
        c for c in df.columns
        if ("Return" in c or "Vol" in c or "Drawdown" in c or "DD" in c)
    ]

    # Write header
    for col_idx, col_name in enumerate(df.columns):
        ws.write(0, col_idx, col_name, header_fmt)

    # Write data
    for row_idx in range(1, len(df) + 1):
        row = df.iloc[row_idx - 1]

        for col_idx, col_name in enumerate(df.columns):
            val = row[col_name]

            if col_name == "Date":
                fmt = formats["date"]
            elif col_name in percent_cols:
                fmt = formats["percent"]
            elif col_name.endswith("_shares"):
                fmt = formats["number"]
            elif col_name.endswith("_value") or "Value" in col_name:
                fmt = formats["dollar"]
            else:
                fmt = fmt_text

            safe_write(ws, row_idx, col_idx, val, fmt, fmt_text)

    ws.set_column(0, len(df.columns), 18)


# ------------------------------------------------------------
# PARAMETERS SHEET
# ------------------------------------------------------------
def write_parameters_sheet(writer, config):

    df = pd.DataFrame([
        {"Parameter": k, "Value": v}
        for k, v in config.items()
        if k != "regimes"
    ])

    df.to_excel(writer, sheet_name="Parameters", index=False)

    ws = writer.sheets["Parameters"]
    ws.freeze_panes(1, 1)


# ------------------------------------------------------------
# REGIMES SHEET
# ------------------------------------------------------------
def write_regimes_sheet(writer, config, formats):

    rows = []
    for regime, vals in config["regimes"].items():
        row = {"Regime": regime}
        row.update(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Regimes", index=False)

    ws = writer.sheets["Regimes"]
    ws.freeze_panes(1, 1)

    header_fmt = formats["header"]

    # Header row
    for c, name in enumerate(df.columns):
        ws.write(0, c, name, header_fmt)

    # Data rows
    for r in range(1, len(df) + 1):
        row = df.iloc[r - 1]

        for c, name in enumerate(df.columns):
            val = row[name]

            if name == "Regime" or name in (
                "rebalance_on_downward",
                "rebalance_on_upward",
            ) or isinstance(val, str):
                fmt = formats["text"]
            else:
                fmt = formats["percent"]

            safe_write(ws, r, c, val, fmt, formats["text"])

    ws.set_column(0, len(df.columns), 14)


# ------------------------------------------------------------
# RESULTS SHEET
# ------------------------------------------------------------
def write_results_sheet(writer, equity_df, formats):

    last = equity_df.tail(1).copy()

    start_val = equity_df["Value"].iloc[0]
    end_val = equity_df["Value"].iloc[-1]

    years = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0

    last.insert(len(last.columns), "Avg_YoY_Growth", cagr)
    # Worst peak-to-trough over full equity path (not the same as final-row Portfolio_DD)
    sim_mdd = max_drawdown_from_equity_curve(equity_df["Value"])
    last.insert(len(last.columns), "Max_Drawdown_Simulation", sim_mdd)

    sheet_name = "Results"
    last.to_excel(writer, sheet_name=sheet_name, index=False)

    ws = writer.sheets["Results"]

    header_fmt = formats["header"]
    fmt_text = formats["text"]

    # Header
    for c, name in enumerate(last.columns):
        ws.write(0, c, name, header_fmt)

    # Data row
    for c, name in enumerate(last.columns):
        val = last.iloc[0][name]

        if name == "Date":
            fmt = formats["date"]
        elif ("Value" in name or "_price" in name or "_value" in name):
            fmt = formats["dollar"]
        elif (
            "Pct" in name
            or "Growth" in name
            or "YoY" in name
            or "DD" in name
            or "Drawdown" in name
        ):
            fmt = formats["percent"]
        elif name.endswith("_shares"):
            fmt = formats["number"]
        elif name.endswith("_norm"):
            fmt = formats["dollar"]
        else:
            fmt = fmt_text

        safe_write(ws, 1, c, val, fmt, fmt_text)

    # Highlight final value
    val_idx = last.columns.get_loc("Value")
    safe_write(ws, 1, val_idx, last.iloc[0]["Value"], formats["final_value"], fmt_text)

    ws.set_column(0, len(last.columns), 16)


# ------------------------------------------------------------
# MAIN EXPORT FUNCTION
# ------------------------------------------------------------
def export_to_excel(equity_df, quarterly_df, config, mode="normal"):

    filename = (
        get_unique_filename("worst_case_results")
        if mode == "worst_case"
        else get_unique_filename("backtest_results")
    )

    print(f"Exporting to: {filename}")

    writer = pd.ExcelWriter(
        filename,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}},
    )
    workbook = writer.book

    formats = create_formats(workbook)

    write_equity_sheet(writer, equity_df, formats, config["tickers"])
    write_chart_sheet(workbook, equity_df)
    write_quarterly_sheet(writer, quarterly_df, formats, config)
    write_parameters_sheet(writer, config)
    write_regimes_sheet(writer, config, formats)
    write_results_sheet(writer, equity_df, formats)

    writer.close()
    print(f"Excel export complete: {filename}\n")

