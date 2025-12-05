import pandas as pd
import xlsxwriter
import os
import math


# ------------------------------------------------------------
# Unique output filename
# ------------------------------------------------------------
def get_unique_filename(base="backtest_results", ext="xlsx"):
    i = 1
    while True:
        filename = f"{base}_{i}.{ext}"
        if not os.path.exists(filename):
            return filename
        i += 1


# ------------------------------------------------------------
# Safe write helper
# ------------------------------------------------------------
def safe_write(ws, r, c, val, fmt, fmt_text):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        ws.write(r, c, "", fmt_text)
    else:
        ws.write(r, c, val, fmt)


# ------------------------------------------------------------
# Formats
# ------------------------------------------------------------
def create_formats(workbook):
    return {
        "header": workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#003366',
            'border': 1
        }),
        "zebra_light": workbook.add_format({'bg_color': '#F2F2F2'}),
        "zebra_white": workbook.add_format({'bg_color': 'white'}),
        "text": workbook.add_format({'align': 'left'}),
        "dollar": workbook.add_format({
            'num_format': '$#,##0.00',
            'align': 'right'
        }),
        "percent": workbook.add_format({
            'num_format': '0.00%',
            'align': 'right'
        }),
        "date": workbook.add_format({
            'num_format': 'yyyy-mm-dd',
            'align': 'left'
        }),
        "number": workbook.add_format({
            'num_format': '#,##0.00',
            'align': 'right'
        }),
        "final_value": workbook.add_format({
            'num_format': '$#,##0.00',
            'bg_color': '#D9EAD3',
            'bold': True,
            'align': 'right'
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

    price_cols = [
        c for c in equity_df.columns
        if c.endswith("_price") or c == "Value"
    ]
    share_cols = [c for c in equity_df.columns if c.endswith("_shares")]
    value_cols = [c for c in equity_df.columns if c.endswith("_value")]

    percent_cols = [
        c for c in equity_df.columns
        if ("Pct" in c or "Return" in c or "Vol" in c
            or "Drawdown" in c or "Market_DD" in c)
    ]

    # Headers
    for col_num, col_name in enumerate(equity_df.columns):
        ws.write(0, col_num, col_name, header_fmt)

    # Data
    for row in range(1, len(equity_df) + 1):
        current = equity_df.iloc[row - 1]

        for col_num, col_name in enumerate(equity_df.columns):
            val = current[col_name]

            # Determine format
            if col_name == "Date":
                fmt = formats["date"]
            elif col_name in percent_cols:
                fmt = formats["percent"]
            elif col_name in price_cols or col_name in value_cols:
                fmt = formats["dollar"]
            elif col_name in share_cols:
                fmt = formats["number"]
            else:
                fmt = fmt_text

            safe_write(ws, row, col_num, val, fmt, fmt_text)

    # Highlight final value
    val_col = equity_df.columns.get_loc("Value")
    final_val = equity_df.iloc[-1]["Value"]
    safe_write(ws, len(equity_df), val_col, final_val, formats["final_value"], fmt_text)

    ws.set_column(0, len(equity_df.columns), 14)


# ------------------------------------------------------------
# CHART SHEET (EXACTLY AS ORIGINAL)
# ------------------------------------------------------------
def write_chart_sheet(workbook, equity_df):

    chart_sheet = workbook.add_worksheet("Chart")
    chart = workbook.add_chart({"type": "line"})

    # Portfolio value line
    val_idx = equity_df.columns.get_loc("Value")

    chart.add_series({
        "name": "Portfolio Value",
        "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
        "values": ["Equity Curve", 1, val_idx, len(equity_df), val_idx],
        "line": {"color": "#003366", "width": 2},
    })

    # Benchmark normalized lines (unchanged)
    for col in [c for c in equity_df.columns if c.endswith("_norm")]:
        idx = equity_df.columns.get_loc(col)
        chart.add_series({
            "name": col.replace("_norm", "") + " (Normalized)",
            "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
            "values": ["Equity Curve", 1, idx, len(equity_df), idx],
            "line": {"dash_type": "dash"},
        })

    chart.set_title({"name": "Portfolio vs Benchmarks"})
    chart.set_y_axis({"name": "Value ($)"})
    chart.set_x_axis({"name": "Date"})
    chart.set_legend({"position": "bottom"})

    chart_sheet.insert_chart("B2", chart)


# ------------------------------------------------------------
# SUMMARY SHEET
# ------------------------------------------------------------
def write_quarterly_sheet(writer, quarterly_df, formats, config):

    if quarterly_df is None or quarterly_df.empty:
        return

    freq = config.get("rebalance_frequency", "quarterly")

    name_map = {
        "daily": "Daily Summary",
        "weekly": "Weekly Summary",
        "monthly": "Monthly Summary",
        "quarterly": "Quarterly Summary",
        "semiannual": "Semiannual Summary",
        "annual": "Annual Summary",
    }
    sheet_name = name_map.get(freq, "Summary")

    quarterly_df.to_excel(writer, sheet_name=sheet_name, index=False)

    ws = writer.sheets[sheet_name]
    ws.freeze_panes(1, 1)

    header_fmt = formats["header"]
    fmt_text = formats["text"]

    percent_cols = [
        c for c in quarterly_df.columns
        if ("Return" in c or "Vol" in c or "Drawdown" in c or "Market_DD" in c)
    ]

    for c, name in enumerate(quarterly_df.columns):
        ws.write(0, c, name, header_fmt)

    for r in range(1, len(quarterly_df) + 1):
        current = quarterly_df.iloc[r - 1]

        for c, name in enumerate(quarterly_df.columns):
            val = current[name]

            if name == "Date":
                fmt = formats["date"]
            elif name in percent_cols:
                fmt = formats["percent"]
            elif "Value" in name:
                fmt = formats["dollar"]
            else:
                fmt = fmt_text

            safe_write(ws, r, c, val, fmt, fmt_text)

    ws.set_column(0, len(quarterly_df.columns), 18)


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

    for c, name in enumerate(df.columns):
        ws.write(0, c, name, header_fmt)

    for r in range(1, len(df) + 1):
        row = df.iloc[r - 1]
        for c, name in enumerate(df.columns):
            val = row[name]
            fmt = formats["text"] if name == "Regime" else formats["percent"]
            safe_write(ws, r, c, val, fmt, formats["text"])

    ws.set_column(0, len(df.columns), 14)


# ------------------------------------------------------------
# FINAL RESULTS SHEET
# ------------------------------------------------------------
def write_results_sheet(writer, equity_df, formats):

    last = equity_df.tail(1).copy()

    start_val = equity_df["Value"].iloc[0]
    end_val = equity_df["Value"].iloc[-1]

    years = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0

    last.insert(len(last.columns), "Avg_YoY_Growth", cagr)

    sheet_name = "Results"
    last.to_excel(writer, sheet_name=sheet_name, index=False)

    ws = writer.sheets[sheet_name]

    header_fmt = formats["header"]
    fmt_text = formats["text"]

    for c, name in enumerate(last.columns):
        ws.write(0, c, name, header_fmt)

    for c, name in enumerate(last.columns):
        val = last.iloc[0][name]

        if name == "Date":
            fmt = formats["date"]
        elif ("Value" in name or "_price" in name or "_value" in name):
            fmt = formats["dollar"]
        elif ("Pct" in name or "Growth" in name or "YoY" in name
              or "Drawdown" in name or "Market_DD" in name):
            fmt = formats["percent"]
        elif "_norm" in name:
            fmt = formats["dollar"]
        elif "_shares" in name:
            fmt = formats["number"]
        else:
            fmt = fmt_text

        safe_write(ws, 1, c, val, fmt, fmt_text)

    val_idx = last.columns.get_loc("Value")
    safe_write(ws, 1, val_idx, last.iloc[0]["Value"], formats["final_value"], fmt_text)

    ws.set_column(0, len(last.columns), 16)


# ------------------------------------------------------------
# MAIN EXPORT FUNCTION
# ------------------------------------------------------------
def export_to_excel(equity_df, quarterly_df, config, mode="normal"):

    filename = (
        get_unique_filename(base="worst_case_results")
        if mode == "worst_case"
        else get_unique_filename(base="backtest_results")
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
    write_chart_sheet(workbook, equity_df)   # ← Chart restored EXACTLY as before
    write_quarterly_sheet(writer, quarterly_df, formats, config)
    write_parameters_sheet(writer, config)
    write_regimes_sheet(writer, config, formats)
    write_results_sheet(writer, equity_df, formats)

    writer.close()
    print(f"Excel export complete: {filename}\n")
