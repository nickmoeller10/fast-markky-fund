
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
# Safe write helper (shared across sheets)
# ------------------------------------------------------------
def safe_write(ws, r, c, val, fmt, fmt_text):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        ws.write(r, c, "", fmt_text)
    else:
        ws.write(r, c, val, fmt)


# ------------------------------------------------------------
# Create all formatting in one place
# ------------------------------------------------------------
def create_formats(workbook):
    return {
        "header": workbook.add_format({
            'bold': True, 'font_color': 'white',
            'bg_color': '#003366', 'border': 1
        }),
        "zebra_light": workbook.add_format({'bg_color': '#F2F2F2'}),
        "zebra_white": workbook.add_format({'bg_color': 'white'}),
        "bold": workbook.add_format({'bold': True}),
        "text": workbook.add_format({'align': 'left'}),
        "dollar": workbook.add_format({'num_format': '$#,##0.00', 'align': 'right'}),
        "percent": workbook.add_format({'num_format': '0.00%', 'align': 'right'}),
        "date": workbook.add_format({'num_format': 'yyyy-mm-dd', 'align': 'left'}),
        "number": workbook.add_format({'num_format': '#,##0.00', 'align': 'right'}),
        "final_value": workbook.add_format({
            'num_format': '$#,##0.00',
            'bg_color': '#D9EAD3',
            'bold': True,
            'align': 'right'
        })
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

    # Determine formatting groups
    norm_cols = [c for c in equity_df.columns if c.endswith("_norm")]
    price_cols = [c for c in equity_df.columns if c.endswith("_price") or c == "Value"]
    share_cols = [c for c in equity_df.columns if c.endswith("_shares")]
    value_cols = [c for c in equity_df.columns if c.endswith("_value")]
    percent_cols = [c for c in equity_df.columns if "Pct" in c or "Return" in c or "Vol" in c]

    # Headers
    for col_num, col_name in enumerate(equity_df.columns):
        ws.write(0, col_num, col_name, header_fmt)

    # Data rows
    for row in range(1, len(equity_df) + 1):
        row_fmt = formats["zebra_light"] if row % 2 == 0 else formats["zebra_white"]
        current = equity_df.iloc[row - 1]

        for col_num, col_name in enumerate(equity_df.columns):
            val = current[col_name]

            # Format selector
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
    value_col = equity_df.columns.get_loc("Value")
    final_val = equity_df.iloc[-1]["Value"]
    safe_write(ws, len(equity_df), value_col, final_val, formats["final_value"], fmt_text)

    ws.set_column(0, len(equity_df.columns), 14)


# ------------------------------------------------------------
# CHART SHEET
# ------------------------------------------------------------
def write_chart_sheet(workbook, equity_df):

    chart_sheet = workbook.add_worksheet("Chart")
    chart = workbook.add_chart({"type": "line"})

    # Portfolio value line
    val_idx = equity_df.columns.get_loc("Value")
    chart.add_series({
        "name": "Portfolio Value",
        "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
        "values":     ["Equity Curve", 1, val_idx, len(equity_df), val_idx],
        "line": {"color": "#003366", "width": 2},
    })

    # Normalized lines
    for col in [c for c in equity_df.columns if c.endswith("_norm")]:
        idx = equity_df.columns.get_loc(col)
        chart.add_series({
            "name": col.replace("_norm", "") + " (Normalized)",
            "categories": ["Equity Curve", 1, 0, len(equity_df), 0],
            "values":     ["Equity Curve", 1, idx, len(equity_df), idx],
            "line": {"dash_type": "dash"},
        })

    chart.set_title({"name": "Portfolio vs Benchmarks"})
    chart.set_y_axis({"name": "Value ($)"})
    chart.set_x_axis({"name": "Date"})
    chart.set_legend({"position": "bottom"})

    chart_sheet.insert_chart("B2", chart)


# ------------------------------------------------------------
# QUARTERLY SUMMARY SHEET
# ------------------------------------------------------------
def write_quarterly_sheet(writer, quarterly_df, formats):
    if quarterly_df is None or quarterly_df.empty:
        return

    sheet_name = "Quarterly Summary"
    quarterly_df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    ws.freeze_panes(1, 1)

    fmt_text = formats["text"]

    # Headers
    for col_num, col_name in enumerate(quarterly_df.columns):
        ws.write(0, col_num, col_name, formats["header"])

    # Rows
    for row in range(1, len(quarterly_df) + 1):
        current = quarterly_df.iloc[row - 1]

        for col_num, col_name in enumerate(quarterly_df.columns):
            val = current[col_name]

            if "Date" in col_name:
                fmt = formats["date"]
            elif "Return" in col_name or "Vol" in col_name:
                fmt = formats["percent"]
            elif "Value" in col_name:
                fmt = formats["dollar"]
            else:
                fmt = fmt_text

            safe_write(ws, row, col_num, val, fmt, fmt_text)

    ws.set_column(0, len(quarterly_df.columns), 18)


# ------------------------------------------------------------
# PARAMETERS SHEET
# ------------------------------------------------------------
def write_parameters_sheet(writer, config):
    df = pd.DataFrame(
        [{"Parameter": k, "Value": v} for k, v in config.items() if k != "regimes"]
    )
    df.to_excel(writer, sheet_name="Parameters", index=False)


# ------------------------------------------------------------
# REGIMES SHEET
# ------------------------------------------------------------
def write_regimes_sheet(writer, config, formats):
    regime_rows = []
    for regime, vals in config["regimes"].items():
        row = {"Regime": regime}
        row.update(vals)
        regime_rows.append(row)

    df = pd.DataFrame(regime_rows)
    df.to_excel(writer, sheet_name="Regimes", index=False)
    ws = writer.sheets["Regimes"]
    ws.freeze_panes(1, 1)

    header_fmt = formats["header"]
    percent_fmt = formats["percent"]
    text_fmt = formats["text"]

    # Write headers
    for col_num, col_name in enumerate(df.columns):
        ws.write(0, col_num, col_name, header_fmt)

    # Write rows w/ formatting
    for r in range(1, len(df) + 1):
        row = df.iloc[r - 1]
        for c, col_name in enumerate(df.columns):
            val = row[col_name]

            if col_name == "Regime":
                fmt = text_fmt
            else:
                fmt = percent_fmt  # ALL regime fields are %

            safe_write(ws, r, c, val, fmt, text_fmt)

    ws.set_column(0, len(df.columns), 14)

# ------------------------------------------------------------
# FINAL RESULTS SHEET
# ------------------------------------------------------------
def write_results_sheet(writer, equity_df, formats):
    last = equity_df.tail(1).copy()

    # --- Calculate YoY growth (CAGR)
    start_val = equity_df["Value"].iloc[0]
    end_val = equity_df["Value"].iloc[-1]
    years = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days / 365.25

    if years > 0:
        cagr = (end_val / start_val) ** (1 / years) - 1
    else:
        cagr = 0

    # Add CAGR to the results row
    last.insert(len(last.columns), "Avg_YoY_Growth", cagr)

    # Create sheet
    sheet_name = "Results"
    last.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    fmt_text = formats["text"]
    header_fmt = formats["header"]

    # Header row
    for col_num, col_name in enumerate(last.columns):
        ws.write(0, col_num, col_name, header_fmt)

    # Format correct column types
    for col_num, col_name in enumerate(last.columns):
        val = last.iloc[0][col_name]

        if col_name == "Date":
            fmt = formats["date"]
        elif col_name == "Value" or "_price" in col_name or "_value" in col_name:
            fmt = formats["dollar"]
        elif "Pct" in col_name or "Growth" in col_name or "YoY" in col_name:
            fmt = formats["percent"]
        elif "_norm" in col_name:
            fmt = formats["dollar"]
        elif "_shares" in col_name:
            fmt = formats["number"]
        else:
            fmt = fmt_text

        safe_write(ws, 1, col_num, val, fmt, fmt_text)

    ws.set_column(0, len(last.columns), 16)

    # Highlight final portfolio value cell
    val_idx = last.columns.get_loc("Value")
    safe_write(ws, 1, val_idx, last.iloc[0]["Value"], formats["final_value"], fmt_text)



# ======================================================================
# MAIN EXPORT FUNCTION — CLEAN, SIMPLE, MODULAR
# ======================================================================
def export_to_excel(equity_df, quarterly_df, config, mode="normal"):
    # Pick different base name depending on mode
    if mode == "worst_case":
        filename = get_unique_filename(base="worst_case_results")
    else:
        filename = get_unique_filename(base="backtest_results")

    print(f"Exporting to: {filename}")

    writer = pd.ExcelWriter(
        filename,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}}
    )
    workbook = writer.book

    formats = create_formats(workbook)

    write_equity_sheet(writer, equity_df, formats, config["tickers"])
    write_chart_sheet(workbook, equity_df)
    write_quarterly_sheet(writer, quarterly_df, formats)
    write_parameters_sheet(writer, config)
    write_regimes_sheet(writer, config, formats)  # <-- FIXED
    write_results_sheet(writer, equity_df, formats)

    writer.close()
    print(f"Excel export complete: {filename}\n")
