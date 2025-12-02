#
# import pandas as pd
# import xlsxwriter
# import os
#
#
# def get_unique_filename(base="backtest_results", ext="xlsx"):
#     i = 1
#     while True:
#         filename = f"{base}_{i}.{ext}"
#         if not os.path.exists(filename):
#             return filename
#         i += 1
#
#
# def export_to_excel(equity_df, quarterly_df, config):
#     filename = get_unique_filename()
#     print(f"Exporting to: {filename}")
#
#     writer = pd.ExcelWriter(filename, engine="xlsxwriter")
#     workbook = writer.book
#
#     # Create formatting styles
#     fmt_dollar = workbook.add_format({'num_format': '$#,##0.00'})
#     fmt_percent = workbook.add_format({'num_format': '0.00%'})
#     fmt_date = workbook.add_format({'num_format': 'yyyy-mm-dd'})
#     fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
#
#     # ============================================================
#     # CLEAN EQUITY DF
#     # ============================================================
#     eq = equity_df.copy()
#
#     tickers = config.get("tickers", [])
#
#     # Rename raw prices to *_price if not already
#     for t in tickers:
#         if t in eq.columns:
#             eq.rename(columns={t: f"{t}_price"}, inplace=True)
#
#     # Identify normalized, share, and value columns
#     norm_cols = [c for c in eq.columns if c.endswith("_norm")]
#     price_cols = [c for c in eq.columns if c.endswith("_price") or c == "Value"]
#     share_cols = [c for c in eq.columns if c.endswith("_shares")]
#     value_cols = [c for c in eq.columns if c.endswith("_value")]
#     percent_cols = [c for c in eq.columns if "Pct" in c or "Return" in c or "Vol" in c]
#
#     # ============================================================
#     # EQUITY CURVE SHEET
#     # ============================================================
#     eq.to_excel(writer, sheet_name="Equity Curve", index=False)
#     sheet = writer.sheets["Equity Curve"]
#
#     # Format header row
#     for col_num, col_name in enumerate(eq.columns):
#         sheet.write(0, col_num, col_name, fmt_header)
#
#     # Apply formatting by column type
#     for col_num, col_name in enumerate(eq.columns):
#         if col_name == "Date":
#             sheet.set_column(col_num, col_num, 15, fmt_date)
#         elif col_name in price_cols or col_name in value_cols:
#             sheet.set_column(col_num, col_num, 14, fmt_dollar)
#         elif col_name in percent_cols:
#             sheet.set_column(col_num, col_num, 12, fmt_percent)
#         elif col_name in share_cols:
#             sheet.set_column(col_num, col_num, 12)
#         else:
#             sheet.set_column(col_num, col_num, 14)
#
#     # ============================================================
#     # CHART SHEET
#     # ============================================================
#     chart_sheet = workbook.add_worksheet("Chart")
#     chart = workbook.add_chart({'type': 'line'})
#
#     # Portfolio line
#     val_idx = eq.columns.get_loc("Value")
#     chart.add_series({
#         'name': 'Portfolio Value',
#         'categories': ['Equity Curve', 1, 0, len(eq), 0],
#         'values': ['Equity Curve', 1, val_idx, len(eq), val_idx],
#         'y2_axis': False
#     })
#
#     # Add normalized series
#     for col in norm_cols:
#         idx = eq.columns.get_loc(col)
#         nice_label = col.replace("_norm", "").upper() + " (Normalized)"
#
#         chart.add_series({
#             'name': nice_label,
#             'categories': ['Equity Curve', 1, 0, len(eq), 0],
#             'values': ['Equity Curve', 1, idx, len(eq), idx],
#             'y2_axis': True
#         })
#
#     chart.set_title({'name': 'Portfolio vs Normalized Benchmarks'})
#     chart.set_x_axis({'name': 'Date'})
#     chart.set_y_axis({'name': 'Portfolio Value ($)', 'log_base': 10})
#     chart.set_y2_axis({'name': f"Normalized Performance (${config['starting_balance']} start)", 'log_base': 10})
#     chart.set_legend({'position': 'bottom'})
#
#     chart_sheet.insert_chart('B2', chart)
#
#     # ============================================================
#     # QUARTERLY SUMMARY SHEET
#     # ============================================================
#     if quarterly_df is not None and len(quarterly_df) > 0:
#         quarterly_df.to_excel(writer, sheet_name="Quarterly Summary", index=False)
#         qsheet = writer.sheets["Quarterly Summary"]
#
#         # Format header row
#         for col_num, col_name in enumerate(quarterly_df.columns):
#             qsheet.write(0, col_num, col_name, fmt_header)
#
#             if "Date" in col_name:
#                 qsheet.set_column(col_num, col_num, 15, fmt_date)
#             elif "Return" in col_name or "Vol" in col_name or "Growth" in col_name:
#                 qsheet.set_column(col_num, col_num, 12, fmt_percent)
#             elif "Value" in col_name:
#                 qsheet.set_column(col_num, col_num, 14, fmt_dollar)
#             else:
#                 qsheet.set_column(col_num, col_num, 14)
#
#     # ============================================================
#     # PARAMETERS SHEET
#     # ============================================================
#     params_df = pd.DataFrame(
#         [{"Parameter": k, "Value": v} for k, v in config.items() if k != "regimes"]
#     )
#     params_df.to_excel(writer, sheet_name="Parameters", index=False)
#
#     psheet = writer.sheets["Parameters"]
#     psheet.set_column(0, 0, 25)
#     psheet.set_column(1, 1, 40)
#
#     # ============================================================
#     # REGIMES SHEET
#     # ============================================================
#     regime_records = []
#     for name, vals in config.get("regimes", {}).items():
#         row = {"Regime": name}
#         row.update(vals)
#         regime_records.append(row)
#
#     pd.DataFrame(regime_records).to_excel(writer, sheet_name="Regimes", index=False)
#
#     rsheet = writer.sheets["Regimes"]
#     rsheet.set_column(0, len(regime_records[0]), 18)
#
#     # ============================================================
#     # FINAL RESULTS SHEET
#     # ============================================================
#     last = eq.tail(1)
#     last.to_excel(writer, sheet_name="Results", index=False)
#
#     results_sheet = writer.sheets["Results"]
#     for col_num, col_name in enumerate(last.columns):
#
#         if "Date" in col_name:
#             results_sheet.set_column(col_num, col_num, 15, fmt_date)
#         elif col_name in price_cols or col_name in value_cols or col_name == "Value":
#             results_sheet.set_column(col_num, col_num, 14, fmt_dollar)
#         elif col_name in percent_cols:
#             results_sheet.set_column(col_num, col_num, 12, fmt_percent)
#         else:
#             results_sheet.set_column(col_num, col_num, 14)
#
#     writer.close()
#     print(f"Excel export complete: {filename}\n")
#

import pandas as pd
import xlsxwriter
import os
import math


def get_unique_filename(base="backtest_results", ext="xlsx"):
    i = 1
    while True:
        filename = f"{base}_{i}.{ext}"
        if not os.path.exists(filename):
            return filename
        i += 1


def export_to_excel(equity_df, quarterly_df, config):
    filename = get_unique_filename()
    print(f"Exporting to: {filename}")

    # IMPORTANT FIX: allow NaN/Inf in numeric cells
    writer = pd.ExcelWriter(
        filename,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}}
    )
    workbook = writer.book

    # -----------------------------------------------------------------
    # PROFESSIONAL FINANCIAL FORMATTING
    # -----------------------------------------------------------------
    header_fmt = workbook.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': '#003366',
        'border': 1
    })

    zebra_light = workbook.add_format({'bg_color': '#F2F2F2'})
    zebra_white = workbook.add_format({'bg_color': 'white'})
    bold_fmt = workbook.add_format({'bold': True})

    fmt_text = workbook.add_format({'align': 'left'})

    fmt_dollar = workbook.add_format({
        'num_format': '$#,##0.00',
        'align': 'right'
    })

    fmt_percent = workbook.add_format({
        'num_format': '0.00%',
        'align': 'right'
    })

    fmt_date = workbook.add_format({
        'num_format': 'yyyy-mm-dd',
        'align': 'left'
    })

    fmt_number = workbook.add_format({
        'num_format': '#,##0.00',
        'align': 'right'
    })

    fmt_value_highlight = workbook.add_format({
        'num_format': '$#,##0.00',
        'bg_color': '#D9EAD3',
        'bold': True,
        'align': 'right'
    })

    # =================================================================
    # CLEAN EQUITY DF
    # =================================================================
    eq = equity_df.copy()
    tickers = config.get("tickers", [])

    for t in tickers:
        if t in eq.columns:
            eq.rename(columns={t: f"{t}_price"}, inplace=True)

    norm_cols = [c for c in eq.columns if c.endswith("_norm")]
    price_cols = [c for c in eq.columns if c.endswith("_price") or c == "Value"]
    share_cols = [c for c in eq.columns if c.endswith("_shares")]
    value_cols = [c for c in eq.columns if c.endswith("_value")]
    percent_cols = [c for c in eq.columns if "Pct" in c or "Return" in c or "Vol" in c]

    # =================================================================
    # EQUITY CURVE SHEET
    # =================================================================
    eq.to_excel(writer, sheet_name="Equity Curve", index=False)
    sheet = writer.sheets["Equity Curve"]
    sheet.freeze_panes(1, 1)

    # Write header
    for col_num, col_name in enumerate(eq.columns):
        sheet.write(0, col_num, col_name, header_fmt)

    # HELPER: Safe write that handles NaN
    def safe_write(ws, r, c, val, fmt):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            ws.write(r, c, "", fmt_text)
        else:
            ws.write(r, c, val, fmt)

    # Data rows + zebra striping
    for row in range(1, len(eq) + 1):
        fmt_row = zebra_light if row % 2 == 0 else zebra_white
        current = eq.iloc[row - 1]

        for col_num, col_name in enumerate(eq.columns):
            value = current[col_name]

            # Determine formatting
            if col_name == "Date":
                safe_write(sheet, row, col_num, value, fmt_date)
            elif col_name in price_cols or col_name in value_cols:
                safe_write(sheet, row, col_num, value, fmt_dollar)
            elif col_name in percent_cols:
                safe_write(sheet, row, col_num, value, fmt_percent)
            elif col_name in share_cols:
                safe_write(sheet, row, col_num, value, fmt_number)
            else:
                safe_write(sheet, row, col_num, value, fmt_text)

    # Highlight last portfolio value
    last_row = len(eq)
    value_col = eq.columns.get_loc("Value")
    safe_write(sheet, last_row, value_col, eq.iloc[-1]["Value"], fmt_value_highlight)

    sheet.set_column(0, len(eq.columns), 14)

    # =================================================================
    # CHART SHEET (unchanged)
    # =================================================================
    chart_sheet = workbook.add_worksheet("Chart")
    chart = workbook.add_chart({'type': 'line'})

    val_idx = eq.columns.get_loc("Value")
    chart.add_series({
        'name': 'Portfolio Value',
        'categories': ['Equity Curve', 1, 0, len(eq), 0],
        'values':     ['Equity Curve', 1, val_idx, len(eq), val_idx],
        'line': {'color': '#003366', 'width': 2},
    })

    for col in norm_cols:
        idx = eq.columns.get_loc(col)
        chart.add_series({
            'name': col.replace("_norm", "") + " (Normalized)",
            'categories': ['Equity Curve', 1, 0, len(eq), 0],
            'values': ['Equity Curve', 1, idx, len(eq), idx],
            'line': {'dash_type': 'dash'}
        })

    chart.set_title({'name': 'Portfolio vs Benchmarks'})
    chart.set_y_axis({'name': 'Value ($)'})
    chart.set_x_axis({'name': 'Date'})
    chart.set_legend({'position': 'bottom'})
    chart_sheet.insert_chart('B2', chart)

    # =================================================================
    # QUARTERLY SUMMARY (same NaN-safe writer)
    # =================================================================
    if quarterly_df is not None and len(quarterly_df) > 0:
        quarterly_df.to_excel(writer, sheet_name="Quarterly Summary", index=False)
        qsheet = writer.sheets["Quarterly Summary"]
        qsheet.freeze_panes(1, 1)

        for col_num, col_name in enumerate(quarterly_df.columns):
            qsheet.write(0, col_num, col_name, header_fmt)

        for row in range(1, len(quarterly_df) + 1):
            current = quarterly_df.iloc[row - 1]

            for col_num, col_name in enumerate(quarterly_df.columns):
                value = current[col_name]

                if "Date" in col_name:
                    safe_write(qsheet, row, col_num, value, fmt_date)
                elif "Return" in col_name or "Vol" in col_name:
                    safe_write(qsheet, row, col_num, value, fmt_percent)
                elif "Value" in col_name:
                    safe_write(qsheet, row, col_num, value, fmt_dollar)
                else:
                    safe_write(qsheet, row, col_num, value, fmt_text)

        qsheet.set_column(0, len(quarterly_df.columns), 18)

    # =================================================================
    # PARAMETERS SHEET (unchanged)
    # =================================================================
    params = pd.DataFrame(
        [{"Parameter": k, "Value": v} for k, v in config.items() if k != "regimes"]
    )
    params.to_excel(writer, sheet_name="Parameters", index=False)
    psheet = writer.sheets["Parameters"]
    psheet.freeze_panes(1, 1)

    # =================================================================
    # REGIMES SHEET
    # =================================================================
    regime_rows = []
    for r, vals in config["regimes"].items():
        row = {"Regime": r}
        row.update(vals)
        regime_rows.append(row)

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_excel(writer, sheet_name="Regimes", index=False)

    rsheet = writer.sheets["Regimes"]
    rsheet.freeze_panes(1, 1)

    # =================================================================
    # FINAL RESULTS SHEET
    # =================================================================
    last = eq.tail(1)
    last.to_excel(writer, sheet_name="Results", index=False)
    results = writer.sheets["Results"]

    for col_num, col_name in enumerate(last.columns):
        results.write(0, col_num, col_name, header_fmt)

    # Highlight final value again
    value_idx = last.columns.get_loc("Value")
    safe_write(results, 1, value_idx, last.iloc[0]["Value"], fmt_value_highlight)

    writer.close()
    print(f"Excel export complete: {filename}\n")
