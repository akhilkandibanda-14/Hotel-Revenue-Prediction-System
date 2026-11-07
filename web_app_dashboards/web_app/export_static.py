import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

CSS_PATH = Path("assets/style.css")
OUTPUT = Path("assets/hotel_report.html")

if __name__ == "__main__":
    if Path("assets/demo_data.csv").exists():
        df = pd.read_csv("assets/demo_data.csv", parse_dates=["Date"]) 
    elif Path("Hotel book.csv").exists():
        df = pd.read_csv("Hotel book.csv", parse_dates=["Date"]) 
    else:
        raise SystemExit("No data file found (assets/demo_data.csv or Hotel book.csv)")

    # prepare monthly revenue
    if "Date" in df.columns and "Total_Revenue" in df.columns:
        df["Month_Name"] = pd.to_datetime(df["Date"]).dt.month_name()
        monthly = df.groupby("Month_Name")["Total_Revenue"].sum()
        rev_df = monthly.reset_index()
        rev_df.columns = ["Month", "Revenue"]
    else:
        rev_df = pd.DataFrame()

    figs = []
    if not rev_df.empty:
        figs.append(pio.to_html(px.line(rev_df, x="Month", y="Revenue", title="Monthly Revenue Trend"), full_html=False, include_plotlyjs='cdn'))

    if {"Total_Revenue", "Total_Costs"}.issubset(df.columns):
        df["Profit"] = df["Total_Revenue"] - df["Total_Costs"]
        figs.append(pio.to_html(px.scatter(df, x="Total_Revenue", y="Profit", title="Revenue vs Profit"), full_html=False, include_plotlyjs='cdn'))

    # build HTML
    css = CSS_PATH.read_text() if CSS_PATH.exists() else ""
    html_parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<style>{css}</style>",
        "<title>Hotel Revenue Report</title>",
        "</head>",
        "<body>",
        "<div class=\"container\">",
        "<h1>Hotel Revenue Report (Static)</h1>",
    ]

    for fig_html in figs:
        html_parts.append("<div class=\"report-figure\">")
        html_parts.append(fig_html)
        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</body></html>")

    OUTPUT.write_text('\n'.join(html_parts), encoding='utf-8')
    print(f"Wrote static report to {OUTPUT}")
