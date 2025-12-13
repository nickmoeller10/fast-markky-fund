# data_loader.py
from utils import log
import yfinance as yf
from datetime import datetime

def load_price_data(tickers, start_date, end_date=None):
    if end_date is None:
        log(f"Downloading price data for: {tickers}, starting {start_date} (end: current date)")
        data = yf.download(tickers, start=start_date, auto_adjust=True)
    else:
        log(f"Downloading price data for: {tickers}, from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    log(f"Download complete. Shape: {data.shape}")
    closes = data["Close"].dropna(how="all")
    log(f"Cleaned price data shape: {closes.shape}")

    return closes
