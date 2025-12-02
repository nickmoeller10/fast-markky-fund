# data_loader.py
from utils import log
import yfinance as yf

def load_price_data(tickers, start_date):
    log(f"Downloading price data for: {tickers}, starting {start_date}")

    data = yf.download(tickers, start=start_date, auto_adjust=True)

    log(f"Download complete. Shape: {data.shape}")
    closes = data["Close"].dropna(how="all")
    log(f"Cleaned price data shape: {closes.shape}")

    return closes
