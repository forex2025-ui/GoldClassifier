import yfinance as yf
import pandas as pd

def fetch_intraday_xauusd(interval="5m", period="5d"):
    """
    Fetch intraday XAU/USD data from Yahoo Finance
    """
    df = yf.download(
        "XAUUSD=X",
        interval=interval,
        period=period,
        progress=False
    )

    df = df.reset_index()
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df
