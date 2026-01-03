import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = yf.download(
    "GC=F",
    interval="1h",
    period="2y"   # Yahoo limit for hourly data
)

df.reset_index(inplace=True)
df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

df.to_csv("data/gold_hourly.csv", index=False)

print("Hourly gold data saved")
