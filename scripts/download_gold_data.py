import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Download gold futures data
df = yf.download("GC=F", start="2015-01-01", auto_adjust=False)

# Ensure proper structure
df = df.reset_index()

# Keep only required columns
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Save clean CSV
df.to_csv("data/gold.csv", index=False)

print("gold.csv created cleanly")
