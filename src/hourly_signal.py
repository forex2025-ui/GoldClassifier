import pandas as pd
from src.intraday_data import fetch_intraday_xauusd
from src.feature_engineering import add_features
from src.config import FEATURE_COLUMNS

def get_hourly_signal(model, threshold=0.55):
    """
    Builds hourly candles from intraday data
    and predicts NEXT hour direction
    """

    # 1. Fetch intraday data (5-min)
    df = fetch_intraday_xauusd()

    # 2. Set datetime index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
    df.columns = [
    col[0].capitalize() if isinstance(col, tuple) else col.capitalize()
    for col in df.columns
]



    # 3. Resample to HOURLY candles
    hourly = df.resample("1H").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # 4. Drop current forming hour (CRITICAL)
    hourly = hourly.iloc[:-1]

    # 5. Featur
    hourly = add_features(hourly)

# 🚨 SAFETY CHECK: not enough data after indicators
    if hourly.empty:
        return {
        "timeframe": "1H",
        "signal": "NO DATA",
        "probability_up": None,
        "last_completed_hour": None
    }

    latest = hourly.iloc[-1:]
    X = latest[FEATURE_COLUMNS]


    # 7. Predict probability
    if X.empty:
        return {
        "timeframe": "1H",
        "signal": "NO DATA",
        "probability_up": None,
        "last_completed_hour": None
        }

    prob_up = model.predict_proba(X)[0, 1]

    signal = "BUY" if prob_up >= threshold else "NO TRADE"

    return {
        "timeframe": "1H",
        "probability_up": round(float(prob_up), 4),
        "signal": signal,
        "last_completed_hour": str(latest.index[-1])
    }
