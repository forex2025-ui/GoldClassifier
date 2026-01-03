import ta

def add_features(df):

    # =====================
    # PRICE ACTION FEATURES (REQUIRED)
    # =====================
    df["return_1"] = df["Close"].pct_change(1)
    df["return_2"] = df["Close"].pct_change(2)
    df["hl_range"] = df["High"] - df["Low"]

    # =====================
    # MOMENTUM (Hourly-friendly)
    # =====================
    df["rsi"] = ta.momentum.RSIIndicator(
        df["Close"], window=9
    ).rsi()

    # =====================
    # TREND
    # =====================
    df["ema_10"] = ta.trend.EMAIndicator(
        df["Close"], window=10
    ).ema_indicator()

    df["ema_20"] = ta.trend.EMAIndicator(
        df["Close"], window=20
    ).ema_indicator()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()

    # =====================
    # VOLATILITY
    # =====================
    bb = ta.volatility.BollingerBands(
        df["Close"], window=10
    )
    df["bb_lower"] = bb.bollinger_lband()

    # =====================
    # CLEANUP
    # =====================
    df.dropna(inplace=True)
    return df
