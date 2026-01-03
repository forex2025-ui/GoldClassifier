import ta

def add_features(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    df["ema_10"] = ta.trend.EMAIndicator(df["Close"], 10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["return_1"] = df["Close"].pct_change(1)
    df["return_2"] = df["Close"].pct_change(2)

    df["hl_range"] = df["High"] - df["Low"]

    df.dropna(inplace=True)
    return df
