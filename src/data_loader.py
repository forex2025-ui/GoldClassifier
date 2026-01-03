import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    price_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df
