from src.model_io import load_model
from src.feature_engineering import add_features
from src.data_loader import load_data
from src.config import FEATURE_COLUMNS, DATA_PATH

def generate_latest_signal(threshold=0.55):
    # Load trained model
    model = load_model()

    # Load & clean latest data (IMPORTANT FIX)
    df = load_data(DATA_PATH)

    # Feature engineering
    df = add_features(df)

    # Latest row
    latest = df.iloc[-1:]
    X_latest = latest[FEATURE_COLUMNS]

    # Predict probability
    prob_up = model.predict_proba(X_latest)[0, 1]

    # Generate signal
    signal = "BUY" if prob_up >= threshold else "NO TRADE"

    print("\n📈 LIVE SIGNAL")
    print(f"Probability UP : {prob_up:.2%}")
    print(f"Signal         : {signal}")

    return signal
