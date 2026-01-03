from src.model_io import save_model
from src.data_loader import load_data
from src.feature_engineering import add_features
from src.target_builder import create_target
from src.split import time_series_split
from src.train_model import train_model
from src.evaluate import evaluate_model
from src.backtest import backtest_strategy
from src.walk_forward import walk_forward_validation
from src.config import DATA_PATH, FEATURE_COLUMNS


def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    df = add_features(df)

    # 3. Target creation
    df = create_target(df)

    # 4. Train-test split
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    X_train, X_test, y_train, y_test = time_series_split(X, y)

    # 5. Train model
    model = train_model(X_train, y_train)

    # 6. Evaluate with threshold
    evaluate_model(model, X_test, y_test, threshold=0.55)

    # 7. Backtest
    backtest_strategy(
        df,
        model,
        FEATURE_COLUMNS,
        threshold=0.55
    )

    # 8. Walk-forward validation  ✅ df IS DEFINED HERE
    walk_forward_validation(
        df,
        FEATURE_COLUMNS,
        threshold=0.55
    )
    # STEP-25: Save trained model
    save_model(model)



if __name__ == "__main__":
    main()
