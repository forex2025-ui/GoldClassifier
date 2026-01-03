import numpy as np
from src.train_model import train_model

def walk_forward_validation(
    df,
    feature_cols,
    target_col="target",
    train_size=0.6,
    step_size=0.1,
    threshold=0.55
):
    """
    Walk-forward validation:
    - Train on expanding window
    - Test on next chunk
    """

    n = len(df)
    train_end = int(n * train_size)
    step = int(n * step_size)

    results = []

    while train_end + step <= n:
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + step]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = train_model(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        accuracy = (preds == y_test).mean()
        results.append(accuracy)

        print(
            f"Train end: {train_end}, "
            f"Test size: {len(test_df)}, "
            f"Accuracy: {accuracy:.2%}"
        )

        train_end += step

    print("\n📊 WALK-FORWARD SUMMARY")
    print(f"Average Accuracy: {np.mean(results):.2%}")

    return results
