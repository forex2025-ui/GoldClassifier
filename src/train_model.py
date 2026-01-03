from catboost import CatBoostClassifier

def train_model(X_train, y_train):
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=[1.0, 1.8],  
        verbose=False,
        random_seed=42
    )

    model.fit(X_train, y_train)
    return model
