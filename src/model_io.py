import joblib

def save_model(model, path="models/gold_model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved at {path}")

def load_model(path="models/gold_model.pkl"):
    return joblib.load(path)
