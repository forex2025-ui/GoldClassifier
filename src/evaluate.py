import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Get probability of class 1 (UP)
    probs = model.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (probs >= threshold).astype(int)

    print(f"\nThreshold used: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
