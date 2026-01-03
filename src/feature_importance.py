import pandas as pd

def show_feature_importance(model, feature_names):
    importance = model.feature_importances_

    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n")
    print(fi)
