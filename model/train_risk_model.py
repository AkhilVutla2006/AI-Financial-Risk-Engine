import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import pickle
import json


def train_and_evaluate_model(
    data_path='dataset/financial_risk_data.csv',
    model_save_path='model/risk_classifier.pkl',
    metadata_path='model/model_metadata.json'
):

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['will_fall_in_debt'])
    y = df['will_fall_in_debt']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature metadata
    metadata = {
        "feature_names": list(model.feature_names_in_)
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"\nModel saved at {model_save_path}")
    print(f"Metadata saved at {metadata_path}")

    # SHAP
    print("\nGenerating SHAP summary...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, X_test)


if __name__ == "__main__":
    train_and_evaluate_model()