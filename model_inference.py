# model_inference.py

import shap
import pickle
import json
import pandas as pd
import os

def load_model(model_path="model/risk_classifier.pkl"):
    """Loads the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def explain_dataframe(input_df: pd.DataFrame, top_k=5):
    """
    Accepts a pandas DataFrame (single row).
    Returns structured explanation including SHAP values.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if len(input_df) != 1:
        raise ValueError("This function expects a single-row DataFrame")

    # Load model and metadata
    model_path = "model/risk_classifier.pkl"
    metadata_path = "model/model_metadata.json"
    
    model = load_model(model_path)
    
    if not os.path.exists(metadata_path):
        # Fallback if metadata is missing, just use what's in the DF
        required_features = list(model.feature_names_in_)
    else:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        required_features = metadata["feature_names"]

    # Align features: Add missing with 0.0, remove extras
    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0.0
    
    input_df = input_df[required_features]

    # Prediction
    prediction_prob = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])

    # SHAP Explanations
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)

    # SHAP output format varies by sklearn/shap version
    if isinstance(shap_vals, list):
        # Multi-class output: class 1 is risk
        impacts = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        # Newer SHAP format: [sample, feature, class]
        impacts = shap_vals[0, :, 1]
    else:
        # Binary case with single output array
        impacts = shap_vals[0]

    feature_data = []
    for feature, value, impact in zip(input_df.columns,
                                      input_df.iloc[0].values,
                                      impacts):
        feature_data.append({
            "feature": feature,
            "value": float(value),
            "impact": float(impact)
        })

    # Sort by absolute importance
    feature_data = sorted(
        feature_data,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )[:top_k]

    return {
        "predicted_probability": prediction_prob,
        "prediction": prediction,
        "top_features": feature_data
    }