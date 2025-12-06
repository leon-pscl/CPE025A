import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from catboost import CatBoostClassifier

# ==== Paths ====
model_path = r"catboost_water_model.cbm"
scaler_path = r"catboost_colorimetry_scaler.pkl"
feature_names_path = r"catboost_feature_names.pkl"
cluster_map_path = r"cluster_names/cluster_to_class_name.pkl"


def load_artifacts():
    # Load CatBoost model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Load scaler, feature names, and cluster→class map
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    cluster_map = joblib.load(cluster_map_path)

    return model, scaler, feature_names, cluster_map


# ==== Prediction from User Values ====
def predict_from_values(values_dict):
    model, scaler, feature_names, cluster_map = load_artifacts()

    # Make row with missing values defaulting to 0.0
    row = {f: values_dict.get(f, 0.0) for f in feature_names}

    # Convert to DataFrame
    X_input = pd.DataFrame([row])

    # Scale
    X_scaled = scaler.transform(X_input)

    # Predict cluster
    pred_cluster_raw = model.predict(X_scaled)[0]

    # CatBoost may return float, ndarray, or string depending on model setup
    pred_cluster = int(pred_cluster_raw) if not isinstance(pred_cluster_raw, str) else pred_cluster_raw

    # Convert cluster → class name
    class_name = cluster_map.get(pred_cluster, f"Unknown class ({pred_cluster})")

    return class_name


# ==== Save prediction results to CSV ====
def save_result_to_csv(user_inputs, predicted_class_name, csv_path="predictions_log.csv"):
    row = dict(user_inputs)

    # Add timestamp + prediction
    row["Timestamp"] = datetime.now().isoformat(timespec='seconds')
    row["Predicted_Class"] = predicted_class_name

    df = pd.DataFrame([row])

    # Create or append
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


# ==== Main interactive input mode ====
if __name__ == "__main__":
    model, scaler, feature_names, cluster_map = load_artifacts()

    print("\nPlease input the following water colorimetry concentration values:\n")
    user_inputs = {}

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_inputs[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.\n")

    predicted_class_name = predict_from_values(user_inputs)
    print("\nPredicted water class:", predicted_class_name)

    # Save to CSV and print file size
    csv_path = "predictions_log.csv"
    save_result_to_csv(user_inputs, predicted_class_name, csv_path)

    file_size = os.path.getsize(csv_path)
    print(f"\nPrediction saved to {csv_path}")
    print(f"Current file size: {file_size} bytes\n")
