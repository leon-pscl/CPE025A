# catboost_predict_water_class_with_input.py  (LGBM Version)

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from datetime import datetime

# ==== Paths ====
model_path = r"lgbm_water_model.pkl"
scaler_path = r"lgbm_colorimetry_scaler.pkl"
feature_names_path = r"lgbm_feature_names.pkl"
cluster_map_path = r"cluster_names/cluster_to_class_name.pkl"


# ==== Load all files ====
def load_artifacts():
    model = joblib.load(model_path)          # LGBMClassifier loaded as sklearn object
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    cluster_map = joblib.load(cluster_map_path)
    return model, scaler, feature_names, cluster_map


# ==== Prediction ====
def predict_from_values(values_dict):
    model, scaler, feature_names, cluster_map = load_artifacts()

    # Construct a row with missing features defaulting to 0.0
    row = {f: values_dict.get(f, 0.0) for f in feature_names}

    # DataFrame with **ordered columns**
    X_input = pd.DataFrame([row], columns=feature_names)

    # Scale data
    X_scaled = scaler.transform(X_input)

    # Predict cluster
    pred_raw = model.predict(X_scaled)[0]

    # LGBM returns numpy scalar -> convert safely
    try:
        pred_cluster = int(pred_raw)
    except:
        pred_cluster = pred_raw  # fallback for string models

    # Map cluster -> human label
    class_name = cluster_map.get(pred_cluster, f"Unknown class ({pred_cluster})")
    return class_name


# ==== CSV logging ====
def save_result_to_csv(user_inputs, predicted_class_name, csv_path="predictions_log.csv"):

    row = dict(user_inputs)
    row["Timestamp"] = datetime.now().isoformat(timespec='seconds')
    row["Predicted_Class"] = predicted_class_name

    df = pd.DataFrame([row])

    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


# ==== Main interactive console input ====
if __name__ == "__main__":
    model, scaler, feature_names, cluster_map = load_artifacts()

    print("\nPlease input the following water colorimetry concentration values:\n")
    user_inputs = {}

    for feature in feature_names:
        while True:
            try:
                val = float(input(f"{feature}: "))
                user_inputs[feature] = val
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    predicted_class_name = predict_from_values(user_inputs)
    print("\nPredicted water class:", predicted_class_name)

    csv_path = "predictions_log.csv"
    save_result_to_csv(user_inputs, predicted_class_name, csv_path)
    file_size = os.path.getsize(csv_path)

    print(f"\nPrediction saved to {csv_path}")
    print(f"Current file size: {file_size} bytes\n")
