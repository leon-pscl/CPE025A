# XGBoost_train_and_save.py

import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # pip install joblib

parquet_path = "C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet"

df_parquet = pd.read_parquet(parquet_path)

# Features and target
X = df_parquet.drop(columns=['cluster', 'class_label'], axis=1)
y = df_parquet['cluster']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler on training data (optional but typical)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize CatBoost
xgb = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.075, 
                    max_depth=8,
                    objective='multi:softprob',  
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                    )

# Train (note: here using scaled features; you can also use raw X_train)
start_time_train = time.time()
xgb.fit(X_train_scaled, y_train)
end_time_train = time.time()

print("Training finished!")
print(f"Training time: {end_time_train - start_time_train:.2f} seconds")

# Save model and scaler
joblib.dump(xgb, "xgb_water_model.pkl")
joblib.dump(scaler, "xgb_colorimetry_scaler.pkl")
joblib.dump(X.columns.tolist(), "xgb_feature_names.pkl")
print("Model, scaler, and feature names saved.")
