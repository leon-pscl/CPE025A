# CatBoost_train_and_save.py

import pandas as pd
import time
from catboost import CatBoostClassifier
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
cb = CatBoostClassifier(
                        iterations=200,
                        learning_rate=0.15,
                        depth=8,
                        eval_metric='Accuracy',
                        random_state=42,
                        thread_count=-1,
                        subsample=0.8,           # Only works with Bernoulli bootstrap
                        bootstrap_type='Bernoulli',
                        verbose=0
                        )

# Train (note: here using scaled features; you can also use raw X_train)
start_time_train = time.time()
cb.fit(X_train_scaled, y_train)
end_time_train = time.time()

print("Training finished!")
print(f"Training time: {end_time_train - start_time_train:.2f} seconds")

# Save model and scaler
cb.save_model("catboost_water_model.cbm")
joblib.dump(scaler, "catboost_colorimetry_scaler.pkl")
joblib.dump(X.columns.tolist(), "catboost_feature_names.pkl")
print("Model, scaler, and feature names saved.")
