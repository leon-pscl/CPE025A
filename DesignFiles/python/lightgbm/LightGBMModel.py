# lgbm_train_and_save.py

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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
# After scaling, restore feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df  = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
# Initialize CatBoost
lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=63,
        max_depth=10,
        objective='multiclass',
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

# Train (note: here using scaled features; you can also use raw X_train)
start_time_train = time.time()
lgbm.fit(X_train_scaled, y_train)
end_time_train = time.time()

print("Training finished!")
print(f"Training time: {end_time_train - start_time_train:.2f} seconds")

# Save model and scaler
joblib.dump(lgbm, "lgbm_water_model.pkl")
joblib.dump(scaler, "lgbm_colorimetry_scaler.pkl")
joblib.dump(X.columns.tolist(), "lgbm_feature_names.pkl")
print("Model, scaler, and feature names saved.")
