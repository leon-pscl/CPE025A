import pandas as pd 
parquet_path = "C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet"
df_parquet = pd.read_parquet(parquet_path)
df_parquet.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df_parquet.drop(columns=['cluster','class_label'], axis=1)
y = df_parquet['cluster']

# Split sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import time
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize LightGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    objective='multiclass',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Measure training time
start_time_train = time.time()
lgbm.fit(X_train, y_train)
end_time_train = time.time()
print("Training finished!")

# Predictions and probabilities
start_time_predict = time.time()
predictions = lgbm.predict(X_test)
probs = lgbm.predict_proba(X_test)[:, 1]
end_time_predict = time.time()

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

train_preds = lgbm.predict(X_train)
print("LightGBM Train Accuracy:", accuracy_score(y_train, train_preds))
print("LightGBM Test Accuracy:", accuracy_score(y_test, predictions))
print(f"LightGBM training time: {end_time_train - start_time_train:.2f} seconds")
print(f"LightGBM prediction time: {end_time_predict - start_time_predict:.2f} seconds")


#Ciconfusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("LightGBM Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

