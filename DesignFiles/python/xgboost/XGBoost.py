import pandas as pd
from sklearn.model_selection import train_test_split 
import time
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
parquet_path = "C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet"
df_parquet = pd.read_parquet(parquet_path)

X = df_parquet.drop(columns=['cluster','class_label'], axis=1)
y = df_parquet['cluster']

# Split sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    objective='multi:softprob',  
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Measure training time
start_time_train = time.time()
xgb.fit(X_train, y_train)
end_time_train = time.time()
print("Training finished!")

# Predictions and probabilities
start_time_predict = time.time()
predictions = xgb.predict(X_test)
probs = xgb.predict_proba(X_test)[:, 1]
end_time_predict = time.time()

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

print("XGBoost Train Accuracy:", accuracy_score(y_train, train_preds))
print("XGBoost Test Accuracy:", accuracy_score(y_test, predictions))
print(f"XGBoost training time: {end_time_train - start_time_train:.2f} seconds")
print(f"XGBoost prediction time: {end_time_predict - start_time_predict:.2f} seconds")

#Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


