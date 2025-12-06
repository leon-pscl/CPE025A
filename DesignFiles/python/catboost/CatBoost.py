import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
parquet_path = "C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet"
df_parquet = pd.read_parquet(parquet_path)
df_parquet.head()
scaler = StandardScaler()
X = df_parquet.drop(columns=['cluster','class_label','CCME_Values'], axis=1)
y = df_parquet['cluster']
# Split sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize CatBoost
cb = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=8,
    eval_metric='MultiClass',
    verbose=0,
    random_state=42
)
# Measure training time
start_time_train = time.time()
cb.fit(X_train, y_train)
print("Training finished!")
end_time_train = time.time()

# Predictions
train_preds = cb.predict(X_train)
start_time_predict = time.time()
predictions = cb.predict(X_test)
end_time_predict = time.time()

# Metrics
print("Classification Report:\n", classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

print("CatBoost Train Accuracy:", accuracy_score(y_train, train_preds))
print("CatBoost Test Accuracy:", accuracy_score(y_test, predictions))
print(f"CatBoost training time: {end_time_train - start_time_train:.2f} seconds")
print(f"CatBoost prediction time: {end_time_predict - start_time_predict:.2f} seconds")

#Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=cb.classes_,
            yticklabels=cb.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CatBoost Confusion Matrix")
plt.show()
