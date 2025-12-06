import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------- Configuration ------------------
DATA_PATH = (
    'C:/Users/Leon/Documents/Jupyter/softdes/27800394/Dataset/Combined Data/Combined_dataset.csv'
)
PARQUET_PATH = 'C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet'
FRESHWATER_TYPES = ['River', 'Effluent', 'Sewage', 'Lake', 'Canal', 'Drainage']
FEATURE_COLS = ['Ammonia (mg/l)', 'pH (ph units)', 'Nitrate (mg/l)', 'CCME_Values']
QUALITY_MAPPING = {"Excellent": 5, "Good": 4, "Fair": 3, "Marginal": 2, "Poor": 1}
RANDOM_STATE = 42


def load_and_preprocess_data(
    path: str, freshwater_types: list[str], quality_mapping: dict
) -> pd.DataFrame:
    """
    Load CSV data, convert dates, filter by freshwater types and year, and map quality labels.

    Args:
        path: Filepath to the dataset CSV.
        freshwater_types: List of waterbody types to include.
        quality_mapping: Dictionary mapping CCME_WQI text to numeric scores.

    Returns:
        Preprocessed DataFrame.
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)
    df = df[df['Waterbody Type'].isin(freshwater_types) & (df['Year'] >= 2000)].copy()
    df['CCME_WQI'] = df['CCME_WQI'].map(quality_mapping)
    return df


def iqr_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply IQR filtering to remove outliers from a DataFrame column.

    Args:
        df: DataFrame to filter.
        column: Column name to apply filtering.

    Returns:
        Filtered DataFrame.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1 * iqr
    upper_bound = q3 + 1 * iqr
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


def apply_iqr_filtering(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Clean data by applying IQR filtering on multiple columns.

    Args:
        df: DataFrame to clean.
        columns: List of column names to apply IQR filtering.

    Returns:
        Cleaned DataFrame.
    """
    for col in columns:
        df = iqr_filter(df, col)
    return df


def perform_clustering(
    df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 16
) -> Tuple[pd.DataFrame, StandardScaler, KMeans]:
    """
    Scale features and perform KMeans clustering.

    Args:
        df: DataFrame containing features.
        feature_cols: List of feature column names.
        n_clusters: Number of clusters.

    Returns:
        Tuple containing DataFrame with clusters, fitted scaler, and KMeans model.
    """
    scaler = StandardScaler()
    features = df[feature_cols].copy()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(scaled_features)
    df = df.copy()
    df['cluster'] = cluster_labels
    return df, scaler, kmeans


def label_clusters(
    df: pd.DataFrame, feature_cols: list[str], label_prefixes=('A', 'B', 'C', 'D')
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign human-readable labels to clusters based on scoring.

    Args:
        df: DataFrame with 'cluster' column.
        feature_cols: List of feature columns used for scoring.
        label_prefixes: Tuple of prefixes for labeling groups of clusters.

    Returns:
        DataFrame with class labels, and DataFrame of cluster statistics.
    """
    agg_func = {
        feature_cols[1]: 'std',  # pH std deviation
        feature_cols[0]: 'mean',  # Ammonia mean
        feature_cols[2]: 'mean',  # Nitrate mean
    }
    stats = df.groupby('cluster').agg(agg_func)
    stats.rename(
        columns={
            feature_cols[1]: 'pH_std',
            feature_cols[0]: 'Ammonia_mean',
            feature_cols[2]: 'Nitrate_mean',
        },
        inplace=True,
    )
    stats['score'] = -stats['Nitrate_mean'] - stats['Ammonia_mean'] - stats['pH_std']
    stats_sorted = stats.sort_values('score', ascending=False)
    labels = []
    for i in range(len(stats_sorted)):
        if i < 4:
            labels.append(f'{label_prefixes[0]}{i + 1}')
        elif i < 8:
            labels.append(f'{label_prefixes[1]}{i - 3}')
        elif i < 12:
            labels.append(f'{label_prefixes[2]}{i - 7}')
        else:
            labels.append(f'{label_prefixes[3]}{i - 11}')
    stats_sorted['label'] = labels
    label_map = dict(zip(stats_sorted.index, stats_sorted['label']))
    df = df.copy()
    df['class_label'] = df['cluster'].map(label_map)
    return df, stats_sorted


def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet file, creating directories if needed.

    Args:
        df: DataFrame to save.
        path: File path for parquet file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"Saved preprocessed data to {path}")


def train_and_evaluate_classifier(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Train Random Forest classifier and evaluate performance on train and test data.

    Args:
        X: Feature DataFrame.
        y: Target Series.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
        min_samples_split=5,
        min_samples_leaf=10,
        max_depth=10,
    )
    start_train = time.time()
    rf.fit(X_train, y_train)
    end_train = time.time()

    start_predict = time.time()
    y_pred = rf.predict(X_test)
    end_predict = time.time()

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        "Random Forest Train Accuracy:",
        accuracy_score(y_train, rf.predict(X_train)),
    )
    print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Training time: {end_train - start_train:.2f} seconds")
    print(f"Prediction time: {end_predict - start_predict:.2f} seconds")


# ---------------- Pipeline Execution ----------------
if __name__ == "__main__":
    data_frame = load_and_preprocess_data(DATA_PATH, FRESHWATER_TYPES, QUALITY_MAPPING)
    data_frame = apply_iqr_filtering(data_frame, FEATURE_COLS[:-1])  # Exclude non-numeric last feature
    clustered_df, scaler_model, kmeans_model = perform_clustering(data_frame, FEATURE_COLS)
    labeled_df, stats_df = label_clusters(clustered_df, FEATURE_COLS)
    save_to_parquet(labeled_df, PARQUET_PATH)
    loaded_df = pd.read_parquet(PARQUET_PATH)
    features = loaded_df.drop(columns=["cluster", "class_label"])
    targets = loaded_df["cluster"]
    train_and_evaluate_classifier(features, targets)
