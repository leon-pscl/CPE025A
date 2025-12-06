import os
import pandas as pd
import joblib

output_folder = "cluster_names"
os.makedirs(output_folder, exist_ok=True)  # Ensures the directory exists

df_parquet = pd.read_parquet("C:/Users/Leon/Documents/Github/CPE025A/df_parquet.parquet")
mapping = dict(df_parquet[['cluster', 'class_label']].drop_duplicates().values)
joblib.dump(mapping, os.path.join(output_folder, "cluster_to_class_name.pkl"))
