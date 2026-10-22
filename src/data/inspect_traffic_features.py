import pandas as pd

file_path = (
    "data/raw/idd_traffic_features.csv"
)

df = pd.read_csv(file_path)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)