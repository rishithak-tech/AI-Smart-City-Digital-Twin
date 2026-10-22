import pandas as pd
from pathlib import Path


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

weather_file = (
    project_root
    / "data"
    / "raw"
    / "open-meteo-17.40N78.46E505m.csv"
)


# ========================================
# 2. LOAD WEATHER DATA
# ========================================

data = pd.read_csv(weather_file, skiprows=2)


print("\n========================================")
print("WEATHER DATA INSPECTION")
print("========================================")

print("\nNumber of records:", len(data))

print("\nNumber of columns:", len(data.columns))

print("\nColumns:")
print(list(data.columns))


# ========================================
# 3. FIRST 10 ROWS
# ========================================

print("\nFirst 10 rows:")
print(data.head(10))


# ========================================
# 4. DATA TYPES
# ========================================

print("\nData types:")
print(data.dtypes)


# ========================================
# 5. MISSING VALUES
# ========================================

print("\nMissing values:")
print(data.isnull().sum())


# ========================================
# 6. BASIC INFORMATION
# ========================================

print("\nDataset information:")
print(data.info())


print("\n========================================")
print("WEATHER INSPECTION COMPLETED!")
print("========================================")