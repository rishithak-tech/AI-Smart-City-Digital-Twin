import pandas as pd
from pathlib import Path


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

raw_file = (
    project_root
    / "data"
    / "raw"
    / "open-meteo-17.40N78.46E505m.csv"
)

processed_folder = (
    project_root
    / "data"
    / "processed"
)

processed_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ========================================
# 2. LOAD WEATHER DATA
# ========================================

data = pd.read_csv(
    raw_file,
    skiprows=2
)

print("\n========================================")
print("WEATHER DATA PROCESSING")
print("========================================")

print(f"\nRaw weather records: {len(data)}")


# ========================================
# 3. RENAME COLUMNS
# ========================================

data = data.rename(
    columns={
        "time": "datetime",
        "temperature_2m (°C)": "temperature",
        "relative_humidity_2m (%)": "humidity",
        "wind_speed_10m (km/h)": "wind_speed",
        "wind_direction_10m (°)": "wind_direction",
        "surface_pressure (hPa)": "surface_pressure"
    }
)


# ========================================
# 4. CONVERT DATETIME
# ========================================

data["datetime"] = pd.to_datetime(
    data["datetime"]
)


# ========================================
# 5. CONVERT UTC → INDIA TIME
# ========================================

data["datetime"] = (
    data["datetime"]
    .dt.tz_localize("UTC")
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
)


# ========================================
# 6. CONVERT NUMERIC COLUMNS
# ========================================

numeric_columns = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "surface_pressure"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ========================================
# 7. REMOVE INVALID RECORDS
# ========================================

data = data.dropna()

data = data.sort_values(
    "datetime"
)

data = data.drop_duplicates(
    subset=["datetime"]
)

data = data.reset_index(
    drop=True
)


# ========================================
# 8. SAVE CLEAN WEATHER DATA
# ========================================

output_file = (
    processed_folder
    / "hyderabad_weather_cleaned.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ========================================
# 9. DISPLAY RESULTS
# ========================================

print("\n========================================")
print("WEATHER PROCESSING COMPLETED!")
print("========================================")

print(f"\nClean records: {len(data)}")

print("\nColumns:")
print(list(data.columns))

print("\nDate range:")
print("Start:", data["datetime"].min())
print("End:", data["datetime"].max())

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst 5 records:")
print(data.head())

print("\nSaved to:")
print(output_file)

print("\n========================================")
print("DONE!")
print("========================================")