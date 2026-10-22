import pandas as pd
from pathlib import Path


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

merged_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_merged.csv"
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
# 2. LOAD MERGED DATA
# ========================================

data = pd.read_csv(merged_file)

data["datetime"] = pd.to_datetime(
    data["datetime"]
)

data = data.sort_values(
    "datetime"
)

data = data.reset_index(
    drop=True
)


print("\n========================================")
print("WEATHER FEATURE ENGINEERING")
print("========================================")

print("\nInput records:", len(data))


# ========================================
# 3. TIME FEATURES
# ========================================

data["hour"] = data["datetime"].dt.hour
data["day"] = data["datetime"].dt.day
data["month"] = data["datetime"].dt.month
data["year"] = data["datetime"].dt.year
data["day_of_week"] = data["datetime"].dt.dayofweek

data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)


# ========================================
# 4. AQI LAG FEATURES
# ========================================

data["aqi_previous_hour"] = (
    data["aqi"].shift(1)
)

data["aqi_previous_3_hours"] = (
    data["aqi"].shift(3)
)

data["aqi_previous_6_hours"] = (
    data["aqi"].shift(6)
)

data["aqi_previous_24_hours"] = (
    data["aqi"].shift(24)
)


# ========================================
# 5. AQI ROLLING FEATURES
# ========================================

data["aqi_rolling_6h"] = (
    data["aqi"]
    .shift(1)
    .rolling(window=6)
    .mean()
)

data["aqi_rolling_24h"] = (
    data["aqi"]
    .shift(1)
    .rolling(window=24)
    .mean()
)


# ========================================
# 6. SEASON
# ========================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


data["season"] = (
    data["month"].apply(get_season)
)


# ========================================
# 7. WEATHER FEATURES
# ========================================

# Current weather variables are already aligned
# with the AQI timestamp.

data["temperature_current"] = (
    data["temperature"]
)

data["humidity_current"] = (
    data["humidity"]
)

data["wind_speed_current"] = (
    data["wind_speed"]
)

data["wind_direction_current"] = (
    data["wind_direction"]
)

data["pressure_current"] = (
    data["surface_pressure"]
)


# ========================================
# 8. WEATHER LAG FEATURES
# ========================================

data["temperature_previous_hour"] = (
    data["temperature"].shift(1)
)

data["humidity_previous_hour"] = (
    data["humidity"].shift(1)
)

data["wind_speed_previous_hour"] = (
    data["wind_speed"].shift(1)
)


# ========================================
# 9. WEATHER CHANGE FEATURES
# ========================================

data["temperature_change"] = (
    data["temperature"]
    - data["temperature"].shift(1)
)

data["humidity_change"] = (
    data["humidity"]
    - data["humidity"].shift(1)
)

data["wind_speed_change"] = (
    data["wind_speed"]
    - data["wind_speed"].shift(1)
)


# ========================================
# 10. TARGET
# ========================================

data["target_aqi_next_hour"] = (
    data["aqi"].shift(-1)
)


# ========================================
# 11. REMOVE MISSING VALUES
# ========================================

data = data.dropna()

data = data.reset_index(
    drop=True
)


# ========================================
# 12. SAVE DATASET
# ========================================

output_file = (
    processed_folder
    / "hyderabad_aqi_weather_features.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ========================================
# 13. DISPLAY RESULTS
# ========================================

print("\n========================================")
print("WEATHER FEATURE ENGINEERING COMPLETED!")
print("========================================")

print("\nRecords:", len(data))

print("\nColumns:", len(data.columns))

print("\nFeatures:")

for column in data.columns:
    print("-", column)


print("\nFirst 5 rows:")
print(data.head())


print("\nSaved to:")
print(output_file)


print("\n========================================")
print("DONE!")
print("========================================")