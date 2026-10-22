import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Find the project and processed dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

processed_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_air_quality_cleaned.csv"
)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

data = pd.read_csv(processed_file)

data["datetime"] = pd.to_datetime(data["datetime"])


# --------------------------------------------------
# 3. Sort data by datetime
# --------------------------------------------------

data = data.sort_values("datetime")

data = data.reset_index(drop=True)


# --------------------------------------------------
# 4. Create time-based features
# --------------------------------------------------

data["hour"] = data["datetime"].dt.hour

data["day"] = data["datetime"].dt.day

data["month"] = data["datetime"].dt.month

data["year"] = data["datetime"].dt.year

data["day_of_week"] = data["datetime"].dt.dayofweek


# --------------------------------------------------
# 5. Create weekend feature
# --------------------------------------------------

data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)


# --------------------------------------------------
# 6. Create previous AQI features
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Create rolling AQI features
# --------------------------------------------------

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


# --------------------------------------------------
# 8. Create seasonal feature
# --------------------------------------------------

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


data["season"] = data["month"].apply(get_season)


# --------------------------------------------------
# 9. Create next-hour AQI target
# --------------------------------------------------

data["target_aqi_next_hour"] = (
    data["aqi"].shift(-1)
)


# --------------------------------------------------
# 10. Remove rows with unavailable features/target
# --------------------------------------------------

data = data.dropna()

data = data.reset_index(drop=True)


# --------------------------------------------------
# 11. Save feature-engineered dataset
# --------------------------------------------------

output_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_air_quality_features.csv"
)

data.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 12. Display results
# --------------------------------------------------

print("\n========================================")
print("FEATURE ENGINEERING COMPLETED!")
print("========================================")

print(f"\nNumber of records: {len(data)}")

print(f"Number of columns: {len(data.columns)}")

print("\nFeatures:")

for column in data.columns:
    print("-", column)

print("\nFirst 5 rows:")

print(data.head())

print("\nSaved to:")

print(output_file)