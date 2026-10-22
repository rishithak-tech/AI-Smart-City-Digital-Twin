import pandas as pd
from pathlib import Path


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

aqi_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_air_quality_cleaned.csv"
)

weather_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_weather_cleaned.csv"
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
# 2. LOAD DATA
# ========================================

aqi = pd.read_csv(aqi_file)
weather = pd.read_csv(weather_file)

aqi["datetime"] = pd.to_datetime(aqi["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])


print("\n========================================")
print("AQI + WEATHER DATA MERGING")
print("========================================")

print("\nAQI records:", len(aqi))
print("Weather records:", len(weather))


# ========================================
# 3. ALIGN WEATHER TIME
# ========================================

# Weather timestamps are currently at :30 minutes
# because the downloaded data was converted from GMT to IST.
#
# We round them to the nearest hour so they can
# align with the AQI hourly timestamps.

weather["datetime"] = (
    weather["datetime"] - pd.Timedelta(minutes=30)
)


# ========================================
# 4. REMOVE POSSIBLE DUPLICATES
# ========================================

weather = (
    weather
    .sort_values("datetime")
    .drop_duplicates(subset=["datetime"])
    .reset_index(drop=True)
)


# ========================================
# 5. MERGE AQI + WEATHER
# ========================================

merged_data = pd.merge(
    aqi,
    weather,
    on="datetime",
    how="inner"
)


# ========================================
# 6. SORT DATA
# ========================================

merged_data = (
    merged_data
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ========================================
# 7. CHECK MISSING VALUES
# ========================================

print("\nMerged records:", len(merged_data))

print("\nMissing values:")
print(merged_data.isnull().sum())


# ========================================
# 8. DISPLAY DATA
# ========================================

print("\nColumns:")
print(list(merged_data.columns))

print("\nFirst 10 records:")
print(merged_data.head(10))


# ========================================
# 9. DATE RANGE
# ========================================

print("\nDate range:")

print(
    "Start:",
    merged_data["datetime"].min()
)

print(
    "End:",
    merged_data["datetime"].max()
)


# ========================================
# 10. SAVE MERGED DATA
# ========================================

output_file = (
    processed_folder
    / "hyderabad_aqi_weather_merged.csv"
)

merged_data.to_csv(
    output_file,
    index=False
)


print("\nSaved to:")
print(output_file)


print("\n========================================")
print("AQI + WEATHER MERGE COMPLETED!")
print("========================================")