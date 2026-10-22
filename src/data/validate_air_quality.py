import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Find the processed dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

processed_folder = project_root / "data" / "processed"

csv_files = list(processed_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No processed CSV file found."
    )

data_file = csv_files[0]

print(f"Reading: {data_file.name}")


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

data = pd.read_csv(data_file)

data["datetime"] = pd.to_datetime(data["datetime"])


# --------------------------------------------------
# 3. Basic information
# --------------------------------------------------

print("\n========================================")
print("DATASET VALIDATION")
print("========================================")

print(f"\nNumber of records: {len(data)}")

print(f"Number of columns: {len(data.columns)}")

print("\nColumns:")
print(list(data.columns))


# --------------------------------------------------
# 4. Date range
# --------------------------------------------------

print("\nDate range:")

print("Start:", data["datetime"].min())

print("End:", data["datetime"].max())


# --------------------------------------------------
# 5. AQI statistics
# --------------------------------------------------

print("\nAQI statistics:")

print("Minimum:", data["aqi"].min())

print("Maximum:", data["aqi"].max())

print("Average:", round(data["aqi"].mean(), 2))

print("Median:", data["aqi"].median())


# --------------------------------------------------
# 6. Missing values
# --------------------------------------------------

print("\nMissing values:")

print(data.isnull().sum())


# --------------------------------------------------
# 7. Duplicate timestamps
# --------------------------------------------------

duplicate_count = data["datetime"].duplicated().sum()

print("\nDuplicate timestamps:", duplicate_count)


# --------------------------------------------------
# 8. Check negative AQI values
# --------------------------------------------------

negative_count = (data["aqi"] < 0).sum()

print("Negative AQI values:", negative_count)


# --------------------------------------------------
# 9. Dataset statistics
# --------------------------------------------------

print("\nFull AQI statistics:")

print(data["aqi"].describe())


# --------------------------------------------------
# 10. Final message
# --------------------------------------------------

print("\n========================================")
print("VALIDATION COMPLETED!")
print("========================================")