import pandas as pd
import csv
import re
from pathlib import Path


# --------------------------------------------------
# 1. Find project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

raw_folder = project_root / "data" / "raw"
processed_folder = project_root / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Find the CSV file
# --------------------------------------------------

csv_files = list(raw_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found inside data/raw")

raw_file = csv_files[0]

print(f"Reading file: {raw_file.name}")


# --------------------------------------------------
# 3. Read the unusual CSV structure
# --------------------------------------------------

with open(
    raw_file,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.reader(file)
    raw_data = list(reader)

print(f"Raw data rows: {len(raw_data)}")


# --------------------------------------------------
# 4. Prepare variables
# --------------------------------------------------

current_year = None
current_month = None

records = []


month_names = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


# --------------------------------------------------
# 5. Process every row
# --------------------------------------------------

for row in raw_data:

    if not row:
        continue

    first_value = row[0].strip()


    # ----------------------------------------------
    # Detect year
    # ----------------------------------------------

    if first_value == "Year":

        if len(row) > 1:
            current_year = int(row[1])

        continue


    # ----------------------------------------------
    # Detect month
    # ----------------------------------------------

    month_match = re.match(
        r"([A-Za-z]+)-(\d{4})",
        first_value
    )

    if month_match:

        month_name = month_match.group(1)
        current_year = int(month_match.group(2))

        if month_name in month_names:
            current_month = month_names[month_name]

        continue


    # ----------------------------------------------
    # Detect day
    # ----------------------------------------------

    try:
        current_day = int(float(first_value))
    except ValueError:
        continue


    # ----------------------------------------------
    # Process hourly AQI values
    # ----------------------------------------------

    if current_year is None or current_month is None:
        continue

    hourly_values = row[1:25]

    for hour, value in enumerate(hourly_values):

        value = value.strip()

        # Skip missing values
        if value == "":
            continue

        try:
            aqi = float(value)
        except ValueError:
            continue

        try:
            timestamp = pd.Timestamp(
                year=current_year,
                month=current_month,
                day=current_day,
                hour=hour
            )
        except ValueError:
            continue

        records.append({
            "datetime": timestamp,
            "aqi": aqi
        })


# --------------------------------------------------
# 6. Create clean DataFrame
# --------------------------------------------------

clean_data = pd.DataFrame(records)


if clean_data.empty:
    raise ValueError(
        "No valid AQI records were extracted from the CSV."
    )


# --------------------------------------------------
# 7. Clean and sort the data
# --------------------------------------------------

clean_data["datetime"] = pd.to_datetime(
    clean_data["datetime"]
)

clean_data["aqi"] = pd.to_numeric(
    clean_data["aqi"],
    errors="coerce"
)

clean_data = clean_data.dropna(
    subset=["datetime", "aqi"]
)

clean_data = clean_data.sort_values(
    "datetime"
)

clean_data = clean_data.drop_duplicates(
    subset=["datetime"]
)

clean_data = clean_data.reset_index(
    drop=True
)


# --------------------------------------------------
# 8. Save processed dataset
# --------------------------------------------------

output_file = (
    processed_folder /
    "hyderabad_air_quality_cleaned.csv"
)

clean_data.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========================================")
print("AIR QUALITY PROCESSING COMPLETED!")
print("========================================")

print(f"Clean records: {len(clean_data)}")

print(f"Output file:")
print(output_file)

print("\nFirst 10 records:")
print(clean_data.head(10))

print("\nDataset information:")
print(clean_data.info())