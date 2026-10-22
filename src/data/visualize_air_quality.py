import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Find the processed dataset
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
# 3. Calculate monthly average AQI
# --------------------------------------------------

monthly_aqi = (
    data
    .set_index("datetime")["aqi"]
    .resample("ME")
    .mean()
    .reset_index()
)


# --------------------------------------------------
# 4. Display monthly averages
# --------------------------------------------------

print("\nFirst 10 monthly averages:")
print(monthly_aqi.head(10))


# --------------------------------------------------
# 5. Calculate average AQI for each calendar month
# --------------------------------------------------

data["month"] = data["datetime"].dt.month

monthly_pattern = (
    data
    .groupby("month")["aqi"]
    .mean()
    .reset_index()
)


# --------------------------------------------------
# 6. Add month names
# --------------------------------------------------

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

monthly_pattern["month_name"] = (
    monthly_pattern["month"].map(month_names)
)


# --------------------------------------------------
# 7. Display seasonal pattern
# --------------------------------------------------

print("\nAverage AQI by month:")

print(
    monthly_pattern[
        ["month", "month_name", "aqi"]
    ]
)


# --------------------------------------------------
# 8. Create seasonal AQI graph
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_pattern["month_name"],
    monthly_pattern["aqi"],
    marker="o",
    linewidth=1.5
)

plt.title(
    "Average AQI by Month — Hyderabad (2017–2023)"
)

plt.xlabel("Month")
plt.ylabel("Average AQI")

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()