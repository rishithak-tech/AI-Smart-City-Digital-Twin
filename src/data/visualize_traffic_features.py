from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# LOAD DATA
# ============================================

file_path = Path(
    "C:/Users/LENOVO/OneDrive/Documents/"
    "AI smart city/data/raw/"
    "idd_traffic_features.csv"
)

data = pd.read_csv(file_path)


# ============================================
# DISPLAY BASIC INFORMATION
# ============================================

print("Number of images:", len(data))

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

print("\nFeature means:")
print(data[
    [
        "traffic_pixel_ratio",
        "living_pixel_ratio",
        "drivable_pixel_ratio",
        "construction_pixel_ratio",
        "sky_pixel_ratio",
        "traffic_density_score"
    ]
].mean())


# ============================================
# TRAFFIC PIXEL RATIO
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(
    data["traffic_pixel_ratio"],
    bins=30
)

plt.title("Traffic Pixel Ratio Distribution")

plt.xlabel("Traffic Pixel Ratio")

plt.ylabel("Number of Images")

plt.tight_layout()

plt.show()


# ============================================
# TRAFFIC DENSITY SCORE
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(
    data["traffic_density_score"],
    bins=30
)

plt.title("Traffic Density Score Distribution")

plt.xlabel("Traffic Density Score")

plt.ylabel("Number of Images")

plt.tight_layout()

plt.show()


# ============================================
# FEATURE SUMMARY
# ============================================

print("\nFeature statistics:")

print(
    data[
        [
            "traffic_pixel_ratio",
            "living_pixel_ratio",
            "drivable_pixel_ratio",
            "construction_pixel_ratio",
            "sky_pixel_ratio",
            "traffic_density_score"
        ]
    ].describe()
)