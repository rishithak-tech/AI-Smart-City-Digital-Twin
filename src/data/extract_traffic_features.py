from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd


# ============================================
# DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

images_path = dataset_path / "leftImg8bit"
labels_path = dataset_path / "gtFine"


# ============================================
# FIND SEMANTIC LABELS
# ============================================

label_files = [
    file for file in labels_path.rglob("*_label.png")
    if not file.name.endswith("_inst_label.png")
]

print("Semantic labels found:", len(label_files))


# ============================================
# CLASS IDs
# ============================================

# IDD Lite Level-1 classes
# 0 = Drivable
# 1 = Non-drivable
# 2 = Living things
# 3 = Vehicles / traffic
# 4 = Barrier
# 5 = Construction
# 6 = Sky
# 255 = Ignore

TRAFFIC_CLASS = 3
LIVING_CLASS = 2
DRIVABLE_CLASS = 0
CONSTRUCTION_CLASS = 5
SKY_CLASS = 6


# ============================================
# STORE RESULTS
# ============================================

results = []


# ============================================
# PROCESS EACH LABEL
# ============================================

for index, label_file in enumerate(label_files):

    image_id = label_file.stem.replace(
        "_label",
        ""
    )

    # Find corresponding image
    matches = list(
        images_path.rglob(
            f"{image_id}_image.jpg"
        )
    )

    if not matches:
        continue

    image_file = matches[0]

    # Load image
    image = Image.open(image_file)

    # Load segmentation mask
    label = Image.open(label_file)

    label_array = np.array(label)

    # ----------------------------------------
    # VALID PIXELS
    # ----------------------------------------

    valid_pixels = label_array != 255

    total_valid_pixels = valid_pixels.sum()

    if total_valid_pixels == 0:
        continue

    # ----------------------------------------
    # CLASS PIXEL COUNTS
    # ----------------------------------------

    traffic_pixels = np.sum(
        label_array[valid_pixels] == TRAFFIC_CLASS
    )

    living_pixels = np.sum(
        label_array[valid_pixels] == LIVING_CLASS
    )

    drivable_pixels = np.sum(
        label_array[valid_pixels] == DRIVABLE_CLASS
    )

    construction_pixels = np.sum(
        label_array[valid_pixels] == CONSTRUCTION_CLASS
    )

    sky_pixels = np.sum(
        label_array[valid_pixels] == SKY_CLASS
    )

    # ----------------------------------------
    # RATIOS
    # ----------------------------------------

    traffic_ratio = (
        traffic_pixels / total_valid_pixels
    )

    living_ratio = (
        living_pixels / total_valid_pixels
    )

    drivable_ratio = (
        drivable_pixels / total_valid_pixels
    )

    construction_ratio = (
        construction_pixels / total_valid_pixels
    )

    sky_ratio = (
        sky_pixels / total_valid_pixels
    )

    # ----------------------------------------
    # TRAFFIC DENSITY SCORE
    # ----------------------------------------

    traffic_density_score = (
        traffic_ratio * 0.7
        +
        living_ratio * 0.3
    )

    # ----------------------------------------
    # SAVE RESULT
    # ----------------------------------------

    results.append({

        "image_id": image_id,

        "image_width": image.width,

        "image_height": image.height,

        "traffic_pixel_ratio": traffic_ratio,

        "living_pixel_ratio": living_ratio,

        "drivable_pixel_ratio": drivable_ratio,

        "construction_pixel_ratio": construction_ratio,

        "sky_pixel_ratio": sky_ratio,

        "traffic_density_score":
            traffic_density_score
    })

    # Progress
    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1} / "
            f"{len(label_files)}"
        )


# ============================================
# CREATE DATAFRAME
# ============================================

features = pd.DataFrame(results)


# ============================================
# SAVE CSV
# ============================================

output_file = Path(
    "C:/Users/LENOVO/OneDrive/Documents/"
    "AI smart city/data/raw/"
    "idd_traffic_features.csv"
)

features.to_csv(
    output_file,
    index=False
)


# ============================================
# SUMMARY
# ============================================

print("\n====================================")
print("TRAFFIC FEATURE EXTRACTION COMPLETE")
print("====================================")

print(
    "Images processed:",
    len(features)
)

print(
    "\nFeature columns:"
)

print(
    features.columns.tolist()
)

print(
    "\nFirst 5 rows:"
)

print(
    features.head()
)

print(
    "\nFeature statistics:"
)

print(
    features.describe()
)

print(
    "\nSaved to:"
)

print(output_file)