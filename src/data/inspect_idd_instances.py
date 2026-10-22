from pathlib import Path
from PIL import Image
import numpy as np


# ============================================
# DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

images_path = dataset_path / "leftImg8bit"
labels_path = dataset_path / "gtFine"


# ============================================
# FIND INSTANCE LABELS
# ============================================

instance_files = list(
    labels_path.rglob("*_inst_label.png")
)

print("Total instance labels:", len(instance_files))


# ============================================
# FIND A MATCHING IMAGE
# ============================================

selected_instance = None
selected_image = None

for instance_file in instance_files:

    image_name = instance_file.stem.replace(
        "_inst_label",
        ""
    )

    matches = list(
        images_path.rglob(
            f"{image_name}_image.jpg"
        )
    )

    if matches:

        selected_instance = instance_file
        selected_image = matches[0]

        break


# ============================================
# CHECK
# ============================================

if selected_instance is None:

    print("ERROR: No matching image found.")

    raise SystemExit


# ============================================
# LOAD INSTANCE MASK
# ============================================

instance_image = Image.open(
    selected_instance
)

instance_array = np.array(
    instance_image
)


# ============================================
# INFORMATION
# ============================================

print("\nImage:")
print(selected_image)

print("\nInstance mask:")
print(selected_instance)

print("\nImage size:")
print(selected_image.stat().st_size, "bytes")

print("\nMask size:")
print(instance_image.size)

print("\nData type:")
print(instance_array.dtype)


# ============================================
# UNIQUE INSTANCE IDs
# ============================================

unique_ids = np.unique(
    instance_array
)

print("\nNumber of unique IDs:")
print(len(unique_ids))

print("\nFirst 100 unique instance IDs:")

print(unique_ids[:100])


# ============================================
# MIN / MAX
# ============================================

print("\nMinimum ID:")
print(instance_array.min())

print("\nMaximum ID:")
print(instance_array.max())