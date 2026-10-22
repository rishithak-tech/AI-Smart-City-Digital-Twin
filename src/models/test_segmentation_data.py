from pathlib import Path
from PIL import Image
import numpy as np
import torch


# ============================================
# DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

images_path = dataset_path / "leftImg8bit"
labels_path = dataset_path / "gtFine"


# ============================================
# FIND LABELED IMAGES
# ============================================

label_files = [
    file for file in labels_path.rglob("*_label.png")
    if not file.name.endswith("_inst_label.png")
]


print("Labeled images found:", len(label_files))


# ============================================
# SELECT FIRST MATCHING IMAGE
# ============================================

label_file = label_files[0]

image_id = label_file.stem.replace(
    "_label",
    ""
)

image_matches = list(
    images_path.rglob(
        f"{image_id}_image.jpg"
    )
)


if not image_matches:

    print("ERROR: Matching image not found.")

    raise SystemExit


image_file = image_matches[0]


# ============================================
# LOAD IMAGE
# ============================================

image = Image.open(image_file).convert("RGB")

label = Image.open(label_file)


# ============================================
# CONVERT TO NUMPY
# ============================================

image_array = np.array(image)

label_array = np.array(label)


# ============================================
# CONVERT IMAGE TO PYTORCH TENSOR
# ============================================

image_tensor = torch.tensor(
    image_array,
    dtype=torch.float32
)

# Change:
# Height × Width × Channels
#
# to:
# Channels × Height × Width

image_tensor = image_tensor.permute(
    2, 0, 1
)

# Normalize pixel values
image_tensor = image_tensor / 255.0


# ============================================
# CONVERT LABEL TO PYTORCH TENSOR
# ============================================

label_tensor = torch.tensor(
    label_array,
    dtype=torch.long
)


# ============================================
# PRINT RESULTS
# ============================================

print("\n====================================")
print("SEGMENTATION DATA TEST")
print("====================================")

print("\nImage file:")
print(image_file)

print("\nLabel file:")
print(label_file)

print("\nOriginal image shape:")
print(image_array.shape)

print("\nImage tensor shape:")
print(image_tensor.shape)

print("\nLabel shape:")
print(label_array.shape)

print("\nLabel tensor shape:")
print(label_tensor.shape)

print("\nImage tensor data type:")
print(image_tensor.dtype)

print("\nLabel tensor data type:")
print(label_tensor.dtype)

print("\nImage pixel range:")
print(
    image_tensor.min().item(),
    "to",
    image_tensor.max().item()
)

print("\nLabel classes:")
print(
    torch.unique(label_tensor)
)

print("\n====================================")
print("DATA LOADING TEST PASSED")
print("====================================")