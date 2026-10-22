from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

images_path = dataset_path / "leftImg8bit"
labels_path = dataset_path / "gtFine"


# ============================================
# FIND A MATCHING IMAGE + LABEL
# ============================================

label_files = [
    file for file in labels_path.rglob("*_label.png")
    if not file.name.endswith("_inst_label.png")
]

selected_image = None
selected_label = None

for label_file in label_files:

    image_name = label_file.stem.replace("_label", "")

    matches = list(
        images_path.rglob(f"{image_name}_image.jpg")
    )

    if matches:
        selected_image = matches[0]
        selected_label = label_file
        break


if selected_image is None:
    print("ERROR: No matching image and label found.")
    raise SystemExit


# ============================================
# LOAD IMAGE + LABEL
# ============================================

image = Image.open(selected_image)
label = Image.open(selected_label)

image_array = np.array(image)
label_array = np.array(label)


# ============================================
# PRINT INFORMATION
# ============================================

print("Image:")
print(selected_image)

print("\nLabel:")
print(selected_label)

print("\nImage size:", image.size)

print("Label size:", label.size)

print("\nClass pixel counts:")

unique_values, counts = np.unique(
    label_array,
    return_counts=True
)

for class_id, count in zip(unique_values, counts):

    print(
        f"Class ID {class_id}: {count:,} pixels"
    )


# ============================================
# CREATE CATEGORICAL MASK
# ============================================

# 7 distinct colors for IDs 0-6
class_colors = [
    [0, 0, 0],        # 0
    [255, 0, 0],      # 1
    [0, 255, 0],      # 2
    [0, 0, 255],      # 3
    [255, 255, 0],    # 4
    [255, 0, 255],    # 5
    [0, 255, 255],    # 6
]

colored_mask = np.zeros(
    (label_array.shape[0], label_array.shape[1], 3),
    dtype=np.uint8
)


for class_id in range(7):

    colored_mask[label_array == class_id] = class_colors[class_id]


# Ignore pixels = black
colored_mask[label_array == 255] = [80, 80, 80]


# ============================================
# DISPLAY
# ============================================

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.imshow(image_array)

plt.title("Original Traffic Image")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(colored_mask)

plt.title("Categorical Segmentation Mask")

plt.axis("off")


plt.tight_layout()

plt.show()