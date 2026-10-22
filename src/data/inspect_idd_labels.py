from pathlib import Path
from PIL import Image
import numpy as np


# ============================================
# DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

labels_path = dataset_path / "gtFine"


# ============================================
# FIND SEMANTIC LABELS ONLY
# ============================================

label_files = [
    file for file in labels_path.rglob("*_label.png")
    if not file.name.endswith("_inst_label.png")
]

print("Semantic label files:", len(label_files))


# ============================================
# INSPECT FIRST 10 LABELS
# ============================================

for label_file in label_files[:10]:

    image = Image.open(label_file)

    label_array = np.array(image)

    unique_values = np.unique(label_array)

    print("\nFile:", label_file.name)
    print("Image size:", image.size)
    print("Unique class IDs:", unique_values)


# ============================================
# UNIQUE CLASS IDs ACROSS DATASET
# ============================================

all_classes = set()

for label_file in label_files:

    image = Image.open(label_file)

    label_array = np.array(image)

    unique_values = np.unique(label_array)

    all_classes.update(unique_values.tolist())


print("\n====================================")
print("ALL CLASS IDs FOUND")
print("====================================")

print(sorted(all_classes))