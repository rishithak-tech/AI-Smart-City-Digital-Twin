from pathlib import Path


# ============================================
# 1. DATASET PATH
# ============================================

dataset_path = Path(
    "C:/Users/LENOVO/Downloads/idd-lite/idd20k_lite"
)

images_path = dataset_path / "leftImg8bit"
labels_path = dataset_path / "gtFine"


# ============================================
# 2. FIND ALL IMAGES
# ============================================

image_files = list(images_path.rglob("*_image.jpg"))

print("Total images found:", len(image_files))


# ============================================
# 3. FIND ALL SEMANTIC LABELS
# ============================================

label_files = list(labels_path.rglob("*_label.png"))

print("Total semantic labels found:", len(label_files))


# ============================================
# 4. FIND ALL INSTANCE LABELS
# ============================================

instance_files = list(
    labels_path.rglob("*_inst_label.png")
)

print("Total instance labels found:", len(instance_files))


# ============================================
# 5. CHECK IMAGE-LABEL PAIRS
# ============================================

image_names = {
    image.stem.replace("_image", "")
    for image in image_files
}

label_names = {
    label.stem.replace("_label", "")
    for label in label_files
}

instance_names = {
    label.stem.replace("_inst_label", "")
    for label in instance_files
}


missing_labels = image_names - label_names

missing_instances = image_names - instance_names


print("\nMissing semantic labels:", len(missing_labels))
print("Missing instance labels:", len(missing_instances))


# ============================================
# 6. DISPLAY SAMPLE FILES
# ============================================

print("\nSample images:")

for image in image_files[:5]:
    print(image)

print("\nSample semantic labels:")

for label in label_files[:5]:
    print(label)

print("\nSample instance labels:")

for label in instance_files[:5]:
    print(label)