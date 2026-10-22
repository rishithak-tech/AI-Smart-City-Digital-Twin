from idd_dataset import IDDLiteDataset


# ============================================
# DATASET PATH
# ============================================

dataset_path = (
    "C:/Users/LENOVO/Downloads/"
    "idd-lite/idd20k_lite"
)


# ============================================
# CREATE DATASET
# ============================================

dataset = IDDLiteDataset(
    dataset_path
)


# ============================================
# DATASET SIZE
# ============================================

print("\nDataset size:")

print(len(dataset))


# ============================================
# GET ONE SAMPLE
# ============================================

image, label = dataset[0]


# ============================================
# PRINT INFORMATION
# ============================================

print("\n====================================")
print("DATASET TEST")
print("====================================")

print("\nImage shape:")
print(image.shape)

print("\nLabel shape:")
print(label.shape)

print("\nImage data type:")
print(image.dtype)

print("\nLabel data type:")
print(label.dtype)

print("\nImage range:")
print(
    image.min().item(),
    "to",
    image.max().item()
)

print("\nClasses:")
print(
    label.unique()
)

print("\n====================================")
print("DATASET LOADER TEST PASSED")
print("====================================")