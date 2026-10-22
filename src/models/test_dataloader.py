from idd_dataset import IDDLiteDataset

from torch.utils.data import DataLoader, random_split


# ============================================
# DATASET PATH
# ============================================

dataset_path = (
    "C:/Users/LENOVO/Downloads/"
    "idd-lite/idd20k_lite"
)


# ============================================
# LOAD DATASET
# ============================================

dataset = IDDLiteDataset(
    dataset_path
)


# ============================================
# TRAIN / VALIDATION SPLIT
# ============================================

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset) - train_size
)


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# ============================================
# CREATE DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================
# GET ONE BATCH
# ============================================

images, labels = next(
    iter(train_loader)
)


# ============================================
# PRINT INFORMATION
# ============================================

print("\n====================================")
print("DATALOADER TEST")
print("====================================")

print("\nTotal dataset:")
print(len(dataset))

print("\nTraining samples:")
print(len(train_dataset))

print("\nValidation samples:")
print(len(val_dataset))

print("\nBatch image shape:")
print(images.shape)

print("\nBatch label shape:")
print(labels.shape)

print("\nBatch image data type:")
print(images.dtype)

print("\nBatch label data type:")
print(labels.dtype)

print("\n====================================")
print("DATALOADER TEST PASSED")
print("====================================")