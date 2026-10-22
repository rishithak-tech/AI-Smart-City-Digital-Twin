from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from idd_dataset import IDDLiteDataset
from unet_model import LightweightUNet


# ============================================
# SETTINGS
# ============================================

DATASET_PATH = (
    "C:/Users/LENOVO/Downloads/"
    "idd-lite/idd20k_lite"
)

MODEL_PATH = (
    "C:/Users/LENOVO/OneDrive/Documents/"
    "AI smart city/data/raw/"
    "best_unet.pth"
)

BATCH_SIZE = 8
NUM_CLASSES = 7


# ============================================
# CLASS NAMES
# ============================================

CLASS_NAMES = {
    0: "Drivable",
    1: "Non-drivable",
    2: "Living-thing",
    3: "Vehicles",
    4: "Road-side objects",
    5: "Construction",
    6: "Sky"
}


# ============================================
# DEVICE
# ============================================

device = torch.device("cpu")

print("Using device:", device)


# ============================================
# LOAD DATASET
# ============================================

dataset = IDDLiteDataset(DATASET_PATH)

print("Total dataset:", len(dataset))


# ============================================
# SAME TRAIN / VALIDATION SPLIT
# ============================================

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Validation samples:", len(val_dataset))


# ============================================
# VALIDATION DATALOADER
# ============================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# CREATE MODEL
# ============================================

model = LightweightUNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================
# LOAD TRAINED MODEL
# ============================================

if not Path(MODEL_PATH).exists():

    raise FileNotFoundError(
        "best_unet.pth was not found!"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("✓ Trained U-Net loaded successfully!")


# ============================================
# CONFUSION MATRIX
# ============================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================
# EVALUATION
# ============================================

print("\nEvaluating U-Net...")

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        predictions = predictions.cpu().numpy()

        labels = labels.cpu().numpy()

        for true_mask, pred_mask in zip(
            labels,
            predictions
        ):

            valid = (
                true_mask != 255
            )

            true_pixels = true_mask[valid]

            pred_pixels = pred_mask[valid]

            for true_class, pred_class in zip(
                true_pixels,
                pred_pixels
            ):

                if 0 <= true_class < NUM_CLASSES:

                    confusion_matrix[
                        true_class,
                        pred_class
                    ] += 1


# ============================================
# CALCULATE METRICS
# ============================================

print("\n====================================")
print("U-NET EVALUATION RESULTS")
print("====================================")


# Pixel accuracy

total_correct = np.trace(
    confusion_matrix
)

total_pixels = confusion_matrix.sum()

pixel_accuracy = (
    total_correct / total_pixels
)


print(
    f"\nPixel Accuracy: "
    f"{pixel_accuracy:.4f}"
)


# ============================================
# IoU
# ============================================

ious = []

print("\nClass-wise IoU:")

for class_id in range(NUM_CLASSES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ]

    false_positive = (
        confusion_matrix[:, class_id].sum()
        - true_positive
    )

    false_negative = (
        confusion_matrix[class_id, :].sum()
        - true_positive
    )

    denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if denominator == 0:

        iou = float("nan")

    else:

        iou = (
            true_positive
            / denominator
        )

        ious.append(iou)

    print(
        f"{class_id} - "
        f"{CLASS_NAMES[class_id]:20s}: "
        f"{iou:.4f}"
    )


# ============================================
# MEAN IoU
# ============================================

mean_iou = np.nanmean(
    ious
)

print(
    f"\nMean IoU (mIoU): "
    f"{mean_iou:.4f}"
)


# ============================================
# CONFUSION MATRIX
# ============================================

print("\nConfusion Matrix:")

print(confusion_matrix)

print("\n====================================")
print("EVALUATION COMPLETE")
print("====================================")