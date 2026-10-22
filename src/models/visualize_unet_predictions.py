from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

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

OUTPUT_DIR = (
    "C:/Users/LENOVO/OneDrive/Documents/"
    "AI smart city/data/raw/"
    "unet_predictions"
)

NUM_CLASSES = 7


# ============================================
# CLASS COLORS
# ============================================

CLASS_COLORS = np.array([
    [128, 64, 128],    # Drivable
    [244, 35, 232],    # Non-drivable
    [220, 20, 60],     # Living-thing
    [0, 0, 142],       # Vehicles
    [190, 153, 153],   # Road-side objects
    [70, 70, 70],      # Construction
    [70, 130, 180]     # Sky
], dtype=np.uint8)


# ============================================
# DEVICE
# ============================================

device = torch.device("cpu")

print("Using device:", device)


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# LOAD DATASET
# ============================================

dataset = IDDLiteDataset(
    DATASET_PATH
)

print(
    "Dataset size:",
    len(dataset)
)


# ============================================
# LOAD MODEL
# ============================================

model = LightweightUNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "✓ Trained U-Net loaded successfully!"
)


# ============================================
# SELECT IMAGES
# ============================================

sample_indices = [
    0,
    50,
    100,
    150,
    200
]


# ============================================
# GENERATE PREDICTIONS
# ============================================

for count, index in enumerate(sample_indices):

    print(
        f"\nProcessing image "
        f"{count + 1}/{len(sample_indices)}"
    )

    image, label = dataset[index]

    # Add batch dimension

    input_image = image.unsqueeze(0).to(device)


    # ----------------------------------------
    # PREDICTION
    # ----------------------------------------

    with torch.no_grad():

        output = model(
            input_image
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = prediction.squeeze().cpu().numpy()

    image_np = image.permute(
        1,
        2,
        0
    ).numpy()

    label_np = label.numpy()


    # ----------------------------------------
    # CONVERT MASK TO COLORS
    # ----------------------------------------

    actual_color = np.zeros(
        (
            label_np.shape[0],
            label_np.shape[1],
            3
        ),
        dtype=np.uint8
    )

    predicted_color = np.zeros(
        (
            prediction.shape[0],
            prediction.shape[1],
            3
        ),
        dtype=np.uint8
    )


    for class_id in range(NUM_CLASSES):

        actual_color[
            label_np == class_id
        ] = CLASS_COLORS[class_id]

        predicted_color[
            prediction == class_id
        ] = CLASS_COLORS[class_id]


    # ----------------------------------------
    # IGNORE VOID PIXELS
    # ----------------------------------------

    actual_color[
        label_np == 255
    ] = [0, 0, 0]


    # ========================================
    # CREATE FIGURE
    # ========================================

    plt.figure(
        figsize=(15, 5)
    )


    # Original image

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image_np
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    # Ground truth

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        actual_color
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis("off")


    # Prediction

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        predicted_color
    )

    plt.title(
        "U-Net Prediction"
    )

    plt.axis("off")


    plt.tight_layout()


    # ========================================
    # SAVE RESULT
    # ========================================

    output_path = (
        Path(OUTPUT_DIR)
        / f"prediction_{count + 1}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        output_path
    )


# ============================================
# COMPLETE
# ============================================

print("\n====================================")
print("VISUALIZATION COMPLETE")
print("====================================")

print(
    "\nPredictions saved in:"
)

print(OUTPUT_DIR)