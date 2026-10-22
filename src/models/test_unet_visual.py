import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src" / "models")
)

from unet_model import LightweightUNet


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

IMAGE_PATH = Path(
    r"C:\Users\LENOVO\Downloads\idd-lite\idd20k_lite"
    r"\leftImg8bit\test\103\206393_image.jpg"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "best_unet.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "unet_color_segmentation.png"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

device = torch.device("cpu")

model = LightweightUNet(num_classes=7)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("U-Net loaded successfully.")


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_array = (
    np.array(image)
    .astype(np.float32)
    / 255.0
)

image_tensor = torch.from_numpy(
    image_array.transpose(2, 0, 1)
).unsqueeze(0)

image_tensor = image_tensor.to(device)

print(
    "Image loaded:",
    image.size
)


# ---------------------------------------------------------
# U-Net prediction
# ---------------------------------------------------------

with torch.no_grad():

    output = model(
        image_tensor
    )

    prediction = torch.argmax(
        output,
        dim=1
    )[0].cpu().numpy()

print(
    "Prediction generated."
)

print(
    "Classes:",
    np.unique(prediction)
)


# ---------------------------------------------------------
# Color map
# ---------------------------------------------------------
# 0 = Drivable
# 1 = Non-drivable
# 2 = Living thing
# 3 = Vehicles
# 4 = Road-side objects
# 5 = Construction
# 6 = Sky

color_map = np.array(
    [
        [0, 255, 0],       # Drivable
        [255, 255, 0],     # Non-drivable
        [255, 0, 255],     # Living thing
        [255, 0, 0],       # Vehicles
        [0, 165, 255],     # Road-side
        [180, 0, 255],     # Construction
        [0, 191, 255],     # Sky
    ],
    dtype=np.uint8
)


# ---------------------------------------------------------
# Create color segmentation
# ---------------------------------------------------------

segmentation = color_map[
    prediction
]

segmentation_image = Image.fromarray(
    segmentation
)

segmentation_image.save(
    OUTPUT_PATH
)


# ---------------------------------------------------------
# Create overlay
# ---------------------------------------------------------

original = np.array(
    image
).astype(np.float32)

segmentation_float = (
    segmentation
    .astype(np.float32)
)

overlay = (
    0.55 * original
    + 0.45 * segmentation_float
)

overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)

overlay_image = Image.fromarray(
    overlay
)

OVERLAY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "unet_overlay.png"
)

overlay_image.save(
    OVERLAY_PATH
)


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------

print()
print(
    "Color segmentation saved to:"
)
print(OUTPUT_PATH)

print(
    "Overlay saved to:"
)
print(OVERLAY_PATH)