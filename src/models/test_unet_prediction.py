import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from unet_model import LightweightUNet


IMAGE_PATH = Path(
    r"C:\Users\LENOVO\Downloads\idd-lite\idd20k_lite"
    r"\leftImg8bit\test\103\206393_image.jpg"
)

MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "best_unet.pth"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "unet_dashboard_test.png"


device = torch.device("cpu")

model = LightweightUNet(num_classes=7)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("U-Net loaded successfully.")

image = Image.open(IMAGE_PATH).convert("RGB")

image_array = np.array(image).astype(np.float32) / 255.0

image_tensor = torch.from_numpy(
    image_array.transpose(2, 0, 1)
).unsqueeze(0)

image_tensor = image_tensor.to(device)

print("Image loaded successfully.")
print("Image size:", image.size)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(
        output,
        dim=1
    )[0].cpu().numpy()

print("Prediction generated successfully.")
print("Predicted classes:", np.unique(prediction))

mask = (prediction * 36).astype(np.uint8)

mask_image = Image.fromarray(mask)

mask_image.save(OUTPUT_PATH)

print("Segmentation mask saved to:")
print(OUTPUT_PATH)
