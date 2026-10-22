from pathlib import Path

import torch
import torch.nn as nn

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

# We already trained approximately 5 epochs
# So train for 10 more epochs
EPOCHS = 10

LEARNING_RATE = 0.001

NUM_CLASSES = 7


# ============================================
# DEVICE
# ============================================

device = torch.device("cpu")

print("Using device:", device)


# ============================================
# LOAD DATASET
# ============================================

dataset = IDDLiteDataset(
    DATASET_PATH
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

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


# ============================================
# DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================
# CREATE MODEL
# ============================================

model = LightweightUNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================
# LOAD EXISTING BEST MODEL
# ============================================

if Path(MODEL_PATH).exists():

    print("\nLoading existing trained model...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print("✓ Existing model loaded successfully!")

else:

    print("\nNo existing model found.")
    print("Starting training from scratch.")


# ============================================
# LOSS FUNCTION
# ============================================

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================
# TRAINING VARIABLES
# ============================================

best_val_loss = float("inf")


# ============================================
# TRAINING LOOP
# ============================================

for epoch in range(EPOCHS):

    print(
        f"\nAdditional Epoch {epoch + 1}/{EPOCHS}"
    )

    # ----------------------------------------
    # TRAIN
    # ----------------------------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ----------------------------------------
    # VALIDATION
    # ----------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # ----------------------------------------
    # PRINT RESULTS
    # ----------------------------------------

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )


    # ----------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved!"
        )


# ============================================
# TRAINING COMPLETE
# ============================================

print("\n====================================")
print("U-NET ADDITIONAL TRAINING COMPLETE")
print("====================================")

print(
    "\nBest validation loss:",
    f"{best_val_loss:.4f}"
)

print(
    "\nModel saved to:"
)

print(MODEL_PATH)