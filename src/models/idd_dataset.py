from pathlib import Path

from PIL import Image

import numpy as np
import torch

from torch.utils.data import Dataset


class IDDLiteDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.images_path = (
            self.dataset_path / "leftImg8bit"
        )

        self.labels_path = (
            self.dataset_path / "gtFine"
        )

        # Find semantic labels only
        self.label_files = [
            file
            for file in self.labels_path.rglob(
                "*_label.png"
            )
            if not file.name.endswith(
                "_inst_label.png"
            )
        ]

        # Keep only labels that have a matching image
        self.samples = []

        for label_file in self.label_files:

            image_id = label_file.stem.replace(
                "_label",
                ""
            )

            image_matches = list(
                self.images_path.rglob(
                    f"{image_id}_image.jpg"
                )
            )

            if image_matches:

                self.samples.append(
                    (
                        image_matches[0],
                        label_file
                    )
                )

        print(
            "Valid image-label pairs:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_file, label_file = self.samples[index]

        # Load image
        image = Image.open(
            image_file
        ).convert("RGB")

        # Load segmentation mask
        label = Image.open(
            label_file
        )

        # Convert to NumPy
        image = np.array(image)
        label = np.array(label)

        # Image → PyTorch tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # H × W × C → C × H × W
        image = image.permute(
            2,
            0,
            1
        )

        # Normalize image
        image = image / 255.0

        # Label → PyTorch tensor
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label