from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

import joblib


# ============================================
# PROJECT PATH
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IDD_PATH = (
    Path("C:/Users/LENOVO/Downloads")
    / "idd-lite"
    / "idd20k_lite"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "best_unet.pth"
)

TRAFFIC_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic_density_model.pkl"
)


# ============================================
# IMPORT U-NET AND DATASET
# ============================================

sys.path.append(
    str(
        PROJECT_ROOT
        / "src"
        / "models"
    )
)

from unet_model import LightweightUNet
from idd_dataset import IDDLiteDataset


# ============================================
# LOAD U-NET
# ============================================

def load_unet():

    model = LightweightUNet(
        num_classes=7
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# ============================================
# LOAD TRAFFIC MODEL
# ============================================

def load_traffic_model():

    model = joblib.load(
        TRAFFIC_MODEL_PATH
    )

    return model


# ============================================
# EXTRACT FEATURES FROM U-NET PREDICTION
# ============================================

def extract_scene_features(
    prediction
):

    prediction = prediction.numpy()


    # Ignore void pixels

    valid_pixels = (
        prediction != 255
    )


    total_pixels = (
        valid_pixels.sum()
    )


    if total_pixels == 0:

        return {
            "traffic_pixel_ratio": 0.0,
            "living_pixel_ratio": 0.0,
            "drivable_pixel_ratio": 0.0,
            "construction_pixel_ratio": 0.0,
            "sky_pixel_ratio": 0.0
        }


    traffic_ratio = (
        (prediction == 3).sum()
        / total_pixels
    )


    living_ratio = (
        (prediction == 2).sum()
        / total_pixels
    )


    drivable_ratio = (
        (prediction == 0).sum()
        / total_pixels
    )


    construction_ratio = (
        (prediction == 5).sum()
        / total_pixels
    )


    sky_ratio = (
        (prediction == 6).sum()
        / total_pixels
    )


    return {

        "traffic_pixel_ratio":
            traffic_ratio,

        "living_pixel_ratio":
            living_ratio,

        "drivable_pixel_ratio":
            drivable_ratio,

        "construction_pixel_ratio":
            construction_ratio,

        "sky_pixel_ratio":
            sky_ratio
    }


# ============================================
# ANALYZE TRAFFIC SCENE
# ============================================

def analyze_traffic_scene(
    unet,
    traffic_model,
    image,
    label
):

    # ----------------------------------------
    # Prepare image
    # ----------------------------------------

    image_tensor = (
        image
        .unsqueeze(0)
    )


    # ----------------------------------------
    # U-Net prediction
    # ----------------------------------------

    with torch.no_grad():

        output = unet(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0]


    # ----------------------------------------
    # Extract scene features
    # ----------------------------------------

    features = (
        extract_scene_features(
            prediction
        )
    )


    # ----------------------------------------
    # Prepare ML input
    # ----------------------------------------

    feature_names = [

        "traffic_pixel_ratio",

        "living_pixel_ratio",

        "drivable_pixel_ratio",

        "construction_pixel_ratio",

        "sky_pixel_ratio"

    ]


    X = np.array(
        [
            [
                features[name]
                for name in feature_names
            ]
        ]
    )


    # ----------------------------------------
    # Predict traffic density
    # ----------------------------------------

    predicted_density = (
        traffic_model.predict(X)[0]
    )


    predicted_density = float(
        predicted_density
    )


    # Keep value between 0 and 1

    predicted_density = max(
        0.0,
        min(
            1.0,
            predicted_density
        )
    )


    # ----------------------------------------
    # Traffic risk
    # ----------------------------------------

    if predicted_density < 0.05:

        traffic_risk = "Low"

    elif predicted_density < 0.12:

        traffic_risk = "Moderate"

    elif predicted_density < 0.20:

        traffic_risk = "High"

    else:

        traffic_risk = "Very High"


    # ----------------------------------------
    # Return result
    # ----------------------------------------

    return {

        "traffic_pixel_ratio":
            features[
                "traffic_pixel_ratio"
            ],

        "living_pixel_ratio":
            features[
                "living_pixel_ratio"
            ],

        "drivable_pixel_ratio":
            features[
                "drivable_pixel_ratio"
            ],

        "construction_pixel_ratio":
            features[
                "construction_pixel_ratio"
            ],

        "sky_pixel_ratio":
            features[
                "sky_pixel_ratio"
            ],

        "traffic_density_score":
            predicted_density,

        "traffic_risk":
            traffic_risk

    }


# ============================================
# MAIN TEST
# ============================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "AI TRAFFIC STATE ANALYSIS"
    )

    print(
        "===================================="
    )


    # ----------------------------------------
    # Load U-Net
    # ----------------------------------------

    print(
        "\nLoading U-Net..."
    )

    unet = load_unet()

    print(
        "U-Net loaded successfully."
    )


    # ----------------------------------------
    # Load traffic model
    # ----------------------------------------

    print(
        "\nLoading traffic model..."
    )

    traffic_model = (
        load_traffic_model()
    )

    print(
        "Traffic model loaded successfully."
    )


    # ----------------------------------------
    # Load dataset
    # ----------------------------------------

    print(
        "\nLoading IDD Lite dataset..."
    )

    dataset = IDDLiteDataset(
        IDD_PATH
    )

    print(
        f"Dataset size: {len(dataset)}"
    )


    # ----------------------------------------
    # Select first scene
    # ----------------------------------------

    image, label = dataset[0]


    # ----------------------------------------
    # Analyze scene
    # ----------------------------------------

    result = analyze_traffic_scene(

        unet,

        traffic_model,

        image,

        label
    )


    # ----------------------------------------
    # Display result
    # ----------------------------------------

    print(
        "\n===================================="
    )

    print(
        "TRAFFIC ANALYSIS RESULT"
    )

    print(
        "===================================="
    )


    print(
        f"Vehicle pixel ratio: "
        f"{result['traffic_pixel_ratio']:.4f}"
    )


    print(
        f"Living-thing ratio: "
        f"{result['living_pixel_ratio']:.4f}"
    )


    print(
        f"Drivable ratio: "
        f"{result['drivable_pixel_ratio']:.4f}"
    )


    print(
        f"Construction ratio: "
        f"{result['construction_pixel_ratio']:.4f}"
    )


    print(
        f"Sky ratio: "
        f"{result['sky_pixel_ratio']:.4f}"
    )


    print(
        "\nPredicted traffic density: "
        f"{result['traffic_density_score']:.4f}"
    )


    print(
        "Traffic risk: "
        f"{result['traffic_risk']}"
    )


    print(
        "\n===================================="
    )

    print(
        "TRAFFIC ANALYSIS COMPLETE"
    )

    print(
        "===================================="
    )