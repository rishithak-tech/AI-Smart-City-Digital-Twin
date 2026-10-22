from pathlib import Path
import sys

import pandas as pd


# ============================================
# PROJECT PATH
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)

IDD_PATH = (
    Path("C:/Users/LENOVO/Downloads")
    / "idd-lite"
    / "idd20k_lite"
)


# ============================================
# IMPORT DIGITAL TWIN COMPONENTS
# ============================================

sys.path.append(
    str(
        PROJECT_ROOT
        / "src"
        / "models"
    )
)

sys.path.append(
    str(
        PROJECT_ROOT
        / "src"
        / "digital_twin"
    )
)


from city_state import CityState

from traffic_state import (
    load_unet,
    load_traffic_model,
    analyze_traffic_scene
)

from idd_dataset import IDDLiteDataset


# ============================================
# AIR QUALITY RISK
# ============================================

def get_air_quality_risk(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# ============================================
# BUILD ENVIRONMENTAL STATE
# ============================================

def load_environment_state():

    data = pd.read_csv(
        DATA_PATH
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    latest = data.iloc[-1]

    state = CityState(

        city="Hyderabad",

        current_aqi=float(
            latest["aqi"]
        ),

        predicted_aqi_next_hour=float(
            latest["target_aqi_next_hour"]
        ),

        temperature=float(
            latest["temperature"]
        ),

        humidity=float(
            latest["humidity"]
        ),

        wind_speed=float(
            latest["wind_speed"]
        ),

        wind_direction=float(
            latest["wind_direction"]
        ),

        surface_pressure=float(
            latest["surface_pressure"]
        )

    )

    state.air_quality_risk = (
        get_air_quality_risk(
            state.current_aqi
        )
    )

    return state, latest["datetime"]


# ============================================
# MAIN DIGITAL TWIN
# ============================================

if __name__ == "__main__":

    print(
        "\n"
        "===================================="
    )

    print(
        "AI SMART CITY DIGITAL TWIN"
    )

    print(
        "===================================="
    )


    # ========================================
    # ENVIRONMENT
    # ========================================

    print(
        "\nLoading environmental state..."
    )

    city_state, timestamp = (
        load_environment_state()
    )

    print(
        "Environmental data loaded."
    )


    # ========================================
    # LOAD U-NET
    # ========================================

    print(
        "\nLoading U-Net..."
    )

    unet = load_unet()

    print(
        "U-Net loaded successfully."
    )


    # ========================================
    # LOAD TRAFFIC MODEL
    # ========================================

    print(
        "\nLoading traffic model..."
    )

    traffic_model = (
        load_traffic_model()
    )

    print(
        "Traffic model loaded successfully."
    )


    # ========================================
    # LOAD IDD DATASET
    # ========================================

    print(
        "\nLoading traffic scene..."
    )

    dataset = IDDLiteDataset(
        IDD_PATH
    )

    print(
        f"Traffic scenes available: "
        f"{len(dataset)}"
    )


    # ========================================
    # ANALYZE TRAFFIC SCENE
    # ========================================

    image, label = dataset[0]

    traffic_result = (
        analyze_traffic_scene(

            unet,

            traffic_model,

            image,

            label

        )
    )


    # ========================================
    # UPDATE CITY STATE
    # ========================================

    city_state.vehicle_pixel_ratio = (
        traffic_result[
            "traffic_pixel_ratio"
        ]
    )

    city_state.traffic_density_score = (
        traffic_result[
            "traffic_density_score"
        ]
    )

    city_state.traffic_risk = (
        traffic_result[
            "traffic_risk"
        ]
    )


    # ========================================
    # DISPLAY UNIFIED DIGITAL TWIN
    # ========================================

    print(
        "\n"
        "===================================="
    )

    print(
        "UNIFIED CITY STATE"
    )

    print(
        "===================================="
    )


    print(
        f"City: "
        f"{city_state.city}"
    )

    print(
        f"Timestamp: "
        f"{timestamp}"
    )


    print(
        "\n--- AIR QUALITY ---"
    )

    print(
        f"Current AQI: "
        f"{city_state.current_aqi:.1f}"
    )

    print(
        f"Next-hour AQI: "
        f"{city_state.predicted_aqi_next_hour:.1f}"
    )

    print(
        f"Air quality risk: "
        f"{city_state.air_quality_risk}"
    )


    print(
        "\n--- WEATHER ---"
    )

    print(
        f"Temperature: "
        f"{city_state.temperature:.1f} °C"
    )

    print(
        f"Humidity: "
        f"{city_state.humidity:.1f} %"
    )

    print(
        f"Wind speed: "
        f"{city_state.wind_speed:.1f} km/h"
    )

    print(
        f"Wind direction: "
        f"{city_state.wind_direction:.1f}°"
    )

    print(
        f"Surface pressure: "
        f"{city_state.surface_pressure:.1f} hPa"
    )


    print(
        "\n--- TRAFFIC ---"
    )

    print(
        f"Vehicle pixel ratio: "
        f"{city_state.vehicle_pixel_ratio:.4f}"
    )

    print(
        f"Traffic density: "
        f"{city_state.traffic_density_score:.4f}"
    )

    print(
        f"Traffic risk: "
        f"{city_state.traffic_risk}"
    )


    print(
        "\n"
        "===================================="
    )

    print(
        "DIGITAL TWIN READY"
    )

    print(
        "===================================="
    )