from pathlib import Path
import sys

import pandas as pd


# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)


# ============================================
# IMPORT DIGITAL TWIN COMPONENTS
# ============================================

sys.path.append(
    str(
        PROJECT_ROOT
        / "src"
        / "digital_twin"
    )
)

from city_state import CityState


# ============================================
# LOAD LATEST CITY STATE
# ============================================

def load_latest_city_state():

    data = pd.read_csv(DATA_PATH)

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    latest = data.iloc[-1]

    return {
        "datetime": latest["datetime"],
        "aqi": float(latest["aqi"]),
        "predicted_aqi": float(
            latest["target_aqi_next_hour"]
        ),
        "temperature": float(
            latest["temperature"]
        ),
        "humidity": float(
            latest["humidity"]
        ),
        "wind_speed": float(
            latest["wind_speed"]
        ),
        "wind_direction": float(
            latest["wind_direction"]
        ),
        "surface_pressure": float(
            latest["surface_pressure"]
        )
    }


# ============================================
# TRAFFIC RISK
# ============================================

def get_traffic_risk(density):

    if density < 0.05:

        return "Low"

    elif density < 0.12:

        return "Moderate"

    elif density < 0.20:

        return "High"

    else:

        return "Very High"


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
# RUN WHAT-IF SIMULATION
# ============================================

def run_simulation(

    current_density,

    traffic_change_percent,

    current_weather,

    temperature_change,

    humidity_change,

    wind_change

):

    # ----------------------------------------
    # Traffic simulation
    # ----------------------------------------

    new_density = (
        current_density
        * (
            1
            + traffic_change_percent / 100
        )
    )


    # Keep density between 0 and 1

    new_density = max(
        0,
        min(
            1,
            new_density
        )
    )


    # ----------------------------------------
    # Weather simulation
    # ----------------------------------------

    new_temperature = (
        current_weather["temperature"]
        + temperature_change
    )


    new_humidity = (
        current_weather["humidity"]
        + humidity_change
    )


    new_wind_speed = (
        current_weather["wind_speed"]
        + wind_change
    )


    # Prevent physically impossible values

    new_humidity = max(
        0,
        min(
            100,
            new_humidity
        )
    )


    new_wind_speed = max(
        0,
        new_wind_speed
    )


    # ----------------------------------------
    # Traffic risk
    # ----------------------------------------

    new_traffic_risk = (
        get_traffic_risk(
            new_density
        )
    )


    # ----------------------------------------
    # Return scenario
    # ----------------------------------------

    return {

        "traffic_density":
            new_density,

        "traffic_risk":
            new_traffic_risk,

        "temperature":
            new_temperature,

        "humidity":
            new_humidity,

        "wind_speed":
            new_wind_speed

    }


# ============================================
# DISPLAY COMPARISON
# ============================================

def display_comparison(
    baseline,
    scenario
):

    print(
        "\n"
        "===================================="
    )

    print(
        "BASELINE CITY STATE"
    )

    print(
        "===================================="
    )


    print(
        f"AQI: "
        f"{baseline['aqi']:.1f}"
    )

    print(
        f"Next-hour AQI prediction: "
        f"{baseline['predicted_aqi']:.1f}"
    )

    print(
        f"Temperature: "
        f"{baseline['temperature']:.1f} °C"
    )

    print(
        f"Humidity: "
        f"{baseline['humidity']:.1f} %"
    )

    print(
        f"Wind speed: "
        f"{baseline['wind_speed']:.1f} km/h"
    )

    print(
        f"Traffic density: "
        f"{baseline['traffic_density']:.4f}"
    )

    print(
        f"Traffic risk: "
        f"{baseline['traffic_risk']}"
    )


    print(
        "\n"
        "===================================="
    )

    print(
        "SIMULATED CITY STATE"
    )

    print(
        "===================================="
    )


    print(
        f"Traffic density: "
        f"{scenario['traffic_density']:.4f}"
    )

    print(
        f"Traffic risk: "
        f"{scenario['traffic_risk']}"
    )

    print(
        f"Temperature: "
        f"{scenario['temperature']:.1f} °C"
    )

    print(
        f"Humidity: "
        f"{scenario['humidity']:.1f} %"
    )

    print(
        f"Wind speed: "
        f"{scenario['wind_speed']:.1f} km/h"
    )


    print(
        "\n"
        "===================================="
    )

    print(
        "SCENARIO IMPACT"
    )

    print(
        "===================================="
    )


    density_change = (
        scenario["traffic_density"]
        - baseline["traffic_density"]
    )


    temperature_change = (
        scenario["temperature"]
        - baseline["temperature"]
    )


    humidity_change = (
        scenario["humidity"]
        - baseline["humidity"]
    )


    wind_change = (
        scenario["wind_speed"]
        - baseline["wind_speed"]
    )


    print(
        f"Traffic density change: "
        f"{density_change:+.4f}"
    )

    print(
        f"Temperature change: "
        f"{temperature_change:+.1f} °C"
    )

    print(
        f"Humidity change: "
        f"{humidity_change:+.1f} %"
    )

    print(
        f"Wind speed change: "
        f"{wind_change:+.1f} km/h"
    )


    print(
        "\nNOTE:"
    )

    print(
        "The current AQI forecast is "
        "not changed by traffic in this "
        "simulation because the current "
        "AQI model does not use traffic "
        "as an input feature."
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print(
        "\n"
        "===================================="
    )

    print(
        "HYDERABAD INTERACTIVE "
        "WHAT-IF SIMULATION"
    )

    print(
        "===================================="
    )


    # ----------------------------------------
    # Load baseline
    # ----------------------------------------

    baseline = (
        load_latest_city_state()
    )


    print(
        "\nBaseline timestamp:",
        baseline["datetime"]
    )


    # ----------------------------------------
    # Current traffic state
    # ----------------------------------------

    current_traffic_density = 0.1721

    current_traffic_risk = (
        get_traffic_risk(
            current_traffic_density
        )
    )


    baseline["traffic_density"] = (
        current_traffic_density
    )

    baseline["traffic_risk"] = (
        current_traffic_risk
    )


    current_weather = {

        "temperature":
            baseline["temperature"],

        "humidity":
            baseline["humidity"],

        "wind_speed":
            baseline["wind_speed"]

    }


    # ========================================
    # USER INPUT
    # ========================================

    print(
        "\nEnter your WHAT-IF scenario."
    )

    print(
        "Press Enter to use the default."
    )

    print(
        "Use positive values for increases."
    )

    print(
        "Use negative values for decreases."
    )


    # ----------------------------------------
    # Traffic
    # ----------------------------------------

    traffic_input = input(
        "\nTraffic change (%) [30]: "
    ).strip()


    if traffic_input == "":

        traffic_change = 30

    else:

        traffic_change = float(
            traffic_input
        )


    # ----------------------------------------
    # Temperature
    # ----------------------------------------

    temperature_input = input(
        "Temperature change (°C) [2]: "
    ).strip()


    if temperature_input == "":

        temperature_change = 2

    else:

        temperature_change = float(
            temperature_input
        )


    # ----------------------------------------
    # Humidity
    # ----------------------------------------

    humidity_input = input(
        "Humidity change (%) [5]: "
    ).strip()


    if humidity_input == "":

        humidity_change = 5

    else:

        humidity_change = float(
            humidity_input
        )


    # ----------------------------------------
    # Wind
    # ----------------------------------------

    wind_input = input(
        "Wind speed change (km/h) [-3]: "
    ).strip()


    if wind_input == "":

        wind_change = -3

    else:

        wind_change = float(
            wind_input
        )


    # ========================================
    # RUN SIMULATION
    # ========================================

    scenario = run_simulation(

        current_density=
            current_traffic_density,

        traffic_change_percent=
            traffic_change,

        current_weather=
            current_weather,

        temperature_change=
            temperature_change,

        humidity_change=
            humidity_change,

        wind_change=
            wind_change
    )


    # ========================================
    # DISPLAY RESULTS
    # ========================================

    display_comparison(
        baseline,
        scenario
    )


    print(
        "\n"
        "===================================="
    )

    print(
        "INTERACTIVE SIMULATION COMPLETE"
    )

    print(
        "===================================="
    )