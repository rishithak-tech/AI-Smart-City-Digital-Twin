from pathlib import Path

import pandas as pd

from city_state import CityState


# ============================================
# DATA PATH
# ============================================

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)


# ============================================
# LOAD CITY DATA
# ============================================

def load_city_data():

    data = pd.read_csv(
        DATA_PATH
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.sort_values(
        "datetime"
    )

    return data


# ============================================
# DETERMINE AIR QUALITY RISK
# ============================================

def get_air_quality_risk(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================
# DETERMINE TRAFFIC RISK
# ============================================

def get_traffic_risk(traffic_density):

    if traffic_density is None:
        return "Unknown"

    if traffic_density < 0.05:
        return "Low"

    elif traffic_density < 0.15:
        return "Moderate"

    elif traffic_density < 0.30:
        return "High"

    else:
        return "Very High"


# ============================================
# CREATE CITY STATE
# ============================================

def create_city_state(row):

    state = CityState()

    # ----------------------------------------
    # Air Quality
    # ----------------------------------------

    state.current_aqi = float(
        row["aqi"]
    )

    state.predicted_aqi_next_hour = float(
        row["target_aqi_next_hour"]
    )


    # ----------------------------------------
    # Weather
    # ----------------------------------------

    state.temperature = float(
        row["temperature"]
    )

    state.humidity = float(
        row["humidity"]
    )

    state.wind_speed = float(
        row["wind_speed"]
    )

    state.wind_direction = float(
        row["wind_direction"]
    )

    state.surface_pressure = float(
        row["surface_pressure"]
    )


    # ----------------------------------------
    # Traffic
    # ----------------------------------------

    state.traffic_density_score = None

    state.vehicle_pixel_ratio = None


    # ----------------------------------------
    # Risk Levels
    # ----------------------------------------

    state.air_quality_risk = (
        get_air_quality_risk(
            state.current_aqi
        )
    )

    state.traffic_risk = (
        get_traffic_risk(
            state.traffic_density_score
        )
    )


    return state


# ============================================
# TEST THE DIGITAL TWIN STATE
# ============================================

if __name__ == "__main__":

    print("\n====================================")
    print("SMART CITY DIGITAL TWIN")
    print("====================================")


    # Load data

    data = load_city_data()

    print(
        "\nLoaded records:",
        len(data)
    )


    # Select latest available record

    latest_row = data.iloc[-1]


    print(
        "\nLatest timestamp:",
        latest_row["datetime"]
    )


    # Create city state

    city_state = create_city_state(
        latest_row
    )


    # Display state

    print("\n------------------------------------")
    print("CURRENT CITY STATE")
    print("------------------------------------")


    for key, value in city_state.to_dict().items():

        print(
            f"{key}: {value}"
        )


    print("\n====================================")
    print("DIGITAL TWIN STATE CREATED")
    print("====================================")