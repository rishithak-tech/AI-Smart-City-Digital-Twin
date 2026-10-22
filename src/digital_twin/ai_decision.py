import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_city_dqn.pth"
)

# ---------------------------------------------------------
# Allow imports from digital_twin folder
# ---------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from dqn_agent import DQNAgent


# =========================================================
# LOAD REAL HYDERABAD ENVIRONMENTAL STATE
# =========================================================

def load_latest_environmental_state():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Environmental data not found:\n"
            f"{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    # Sort chronologically
    data = data.sort_values(
        "datetime"
    )

    # Get latest available record
    latest = data.iloc[-1]

    state = {

        "datetime":
            latest["datetime"],

        "aqi":
            float(latest["aqi"]),

        "temperature":
            float(latest["temperature"]),

        "humidity":
            float(latest["humidity"]),

        "wind_speed":
            float(latest["wind_speed"])
    }

    return state


# =========================================================
# LOAD REAL TRAFFIC STATE
# =========================================================

def load_traffic_density():

    traffic_model_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "traffic_density_model.pkl"
    )

    traffic_features_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "idd_traffic_features.csv"
    )

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not traffic_model_path.exists():

        raise FileNotFoundError(
            f"Traffic model not found:\n"
            f"{traffic_model_path}"
        )

    if not traffic_features_path.exists():

        raise FileNotFoundError(
            f"Traffic feature data not found:\n"
            f"{traffic_features_path}"
        )

    # -----------------------------------------------------
    # Load model and traffic features
    # -----------------------------------------------------

    import joblib

    traffic_model = joblib.load(
        traffic_model_path
    )

    traffic_data = pd.read_csv(
        traffic_features_path
    )

    # -----------------------------------------------------
    # Use the latest available traffic feature row
    #
    # IMPORTANT:
    # IDD Lite is not timestamp-aligned with
    # Hyderabad AQI/weather data.
    #
    # Therefore this is treated as a
    # scene-based traffic state for simulation,
    # NOT a real-time traffic measurement.
    # -----------------------------------------------------

    latest_traffic = traffic_data.iloc[-1]

    feature_columns = [
        "traffic_pixel_ratio",
        "living_pixel_ratio",
        "drivable_pixel_ratio",
        "construction_pixel_ratio",
        "sky_pixel_ratio"
    ]

    X = pd.DataFrame(
        [[
            latest_traffic[column]
            for column in feature_columns
        ]],
        columns=feature_columns
    )

    predicted_density = (
        traffic_model.predict(X)[0]
    )

    predicted_density = float(
        np.clip(
            predicted_density,
            0.0,
            1.0
        )
    )

    return predicted_density


# =========================================================
# STATE NORMALIZATION
# =========================================================

def normalize_state(state):

    return np.array(
        [
            state["aqi"] / 300.0,
            state["traffic_density"],
            state["temperature"] / 50.0,
            state["humidity"] / 100.0,
            state["wind_speed"] / 30.0
        ],
        dtype=np.float32
    )


# =========================================================
# RISK LEVELS
# =========================================================

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


def get_traffic_risk(traffic_density):

    if traffic_density < 0.10:
        return "Low"

    elif traffic_density < 0.20:
        return "Moderate"

    elif traffic_density < 0.30:
        return "High"

    else:
        return "Very High"


# =========================================================
# LOAD TRAINED DQN
# =========================================================

def load_dqn():

    # The DQN was trained with:
    # 5 state variables
    # 5 possible actions

    state_size = 5

    action_size = 5

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained DQN model not found:\n"
            f"{MODEL_PATH}"
        )

    agent.model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    agent.model.eval()

    # Disable exploration
    agent.epsilon = 0.0

    return agent


# =========================================================
# GENERATE REAL-DATA AI DECISION
# =========================================================

def generate_ai_decision():

    # -----------------------------------------------------
    # Load Hyderabad environmental state
    # -----------------------------------------------------

    environmental_state = (
        load_latest_environmental_state()
    )

    # -----------------------------------------------------
    # Load traffic state
    # -----------------------------------------------------

    traffic_density = (
        load_traffic_density()
    )

    # -----------------------------------------------------
    # Build complete city state
    # -----------------------------------------------------

    state = {

        "datetime":
            environmental_state["datetime"],

        "aqi":
            environmental_state["aqi"],

        "traffic_density":
            traffic_density,

        "temperature":
            environmental_state["temperature"],

        "humidity":
            environmental_state["humidity"],

        "wind_speed":
            environmental_state["wind_speed"]
    }

    # -----------------------------------------------------
    # Load DQN
    # -----------------------------------------------------

    agent = load_dqn()

    # -----------------------------------------------------
    # Normalize state
    # -----------------------------------------------------

    normalized_state = normalize_state(
        state
    )

    state_tensor = torch.tensor(
        normalized_state,
        dtype=torch.float32
    ).unsqueeze(0)

    # -----------------------------------------------------
    # Get Q-values
    # -----------------------------------------------------

    with torch.no_grad():

        q_values = (
            agent.model(
                state_tensor
            )
            .squeeze(0)
            .numpy()
        )

    # -----------------------------------------------------
    # Available actions
    # -----------------------------------------------------

    actions = [
        0,
        -10,
        -20,
        -30,
        -40
    ]

    # -----------------------------------------------------
    # Select best action
    # -----------------------------------------------------

    best_action_index = int(
        np.argmax(q_values)
    )

    best_action = actions[
        best_action_index
    ]

    # -----------------------------------------------------
    # Calculate simulated new traffic density
    # -----------------------------------------------------

    new_traffic_density = (
        traffic_density
        * (1 + best_action / 100)
    )

    new_traffic_density = float(
        np.clip(
            new_traffic_density,
            0.0,
            1.0
        )
    )

    # -----------------------------------------------------
    # Traffic benefit
    # -----------------------------------------------------

    if traffic_density > 0:

        traffic_benefit = (
            traffic_density
            - new_traffic_density
        ) / traffic_density

    else:

        traffic_benefit = 0.0

    # -----------------------------------------------------
    # Intervention cost
    # -----------------------------------------------------

    intervention_strength = (
        abs(best_action) / 100
    )

    intervention_cost = (
        intervention_strength ** 2
    )

    # -----------------------------------------------------
    # Risk levels
    # -----------------------------------------------------

    air_quality_risk = (
        get_air_quality_risk(
            state["aqi"]
        )
    )

    traffic_risk = (
        get_traffic_risk(
            state["traffic_density"]
        )
    )

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    if best_action == 0:

        recommendation = (
            "No traffic intervention is "
            "recommended for the current "
            "simulated state."
        )

    else:

        recommendation = (
            f"Implement approximately "
            f"{abs(best_action)}% traffic "
            f"reduction measures."
        )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    return {

        "city":
            "Hyderabad",

        "datetime":
            state["datetime"],

        "aqi":
            state["aqi"],

        "traffic_density":
            state["traffic_density"],

        "temperature":
            state["temperature"],

        "humidity":
            state["humidity"],

        "wind_speed":
            state["wind_speed"],

        "air_quality_risk":
            air_quality_risk,

        "traffic_risk":
            traffic_risk,

        "recommended_intervention":
            best_action,

        "recommendation":
            recommendation,

        "q_values":
            q_values.tolist(),

        "traffic_density_after":
            new_traffic_density,

        "traffic_benefit":
            traffic_benefit,

        "intervention_cost":
            intervention_cost
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SMART CITY AI DECISION ENGINE")
    print("=" * 70)

    decision = generate_ai_decision()

    # -----------------------------------------------------
    # Current city state
    # -----------------------------------------------------

    print()
    print("CURRENT HYDERABAD CITY STATE")
    print("-" * 50)

    print(
        f"Data timestamp:    "
        f"{decision['datetime']}"
    )

    print(
        f"City:              "
        f"{decision['city']}"
    )

    print(
        f"AQI:               "
        f"{decision['aqi']:.2f}"
    )

    print(
        f"Traffic density:   "
        f"{decision['traffic_density']:.4f}"
    )

    print(
        f"Temperature:       "
        f"{decision['temperature']:.2f} °C"
    )

    print(
        f"Humidity:          "
        f"{decision['humidity']:.2f}%"
    )

    print(
        f"Wind speed:        "
        f"{decision['wind_speed']:.2f} km/h"
    )

    print(
        f"Air quality risk:  "
        f"{decision['air_quality_risk']}"
    )

    print(
        f"Traffic risk:      "
        f"{decision['traffic_risk']}"
    )

    # -----------------------------------------------------
    # AI recommendation
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("🤖 AI RECOMMENDATION")
    print("=" * 70)

    print()

    print(
        decision["recommendation"]
    )

    print()

    print(
        f"Recommended action: "
        f"{decision['recommended_intervention']}%"
    )

    # -----------------------------------------------------
    # Simulated impact
    # -----------------------------------------------------

    print()
    print("EXPECTED SIMULATED IMPACT")
    print("-" * 50)

    print(
        f"Traffic density: "
        f"{decision['traffic_density']:.4f}"
        f" → "
        f"{decision['traffic_density_after']:.4f}"
    )

    print(
        f"Traffic benefit: "
        f"{decision['traffic_benefit']:.4f}"
    )

    print(
        f"Intervention cost: "
        f"{decision['intervention_cost']:.4f}"
    )

    # -----------------------------------------------------
    # Q-values
    # -----------------------------------------------------

    print()
    print("DQN ACTION VALUES")
    print("-" * 50)

    actions = [
        0,
        -10,
        -20,
        -30,
        -40
    ]

    for action, q_value in zip(
        actions,
        decision["q_values"]
    ):

        print(
            f"{action:>4}% intervention"
            f" → Q-value: "
            f"{q_value:.4f}"
        )

    print()
    print("=" * 70)
    print("REAL-DATA AI DECISION MODULE READY")
    print("=" * 70)