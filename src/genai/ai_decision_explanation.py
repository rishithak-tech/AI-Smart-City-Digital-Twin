# =========================================================
# SMART CITY AI DECISION + EXPLANATION
# =========================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)

TRAFFIC_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "idd_traffic_features.csv"
)

TRAFFIC_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic_density_model.pkl"
)

DQN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_city_dqn.pth"
)


# =========================================================
# IMPORT DQN
# =========================================================

sys.path.append(
    str(PROJECT_ROOT / "src" / "digital_twin")
)

from dqn_agent import DQNAgent


# =========================================================
# IMPORT AI EXPLAINER
# =========================================================

sys.path.append(
    str(PROJECT_ROOT / "src" / "genai")
)

from ai_explainer import SmartCityAIExplainer


# =========================================================
# LOAD LATEST CITY DATA
# =========================================================

def load_city_state():

    data = pd.read_csv(DATA_PATH)

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.sort_values(
        "datetime"
    )

    latest = data.iloc[-1]

    return {
        "datetime": latest["datetime"],
        "aqi": float(latest["aqi"]),
        "temperature": float(
            latest["temperature"]
        ),
        "humidity": float(
            latest["humidity"]
        ),
        "wind_speed": float(
            latest["wind_speed"]
        )
    }


# =========================================================
# LOAD TRAFFIC DENSITY
# =========================================================

def load_traffic_density():

    import joblib

    model = joblib.load(
        TRAFFIC_MODEL_PATH
    )

    data = pd.read_csv(
        TRAFFIC_FEATURES_PATH
    )

    latest = data.iloc[-1]

    feature_columns = [
        "traffic_pixel_ratio",
        "living_pixel_ratio",
        "drivable_pixel_ratio",
        "construction_pixel_ratio",
        "sky_pixel_ratio"
    ]

    X = pd.DataFrame(
        [[
            latest[column]
            for column in feature_columns
        ]],
        columns=feature_columns
    )

    density = model.predict(X)[0]

    return float(
        np.clip(
            density,
            0.0,
            1.0
        )
    )


# =========================================================
# NORMALIZE DQN STATE
# =========================================================

def normalize_state(
    aqi,
    traffic_density,
    temperature,
    humidity,
    wind_speed
):

    return np.array(
        [
            aqi / 300.0,
            traffic_density,
            temperature / 50.0,
            humidity / 100.0,
            wind_speed / 30.0
        ],
        dtype=np.float32
    )


# =========================================================
# WHAT-IF SCENARIO
# =========================================================

def simulate_scenario(
    action,
    traffic_density
):

    new_density = (
        traffic_density
        * (1 + action / 100.0)
    )

    new_density = float(
        np.clip(
            new_density,
            0.0,
            1.0
        )
    )

    if traffic_density > 0:

        traffic_benefit = (
            traffic_density - new_density
        ) / traffic_density

    else:

        traffic_benefit = 0.0

    intervention_cost = (
        abs(action) / 100.0
    ) ** 2

    return {
        "action": action,
        "new_density": new_density,
        "traffic_benefit": traffic_benefit,
        "intervention_cost": intervention_cost
    }


# =========================================================
# LOAD DQN
# =========================================================

def load_dqn():

    agent = DQNAgent(
        state_size=5,
        action_size=5
    )

    agent.model.load_state_dict(
        torch.load(
            DQN_PATH,
            map_location="cpu"
        )
    )

    agent.model.eval()

    return agent


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "SMART CITY DIGITAL TWIN + AI EXPLANATION"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load city state
    # -----------------------------------------------------

    city = load_city_state()

    # -----------------------------------------------------
    # 2. Load traffic state
    # -----------------------------------------------------

    traffic_density = load_traffic_density()

    aqi = city["aqi"]
    temperature = city["temperature"]
    humidity = city["humidity"]
    wind_speed = city["wind_speed"]

    # -----------------------------------------------------
    # 3. Display current state
    # -----------------------------------------------------

    print()
    print("CURRENT CITY STATE")
    print("-" * 50)

    print(
        f"Timestamp:       {city['datetime']}"
    )

    print(
        f"AQI:             {aqi:.2f}"
    )

    print(
        f"Traffic density: {traffic_density:.4f}"
    )

    print(
        f"Temperature:     {temperature:.2f} °C"
    )

    print(
        f"Humidity:        {humidity:.2f}%"
    )

    print(
        f"Wind speed:      {wind_speed:.2f} km/h"
    )

    # -----------------------------------------------------
    # 4. What-if scenarios
    # -----------------------------------------------------

    actions = [
        0,
        -10,
        -20,
        -30,
        -40
    ]

    scenarios = []

    for action in actions:

        scenario = simulate_scenario(
            action,
            traffic_density
        )

        scenarios.append(
            scenario
        )

    # -----------------------------------------------------
    # 5. Load trained DQN
    # -----------------------------------------------------

    agent = load_dqn()

    state = normalize_state(
        aqi,
        traffic_density,
        temperature,
        humidity,
        wind_speed
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    # -----------------------------------------------------
    # 6. Get DQN Q-values
    # -----------------------------------------------------

    with torch.no_grad():

        q_values = (
            agent.model(
                state_tensor
            )
            .squeeze(0)
            .numpy()
        )

    best_index = int(
        np.argmax(q_values)
    )

    recommended_action = actions[
        best_index
    ]

    # -----------------------------------------------------
    # 7. Find recommended scenario
    # -----------------------------------------------------

    recommended_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["action"]
        == recommended_action
    )

    # -----------------------------------------------------
    # 8. Display AI decision
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DQN DECISION")
    print("=" * 70)

    print()

    print(
        f"Recommended action: "
        f"{recommended_action}%"
    )

    print(
        f"Recommended traffic density: "
        f"{recommended_scenario['new_density']:.4f}"
    )

    print(
        f"Traffic benefit: "
        f"{recommended_scenario['traffic_benefit']:.2f}"
    )

    print(
        f"Intervention cost: "
        f"{recommended_scenario['intervention_cost']:.4f}"
    )

    # -----------------------------------------------------
    # 9. Display Q-values
    # -----------------------------------------------------

    print()
    print("DQN Q-VALUES")
    print("-" * 50)

    for action, q_value in zip(
        actions,
        q_values
    ):

        print(
            f"{action:>4}% intervention"
            f" → {q_value:.4f}"
        )

    # -----------------------------------------------------
    # 10. Generate AI explanation
    # -----------------------------------------------------

    explainer = SmartCityAIExplainer(
        aqi=aqi,
        traffic_density=traffic_density,
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        recommended_action=recommended_action,
        traffic_benefit=
            recommended_scenario[
                "traffic_benefit"
            ],
        intervention_cost=
            recommended_scenario[
                "intervention_cost"
            ]
    )

    explanation = (
        explainer.generate_explanation()
    )

    # -----------------------------------------------------
    # 11. Display explanation
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("AI-GENERATED DECISION EXPLANATION")
    print("=" * 70)

    print()
    print(explanation)

    print()
    print("=" * 70)
    print("INTEGRATED AI DECISION SYSTEM COMPLETE")
    print("=" * 70)