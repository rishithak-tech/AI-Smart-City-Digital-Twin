import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)

DQN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_city_dqn.pth"
)

TRAFFIC_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "digital_twin"
    / "traffic_state.py"
)

# ---------------------------------------------------------
# Import DQN
# ---------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from dqn_agent import DQNAgent


# =========================================================
# LOAD ENVIRONMENTAL DATA
# =========================================================

def load_latest_environment():

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
# LOAD TRAFFIC STATE
# =========================================================

def load_traffic_density():

    print()
    print("Running U-Net + traffic model pipeline...")

    result = subprocess.run(
        [
            sys.executable,
            str(TRAFFIC_SCRIPT)
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print(result.stdout)
        print(result.stderr)

        raise RuntimeError(
            "Traffic state pipeline failed."
        )

    match = re.search(
        r"Predicted traffic density:\s*([0-9.]+)",
        result.stdout
    )

    if match is None:

        print(result.stdout)

        raise RuntimeError(
            "Could not extract traffic density "
            "from traffic_state.py."
        )

    density = float(
        match.group(1)
    )

    print(
        f"Traffic density: {density:.4f}"
    )

    return density


# =========================================================
# NORMALIZE STATE
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
# ENVIRONMENTAL PRESSURE
# =========================================================

def environmental_pressure(
    aqi,
    traffic_density,
    humidity,
    wind_speed
):

    aqi_factor = min(
        aqi / 300.0,
        1.0
    )

    traffic_factor = traffic_density

    humidity_factor = humidity / 100.0

    wind_factor = max(
        0.0,
        1.0 - wind_speed / 30.0
    )

    pressure = (
        0.50 * aqi_factor
        + 0.30 * traffic_factor
        + 0.10 * humidity_factor
        + 0.10 * wind_factor
    )

    return pressure


# =========================================================
# SIMULATE SCENARIO
# =========================================================

def simulate_scenario(
    action,
    aqi,
    traffic_density,
    humidity,
    wind_speed
):

    # -----------------------------------------------------
    # New traffic density
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Traffic benefit
    # -----------------------------------------------------

    if traffic_density > 0:

        traffic_benefit = (
            traffic_density
            - new_density
        ) / traffic_density

    else:

        traffic_benefit = 0.0

    # -----------------------------------------------------
    # Intervention cost
    # -----------------------------------------------------

    intervention_strength = (
        abs(action) / 100.0
    )

    intervention_cost = (
        intervention_strength ** 2
    )

    # -----------------------------------------------------
    # Environmental pressure
    # -----------------------------------------------------

    baseline_pressure = (
        environmental_pressure(
            aqi,
            traffic_density,
            humidity,
            wind_speed
        )
    )

    scenario_pressure = (
        environmental_pressure(
            aqi,
            new_density,
            humidity,
            wind_speed
        )
    )

    environmental_improvement = (
        baseline_pressure
        - scenario_pressure
    )

    # -----------------------------------------------------
    # Same reward structure used by RL environment
    # -----------------------------------------------------

    reward = (
        5.0 * environmental_improvement
        + 2.0 * traffic_benefit
        - 2.0 * intervention_cost
    )

    return {
        "action": action,
        "new_density": new_density,
        "traffic_benefit": traffic_benefit,
        "environmental_improvement":
            environmental_improvement,
        "intervention_cost":
            intervention_cost,
        "reward": reward
    }


# =========================================================
# LOAD TRAINED DQN
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
    print("SMART CITY WHAT-IF DECISION SIMULATOR")
    print("=" * 70)

    # -----------------------------------------------------
    # Load current city state
    # -----------------------------------------------------

    environment = (
        load_latest_environment()
    )

    traffic_density = (
        load_traffic_density()
    )

    aqi = environment["aqi"]
    temperature = environment["temperature"]
    humidity = environment["humidity"]
    wind_speed = environment["wind_speed"]

    # -----------------------------------------------------
    # Display current state
    # -----------------------------------------------------

    print()
    print("CURRENT HYDERABAD STATE")
    print("-" * 50)

    print(
        f"Timestamp:         "
        f"{environment['datetime']}"
    )

    print(
        f"AQI:               "
        f"{aqi:.2f}"
    )

    print(
        f"Traffic density:   "
        f"{traffic_density:.4f}"
    )

    print(
        f"Temperature:       "
        f"{temperature:.2f} °C"
    )

    print(
        f"Humidity:          "
        f"{humidity:.2f}%"
    )

    print(
        f"Wind speed:        "
        f"{wind_speed:.2f} km/h"
    )

    # -----------------------------------------------------
    # Evaluate all interventions
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

        result = simulate_scenario(
            action=action,
            aqi=aqi,
            traffic_density=traffic_density,
            humidity=humidity,
            wind_speed=wind_speed
        )

        scenarios.append(
            result
        )

    # -----------------------------------------------------
    # Display scenario comparison
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("WHAT-IF SCENARIO COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Action':>10}"
        f"{'Traffic':>15}"
        f"{'Benefit':>12}"
        f"{'Env. Imp.':>14}"
        f"{'Cost':>10}"
        f"{'Reward':>12}"
    )

    print("-" * 75)

    for scenario in scenarios:

        print(
            f"{scenario['action']:>9}%"
            f"{scenario['new_density']:>15.4f}"
            f"{scenario['traffic_benefit']:>12.4f}"
            f"{scenario['environmental_improvement']:>14.4f}"
            f"{scenario['intervention_cost']:>10.4f}"
            f"{scenario['reward']:>12.4f}"
        )

    # -----------------------------------------------------
    # DQN prediction
    # -----------------------------------------------------

    agent = load_dqn()

    normalized_state = normalize_state(
        aqi=aqi,
        traffic_density=traffic_density,
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed
    )

    state_tensor = torch.tensor(
        normalized_state,
        dtype=torch.float32
    ).unsqueeze(0)

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

    dqn_action = actions[
        best_index
    ]

    # -----------------------------------------------------
    # Best immediate scenario
    # -----------------------------------------------------

    best_scenario = max(
        scenarios,
        key=lambda x: x["reward"]
    )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("AI DECISION")
    print("=" * 70)

    print()

    print(
        f"DQN recommended action: "
        f"{dqn_action}%"
    )

    print(
        f"DQN Q-value: "
        f"{q_values[best_index]:.4f}"
    )

    print()

    print(
        f"Highest immediate simulated reward: "
        f"{best_scenario['action']}%"
    )

    print(
        f"Immediate reward: "
        f"{best_scenario['reward']:.4f}"
    )

    # -----------------------------------------------------
    # Q-value table
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
    # Interpretation
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DECISION INTERPRETATION")
    print("=" * 70)

    if dqn_action == best_scenario["action"]:

        print()
        print(
            "The DQN recommendation matches "
            "the highest immediate simulated "
            "reward scenario."
        )

    else:

        print()
        print(
            "The DQN recommendation differs "
            "from the highest immediate reward."
        )

        print(
            "This demonstrates the difference "
            "between immediate reward optimization "
            "and learned long-term value."
        )

    print()
    print(
        "Recommended intervention:"
    )

    print(
        f"→ {dqn_action}% traffic reduction"
    )

    print()
    print("=" * 70)
    print("WHAT-IF SIMULATOR COMPLETE")
    print("=" * 70)