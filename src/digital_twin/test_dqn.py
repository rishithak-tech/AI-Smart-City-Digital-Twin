import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------
# Allow imports from digital_twin folder
# ---------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from rl_environment import SmartCityEnvironment
from dqn_agent import DQNAgent


# =========================================================
# STATE NORMALIZATION
# =========================================================

def normalize_state(state):
    """
    Normalize city-state values before passing them
    to the DQN.
    """

    return np.array(
        [
            state[0] / 300.0,   # AQI
            state[1],           # Traffic density
            state[2] / 50.0,    # Temperature
            state[3] / 100.0,   # Humidity
            state[4] / 30.0     # Wind speed
        ],
        dtype=np.float32
    )


# =========================================================
# TEST TRAINED DQN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART CITY TRAINED DQN EVALUATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Create environment
    # -----------------------------------------------------

    env = SmartCityEnvironment()

    # -----------------------------------------------------
    # State and action sizes
    # -----------------------------------------------------

    state_size = len(
        env.get_state()
    )

    action_size = len(
        env.actions
    )

    # -----------------------------------------------------
    # Create agent
    # -----------------------------------------------------

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    model_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "smart_city_dqn.pth"
    )

    agent.model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    # Turn off exploration
    agent.epsilon = 0.0

    agent.model.eval()

    print()
    print("Trained DQN loaded successfully.")

    # -----------------------------------------------------
    # Current city state
    # -----------------------------------------------------

    state = env.reset()

    print()
    print("CURRENT CITY STATE")
    print("-" * 40)

    print(
        f"AQI:              {state[0]:.2f}"
    )

    print(
        f"Traffic density:  {state[1]:.4f}"
    )

    print(
        f"Temperature:      {state[2]:.2f} °C"
    )

    print(
        f"Humidity:         {state[3]:.2f}%"
    )

    print(
        f"Wind speed:       {state[4]:.2f} km/h"
    )

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

        q_values = agent.model(
            state_tensor
        ).squeeze(0).numpy()

    # -----------------------------------------------------
    # Display Q-values
    # -----------------------------------------------------

    print()
    print("DQN ACTION VALUES")
    print("-" * 40)

    for index, action in enumerate(
        env.actions
    ):

        print(
            f"{action:>4}% intervention"
            f"  → Q-value: "
            f"{q_values[index]:.4f}"
        )

    # -----------------------------------------------------
    # Select best action
    # -----------------------------------------------------

    best_action_index = int(
        np.argmax(q_values)
    )

    best_action = env.actions[
        best_action_index
    ]

    print()
    print("=" * 60)

    print(
        "🤖 DQN RECOMMENDATION"
    )

    print("=" * 60)

    print()
    print(
        f"Recommended intervention: "
        f"{best_action}%"
    )

    print(
        f"Highest Q-value: "
        f"{q_values[best_action_index]:.4f}"
    )

    # -----------------------------------------------------
    # Evaluate selected action
    # -----------------------------------------------------

    new_state, reward, done, result = (
        env.step(best_action)
    )

    print()
    print("RECOMMENDED ACTION IMPACT")
    print("-" * 40)

    print(
        f"Traffic density: "
        f"{result['baseline_traffic_density']:.4f}"
        f" → "
        f"{result['new_traffic_density']:.4f}"
    )

    print(
        f"Traffic benefit: "
        f"{result['traffic_benefit']:.4f}"
    )

    print(
        f"Environmental improvement: "
        f"{result['environmental_improvement']:.4f}"
    )

    print(
        f"Intervention cost: "
        f"{result['intervention_cost']:.4f}"
    )

    print(
        f"Reward: "
        f"{reward:.4f}"
    )

    print()
    print("=" * 60)
    print("DQN EVALUATION COMPLETE")
    print("=" * 60)