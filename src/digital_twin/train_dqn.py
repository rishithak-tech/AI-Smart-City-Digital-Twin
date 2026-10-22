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
    Normalize city-state values before giving them
    to the neural network.

    State:
    [AQI, traffic_density, temperature,
     humidity, wind_speed]
    """

    aqi = state[0] / 300.0

    traffic = state[1]

    temperature = state[2] / 50.0

    humidity = state[3] / 100.0

    wind_speed = state[4] / 30.0

    return np.array(
        [
            aqi,
            traffic,
            temperature,
            humidity,
            wind_speed
        ],
        dtype=np.float32
    )


# =========================================================
# TRAINING
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART CITY DQN TRAINING")
    print("=" * 60)

    # -----------------------------------------------------
    # Create environment
    # -----------------------------------------------------

    env = SmartCityEnvironment()

    # -----------------------------------------------------
    # Determine state/action sizes
    # -----------------------------------------------------

    state_size = len(
        env.get_state()
    )

    action_size = len(
        env.actions
    )

    print()
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    # -----------------------------------------------------
    # Create DQN agent
    # -----------------------------------------------------

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    # -----------------------------------------------------
    # Training settings
    # -----------------------------------------------------

    episodes = 300

    target_update_frequency = 10

    rewards_history = []

    loss_history = []

    print()
    print(f"Training episodes: {episodes}")
    print()

    # =====================================================
    # EPISODE LOOP
    # =====================================================

    for episode in range(1, episodes + 1):

        # Reset environment
        state = env.reset()

        state = normalize_state(state)

        total_reward = 0.0

        episode_losses = []

        done = False

        # -------------------------------------------------
        # STEP LOOP
        # -------------------------------------------------

        while not done:

            # Choose action
            action_index = agent.choose_action(
                state
            )

            # Convert action index to actual intervention
            actual_action = env.actions[
                action_index
            ]

            # Environment step
            next_state, reward, done, result = (
                env.step(actual_action)
            )

            # Normalize next state
            next_state = normalize_state(
                next_state
            )

            # Store experience
            agent.remember(
                state,
                action_index,
                reward,
                next_state,
                done
            )

            # Train neural network
            loss = agent.replay()

            if loss is not None:

                episode_losses.append(
                    loss
                )

            # Move to next state
            state = next_state

            total_reward += reward

        # -------------------------------------------------
        # Decay exploration
        # -------------------------------------------------

        agent.decay_epsilon()

        # -------------------------------------------------
        # Update target network
        # -------------------------------------------------

        if episode % target_update_frequency == 0:

            agent.update_target_network()

        # -------------------------------------------------
        # Store statistics
        # -------------------------------------------------

        rewards_history.append(
            total_reward
        )

        if episode_losses:

            average_loss = (
                sum(episode_losses)
                / len(episode_losses)
            )

            loss_history.append(
                average_loss
            )

        else:

            average_loss = 0.0

        # -------------------------------------------------
        # Print progress
        # -------------------------------------------------

        if episode == 1 or episode % 10 == 0:

            print(
                f"Episode {episode:3d} | "
                f"Reward: {total_reward:8.4f} | "
                f"Loss: {average_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "smart_city_dqn.pth"
    )

    torch.save(
        agent.model.state_dict(),
        model_path
    )

    # =====================================================
    # SAVE TRAINING HISTORY
    # =====================================================

    history_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "dqn_training_history.npz"
    )

    np.savez(
        history_path,
        rewards=np.array(
            rewards_history
        ),
        losses=np.array(
            loss_history
        )
    )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("DQN TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Final epsilon: "
        f"{agent.epsilon:.4f}"
    )

    print(
        f"Best episode reward: "
        f"{max(rewards_history):.4f}"
    )

    print(
        f"Average reward: "
        f"{np.mean(rewards_history):.4f}"
    )

    print()
    print(
        f"Model saved to:"
    )

    print(model_path)

    print()
    print(
        f"Training history saved to:"
    )

    print(history_path)

    print()
    print("=" * 60)