import sys
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow importing the RL environment
sys.path.append(
    str(Path(__file__).resolve().parent)
)

from rl_environment import SmartCityEnvironment


# =========================================================
# DQN NETWORK
# =========================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):

        return self.network(state)


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size
        self.action_size = action_size

        # ---------------------------------------------
        # Hyperparameters
        # ---------------------------------------------

        self.gamma = 0.95

        self.learning_rate = 0.001

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        self.batch_size = 32

        self.memory_size = 5000

        # ---------------------------------------------
        # Experience replay memory
        # ---------------------------------------------

        self.memory = []

        # ---------------------------------------------
        # Neural network
        # ---------------------------------------------

        self.model = DQN(
            state_size,
            action_size
        )

        self.target_model = DQN(
            state_size,
            action_size
        )

        # Initially copy model weights
        self.update_target_network()

        # ---------------------------------------------
        # Optimizer
        # ---------------------------------------------

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Loss function
        self.loss_function = nn.MSELoss()

    # =====================================================
    # TARGET NETWORK
    # =====================================================

    def update_target_network(self):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    # =====================================================
    # REMEMBER EXPERIENCE
    # =====================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(experience)

        # Keep memory size limited
        if len(self.memory) > self.memory_size:

            self.memory.pop(0)

    # =====================================================
    # CHOOSE ACTION
    # =====================================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        return torch.argmax(
            q_values,
            dim=1
        ).item()

    # =====================================================
    # TRAIN FROM EXPERIENCE
    # =====================================================

    def replay(self):

        if len(self.memory) < self.batch_size:

            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states = torch.tensor(
            np.array([
                experience[0]
                for experience in batch
            ]),
            dtype=torch.float32
        )

        actions = torch.tensor(
            [
                experience[1]
                for experience in batch
            ],
            dtype=torch.long
        )

        rewards = torch.tensor(
            [
                experience[2]
                for experience in batch
            ],
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array([
                experience[3]
                for experience in batch
            ]),
            dtype=torch.float32
        )

        dones = torch.tensor(
            [
                experience[4]
                for experience in batch
            ],
            dtype=torch.float32
        )

        # Current Q-values
        current_q_values = self.model(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Next Q-values
        with torch.no_grad():

            next_q_values = self.target_model(
                next_states
            ).max(
                dim=1
            )[0]

        # Bellman target
        target_q_values = (
            rewards
            + self.gamma
            * next_q_values
            * (1 - dones)
        )

        # Calculate loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    # =====================================================
    # EPSILON DECAY
    # =====================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# =========================================================
# TEST DQN AGENT
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART CITY DQN AGENT")
    print("=" * 60)

    # Create environment
    env = SmartCityEnvironment()

    # State size
    state_size = len(
        env.get_state()
    )

    # Number of actions
    action_size = len(
        env.actions
    )

    print()
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    # Create agent
    agent = DQNAgent(
        state_size,
        action_size
    )

    print()
    print("DQN network:")
    print(agent.model)

    print()
    print(
        f"Initial epsilon: "
        f"{agent.epsilon:.4f}"
    )

    # Test action selection
    state = env.reset()

    action_index = agent.choose_action(
        state
    )

    actual_action = env.actions[
        action_index
    ]

    print()
    print(
        f"Selected action index: "
        f"{action_index}"
    )

    print(
        f"Selected traffic intervention: "
        f"{actual_action}%"
    )

    # Take one environment step
    next_state, reward, done, result = (
        env.step(actual_action)
    )

    # Store experience
    agent.remember(
        state,
        action_index,
        reward,
        next_state,
        done
    )

    print()
    print(
        f"Reward received: "
        f"{reward:.4f}"
    )

    print(
        f"Memory size: "
        f"{len(agent.memory)}"
    )

    print()
    print("=" * 60)
    print("DQN AGENT TEST COMPLETE")
    print("=" * 60)