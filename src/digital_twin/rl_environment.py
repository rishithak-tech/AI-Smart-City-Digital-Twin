import sys
from pathlib import Path

# Allow importing impact_model.py
sys.path.append(
    str(Path(__file__).resolve().parent)
)

from impact_model import CityImpactModel


class SmartCityEnvironment:

    def __init__(self):

        # -------------------------------------------------
        # INITIAL CITY STATE
        # -------------------------------------------------

        self.initial_aqi = 162.0

        self.initial_traffic_density = 0.1700

        self.initial_temperature = 17.5

        self.initial_humidity = 88.0

        self.initial_wind_speed = 8.9

        # -------------------------------------------------
        # AVAILABLE RL ACTIONS
        # -------------------------------------------------

        self.actions = [
            0,
            -10,
            -20,
            -30,
            -40
        ]

        # -------------------------------------------------
        # IMPACT MODEL
        # -------------------------------------------------

        self.impact_model = CityImpactModel()

        # -------------------------------------------------
        # EPISODE SETTINGS
        # -------------------------------------------------

        self.max_steps = 10

        # Start environment
        self.reset()

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset the city to its initial state.
        """

        self.aqi = self.initial_aqi

        self.traffic_density = (
            self.initial_traffic_density
        )

        self.temperature = (
            self.initial_temperature
        )

        self.humidity = (
            self.initial_humidity
        )

        self.wind_speed = (
            self.initial_wind_speed
        )

        self.current_step = 0

        return self.get_state()

    # =====================================================
    # GET STATE
    # =====================================================

    def get_state(self):
        """
        Return the current city state.
        """

        return [
            self.aqi,
            self.traffic_density,
            self.temperature,
            self.humidity,
            self.wind_speed
        ]

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):
        """
        Apply an intervention and move the city
        to the next state.
        """

        # Check whether action is valid
        if action not in self.actions:

            raise ValueError(
                f"Invalid action: {action}. "
                f"Choose from {self.actions}"
            )

        # -------------------------------------------------
        # EVALUATE CURRENT STATE
        # -------------------------------------------------

        result = (
            self.impact_model.evaluate_scenario(
                traffic_change_percent=action,
                aqi=self.aqi,
                humidity=self.humidity,
                wind_speed=self.wind_speed,
                current_traffic_density=(
                    self.traffic_density
                )
            )
        )

        # -------------------------------------------------
        # EXTRACT IMPACT VALUES
        # -------------------------------------------------

        environmental_improvement = (
            result["environmental_improvement"]
        )

        traffic_benefit = (
            result["traffic_benefit"]
        )

        intervention_cost = (
            result["intervention_cost"]
        )

        # -------------------------------------------------
        # CALCULATE RL REWARD
        # -------------------------------------------------

        reward = (
            5.0 * environmental_improvement
            + 2.0 * traffic_benefit
            - 2.0 * intervention_cost
        )

        # -------------------------------------------------
        # UPDATE TRAFFIC STATE
        # -------------------------------------------------

        self.traffic_density = (
            result["new_traffic_density"]
        )

        # -------------------------------------------------
        # SIMULATE NATURAL CITY CHANGES
        # -------------------------------------------------

        # Small AQI variation
        self.aqi = max(
            0.0,
            self.aqi
            + 2.0 * (
                0.5
                - self.traffic_density
            )
        )

        # Temperature changes gradually
        self.temperature += 0.2

        # Humidity changes gradually
        self.humidity = max(
            0.0,
            min(
                100.0,
                self.humidity - 0.5
            )
        )

        # Wind speed changes gradually
        self.wind_speed = max(
            0.0,
            self.wind_speed + 0.1
        )

        # -------------------------------------------------
        # UPDATE STEP
        # -------------------------------------------------

        self.current_step += 1

        # Episode ends after max_steps
        done = (
            self.current_step
            >= self.max_steps
        )

        return (
            self.get_state(),
            reward,
            done,
            result
        )


# =========================================================
# TEST THE ENVIRONMENT
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART CITY RL ENVIRONMENT")
    print("=" * 60)

    # Create environment
    env = SmartCityEnvironment()

    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    print("\nInitial city state:")

    print(env.get_state())

    # -----------------------------------------------------
    # SAMPLE EPISODE
    # -----------------------------------------------------

    print("\nRunning sample episode...")

    actions = [
        -10,
        -20,
        -30,
        -20,
        0
    ]

    total_reward = 0.0

    for action in actions:

        new_state, reward, done, result = (
            env.step(action)
        )

        total_reward += reward

        print()
        print(
            f"Step: {env.current_step}"
        )

        print(
            f"Action: "
            f"{action}% traffic intervention"
        )

        print(
            f"Traffic density: "
            f"{result['baseline_traffic_density']:.4f}"
            f" → "
            f"{result['new_traffic_density']:.4f}"
        )

        print(
            f"AQI: "
            f"{env.aqi:.2f}"
        )

        print(
            f"Temperature: "
            f"{env.temperature:.2f} °C"
        )

        print(
            f"Humidity: "
            f"{env.humidity:.2f}%"
        )

        print(
            f"Wind speed: "
            f"{env.wind_speed:.2f} km/h"
        )

        print(
            f"Environmental improvement: "
            f"{result['environmental_improvement']:.4f}"
        )

        print(
            f"Traffic benefit: "
            f"{result['traffic_benefit']:.4f}"
        )

        print(
            f"Intervention cost: "
            f"{result['intervention_cost']:.4f}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )

        if done:

            print(
                "\nEpisode finished."
            )

            break

    # -----------------------------------------------------
    # TOTAL REWARD
    # -----------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"TOTAL EPISODE REWARD: "
        f"{total_reward:.4f}"
    )

    print("=" * 60)

    print(
        "\nEnvironment test complete."
    )