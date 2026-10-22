# =========================================================
# SMART CITY AI EXPLAINER
# =========================================================

class SmartCityAIExplainer:
    """
    Generates a human-readable explanation of the
    Smart City Digital Twin's AI recommendation.
    """

    def __init__(
        self,
        aqi,
        traffic_density,
        temperature,
        humidity,
        wind_speed,
        recommended_action,
        traffic_benefit,
        intervention_cost
    ):
        self.aqi = aqi
        self.traffic_density = traffic_density
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.recommended_action = recommended_action
        self.traffic_benefit = traffic_benefit
        self.intervention_cost = intervention_cost

    # -----------------------------------------------------
    # AQI explanation
    # -----------------------------------------------------

    def explain_aqi(self):

        if self.aqi <= 50:
            return "Air quality is currently in the Good range."

        elif self.aqi <= 100:
            return (
                "Air quality is currently in the Satisfactory range."
            )

        elif self.aqi <= 200:
            return (
                "Air quality is currently elevated, "
                "indicating increased pollution pressure."
            )

        elif self.aqi <= 300:
            return (
                "Air quality is currently poor, "
                "indicating significant pollution pressure."
            )

        else:
            return (
                "Air quality is currently extremely poor, "
                "indicating very high pollution pressure."
            )

    # -----------------------------------------------------
    # Traffic explanation
    # -----------------------------------------------------

    def explain_traffic(self):

        if self.traffic_density < 0.10:

            return (
                "The estimated traffic density is relatively "
                "low for the analyzed scene."
            )

        elif self.traffic_density < 0.20:

            return (
                "The estimated traffic density indicates "
                "a moderate level of traffic activity."
            )

        else:

            return (
                "The estimated traffic density indicates "
                "a high level of traffic activity."
            )

    # -----------------------------------------------------
    # Weather explanation
    # -----------------------------------------------------

    def explain_weather(self):

        explanations = []

        if self.humidity >= 80:

            explanations.append(
                "Humidity is high."
            )

        elif self.humidity <= 40:

            explanations.append(
                "Humidity is relatively low."
            )

        else:

            explanations.append(
                "Humidity is at a moderate level."
            )

        if self.wind_speed < 10:

            explanations.append(
                "Wind speed is relatively low, "
                "indicating weaker atmospheric dispersion."
            )

        else:

            explanations.append(
                "Wind conditions provide comparatively "
                "better atmospheric dispersion."
            )

        return " ".join(explanations)

    # -----------------------------------------------------
    # AI recommendation explanation
    # -----------------------------------------------------

    def explain_recommendation(self):

        action = self.recommended_action

        if action < 0:

            reduction = abs(action)

            return (
                f"The DQN recommends approximately "
                f"{reduction:.0f}% traffic reduction. "
                f"The simulated intervention produces a "
                f"{self.traffic_benefit * 100:.1f}% relative "
                f"reduction in traffic density."
            )

        else:

            return (
                "The DQN recommends maintaining the current "
                "traffic level under the simulated policy."
            )

    # -----------------------------------------------------
    # Cost-benefit explanation
    # -----------------------------------------------------

    def explain_tradeoff(self):

        return (
            f"The simulated intervention cost score is "
            f"{self.intervention_cost:.4f}. The decision "
            f"balances intervention cost against the "
            f"simulated traffic and environmental benefits."
        )

    # -----------------------------------------------------
    # Generate complete explanation
    # -----------------------------------------------------

    def generate_explanation(self):

        explanation = []

        explanation.append(
            "SMART CITY AI DECISION EXPLANATION"
        )

        explanation.append(
            "-" * 55
        )

        explanation.append(
            f"Current AQI: {self.aqi:.2f}"
        )

        explanation.append(
            f"Traffic density: {self.traffic_density:.4f}"
        )

        explanation.append(
            f"Temperature: {self.temperature:.2f} °C"
        )

        explanation.append(
            f"Humidity: {self.humidity:.2f}%"
        )

        explanation.append(
            f"Wind speed: {self.wind_speed:.2f} km/h"
        )

        explanation.append("")

        explanation.append(
            "WHY THIS CITY STATE MATTERS"
        )

        explanation.append(
            self.explain_aqi()
        )

        explanation.append(
            self.explain_traffic()
        )

        explanation.append(
            self.explain_weather()
        )

        explanation.append("")

        explanation.append(
            "AI RECOMMENDATION"
        )

        explanation.append(
            self.explain_recommendation()
        )

        explanation.append(
            self.explain_tradeoff()
        )

        explanation.append("")

        explanation.append(
            "IMPORTANT NOTE"
        )

        explanation.append(
            "The environmental impact values are "
            "simulation-based indicators produced by "
            "the project's Digital Twin. They should "
            "not be interpreted as measured real-world "
            "AQI changes."
        )

        return "\n".join(explanation)


# =========================================================
# TEST THE EXPLAINER
# =========================================================

if __name__ == "__main__":

    explainer = SmartCityAIExplainer(
        aqi=162.0,
        traffic_density=0.0792,
        temperature=17.5,
        humidity=88.0,
        wind_speed=8.9,
        recommended_action=-40,
        traffic_benefit=0.40,
        intervention_cost=0.16
    )

    print(
        explainer.generate_explanation()
    )