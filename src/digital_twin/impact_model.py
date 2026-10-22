class CityImpactModel:
    """
    Simulation-based policy impact model.

    This is a project-defined simulation model.
    It is NOT a scientifically validated causal
    pollution or AQI prediction model.
    """

    def __init__(self):

        # Default/reference traffic density
        self.baseline_traffic_density = 0.1700

    def simulate_traffic(
        self,
        traffic_change_percent,
        current_traffic_density=None
    ):
        """
        Simulate new traffic density after an intervention.

        If current_traffic_density is provided,
        the intervention is applied to the CURRENT
        traffic state.
        """

        if current_traffic_density is None:
            current_traffic_density = (
                self.baseline_traffic_density
            )

        change_factor = (
            1 + traffic_change_percent / 100
        )

        new_density = (
            current_traffic_density * change_factor
        )

        # Keep traffic density between 0 and 1
        new_density = max(
            0.0,
            min(new_density, 1.0)
        )

        return new_density

    def calculate_traffic_benefit(
        self,
        baseline_density,
        new_density
    ):
        """
        Calculate relative traffic improvement.
        """

        if baseline_density == 0:
            return 0.0

        improvement = (
            (baseline_density - new_density)
            / baseline_density
        )

        return max(0.0, improvement)

    def calculate_intervention_cost(
        self,
        traffic_change_percent
    ):
        """
        Estimate intervention cost/disruption.

        Stronger interventions have higher cost.
        """

        intervention_strength = (
            abs(traffic_change_percent) / 100
        )

        cost = intervention_strength ** 2

        return cost

    def calculate_environmental_pressure(
        self,
        aqi,
        traffic_density,
        humidity,
        wind_speed
    ):
        """
        Calculate a scenario-based environmental
        pressure indicator.

        IMPORTANT:
        This is NOT predicted AQI.
        It is only a simulation indicator used
        by the decision-making environment.
        """

        # AQI severity factor
        aqi_factor = min(
            aqi / 300,
            1.0
        )

        # Traffic contribution
        traffic_factor = traffic_density

        # Humidity contribution
        humidity_factor = humidity / 100

        # Higher wind speed generally improves
        # atmospheric dispersion.
        wind_factor = max(
            0.0,
            1 - (wind_speed / 30)
        )

        pressure = (
            0.50 * aqi_factor
            + 0.30 * traffic_factor
            + 0.10 * humidity_factor
            + 0.10 * wind_factor
        )

        return pressure

    def evaluate_scenario(
        self,
        traffic_change_percent,
        aqi,
        humidity,
        wind_speed,
        current_traffic_density=None
    ):
        """
        Evaluate one traffic intervention scenario.
        """

        # Use current traffic state when supplied.
        if current_traffic_density is None:

            baseline_density = (
                self.baseline_traffic_density
            )

        else:

            baseline_density = (
                current_traffic_density
            )

        # Simulate intervention
        new_density = self.simulate_traffic(
            traffic_change_percent,
            current_traffic_density=baseline_density
        )

        # Traffic improvement
        traffic_benefit = (
            self.calculate_traffic_benefit(
                baseline_density,
                new_density
            )
        )

        # Intervention cost
        intervention_cost = (
            self.calculate_intervention_cost(
                traffic_change_percent
            )
        )

        # Environmental pressure before intervention
        baseline_pressure = (
            self.calculate_environmental_pressure(
                aqi,
                baseline_density,
                humidity,
                wind_speed
            )
        )

        # Environmental pressure after intervention
        scenario_pressure = (
            self.calculate_environmental_pressure(
                aqi,
                new_density,
                humidity,
                wind_speed
            )
        )

        # Improvement in environmental pressure
        environmental_improvement = (
            baseline_pressure
            - scenario_pressure
        )

        return {

            "traffic_change_percent":
                traffic_change_percent,

            "baseline_traffic_density":
                baseline_density,

            "new_traffic_density":
                new_density,

            "traffic_benefit":
                traffic_benefit,

            "baseline_environmental_pressure":
                baseline_pressure,

            "scenario_environmental_pressure":
                scenario_pressure,

            "environmental_improvement":
                environmental_improvement,

            "intervention_cost":
                intervention_cost
        }


if __name__ == "__main__":

    print("=" * 60)
    print("SMART CITY CITY IMPACT MODEL")
    print("=" * 60)

    model = CityImpactModel()

    aqi = 162
    humidity = 88
    wind_speed = 8.9

    scenarios = [
        0,
        -10,
        -20,
        -30,
        -40
    ]

    print("\nTesting interventions from the default baseline:\n")

    for change in scenarios:

        result = model.evaluate_scenario(
            traffic_change_percent=change,
            aqi=aqi,
            humidity=humidity,
            wind_speed=wind_speed
        )

        print(
            f"Traffic intervention: {change}%"
        )

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

        print()
    
    print("=" * 60)
    print("IMPACT MODEL TEST COMPLETE")
    print("=" * 60)