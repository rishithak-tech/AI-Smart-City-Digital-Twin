from pathlib import Path
import sys

import pandas as pd


# ============================================
# PROJECT PATH
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)


# ============================================
# TRAFFIC MODEL RESULT
# ============================================

CURRENT_TRAFFIC_DENSITY = 0.1700


# ============================================
# LOAD CURRENT CITY STATE
# ============================================

def load_city_state():

    data = pd.read_csv(
        DATA_PATH
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    latest = data.iloc[-1]

    city_state = {

        "datetime":
            latest["datetime"],

        "aqi":
            float(latest["aqi"]),

        "predicted_aqi":
            float(
                latest["target_aqi_next_hour"]
            ),

        "temperature":
            float(
                latest["temperature"]
            ),

        "humidity":
            float(
                latest["humidity"]
            ),

        "wind_speed":
            float(
                latest["wind_speed"]
            ),

        "traffic_density":
            CURRENT_TRAFFIC_DENSITY

    }

    return city_state


# ============================================
# CALCULATE PRIORITY
# ============================================

def calculate_priority(
    traffic_density,
    aqi,
    temperature,
    humidity,
    wind_speed
):

    score = 0


    # ----------------------------------------
    # Traffic
    # ----------------------------------------

    if traffic_density >= 0.20:

        score += 40

    elif traffic_density >= 0.12:

        score += 30

    elif traffic_density >= 0.05:

        score += 15


    # ----------------------------------------
    # AQI
    # ----------------------------------------

    if aqi > 300:

        score += 40

    elif aqi > 200:

        score += 35

    elif aqi > 150:

        score += 30

    elif aqi > 100:

        score += 20

    elif aqi > 50:

        score += 10


    # ----------------------------------------
    # Humidity
    # ----------------------------------------

    if humidity > 85:

        score += 10


    # ----------------------------------------
    # Wind
    # ----------------------------------------

    if wind_speed < 5:

        score += 10


    return score


# ============================================
# GENERATE RECOMMENDATION
# ============================================

def recommend_action(
    city_state
):

    score = calculate_priority(

        city_state["traffic_density"],

        city_state["aqi"],

        city_state["temperature"],

        city_state["humidity"],

        city_state["wind_speed"]

    )


    # ----------------------------------------
    # Determine priority
    # ----------------------------------------

    if score >= 80:

        level = "CRITICAL"

        action = (
            "Activate emergency traffic "
            "management and pollution "
            "control measures."
        )


    elif score >= 60:

        level = "HIGH"

        action = (
            "Reduce traffic congestion "
            "and encourage public transport."
        )


    elif score >= 40:

        level = "MODERATE"

        action = (
            "Monitor traffic and air quality "
            "closely and optimize traffic flow."
        )


    else:

        level = "LOW"

        action = (
            "Continue normal city operations "
            "while monitoring conditions."
        )


    return {

        "priority_score":
            score,

        "priority_level":
            level,

        "recommended_action":
            action

    }


# ============================================
# SCENARIO EVALUATION
# ============================================

def evaluate_scenario(
    city_state,
    traffic_change
):

    simulated_density = (
        city_state["traffic_density"]
        * (
            1
            + traffic_change / 100
        )
    )


    simulated_density = max(
        0,
        min(
            1,
            simulated_density
        )
    )


    score = calculate_priority(

        simulated_density,

        city_state["aqi"],

        city_state["temperature"],

        city_state["humidity"],

        city_state["wind_speed"]

    )


    return {

        "traffic_change":
            traffic_change,

        "traffic_density":
            simulated_density,

        "priority_score":
            score

    }


# ============================================
# FIND BEST TRAFFIC INTERVENTION
# ============================================

def find_best_intervention(
    city_state
):

    scenarios = [

        -10,

        -20,

        -30,

        -40

    ]


    results = []


    for change in scenarios:

        result = evaluate_scenario(

            city_state,

            change

        )

        results.append(
            result
        )


    best = min(
        results,
        key=lambda x:
            x["priority_score"]
    )


    return results, best


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print(
        "\n"
        "===================================="
    )

    print(
        "AI SMART CITY DECISION ENGINE"
    )

    print(
        "===================================="
    )


    # ----------------------------------------
    # Load actual city state
    # ----------------------------------------

    print(
        "\nLoading current city state..."
    )


    city_state = load_city_state()


    print(
        "City state loaded successfully."
    )


    # ========================================
    # CURRENT CONDITIONS
    # ========================================

    print(
        "\n"
        "===================================="
    )

    print(
        "CURRENT CITY CONDITIONS"
    )

    print(
        "===================================="
    )


    print(
        f"Timestamp: "
        f"{city_state['datetime']}"
    )


    print(
        f"AQI: "
        f"{city_state['aqi']:.1f}"
    )


    print(
        f"Next-hour AQI prediction: "
        f"{city_state['predicted_aqi']:.1f}"
    )


    print(
        f"Temperature: "
        f"{city_state['temperature']:.1f} °C"
    )


    print(
        f"Humidity: "
        f"{city_state['humidity']:.1f} %"
    )


    print(
        f"Wind speed: "
        f"{city_state['wind_speed']:.1f} km/h"
    )


    print(
        f"Traffic density: "
        f"{city_state['traffic_density']:.4f}"
    )


    # ========================================
    # CURRENT RECOMMENDATION
    # ========================================

    recommendation = (
        recommend_action(
            city_state
        )
    )


    print(
        "\n"
        "===================================="
    )

    print(
        "CURRENT AI RECOMMENDATION"
    )

    print(
        "===================================="
    )


    print(
        f"Priority score: "
        f"{recommendation['priority_score']}"
    )


    print(
        f"Priority level: "
        f"{recommendation['priority_level']}"
    )


    print(
        "\nRecommended action:"
    )


    print(
        recommendation[
            "recommended_action"
        ]
    )


    # ========================================
    # SCENARIO ANALYSIS
    # ========================================

    print(
        "\n"
        "===================================="
    )

    print(
        "TRAFFIC INTERVENTION ANALYSIS"
    )

    print(
        "===================================="
    )


    results, best = (
        find_best_intervention(
            city_state
        )
    )


    for result in results:

        print(
            "\nTraffic change: "
            f"{result['traffic_change']}%"
        )

        print(
            "Simulated density: "
            f"{result['traffic_density']:.4f}"
        )

        print(
            "Priority score: "
            f"{result['priority_score']}"
        )


    # ========================================
    # BEST ACTION
    # ========================================

    print(
        "\n"
        "===================================="
    )

    print(
        "AI OPTIMAL INTERVENTION"
    )

    print(
        "===================================="
    )


    print(
        f"Recommended traffic change: "
        f"{best['traffic_change']}%"
    )


    print(
        f"Resulting traffic density: "
        f"{best['traffic_density']:.4f}"
    )


    print(
        f"Resulting priority score: "
        f"{best['priority_score']}"
    )


    print(
        "\nRecommended strategy:"
    )


    if best["traffic_change"] <= -30:

        print(
            "Implement strong traffic reduction "
            "measures and promote public "
            "transport."
        )

    elif best["traffic_change"] <= -20:

        print(
            "Increase public transport usage "
            "and optimize major traffic routes."
        )

    else:

        print(
            "Apply moderate traffic optimization "
            "measures."
        )


    print(
        "\n"
        "===================================="
    )

    print(
        "DECISION ENGINE COMPLETE"
    )

    print(
        "===================================="
    )