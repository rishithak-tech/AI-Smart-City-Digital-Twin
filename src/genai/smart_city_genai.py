# =========================================================
# AI SMART CITY DIGITAL TWIN
# Integrated GenAI + RAG + DQN Decision Support
# =========================================================

import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ollama


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


DQN_MODEL_PATH = (
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


GENAI_PATH = (
    PROJECT_ROOT
    / "src"
    / "genai"
)


# =========================================================
# ADD GENAI PATH
# =========================================================

if str(GENAI_PATH) not in sys.path:

    sys.path.insert(
        0,
        str(GENAI_PATH)
    )


# =========================================================
# IMPORT RAG RETRIEVER
# =========================================================

from rag_retriever import retrieve


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "llama3.2:3b"


ACTIONS = [
    0,
    -10,
    -20,
    -30,
    -40
]


# =========================================================
# DQN NETWORK
# =========================================================

class DQN(torch.nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()

        self.network = torch.nn.Sequential(

            torch.nn.Linear(
                state_size,
                64
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                64,
                64
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                64,
                action_size
            )
        )


    def forward(self, x):

        return self.network(x)


# =========================================================
# LOAD CITY STATE
# =========================================================

def load_city_state():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"City data file not found:\n{DATA_PATH}"
        )


    data = pd.read_csv(
        DATA_PATH
    )


    if data.empty:

        raise ValueError(
            "City data file is empty."
        )


    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )


    data = (
        data
        .sort_values("datetime")
        .reset_index(drop=True)
    )


    latest = data.iloc[-1]


    city = {

        "timestamp":
            latest["datetime"],

        "aqi":
            float(
                latest["aqi"]
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

        "wind_direction":
            float(
                latest["wind_direction"]
            ),

        "surface_pressure":
            float(
                latest["surface_pressure"]
            )
    }


    return city


# =========================================================
# LOAD TRAFFIC DENSITY
# =========================================================
#
# IMPORTANT:
# This runs the exact same traffic_state.py
# pipeline used by the Streamlit dashboard.
#
# U-Net
#   ↓
# segmentation
#   ↓
# traffic model
#   ↓
# traffic density
#
# Expected result:
# 0.1700
# =========================================================

def load_traffic_density():

    if not TRAFFIC_SCRIPT.exists():

        raise FileNotFoundError(
            f"Traffic script not found:\n"
            f"{TRAFFIC_SCRIPT}"
        )


    result = subprocess.run(

        [
            sys.executable,
            str(TRAFFIC_SCRIPT)
        ],

        capture_output=True,

        text=True,

        cwd=str(PROJECT_ROOT)
    )


    if result.returncode != 0:

        raise RuntimeError(

            "Traffic pipeline failed.\n\n"

            f"STDOUT:\n"
            f"{result.stdout}\n\n"

            f"STDERR:\n"
            f"{result.stderr}"
        )


    match = re.search(

        r"Predicted traffic density:\s*([0-9.]+)",

        result.stdout
    )


    if not match:

        raise RuntimeError(

            "Could not read traffic density "
            "from traffic_state.py.\n\n"

            f"Output:\n{result.stdout}"
        )


    traffic_density = float(
        match.group(1)
    )


    return traffic_density


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

    state = np.array(

        [
            aqi / 300.0,

            traffic_density,

            temperature / 50.0,

            humidity / 100.0,

            wind_speed / 30.0
        ],

        dtype=np.float32
    )


    return state


# =========================================================
# LOAD DQN
# =========================================================

def load_dqn():

    if not DQN_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"DQN model not found:\n"
            f"{DQN_MODEL_PATH}"
        )


    model = DQN(
        state_size=5,
        action_size=5
    )


    checkpoint = torch.load(

        DQN_MODEL_PATH,

        map_location="cpu",

        weights_only=False
    )


    if (

        isinstance(
            checkpoint,
            dict
        )

        and

        "model_state_dict"
        in checkpoint

    ):

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model.eval()


    return model


# =========================================================
# GET DQN DECISION
# =========================================================

def get_dqn_decision(
    city,
    traffic_density
):

    model = load_dqn()


    state = normalize_state(

        city["aqi"],

        traffic_density,

        city["temperature"],

        city["humidity"],

        city["wind_speed"]
    )


    state_tensor = torch.tensor(

        state,

        dtype=torch.float32
    ).unsqueeze(0)


    with torch.no_grad():

        q_values = (

            model(
                state_tensor
            )

            .squeeze(0)

            .cpu()

            .numpy()
        )


    best_index = int(
        np.argmax(q_values)
    )


    recommended_action = ACTIONS[
        best_index
    ]


    return (
        recommended_action,
        q_values
    )


# =========================================================
# WHAT-IF SCENARIO
# =========================================================
#
# THIS FUNCTION IS REQUIRED BY dashboard.py
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

            traffic_density
            - new_density

        ) / traffic_density

    else:

        traffic_benefit = 0.0


    intervention_cost = (

        abs(action) / 100.0
    ) ** 2


    return {

        "action":
            action,

        "new_density":
            new_density,

        "traffic_benefit":
            traffic_benefit,

        "intervention_cost":
            intervention_cost
    }


# =========================================================
# BUILD RAG CONTEXT
# =========================================================
#
# THIS FUNCTION IS REQUIRED BY dashboard.py
# =========================================================

def build_rag_context(
    question
):

    try:

        results = retrieve(

            question,

            top_k=3
        )

    except TypeError:

        # Compatibility with retriever versions
        # that do not accept top_k.

        results = retrieve(
            question
        )


    if not results:

        return (

            "No relevant project knowledge "
            "was retrieved."
        )


    context_parts = []


    for result in results:

        if isinstance(
            result,
            dict
        ):

            # Support both possible key names.

            title = result.get(
                "title",
                result.get(
                    "section",
                    "Project Knowledge"
                )
            )


            content = result.get(
                "content",
                ""
            )


            context_parts.append(

                f"SECTION: {title}\n"
                f"{content}"
            )

        else:

            context_parts.append(
                str(result)
            )


    return "\n\n".join(
        context_parts
    )


# =========================================================
# GENERATE GENAI EXPLANATION
# =========================================================
#
# THIS FUNCTION IS REQUIRED BY dashboard.py
# =========================================================

def generate_explanation(
    city,
    traffic_density,
    scenarios,
    recommended_action,
    q_values,
    context
):

    # -----------------------------------------------------
    # Scenario information
    # -----------------------------------------------------

    scenario_text = ""


    for scenario in scenarios:

        scenario_text += (

            f"Action "
            f"{scenario['action']}%: "

            f"traffic density becomes "
            f"{scenario['new_density']:.4f}, "

            f"traffic benefit = "
            f"{scenario['traffic_benefit']:.2f}, "

            f"intervention cost = "
            f"{scenario['intervention_cost']:.4f}\n"
        )


    # -----------------------------------------------------
    # Q-values
    # -----------------------------------------------------

    q_value_text = ""


    for action, q_value in zip(
        ACTIONS,
        q_values
    ):

        q_value_text += (

            f"{action}% → "
            f"{float(q_value):.4f}\n"
        )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""

You are the GenAI explanation module of an
AI Smart City Digital Twin academic prototype.

Explain the current AI decision clearly,
concisely, and factually.

IMPORTANT LIMITATIONS:

1. The environmental data is historical,
   not real-time.

2. The IDD Lite computer-vision scene is
   independent of the AQI/weather timestamp.

3. Traffic density is a project-defined
   engineered scene-level indicator and is
   NOT validated real-world traffic volume.

4. The what-if simulation is a mathematical
   simulation indicator.

5. Environmental improvement in the simulation
   must NOT be described as a guaranteed AQI
   reduction.

6. The DQN was trained in a synthetic
   simulation environment.

7. The DQN recommendation represents the
   highest estimated Q-value within the
   simulated environment.

8. Do not invent facts.

9. Do not claim that the recommended action
   will definitely improve real-world air quality.

10. Clearly distinguish observed data,
    model outputs, simulations, DQN decisions,
    and GenAI explanations.


=================================================
CURRENT CITY STATE
=================================================

Timestamp:
{city["timestamp"]}

AQI:
{city["aqi"]:.2f}

Temperature:
{city["temperature"]:.2f} °C

Humidity:
{city["humidity"]:.2f} %

Wind speed:
{city["wind_speed"]:.2f} km/h

Traffic density:
{traffic_density:.4f}


=================================================
DQN RECOMMENDATION
=================================================

Recommended action:
{recommended_action}% traffic intervention


=================================================
DQN Q-VALUES
=================================================

{q_value_text}


=================================================
WHAT-IF SCENARIOS
=================================================

{scenario_text}


=================================================
RETRIEVED PROJECT KNOWLEDGE
=================================================

{context}


=================================================
RESPONSE FORMAT
=================================================

Write four short sections:

1. Current Situation

Explain the current AQI,
weather and traffic state.

2. Why the DQN Selected This Action

Explain that the selected action has
the highest estimated Q-value among
the available actions.

3. Simulated Impact

Explain the traffic-density change,
traffic benefit and intervention cost.

4. Important Limitations

Clearly explain that the simulation
does not prove a real-world AQI reduction
and that the DQN operates in a synthetic
simulation environment.

Keep the explanation concise and suitable
for a Smart City dashboard.
"""


    # -----------------------------------------------------
    # Call Llama 3.2
    # -----------------------------------------------------

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[

            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ]
    )


    return response[
        "message"
    ][
        "content"
    ]


# =========================================================
# ALIAS
# =========================================================
#
# This is kept for compatibility with any code
# that may use get_rag_context().
# =========================================================

def get_rag_context(
    city_state,
    traffic_density,
    recommended_action
):

    question = (

        "Why did the DQN select the recommended "
        f"traffic intervention of "
        f"{recommended_action}%?"
    )


    context = build_rag_context(
        question
    )


    runtime_context = f"""

CURRENT RUNTIME CITY STATE

Timestamp:
{city_state["timestamp"]}

AQI:
{city_state["aqi"]:.2f}

Temperature:
{city_state["temperature"]:.2f} °C

Humidity:
{city_state["humidity"]:.2f} %

Wind speed:
{city_state["wind_speed"]:.2f} km/h

Traffic density:
{traffic_density:.4f}

Recommended action:
{recommended_action}%
"""


    return (
        context
        + "\n\n"
        + runtime_context
    )


# =========================================================
# ALIAS FOR OTHER CODE
# =========================================================

generate_llm_explanation = generate_explanation


# =========================================================
# STANDALONE MAIN
# =========================================================

def main():

    print()
    print(
        "================================================="
    )
    print(
        "AI SMART CITY DIGITAL TWIN - GENAI"
    )
    print(
        "Integrated RAG + DQN + Llama 3.2"
    )
    print(
        "================================================="
    )
    print()


    # -----------------------------------------------------
    # CITY STATE
    # -----------------------------------------------------

    print(
        "Loading latest available city state..."
    )


    city = load_city_state()


    print(
        "City state loaded successfully."
    )


    # -----------------------------------------------------
    # TRAFFIC
    # -----------------------------------------------------

    print()

    print(
        "Running U-Net + traffic model pipeline..."
    )


    traffic_density = (
        load_traffic_density()
    )


    print(
        f"Traffic density: "
        f"{traffic_density:.4f}"
    )


    # -----------------------------------------------------
    # DQN
    # -----------------------------------------------------

    print()

    print(
        "Loading trained DQN..."
    )


    (
        recommended_action,
        q_values
    ) = get_dqn_decision(

        city,

        traffic_density
    )


    best_q_value = float(
        np.max(q_values)
    )


    # -----------------------------------------------------
    # SCENARIOS
    # -----------------------------------------------------

    scenarios = []


    for action in ACTIONS:

        scenarios.append(

            simulate_scenario(

                action,

                traffic_density
            )
        )


    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    question = (

        "Why did the DQN select the recommended "
        "traffic intervention action?"
    )


    context = build_rag_context(
        question
    )


    # -----------------------------------------------------
    # GENAI
    # -----------------------------------------------------

    explanation = generate_explanation(

        city=city,

        traffic_density=traffic_density,

        scenarios=scenarios,

        recommended_action=recommended_action,

        q_values=q_values,

        context=context
    )


    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print()
    print(
        "================================================="
    )
    print(
        "LATEST AVAILABLE CITY STATE"
    )
    print(
        "================================================="
    )


    print(
        f"Timestamp: "
        f"{city['timestamp']}"
    )


    print(
        f"AQI: "
        f"{city['aqi']:.2f}"
    )


    print(
        f"Temperature: "
        f"{city['temperature']:.2f} °C"
    )


    print(
        f"Humidity: "
        f"{city['humidity']:.2f}%"
    )


    print(
        f"Wind speed: "
        f"{city['wind_speed']:.2f} km/h"
    )


    print(
        f"Traffic density: "
        f"{traffic_density:.4f}"
    )


    # -----------------------------------------------------
    # Q VALUES
    # -----------------------------------------------------

    print()
    print(
        "================================================="
    )
    print(
        "DQN ACTION VALUES"
    )
    print(
        "================================================="
    )


    for action, value in zip(
        ACTIONS,
        q_values
    ):

        print(

            f"{action}% intervention "
            f"→ Q-value: "
            f"{float(value):.4f}"
        )


    print()

    print(
        f"Recommended intervention: "
        f"{recommended_action}%"
    )


    print(
        f"Highest Q-value: "
        f"{best_q_value:.4f}"
    )


    # -----------------------------------------------------
    # RECOMMENDED SCENARIO
    # -----------------------------------------------------

    recommended_scenario = (
        simulate_scenario(

            recommended_action,

            traffic_density
        )
    )


    print()
    print(
        "================================================="
    )
    print(
        "SIMULATED RECOMMENDATION IMPACT"
    )
    print(
        "================================================="
    )


    print(

        f"Traffic density: "
        f"{traffic_density:.4f}"
        f" → "
        f"{recommended_scenario['new_density']:.4f}"
    )


    print(

        f"Traffic benefit: "
        f"{recommended_scenario['traffic_benefit']:.4f}"
    )


    print(

        f"Intervention cost: "
        f"{recommended_scenario['intervention_cost']:.4f}"
    )


    # -----------------------------------------------------
    # GENAI
    # -----------------------------------------------------

    print()
    print(
        "================================================="
    )
    print(
        "LLAMA 3.2 GENAI EXPLANATION"
    )
    print(
        "================================================="
    )


    print(
        "Generating explanation..."
    )


    print()

    print(
        explanation
    )


    print()
    print(
        "================================================="
    )
    print(
        "INTEGRATED GENAI PIPELINE COMPLETE"
    )
    print(
        "================================================="
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()