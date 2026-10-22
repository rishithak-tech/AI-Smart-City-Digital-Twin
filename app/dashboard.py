import streamlit as st
import pandas as pd
import subprocess
import re
import sys
import numpy as np
import torch
import ollama

from pathlib import Path
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Smart City Digital Twin",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)


TRAFFIC_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "digital_twin"
    / "traffic_state.py"
)


DQN_MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_city_dqn.pth"
)


UNET_MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "best_unet.pth"
)


IDD_IMAGE_FILE = Path(
    r"C:\Users\LENOVO\Downloads\idd-lite\idd20k_lite"
    r"\leftImg8bit\test\103\206393_image.jpg"
)


DQN_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "digital_twin"
)


GENAI_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "genai"
)


MODEL_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
)


# =========================================================
# ADD PROJECT MODULE PATHS
# =========================================================

for module_path in [
    DQN_MODULE_PATH,
    GENAI_MODULE_PATH,
    MODEL_MODULE_PATH
]:

    if str(module_path) not in sys.path:

        sys.path.append(
            str(module_path)
        )


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from dqn_agent import DQNAgent

from smart_city_genai import (
    build_rag_context,
    generate_explanation,
    simulate_scenario
)

from unet_model import LightweightUNet


# =========================================================
# CONSTANTS
# =========================================================

ACTIONS = [
    0,
    -10,
    -20,
    -30,
    -40
]


# =========================================================
# LOAD ENVIRONMENTAL DATA
# =========================================================

@st.cache_data
def load_city_data():

    data = pd.read_csv(
        DATA_FILE
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.sort_values(
        "datetime"
    ).reset_index(drop=True)

    latest = data.iloc[-1]

    return data, {
        "timestamp": latest["datetime"],
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
# RUN TRAFFIC PIPELINE
# =========================================================

@st.cache_data
def run_traffic_pipeline():

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
            result.stderr
        )

    output = result.stdout


    # -----------------------------------------------------
    # Extract traffic features
    # -----------------------------------------------------

    vehicle_match = re.search(
        r"Vehicle pixel ratio:\s*([0-9.]+)",
        output
    )

    living_match = re.search(
        r"Living-thing ratio:\s*([0-9.]+)",
        output
    )

    drivable_match = re.search(
        r"Drivable ratio:\s*([0-9.]+)",
        output
    )

    construction_match = re.search(
        r"Construction ratio:\s*([0-9.]+)",
        output
    )

    sky_match = re.search(
        r"Sky ratio:\s*([0-9.]+)",
        output
    )

    density_match = re.search(
        r"Predicted traffic density:\s*([0-9.]+)",
        output
    )

    risk_match = re.search(
        r"Traffic risk:\s*(.+)",
        output
    )


    if not all(
        [
            vehicle_match,
            living_match,
            drivable_match,
            construction_match,
            sky_match,
            density_match,
            risk_match
        ]
    ):

        raise RuntimeError(
            "Could not read traffic analysis results."
        )


    return {

        "vehicle_ratio":
            float(vehicle_match.group(1)),

        "living_ratio":
            float(living_match.group(1)),

        "drivable_ratio":
            float(drivable_match.group(1)),

        "construction_ratio":
            float(construction_match.group(1)),

        "sky_ratio":
            float(sky_match.group(1)),

        "traffic_density":
            float(density_match.group(1)),

        "traffic_risk":
            risk_match.group(1).strip()
    }


# =========================================================
# LOAD U-NET MODEL
# =========================================================

@st.cache_resource
def load_unet_model():

    device = torch.device(
        "cpu"
    )

    model = LightweightUNet(
        num_classes=7
    )

    checkpoint = torch.load(
        UNET_MODEL_FILE,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(device)

    model.eval()

    return model


# =========================================================
# GENERATE U-NET VISUALIZATION
# =========================================================

@st.cache_data
def generate_unet_visuals():

    model = load_unet_model()

    device = torch.device(
        "cpu"
    )


    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = Image.open(
        IDD_IMAGE_FILE
    ).convert("RGB")


    image_array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )


    image_tensor = torch.from_numpy(
        image_array.transpose(
            2,
            0,
            1
        )
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        device
    )


    # -----------------------------------------------------
    # U-Net prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()


    # -----------------------------------------------------
    # Seven-class color map
    # -----------------------------------------------------

    color_map = np.array(
        [
            [0, 255, 0],       # Drivable
            [255, 255, 0],     # Non-drivable
            [255, 0, 255],     # Living thing
            [255, 0, 0],       # Vehicles
            [0, 165, 255],     # Road-side
            [180, 0, 255],     # Construction
            [0, 191, 255],     # Sky
        ],
        dtype=np.uint8
    )


    segmentation = color_map[
        prediction
    ]


    # -----------------------------------------------------
    # Softer overlay
    # -----------------------------------------------------

    original = np.array(
        image
    ).astype(np.float32)


    segmentation_float = (
        segmentation
        .astype(np.float32)
    )


    overlay = (
        0.75 * original
        + 0.25 * segmentation_float
    )


    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


    return (
        image,
        Image.fromarray(
            overlay
        )
    )


# =========================================================
# LOAD DQN MODEL
# =========================================================

@st.cache_resource
def load_dqn_model():

    agent = DQNAgent(
        state_size=5,
        action_size=5
    )

    checkpoint = torch.load(
        DQN_MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )


    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["model_state_dict"]
        )

    else:

        state_dict = checkpoint


    try:

        agent.load_state_dict(
            state_dict
        )

    except Exception:

        loaded = False

        for attribute in [
            "model",
            "q_network",
            "network"
        ]:

            if hasattr(
                agent,
                attribute
            ):

                network = getattr(
                    agent,
                    attribute
                )

                try:

                    network.load_state_dict(
                        state_dict
                    )

                    loaded = True

                    break

                except Exception:

                    pass


        if not loaded:

            raise


    if hasattr(
        agent,
        "epsilon"
    ):

        agent.epsilon = 0.0


    return agent


# =========================================================
# GET DQN Q-VALUES
# =========================================================

def get_dqn_decision(
    city,
    traffic_density
):

    agent = load_dqn_model()


    # -----------------------------------------------------
    # Normalize state exactly as used during DQN training
    # -----------------------------------------------------

    state = np.array(
        [
            city["aqi"] / 300.0,
            traffic_density,
            city["temperature"] / 50.0,
            city["humidity"] / 100.0,
            city["wind_speed"] / 30.0
        ],
        dtype=np.float32
    )


    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)


    # -----------------------------------------------------
    # Find neural network
    # -----------------------------------------------------

    network = None

    for attribute in [
        "model",
        "q_network",
        "network"
    ]:

        if hasattr(
            agent,
            attribute
        ):

            candidate = getattr(
                agent,
                attribute
            )

            if callable(
                candidate
            ):

                network = candidate

                break


    if network is None:

        raise RuntimeError(
            "Could not find the DQN neural network."
        )


    network.eval()


    with torch.no_grad():

        q_tensor = network(
            state_tensor
        )

        q_values = (
            q_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )


    best_index = int(
        np.argmax(
            q_values
        )
    )


    recommended_action = ACTIONS[
        best_index
    ]


    return (
        recommended_action,
        q_values
    )


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    data, city = load_city_data()

    traffic = run_traffic_pipeline()

except Exception as e:

    st.error(
        f"Unable to load project data: {e}"
    )

    st.stop()


# =========================================================
# VARIABLES
# =========================================================

timestamp = city["timestamp"]

aqi = city["aqi"]

temperature = city["temperature"]

humidity = city["humidity"]

wind_speed = city["wind_speed"]

traffic_density = (
    traffic["traffic_density"]
)

traffic_risk = (
    traffic["traffic_risk"]
)


# =========================================================
# AIR QUALITY RISK
# =========================================================

if aqi <= 50:

    air_quality_risk = "Good"

elif aqi <= 100:

    air_quality_risk = "Moderate"

elif aqi <= 150:

    air_quality_risk = (
        "Unhealthy for Sensitive Groups"
    )

elif aqi <= 200:

    air_quality_risk = "Unhealthy"

elif aqi <= 300:

    air_quality_risk = (
        "Very Unhealthy"
    )

else:

    air_quality_risk = "Hazardous"


# =========================================================
# DQN DECISION
# =========================================================

try:

    (
        recommended_action,
        q_values
    ) = get_dqn_decision(
        city,
        traffic_density
    )

except Exception as e:

    recommended_action = None

    q_values = None

    dqn_error = str(e)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🏙️ AI Smart City Digital Twin"
)

st.subheader(
    "Hyderabad Urban Intelligence & Decision Support System"
)

st.markdown(
    """
    This dashboard integrates **Machine Learning, Deep Learning,
    Reinforcement Learning, Digital Twin simulation, RAG and GenAI**
    for smart-city decision support.
    """
)

st.divider()


# =========================================================
# ASK SMART CITY AI
# =========================================================

st.header("🤖 Ask Smart City AI")

st.caption(
    "Ask questions about AQI, traffic, weather, simulations, "
    "the DQN decision, or the current city state."
)

user_question = st.text_input(
    "Your question",
    placeholder="Example: What happens if traffic is reduced by 30%?"
)

ask_ai = st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True
)

if ask_ai:

    if not user_question.strip():

        st.warning("Please enter a question.")

    else:

        st.session_state["smart_city_question"] = (
            user_question.strip()
        )


# ---------------------------------------------------------
# QUESTION ANSWER ENGINE
# ---------------------------------------------------------

question = st.session_state.get(
    "smart_city_question",
    ""
).strip()


def build_smart_city_ai_context(question_text):
    """Build grounded context from the actual project outputs."""

    try:
        rag_context = build_rag_context(question_text)
    except Exception as e:
        rag_context = f"RAG retrieval unavailable: {e}"

    scenario_lines = []

    for action in ACTIONS:
        try:
            scenario = simulate_scenario(
                action,
                traffic_density
            )
        except TypeError:
            scenario = {
                "action": action,
                "new_density": traffic_density * (1 + action / 100),
                "traffic_benefit": abs(action) / 100,
                "intervention_cost": (abs(action) / 100) ** 2
            }

        new_density = float(
            scenario.get(
                "new_density",
                traffic_density * (1 + action / 100)
            )
        )
        benefit = float(
            scenario.get(
                "traffic_benefit",
                abs(action) / 100
            )
        )
        cost = float(
            scenario.get(
                "intervention_cost",
                (abs(action) / 100) ** 2
            )
        )
        reward = scenario.get("reward")
        reward_text = (
            f", reward={float(reward):.4f}"
            if reward is not None else ""
        )

        scenario_lines.append(
            f"{action}%: density {new_density:.4f}, "
            f"benefit {benefit:.2f}, cost {cost:.4f}{reward_text}"
        )

    if recommended_action is not None and q_values is not None:
        q_lines = []
        for action, q_value in zip(ACTIONS, q_values):
            q_lines.append(
                f"{action}% intervention: Q-value {float(q_value):.4f}"
            )
        dqn_summary = (
            f"Recommended DQN action: {recommended_action}%\n"
            f"Highest Q-value: "
            f"{float(q_values[ACTIONS.index(recommended_action)]):.4f}\n"
            "Q-values:\n" + "\n".join(q_lines)
        )
    else:
        dqn_summary = "DQN decision unavailable."

    return f"""
CURRENT PROJECT STATE
---------------------
Latest available environmental timestamp: {timestamp}
AQI: {aqi:.0f}
Air-quality risk: {air_quality_risk}
Temperature: {temperature:.1f} C
Humidity: {humidity:.0f}%
Wind speed: {wind_speed:.1f} km/h

TRAFFIC / COMPUTER VISION
-------------------------
Traffic density indicator: {traffic_density:.4f}
Traffic risk: {traffic_risk}
Vehicle pixel ratio: {traffic["vehicle_ratio"]:.4f}
Drivable ratio: {traffic["drivable_ratio"]:.4f}
Construction ratio: {traffic["construction_ratio"]:.4f}
Sky ratio: {traffic["sky_ratio"]:.4f}

DQN DECISION
------------
{dqn_summary}

WHAT-IF SIMULATION
------------------
{chr(10).join(scenario_lines)}

PROJECT KNOWLEDGE RETRIEVED BY RAG
----------------------------------
{rag_context}

IMPORTANT INTERPRETATION RULES
------------------------------
- Traffic density is a project-defined scene-level indicator, not a direct
  measurement of real-world vehicle volume.
- The IDD Lite road scene is independent of the AQI/weather timestamp.
- What-If results are outputs of the project's mathematical simulation.
- DQN Q-values represent learned long-term returns inside the project's
  simulated environment.
- The DQN recommendation is not a guarantee of real-world optimality.
- Do not claim that a traffic intervention is proven to reduce real-world
  AQI or pollution.
- If the question asks for something not supported by the supplied context,
  say that the current project data does not establish it.
"""


def generate_smart_city_ai_answer(question_text):
    """Ask local Llama 3.2 to answer using grounded project context."""

    context = build_smart_city_ai_context(question_text)

    prompt = f"""
You are the AI assistant inside an academic project called
"AI Smart City Digital Twin".

Answer the user's question using ONLY the project context below.

USER QUESTION:
{question_text}

PROJECT CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the question directly.
2. Use actual project numbers when relevant.
3. Explain the reasoning clearly in simple language.
4. If the user asks why, compare relevant DQN Q-values or What-If results.
5. Do not invent data, dates, measurements, or model capabilities.
6. Do not present simulated results as measured real-world effects.
7. Mention an important limitation when it matters.
8. Keep the answer concise but useful.
9. If the project context does not establish something, say so.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual, grounded Smart City AI assistant. "
                    "Never invent project results."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


if question:

    st.subheader("💡 AI Response")

    try:

        response = generate_smart_city_ai_answer(
            question
        )

        st.markdown(response)

    except Exception as e:

        st.error(
            f"The AI question could not be processed: {e}"
        )


st.divider()


# =========================================================
# CURRENT CITY STATE
# =========================================================

st.header(
    "📊 Latest Available City State"
)

st.caption(
    f"Latest available environmental data: {timestamp}"
)

col1, col2, col3, col4, col5 = (
    st.columns(5)
)


with col1:

    st.metric(
        "AQI",
        f"{aqi:.0f}"
    )


with col2:

    st.metric(
        "Traffic Density",
        f"{traffic_density:.4f}"
    )


with col3:

    st.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )


with col4:

    st.metric(
        "Humidity",
        f"{humidity:.0f}%"
    )


with col5:

    st.metric(
        "Wind Speed",
        f"{wind_speed:.1f} km/h"
    )


st.divider()


# =========================================================
# AQI TREND
# =========================================================

st.header(
    "📈 AQI Trend"
)

# Convert datetime and sort chronologically
aqi_data = data[
    [
        "datetime",
        "aqi"
    ]
].copy()

aqi_data["datetime"] = pd.to_datetime(
    aqi_data["datetime"]
)

aqi_data = (
    aqi_data
    .sort_values("datetime")
)

# Get the latest available 7-day period
latest_time = aqi_data["datetime"].max()
start_time = latest_time - pd.Timedelta(days=7)

aqi_chart_data = aqi_data[
    (aqi_data["datetime"] >= start_time) &
    (aqi_data["datetime"] <= latest_time)
].copy()

# Set datetime as index
aqi_chart_data = (
    aqi_chart_data
    .set_index("datetime")
)

# Create a continuous hourly timeline.
# Missing observations remain NaN, creating
# a visible gap instead of a misleading line.
full_time_range = pd.date_range(
    start=aqi_chart_data.index.min().floor("h"),
    end=aqi_chart_data.index.max().floor("h"),
    freq="h"
)

aqi_chart_data = (
    aqi_chart_data
    .reindex(full_time_range)
)

aqi_chart_data.index.name = "datetime"

st.line_chart(
    aqi_chart_data["aqi"],
    width="stretch"
)

st.caption(
    "Historical AQI trend for the latest available 7-day period. "
    "Gaps indicate periods without available observations."
)

st.divider()
# =========================================================
# CITY RISK STATUS
# =========================================================

st.header(
    "⚠️ City Risk Status"
)

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Air Quality Risk",
        air_quality_risk
    )


with col2:

    st.metric(
        "Traffic Risk",
        traffic_risk
    )


st.divider()


# =========================================================
# COMPUTER VISION
# =========================================================

st.header(
    "👁️ Computer Vision Analysis"
)


# ---------------------------------------------------------
# U-Net visualization
# ---------------------------------------------------------

try:

    (
        original_image,
        segmentation_image
    ) = generate_unet_visuals()


    st.subheader(
        "🛣️ Road Scene Segmentation"
    )


    st.caption(
        "Sample road scene from IDD Lite processed using the trained "
        "U-Net semantic segmentation model."
    )


    image_col1, image_col2 = (
        st.columns(2)
    )


    with image_col1:

        st.image(
            original_image,
            caption="Original Road Image",
            width="stretch"
        )


    with image_col2:

        st.image(
            segmentation_image,
            caption="U-Net Segmentation Overlay",
            width="stretch"
        )


except Exception as e:

    st.warning(
        f"U-Net visualization could not be loaded: {e}"
    )


# ---------------------------------------------------------
# Traffic features
# ---------------------------------------------------------

st.subheader(
    "🚗 Extracted Traffic Features"
)


col1, col2, col3 = (
    st.columns(3)
)


with col1:

    st.metric(
        "Vehicle Pixel Ratio",
        f"{traffic['vehicle_ratio']:.4f}"
    )


with col2:

    st.metric(
        "Drivable Area",
        f"{traffic['drivable_ratio']:.4f}"
    )


with col3:

    st.metric(
        "Sky Area",
        f"{traffic['sky_ratio']:.4f}"
    )


st.info(
    """
    Traffic density is an engineered scene-level indicator derived
    from the project's U-Net semantic segmentation pipeline and
    traffic model. It is not a direct measurement of real-world
    vehicle volume.
    """
)


st.caption(
    "The IDD Lite road scene is independent of the historical "
    "AQI/weather timestamp and is used for scene-level computer "
    "vision analysis."
)


st.divider()


# =========================================================
# AI DECISION
# =========================================================

st.header(
    "🤖 AI Decision"
)


if recommended_action is not None:

    best_q_value = float(
        q_values[
            ACTIONS.index(
                recommended_action
            )
        ]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Recommended Action",
            f"{recommended_action}% Traffic"
        )


    with col2:

        st.metric(
            "Best DQN Q-Value",
            f"{best_q_value:.4f}"
        )


    st.info(
        f"""
        The DQN selected **{recommended_action}% traffic intervention**
        because this action has the highest estimated Q-value for the
        current tested state within the project's simulated environment.
        """
    )

else:

    st.error(
        f"DQN could not be loaded: {dqn_error}"
    )


st.divider()


# =========================================================
# WHAT-IF TRAFFIC SCENARIOS
# =========================================================

st.header(
    "🔮 What-If Traffic Scenarios"
)


scenario_rows = []


for action in ACTIONS:

    new_density = (
        traffic_density
        * (1 + action / 100)
    )


    benefit = (
        abs(action)
        / 100
    )


    scenario_rows.append(
        {
            "Action":
                f"{action}%",

            "Traffic Density":
                round(
                    new_density,
                    4
                ),

            "Traffic Benefit":
                round(
                    benefit,
                    2
                )
        }
    )


scenario_df = pd.DataFrame(
    scenario_rows
)


# ---------------------------------------------------------
# Traffic density chart
# ---------------------------------------------------------

st.subheader(
    "🚗 Simulated Traffic Density"
)


traffic_chart = scenario_df[
    [
        "Action",
        "Traffic Density"
    ]
].copy()

traffic_chart["Action"] = pd.Categorical(
    traffic_chart["Action"],
    categories=[
        "0%",
        "-10%",
        "-20%",
        "-30%",
        "-40%"
    ],
    ordered=True
)

traffic_chart = (
    traffic_chart
    .sort_values("Action")
    .set_index("Action")
)

st.line_chart(
    traffic_chart,
    width="stretch"
)


# ---------------------------------------------------------
# Scenario table
# ---------------------------------------------------------

st.subheader(
    "📋 Scenario Details"
)


st.dataframe(
    scenario_df,
    width="stretch",
    hide_index=True
)


st.caption(
    "The scenarios represent simulated relative changes in the "
    "engineered traffic-density indicator."
)


st.divider()


# =========================================================
# DQN POLICY
# =========================================================

st.header(
    "🧠 DQN Policy"
)


if q_values is not None:

    q_df = pd.DataFrame(
    {
        "Action": [
            f"{action}%"
            for action in ACTIONS
        ],
        "Q-Value": [
            float(value)
            for value in q_values
        ]
    }
)

q_df["Action"] = pd.Categorical(
    q_df["Action"],
    categories=[
        "0%",
        "-10%",
        "-20%",
        "-30%",
        "-40%"
    ],
    ordered=True
)

q_df = q_df.sort_values("Action")

st.bar_chart(
    q_df.set_index("Action"),
    width="stretch"
)
st.caption(
        "Q-values represent estimated long-term returns within "
        "the project's simulated environment."
    )



# =========================================================
# RAG + GENAI EXPLANATION
# =========================================================

st.header(
    "🧠 RAG-Grounded GenAI Explanation"
)


if recommended_action is not None:

    # -----------------------------------------------------
    # Build scenarios for GenAI
    # -----------------------------------------------------

    scenarios = []


    for action in ACTIONS:

        try:

            scenario = simulate_scenario(
                action,
                traffic_density
            )

        except TypeError:

            # Compatibility fallback
            scenario = {
                "action": action,
                "new_density":
                    traffic_density
                    * (1 + action / 100)
            }


        scenarios.append(
            scenario
        )


    # -----------------------------------------------------
    # Retrieve project knowledge
    # -----------------------------------------------------

    question = (
        "Why did the DQN select the recommended "
        "traffic intervention action?"
    )


    try:

        context = build_rag_context(
            question
        )

    except Exception as e:

        context = (
            "RAG retrieval was unavailable. "
            f"Error: {e}"
        )


    # -----------------------------------------------------
    # Generate explanation
    # -----------------------------------------------------

    try:

        explanation = generate_explanation(
            city=city,
            traffic_density=traffic_density,
            scenarios=scenarios,
            recommended_action=recommended_action,
            q_values=q_values,
            context=context
        )


        st.write(
            explanation
        )


    except Exception as e:

        st.warning(
            f"GenAI explanation could not be generated: {e}"
        )


else:

    st.warning(
        "GenAI explanation requires the DQN decision."
    )


st.divider()


# =========================================================
# IMPORTANT PROJECT INTERPRETATION
# =========================================================

st.header(
    "ℹ️ Interpretation Notes"
)


st.info(
    """
    **Traffic density:** a project-defined scene-level indicator
    derived from the computer-vision pipeline, not a direct
    measurement of real-world traffic volume.

    **What-if simulation:** evaluates hypothetical changes using
    the project's mathematical simulation model.

    **DQN:** selects an action according to learned Q-values in
    the simulated environment.

    **GenAI:** explains the project state and DQN decision using
    retrieved project knowledge.

    The simulation and DQN recommendation should not be interpreted
    as proof of a guaranteed real-world reduction in AQI or pollution.
    """
)


st.divider()


# =========================================================
# AI PIPELINE STATUS
# =========================================================

st.header(
    "⚙️ Integrated AI Pipeline"
)


st.success(
    "✅ AQI Forecasting — XGBoost + Weather"
)

st.success(
    "✅ Computer Vision — U-Net"
)

st.success(
    "✅ Traffic State Estimation"
)

st.success(
    "✅ Digital Twin — City State"
)

st.success(
    "✅ What-If Simulation"
)

st.success(
    "✅ Reinforcement Learning — DQN"
)

st.success(
    "✅ RAG — Knowledge Retrieval"
)

st.success(
    "✅ GenAI — Llama 3.2"
)


st.divider()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "AI Smart City Digital Twin | Academic Prototype | Hyderabad"
)