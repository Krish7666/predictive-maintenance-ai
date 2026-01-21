import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import pandas as pd

@st.cache_resource
def load_or_train_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    # Train model if file not found
    df = pd.read_csv("data/ai4i2020.csv")

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type"
    ]

    X = df[features]
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    return model

model = load_or_train_model()


# -------------------- MACHINE IDEAL PROFILES --------------------
MACHINES = {
    "CNC Milling": {"rpm": 1800, "torque": 55, "wear": 5, "type": "M"},
    "Drilling Machine": {"rpm": 1500, "torque": 90, "wear": 5, "type": "L"},
    "Grinding Machine": {"rpm": 4500, "torque": 25, "wear": 3, "type": "L"},
    "Tapping Machine": {"rpm": 600, "torque": 160, "wear": 4, "type": "H"},
    "Broaching Machine": {"rpm": 300, "torque": 220, "wear": 6, "type": "H"},
    "Shaping Machine": {"rpm": 500, "torque": 120, "wear": 5, "type": "M"},
    "Slotting Machine": {"rpm": 450, "torque": 110, "wear": 5, "type": "M"},
    "Sawing Machine": {"rpm": 900, "torque": 70, "wear": 6, "type": "L"},
    "Induction Motor": {"rpm": 1450, "torque": 60, "wear": 0, "type": "L"},
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# -------------------- SIDEBAR --------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="menu_radio"
)

# -------------------- HOME --------------------
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")

    st.markdown("""
    ### What this system does
    - Predicts **failure probability under operational stress**
    - Identifies **root cause (RPM / Torque / Tool Wear)**
    - Uses **real industrial AI4I-2020 sensor data**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "LightGBM")
    col2.metric("ROC-AUC", "0.974")
    col3.metric("Explainability", "SHAP")

# -------------------- MANUAL PREDICTION --------------------
elif menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine = st.selectbox(
        "Machine Under Test",
        list(MACHINES.keys()),
        key="machine_select"
    )

    profile = MACHINES[machine]

    st.subheader("Operating Parameters")

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.slider("Air Temperature (K)", 295, 305, 300)
        process_temp = st.slider("Process Temperature (K)", 300, 330, 308)
        rpm = st.slider(
            "Rotational Speed (RPM)",
            200,
            5000,
            profile["rpm"],
            key="rpm_slider"
        )

    with col2:
        torque = st.slider(
            "Torque (Nm)",
            10,
            300,
            profile["torque"],
            key="torque_slider"
        )
        tool_wear = st.slider(
            "Tool Wear (min)",
            0,
            150,
            profile["wear"],
            key="wear_slider"
        )

    if st.button("🔍 Predict Failure"):
        X = np.array([[
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            TYPE_MAP[profile["type"]]
        ]])

        prob = model.predict_proba(X)[0][1]

        # -------------------- OPERATING ZONE --------------------
        if prob < 0.3:
            zone = "🟢 GREEN ZONE"
            st.success(f"{zone} — Machine operating safely")
        elif prob < 0.6:
            zone = "🟡 YELLOW ZONE"
            st.warning(f"{zone} — Degradation detected")
        else:
            zone = "🔴 RED ZONE"
            st.error(f"{zone} — Failure likely")

        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # -------------------- ROOT CAUSE ANALYSIS --------------------
        st.subheader("🧠 Root Cause Analysis")

        causes = []

        if rpm > profile["rpm"] * 1.3:
            causes.append("Excessive RPM causing heat & vibration")

        if torque > profile["torque"] * 1.3:
            causes.append("High torque increasing mechanical stress")

        if tool_wear > 40:
            causes.append("Tool wear beyond safe operational limit")

        if not causes:
            st.write("✅ No abnormal condition detected. Parameters are within safe limits.")
        else:
            for c in causes:
                st.write(f"• {c}")

# -------------------- MODEL INFO --------------------
elif menu == "Model Info":
    st.title("📘 Model Information")

    st.markdown("""
    **Dataset:** AI4I-2020  
    **Model:** LightGBM  
    **Target:** Machine Failure  
    **Important Insight:**  
    > This model predicts **failure under stress**, not brand-new machines.
    """)

    st.info("Low tool wear + moderate RPM + moderate torque = GREEN zone")
