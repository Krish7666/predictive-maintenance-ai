# =========================================================
# AI-Driven Predictive Maintenance – Induction Motor Focus
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------------
# Physics-based RPM correction
# ---------------------------------------------------------
def apply_torque_rpm_coupling(rpm, torque, rated_torque=40):
    slip_factor = 1 + (torque / rated_torque) * 0.15
    corrected_rpm = rpm / slip_factor
    return max(corrected_rpm, 100)

# ---------------------------------------------------------
# Ideal induction motor profile
# ---------------------------------------------------------
INDUCTION_MOTOR_PROFILE = {
    "rpm": 1450.0,
    "torque": 35.0,
    "wear": 20.0,
    "air_temp": 300.0,
    "proc_temp": 310.0
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav"
)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train LightGBM model
# ---------------------------------------------------------
@st.cache_data
def train_model(df):
    FEATURES = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]

    X = df[FEATURES].copy()
    y = df["Machine_failure"]

    le = LabelEncoder()
    X["Type"] = le.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, le, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("⚙️ Induction Motor Predictive Maintenance")

    st.markdown("""
    ### What this system does
    - Predicts **failure probability** of induction motors  
    - Uses **real industrial sensor data**  
    - Applies **motor physics (Torque–RPM coupling)**  
    - Provides **engineering-grade root cause analysis**
    """)

    st.metric("Model ROC-AUC", f"{auc_score:.3f}")

    st.info(
        "This model is best suited for **induction-motor-driven systems** such as "
        "conveyors, pumps, fans, blowers, and rotating machinery."
    )

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Induction Motor Failure Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Rated RPM",
            value=float(INDUCTION_MOTOR_PROFILE["rpm"]),
            step=10.0,
            key="rpm"
        )

    with col2:
        torque = st.number_input(
            "Load Torque (Nm)",
            value=float(INDUCTION_MOTOR_PROFILE["torque"]),
            step=1.0,
            key="torque"
        )

    with col3:
        tool_wear = st.number_input(
            "Wear Indicator (min)",
            value=float(INDUCTION_MOTOR_PROFILE["wear"]),
            step=1.0,
            key="wear"
        )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=float(INDUCTION_MOTOR_PROFILE["air_temp"]),
        step=1.0
    )

    proc_temp = st.number_input(
        "Process Temperature (K)",
        value=float(INDUCTION_MOTOR_PROFILE["proc_temp"]),
        step=1.0
    )

    # -----------------------------------------------------
    # Apply physics
    # -----------------------------------------------------
    effective_rpm = apply_torque_rpm_coupling(rpm, torque)

    st.caption(f"⚙️ Effective RPM under load: **{effective_rpm:.0f} RPM**")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Type": "M",
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": effective_rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]

        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.25:
            st.success("🟢 Normal Operation")
        elif prob < 0.6:
            st.warning("🟡 Degrading Condition")
        else:
            st.error("🔴 Failure Likely")

        # -------------------------------------------------
        # Torque vs RPM Graph
        # -------------------------------------------------
        torque_range = np.linspace(5, torque * 1.5 + 10, 50)
        rpm_curve = [apply_torque_rpm_coupling(rpm, t) for t in torque_range]

        fig, ax = plt.subplots()
        ax.plot(torque_range, rpm_curve, label="Motor Characteristic")
        ax.scatter(torque, effective_rpm, color="red", label="Operating Point")
        ax.set_xlabel("Torque (Nm)")
        ax.set_ylabel("RPM")
        ax.set_title("Induction Motor Torque–RPM Behaviour")
        ax.legend()
        st.pyplot(fig)

        # -------------------------------------------------
        # Root Cause Analysis
        # -------------------------------------------------
        st.subheader("🧠 Root Cause Analysis")

        if torque > 1.3 * INDUCTION_MOTOR_PROFILE["torque"]:
            st.info(
                "The motor is operating under **excessive load torque**, increasing slip "
                "and reducing RPM. This condition causes higher rotor current, thermal stress, "
                "and accelerated insulation aging."
            )

        elif effective_rpm < 0.75 * rpm:
            st.info(
                "Significant RPM drop detected. This indicates high slip due to mechanical overload "
                "or possible bearing and alignment issues."
            )

        elif proc_temp - air_temp > 40:
            st.info(
                "Abnormal temperature rise suggests insufficient cooling or continuous overload, "
                "which may lead to winding degradation."
            )

        elif tool_wear > 1.5 * INDUCTION_MOTOR_PROFILE["wear"]:
            st.info(
                "Increased wear indicates long-term mechanical stress, leading to increased power draw "
                "and vibration-related failure risks."
            )

        else:
            st.info(
                "Failure risk is driven by a combination of mechanical load and thermal conditions. "
                "Preventive maintenance is recommended."
            )

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Model:** LightGBM Classifier  
    **Dataset:** AI4I 2020 Predictive Maintenance  
    **Target Machines:** Induction-motor-driven systems  

    This system combines **machine learning** with **electrical machine physics**
    for realistic industrial failure prediction.
    """)
