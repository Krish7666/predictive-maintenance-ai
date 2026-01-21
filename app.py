# =========================================================
# MotorGuard AI – Predictive Maintenance for Induction Motors
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MotorGuard AI",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("⚙️ MotorGuard AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav"
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train LightGBM Model
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

    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, encoder, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("⚙️ MotorGuard AI")
    st.subheader("Physics-Aware Predictive Maintenance for Induction Motors")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **What this system does**
        - Predicts induction motor failure probability  
        - Understands torque–RPM behavior  
        - Explains *why* failure risk increases  
        - Supports industrial rotating machinery  

        **Supported Machines**
        - Conveyor motors  
        - Pumps  
        - Fans & blowers  
        - Gearbox-driven systems  
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Induction Motor Failure Prediction")

    st.info(
        "Increase torque to simulate load. "
        "System automatically reflects RPM drop (motor physics)."
    )

    # -------- Ideal Baseline --------
    BASE_RPM = 1450.0
    BASE_TORQUE = 35.0

    col1, col2, col3 = st.columns(3)

    with col1:
        torque = st.number_input(
            "Load Torque (Nm)",
            value=BASE_TORQUE,
            step=1.0,
            key="torque"
        )

    # Physics: Torque ↑ → RPM ↓
    rpm = max(500.0, BASE_RPM - (torque - BASE_TORQUE) * 6)

    with col2:
        st.number_input(
            "Motor Speed (RPM)",
            value=float(rpm),
            disabled=True
        )

    with col3:
        tool_wear = st.number_input(
            "Operational Wear (min)",
            value=50.0,
            step=5.0
        )

    air_temp = st.slider("Air Temperature (K)", 280.0, 340.0, 300.0)
    proc_temp = st.slider("Process Temperature (K)", 300.0, 400.0, 320.0)

    # -------- Predict Button --------
    if st.button("🔍 Predict Failure Risk"):
        input_df = pd.DataFrame([{
            "Type": "M",
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # -------- Status --------
        if prob < 0.25:
            st.success("🟢 Normal Operating Zone")
        elif prob < 0.6:
            st.warning("🟡 Overload Developing – Monitor Closely")
        else:
            st.error("🔴 Critical Risk – Failure Likely")

        # -------- Root Cause Analysis --------
        shap_vals = explainer.shap_values(input_df)
        shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_arr[0], index=FEATURES).abs().sort_values(ascending=False)

        st.subheader("🧠 Engineering Diagnosis")

        if impact.index[0] == "Torque__Nm_":
            st.write(
                "High torque demand is overloading the motor. "
                "This reduces speed, increases current draw, raises temperature, "
                "and accelerates insulation and bearing degradation."
            )
        elif impact.index[0] == "Rotational_speed__rpm_":
            st.write(
                "Reduced motor speed indicates sustained overload conditions. "
                "Low RPM under load increases thermal stress and mechanical fatigue."
            )
        elif impact.index[0] == "Tool_wear__min_":
            st.write(
                "Prolonged operation without maintenance has increased wear, "
                "causing friction losses and efficiency drop."
            )
        else:
            st.write(
                "Failure risk is driven by combined thermal and mechanical stress."
            )

        # -------- Stress Graph --------
        st.subheader("📉 Torque vs RPM Stress Behavior")

        torque_range = np.linspace(BASE_TORQUE, torque + 20, 20)
        rpm_curve = BASE_RPM - (torque_range - BASE_TORQUE) * 6

        fig, ax = plt.subplots()
        ax.plot(torque_range, rpm_curve)
        ax.set_xlabel("Torque (Nm)")
        ax.set_ylabel("RPM")
        ax.set_title("Induction Motor Load Behavior")
        ax.grid(True)

        st.pyplot(fig)

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Model**: LightGBM Classifier  
    **Explainability**: SHAP  
    **Dataset**: AI4I 2020  

    **Design Focus**
    - Induction motors  
    - Physics-aware reasoning  
    - Failure prevention, not just prediction  
    """)
