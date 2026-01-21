import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    layout="wide"
)

# ---------------- MACHINE IDEAL VALUES ----------------
MACHINE_PROFILES = {
    "CNC Milling":        dict(rpm=3000, torque=35, wear=20),
    "Drilling Machine":  dict(rpm=1500, torque=45, wear=15),
    "Grinding Machine":  dict(rpm=6000, torque=15, wear=10),
    "Tapping Machine":   dict(rpm=800,  torque=55, wear=12),
    "Broaching Machine": dict(rpm=400,  torque=65, wear=18),
    "Shaping Machine":   dict(rpm=600,  torque=50, wear=20),
    "Slotting Machine":  dict(rpm=700,  torque=45, wear=22),
    "Sawing Machine":    dict(rpm=1200, torque=40, wear=25),
    "Induction Motor":   dict(rpm=1450, torque=30, wear=5),
}

# ---------------- LOAD & TRAIN MODEL ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("ai4i2020.csv")

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[features]
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc, features

model, auc_score, FEATURES = load_model()

# ---------------- ROOT CAUSE ANALYSIS ----------------
def root_cause(machine, rpm, torque, wear):
    reasons = []

    ideal = MACHINE_PROFILES[machine]

    if rpm > ideal["rpm"] * 1.25:
        reasons.append("Excessive RPM causing overheating and accelerated tool edge degradation.")

    if torque > ideal["torque"] * 1.25:
        reasons.append("High torque load increasing mechanical stress on spindle and tool.")

    if wear > ideal["wear"] * 1.5:
        reasons.append("Tool wear beyond safe limit indicating prolonged or aggressive operation.")

    if not reasons:
        reasons.append("All operating parameters are within healthy limits. Machine is running optimally.")

    return reasons

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔧 Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav_radio"
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("AI-Driven Predictive Maintenance System")
    st.markdown("""
    **Capabilities**
    - Machine failure probability prediction  
    - Machine-specific root cause diagnosis  
    - Real industrial sensor data (AI4I 2020)  
    - Explainable AI (SHAP ready)  
    """)

    st.metric("Model ROC-AUC Score", round(auc_score, 3))

# ---------------- MANUAL PREDICTION ----------------
elif menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine = st.selectbox(
        "Machine Under Test",
        list(MACHINE_PROFILES.keys()),
        key="machine_select"
    )

    ideal = MACHINE_PROFILES[machine]

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature (K)", 290.0, 330.0, 300.0)
        process_temp = st.number_input("Process Temperature (K)", 300.0, 450.0, 340.0)
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            0, 8000, ideal["rpm"], key="rpm_input"
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            0.0, 100.0, float(ideal["torque"]), key="torque_input"
        )
        wear = st.number_input(
            "Tool Wear (min)",
            0, 300, ideal["wear"], key="wear_input"
        )

    if st.button("Run Prediction"):
        input_df = pd.DataFrame([[
            air_temp, process_temp, rpm, torque, wear
        ]], columns=FEATURES)

        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔮 Prediction Result")
        st.progress(prob)
        st.write(f"**Failure Probability:** `{prob*100:.2f}%`")

        # ---------- Graph ----------
        st.subheader("📈 Sensor Contribution Overview")
        fig, ax = plt.subplots()
        ax.bar(FEATURES, input_df.iloc[0])
        ax.set_xticklabels(FEATURES, rotation=45, ha="right")
        st.pyplot(fig)

        # ---------- Root Cause ----------
        st.subheader("🧠 Root Cause Analysis")
        causes = root_cause(machine, rpm, torque, wear)
        for c in causes:
            st.write("•", c)

# ---------------- MODEL INFO ----------------
elif menu == "Model Info":
    st.title("📌 Model Information")
    st.write("**Algorithm:** LightGBM Classifier")
    st.write("**Dataset:** AI4I 2020 Predictive Maintenance")
    st.write("**Features Used:**")
    for f in FEATURES:
        st.write("•", f)
