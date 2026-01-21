import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="AI Predictive Maintenance – Induction Motors", layout="wide")

# ------------------------------
# Load & Train Model (SAFE)
# ------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("predictive_maintenance.csv")

    FEATURES = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[FEATURES].values  # ✅ FIX: no feature-name conflict
    y = df["Machine failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc, FEATURES

model, auc_score, FEATURES = train_model()

# ------------------------------
# RUL LOGIC (INDUSTRY SAFE)
# ------------------------------
def calculate_rul(failure_prob, torque, rpm):
    base_rul = max(50, (1 - failure_prob) * 1000)

    load_factor = torque / max(rpm, 1)
    degradation = load_factor * 120

    rul = max(30, int(base_rul - degradation))
    return rul

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Manual Input", "RUL Prediction", "Maintenance Advice"]
)

# ------------------------------
# OVERVIEW
# ------------------------------
if page == "Overview":
    st.title("🔧 AI-Based Predictive Maintenance (Induction Motors)")
    st.markdown("""
    **Target Machines**
    - Induction Motors
    - Pumps
    - Fans & Blowers
    - Conveyors
    - Gearbox-driven systems

    **Core Capabilities**
    - Failure Probability Prediction
    - Remaining Useful Life (RUL)
    - Load-based What-If Simulation
    """)

    st.metric("Model ROC-AUC Score", round(auc_score, 3))

# ------------------------------
# MANUAL INPUT
# ------------------------------
if page == "Manual Input":
    st.title("⚙️ Manual Motor Input")

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)
        torque = st.number_input("Torque (Nm)", value=40.0)

    with col2:
        rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
        tool_wear = st.number_input("Wear Time (minutes)", value=120.0)

    # Torque–RPM realism
    rpm = max(300, rpm - (torque * 2))

    input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])

    if st.button("🔮 Predict Motor Failure"):
        prob = model.predict_proba(input_data)[0][1]
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        st.session_state["last_prob"] = prob
        st.session_state["last_rpm"] = rpm
        st.session_state["last_torque"] = torque

# ------------------------------
# RUL PAGE
# ------------------------------
if page == "RUL Prediction":
    st.title("🕒 Remaining Useful Life (RUL)")

    if "last_prob" not in st.session_state:
        st.warning("⚠️ Run a prediction first")
    else:
        rul = calculate_rul(
            st.session_state["last_prob"],
            st.session_state["last_torque"],
            st.session_state["last_rpm"]
        )

        if rul > 600:
            status = "🟢 Healthy"
        elif rul > 300:
            status = "🟡 Warning"
        else:
            status = "🔴 Critical"

        st.metric("Estimated RUL", f"{rul} hours")
        st.write("Motor Status:", status)

        st.subheader("📉 What-If Load Simulation")
        st.write(f"+10% Load → ~{int(rul * 0.7)} hrs")
        st.write(f"+20% Load → ~{int(rul * 0.5)} hrs")

# ------------------------------
# MAINTENANCE ADVICE
# ------------------------------
if page == "Maintenance Advice":
    st.title("🧠 Maintenance Recommendations")

    if "last_prob" not in st.session_state:
        st.warning("⚠️ Predict failure first")
    else:
        prob = st.session_state["last_prob"]

        if prob < 0.3:
            st.success("✅ Motor operating normally. Continue routine monitoring.")
        elif prob < 0.6:
            st.warning("⚠️ Schedule inspection. Check bearings & lubrication.")
        else:
            st.error("🚨 Immediate maintenance required. Risk of breakdown high.")
