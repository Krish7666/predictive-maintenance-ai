# =========================================================
# AI-Driven Predictive Maintenance with Diagnosis & SHAP
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# ---------------------------------------------------------
# Machine ideal operating profiles
# ---------------------------------------------------------
MACHINE_PROFILES = {
    "CNC Milling":      {"type":"M","rpm":2500, "torque":45, "wear":60, "air_temp":300, "proc_temp":310},
    "Drilling Machine": {"type":"H","rpm":1800, "torque":70, "wear":50, "air_temp":295, "proc_temp":320},
    "Grinding Machine": {"type":"H","rpm":3200, "torque":20, "wear":40, "air_temp":300, "proc_temp":315},
    "Tapping Machine":  {"type":"L","rpm":600,  "torque":95, "wear":30, "air_temp":285, "proc_temp":300},
    "Broaching Machine":{"type":"L","rpm":300,  "torque":110,"wear":40, "air_temp":280, "proc_temp":300},
    "Shaping Machine":  {"type":"L","rpm":450,  "torque":80, "wear":50, "air_temp":290, "proc_temp":305},
    "Slotting Machine": {"type":"M","rpm":500,  "torque":75, "wear":50, "air_temp":290, "proc_temp":310},
    "Sawing Machine":   {"type":"M","rpm":1200, "torque":55, "wear":60, "air_temp":300, "proc_temp":310},
    "Induction Motor":  {"type":"M","rpm":1450, "torque":35, "wear":20, "air_temp":295, "proc_temp":305}
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav_menu"
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
@st.cache_data
def train_model(df):
    features = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]

    X = df[features].copy()
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

    return model, le, auc, features

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### System Capabilities
        - Predict failure probability  
        - Engineering-based diagnosis  
        - SHAP explainability  
        - Multi-machine support
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    st.divider()
    st.markdown("""
    This system predicts failures in **rotating & cutting machinery** based on:
    - RPM, Torque, Thermal conditions, Tool wear  
    - Supports manual input & real-time diagnostics
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    # Machine selection
    machine = st.selectbox(
        "Machine Under Test",
        list(MACHINE_PROFILES.keys()),
        key="machine_select"
    )

    profile = MACHINE_PROFILES[machine]

    st.info("Ideal values are preloaded. Modify as needed for testing extremes.")

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        rpm = st.number_input("Rotational Speed (RPM)", 0, 5000, profile["rpm"], key="rpm")
    with col2:
        torque = st.number_input("Torque (Nm)", 0.0, 150.0, profile["torque"], key="torque")
    with col3:
        tool_wear = st.number_input("Tool Wear (min)", 0, 500, profile["wear"], key="wear")

    air_temp = st.slider("Air Temperature (K)", 270, 330, profile["air_temp"])
    proc_temp = st.slider("Process Temperature (K)", 290, 380, profile["proc_temp"])

    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Type": profile["type"],
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])
        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Color-coded status
        status = (
            "🟢 Normal Operation" if prob < 0.25
            else "🟡 Degrading Condition" if prob < 0.6
            else "🔴 Failure Likely"
        )
        st.subheader(status)

        # ---------------- Diagnosis ----------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_array[0], index=FEATURES).abs().sort_values(ascending=False)
        top3 = impact.head(3)

        st.subheader("🧠 Root Cause Analysis (Top 3)")
        st.bar_chart(top3)

        # Detailed cause messages
        cause_msgs = {
            "rpm": "High RPM increases thermal and dynamic stress, accelerates vibration fatigue, and causes tool wear.",
            "torque": "High torque puts excessive mechanical load on components, leading to wear or fracture.",
            "wear": "Tool wear beyond ideal range causes friction, heat, and poor machining performance."
        }

        for feature in top3.index:
            key = feature.lower()
            message = cause_msgs.get(key, "Combined thermal and mechanical factors contribute to failure.")
            st.info(f"**{feature}** → {message}")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown("""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    Designed for **industrial rotating & cutting machinery**.
    """)
