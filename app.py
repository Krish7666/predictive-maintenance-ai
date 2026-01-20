# =========================================================
# AI-Driven Predictive Maintenance Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
st.sidebar.markdown(
    """
    AI-Driven Predictive Maintenance System

    - Failure Probability Prediction
    - Root Cause Analysis (Explainable AI)
    - Industrial Sensor Data

    **Tech Stack**
    - LightGBM
    - SHAP (XAI)
    - Streamlit
    """
)

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"]
)

# -------------------------------
# Machine Defaults
# -------------------------------
machine_defaults = {
    "CNC Milling": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":4500,"Torque":60,"Tool_wear":50,
                    "Reason":"High speed & multi-point cutting → Chipping/Flank wear"},
    "Drilling Machine": {"Type":"H","Air_temp":300,"Process_temp":320,"RPM":2500,"Torque":80,"Tool_wear":40,
                         "Reason":"High thrust & poor cooling → Corner/Edge wear"},
    "Grinding Machine": {"Type":"H","Air_temp":300,"Process_temp":300,"RPM":5000,"Torque":20,"Tool_wear":30,
                         "Reason":"Abrasive grain fracture → Abrasive wear"},
    "Tapping Machine": {"Type":"L","Air_temp":300,"Process_temp":310,"RPM":500,"Torque":100,"Tool_wear":60,
                        "Reason":"High torque → Adhesive wear/Breakage"},
    "Broaching Machine": {"Type":"L","Air_temp":300,"Process_temp":310,"RPM":100,"Torque":120,"Tool_wear":70,
                          "Reason":"Continuous cutting → Progressive wear"},
    "Shaping Machine": {"Type":"L","Air_temp":300,"Process_temp":310,"RPM":200,"Torque":80,"Tool_wear":50,
                        "Reason":"Interrupted cutting → Edge wear"},
    "Slotting Machine": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":150,"Torque":60,"Tool_wear":40,
                         "Reason":"Vertical cutting force → Flank wear"},
    "Sawing Machine": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":1000,"Torque":50,"Tool_wear":30,
                       "Reason":"Tooth friction & vibration → Tooth wear"},
    "Induction Motor": {"Type":"H","Air_temp":300,"Process_temp":300,"RPM":1800,"Torque":40,"Tool_wear":0,
                        "Reason":"Electrical & mechanical load → Insulation/Thermal wear"}
}

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model(df):
    feature_columns = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]
    X = df[feature_columns].copy()
    y = df["Machine_failure"]

    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu=="Home":
    st.title("🔧 Predictive Maintenance Dashboard")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### System Features
        - Predicts machine failure probability
        - Root cause analysis (XAI)
        - Ideal readings auto-populated per machine
        """)
    with col2:
        st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")

    st.divider()
    st.markdown("""
        ### Why Predictive Maintenance?
        - Minimize downtime
        - Reduce maintenance costs
        - Extend machine life
        - Safer operations
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu=="Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine_name = st.selectbox("Select Machine Under Test", list(machine_defaults.keys()), key="machine_select")

    # Reset defaults if machine changes
    if "last_machine" not in st.session_state or st.session_state.last_machine != machine_name:
        st.session_state.last_machine = machine_name
        defaults = machine_defaults[machine_name]
        st.session_state.machine_type = defaults["Type"]
        st.session_state.air_temp = defaults["Air_temp"]
        st.session_state.process_temp = defaults["Process_temp"]
        st.session_state.rpm = defaults["RPM"]
        st.session_state.torque = defaults["Torque"]
        st.session_state.tool_wear = defaults["Tool_wear"]
        st.session_state.failure_reason = defaults["Reason"]

    # Input parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        machine_type = st.selectbox("Machine Class", ["L","M","H"], 
                                    index=["L","M","H"].index(st.session_state.machine_type), key="machine_type")
        air_temp = st.number_input("Air Temperature (K)", 250, 400, value=st.session_state.air_temp, key="air_temp")
    with col2:
        process_temp = st.number_input("Process Temperature (K)", 250, 400, value=st.session_state.process_temp, key="process_temp")
        rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=st.session_state.rpm, key="rpm")
    with col3:
        torque = st.number_input("Torque (Nm)", 0, 200, value=st.session_state.torque, key="torque")
        tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=st.session_state.tool_wear, key="tool_wear")

    submit = st.button("Predict Failure")

    if submit:
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": process_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])
        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader(f"Prediction for `{machine_name}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Machine capability
        if prob < 0.1:
            capability = "Running at Full Capability"
        elif prob < 0.5:
            capability = "Running at Partial Capability"
        else:
            capability = "Failure Likely / Needs Attention"
        st.metric("Machine Status", capability)

        # Root cause analysis
        st.divider()
        st.subheader("Root Cause Analysis")
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)

        st.info(f"**Main Reason for Failure:** {st.session_state.failure_reason}")

# =========================================================
# MODEL INFO
# =========================================================
if menu=="Model Info":
    st.title("📚 Model Information")
    st.markdown("""
    ### Model
    LightGBM Classifier - Gradient boosting trees, fast & accurate

    ### Explainability
    SHAP (XAI) - Shows feature impact & root causes

    ### Dataset
    AI4I 2020 Predictive Maintenance Dataset
    10,000 industrial samples
    """)
    st.success("Industrial-grade predictive maintenance dashboard ready for real-world use.")
