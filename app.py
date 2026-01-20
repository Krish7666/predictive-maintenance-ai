# =========================================================
# AI-Driven Predictive Maintenance Dashboard
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

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
st.sidebar.markdown("""
**Industrial Machine Failure Prediction**

- Failure Probability
- Root Cause Analysis (Explainable AI)
- Dynamic Machine Defaults

**Tech Stack:**
- LightGBM
- SHAP (XAI)
- Streamlit
""")

menu = st.sidebar.radio("Navigation", ["Home", "Manual Prediction", "Model Info"])

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
    features = ["Type","Air_temperature__K_","Process_temperature__K_",
                "Rotational_speed__rpm_","Torque__Nm_","Tool_wear__min_"]
    X = df[features].copy()
    y = df["Machine_failure"]

    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    return model, encoder, auc_score, features

model, encoder, auc_score, feature_columns = train_model(df)
explainer = shap.TreeExplainer(model)

# -------------------------------
# Machine Defaults & Root Causes
# -------------------------------
machine_defaults = {
    "CNC Lathe Machine": {"Type":"L","Air_temp":300,"Process_temp":320,"RPM":2500,"Torque":50,"Tool_wear":50},
    "CNC Milling Machine": {"Type":"M","Air_temp":310,"Process_temp":330,"RPM":4000,"Torque":80,"Tool_wear":80},
    "Drilling Machine": {"Type":"L","Air_temp":300,"Process_temp":310,"RPM":1500,"Torque":100,"Tool_wear":60},
    "Grinding Machine": {"Type":"M","Air_temp":295,"Process_temp":305,"RPM":5000,"Torque":20,"Tool_wear":30},
    "Tapping Machine": {"Type":"H","Air_temp":300,"Process_temp":315,"RPM":200,"Torque":120,"Tool_wear":40},
    "Broaching Machine": {"Type":"H","Air_temp":305,"Process_temp":325,"RPM":100,"Torque":150,"Tool_wear":70},
    "Shaping Machine": {"Type":"L","Air_temp":298,"Process_temp":312,"RPM":500,"Torque":80,"Tool_wear":45},
    "Slotting Machine": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":600,"Torque":70,"Tool_wear":50},
    "Sawing Machine": {"Type":"M","Air_temp":300,"Process_temp":315,"RPM":1200,"Torque":60,"Tool_wear":55},
    "Conveyor Belt Motor": {"Type":"L","Air_temp":310,"Process_temp":320,"RPM":900,"Torque":40,"Tool_wear":10},
    "Induction Motor": {"Type":"M","Air_temp":300,"Process_temp":330,"RPM":4000,"Torque":40,"Tool_wear":100},
    "Pump Motor": {"Type":"M","Air_temp":300,"Process_temp":330,"RPM":1500,"Torque":50,"Tool_wear":20},
    "Fan/Blower Motor": {"Type":"L","Air_temp":295,"Process_temp":305,"RPM":900,"Torque":30,"Tool_wear":10},
    "Compressor Motor": {"Type":"H","Air_temp":310,"Process_temp":325,"RPM":1200,"Torque":60,"Tool_wear":25}
}

machine_failure_reason = {
    "CNC Lathe Machine":"High RPM increases heat → tool edge wear (Flank/Crater wear)",
    "CNC Milling Machine":"High speed + multi-point cutting → chipping & flank wear",
    "Drilling Machine":"High thrust force + poor cooling → corner & edge wear",
    "Grinding Machine":"Abrasive grain fracture → abrasive wear",
    "Tapping Machine":"High torque → adhesive wear or breakage",
    "Broaching Machine":"Continuous cutting load → progressive wear",
    "Shaping Machine":"Interrupted cutting → edge wear",
    "Slotting Machine":"Vertical cutting → flank wear",
    "Sawing Machine":"Tooth friction & vibration → tooth wear",
    "Conveyor Belt Motor":"Prolonged operation → mechanical stress",
    "Induction Motor":"Overload / high RPM → winding/tool wear",
    "Pump Motor":"High pressure & cavitation → pump wear",
    "Fan/Blower Motor":"Continuous operation → blade & bearing wear",
    "Compressor Motor":"High load → progressive mechanical wear"
}

# =========================================================
# HOME
# =========================================================
if menu=="Home":
    st.title("Industrial Predictive Maintenance Dashboard")
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### System Features
        - Predicts machine failure probability
        - Dynamic machine defaults per type
        - Root cause analysis & reason
        - Supports manual and real-time input
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu=="Manual Prediction":
    st.title("Machine Failure Prediction & Diagnosis")
    st.info("Select machine to test. Ideal values will auto-fill. You can modify before prediction.")

    with st.form("manual_form"):
        col0,col1,col2,col3 = st.columns([1,1,1,1])

        with col0:
            machine_name = st.selectbox("Machine Under Test", list(machine_defaults.keys()))

        defaults = machine_defaults[machine_name]

        with col1:
            machine_type = st.selectbox("Machine Class", ["L","M","H"], index=["L","M","H"].index(defaults["Type"]))
            air_temp = st.number_input("Air Temperature (K)", 250, 400, value=defaults["Air_temp"])
        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250, 400, value=defaults["Process_temp"])
            rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=defaults["RPM"])
        with col3:
            torque = st.number_input("Torque (Nm)", 0, 200, value=defaults["Torque"])
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=defaults["Tool_wear"])

        submit = st.form_submit_button("Predict & Analyze")

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
        st.subheader(f"Prediction Result for `{machine_name}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Machine Status
        if prob<0.1:
            status = "Running at Full Capacity"
        elif prob<0.5:
            status = "Partial Capability"
        else:
            status = "Failure Likely / Needs Attention"
        st.metric("Machine Status", status)

        # Root Cause
        st.subheader("Root Cause Analysis")
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values,list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)

        st.info(f"Reason for failure: {machine_failure_reason[machine_name]}")

# =========================================================
# MODEL INFO
# =========================================================
if menu=="Model Info":
    st.title("Model & Dataset Information")
    st.markdown("""
    ### Model
    - LightGBM Classifier (Gradient Boosting Trees)
    - High accuracy on tabular industrial data

    ### Explainability
    - SHAP (Explainable AI)
    - Shows feature contribution to failure

    ### Dataset
    - AI4I 2020 Predictive Maintenance Dataset
    - 10,000 real industrial samples
    """)
    st.success("Industrial-grade predictive maintenance system ready for testing.")
