# =========================================================
# AI-Driven Predictive Maintenance Dashboard (Enhanced)
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
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**  

✔ Failure Probability Prediction  
✔ Root Cause Analysis (Explainable AI)  
✔ Real Industrial Sensor Data  

**Tech Stack**  
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

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)

    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)

# -------------------------------
# SHAP Explainer
# -------------------------------
explainer = shap.TreeExplainer(model)

# -------------------------------
# Machine Defaults & Root Cause
# -------------------------------
machine_defaults = {
    "CNC Lathe": {"Type":"L","Air_temp":300,"Process_temp":320,"RPM":2500,"Torque":50,"Tool_wear":50,"Reason":"High RPM → High heat → Tool edge wear"},
    "CNC Milling": {"Type":"M","Air_temp":310,"Process_temp":330,"RPM":4000,"Torque":80,"Tool_wear":80,"Reason":"Multi-point cutting + high speed → Chipping/Flank wear"},
    "Drilling Machine": {"Type":"L","Air_temp":300,"Process_temp":310,"RPM":1500,"Torque":100,"Tool_wear":60,"Reason":"High thrust force + poor cooling → Corner/Edge wear"},
    "Grinding Machine": {"Type":"M","Air_temp":295,"Process_temp":305,"RPM":5000,"Torque":20,"Tool_wear":30,"Reason":"Abrasive grain fracture → Abrasive wear"},
    "Tapping Machine": {"Type":"H","Air_temp":300,"Process_temp":315,"RPM":200,"Torque":120,"Tool_wear":40,"Reason":"High torque → Friction → Adhesive wear/Breakage"},
    "Broaching Machine": {"Type":"H","Air_temp":305,"Process_temp":325,"RPM":100,"Torque":150,"Tool_wear":70,"Reason":"Continuous cutting load → Progressive wear"},
    "Shaping Machine": {"Type":"L","Air_temp":298,"Process_temp":312,"RPM":500,"Torque":80,"Tool_wear":45,"Reason":"Interrupted cutting → Edge wear"},
    "Slotting Machine": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":600,"Torque":70,"Tool_wear":50,"Reason":"Vertical cutting force → Flank wear"},
    "Sawing Machine": {"Type":"M","Air_temp":300,"Process_temp":315,"RPM":1200,"Torque":60,"Tool_wear":55,"Reason":"Tooth friction & vibration → Tooth wear"},
    "Induction Motor": {"Type":"M","Air_temp":300,"Process_temp":330,"RPM":4000,"Torque":40,"Tool_wear":100,"Reason":"Overload & heat → Insulation degradation"}
}

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 System Features
        - Predicts **machine failure probability**
        - Performs **root cause analysis**
        - Uses **real industrial sensor data**
        - Supports **manual input**
        """)
    with col2:
        st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")

    st.divider()
    st.markdown("""
    ### 🏭 Why Predictive Maintenance?
    - Reduced downtime  
    - Lower maintenance cost  
    - Improved machine life  
    - Safer operations  
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")
    st.info("Select a machine. Ideal values will auto-fill. You can modify them.")

    with st.form("manual_form"):
        col0, col1, col2, col3 = st.columns([1,1,1,1])

        with col0:
            machine_name = st.selectbox("Machine Under Test", list(machine_defaults.keys()), key="machine_select")

        defaults = machine_defaults[machine_name]

        with col1:
            machine_type = st.selectbox("Machine Class", ["L","M","H"],
                                        index=["L","M","H"].index(defaults["Type"]),
                                        key="machine_type")
            air_temp = st.number_input("Air Temperature (K)", 250, 400, value=defaults["Air_temp"], key="air_temp")
        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250, 400, value=defaults["Process_temp"], key="process_temp")
            rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=defaults["RPM"], key="rpm")
        with col3:
            torque = st.number_input("Torque (Nm)", 0, 200, value=defaults["Torque"], key="torque")
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=defaults["Tool_wear"], key="tool_wear")

        submit = st.form_submit_button("🔍 Predict & Diagnose")

    if submit:
        # Prepare input for model
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
        st.subheader(f"📈 Prediction Result for `{machine_name}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Machine Status
        if prob < 0.1:
            status = "💚 Running at Full Capability"
        elif prob < 0.5:
            status = "🟡 Running at Partial Capability"
        else:
            status = "🔴 Failure Likely / Needs Attention"
        st.metric("Machine Status", status)

        # Root Cause Analysis
        st.divider()
        st.subheader("🧠 Root Cause Analysis")

        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_array = shap_values[1]
        else:
            shap_array = shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)

        # Show reason from machine defaults + SHAP feature
        st.info(f"**Machine-specific Failure Reason:** {defaults['Reason']}")
        st.info(f"**Top Contributing Feature (SHAP):** {impact.index[0]}")

# =========================================================
# MODEL INFO PAGE
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown("""
    ### 🔹 Model
    **LightGBM Classifier**
    - Gradient boosting decision trees
    - High accuracy on tabular data

    ### 🔹 Explainability
    **SHAP (Explainable AI)**
    - Explains individual predictions
    - Identifies failure root causes

    ### 🔹 Dataset
    - AI4I 2020 Predictive Maintenance Dataset
    - 10,000 industrial samples
    """)
    st.success("Explainable AI-based predictive maintenance system ready for real-world use.")
