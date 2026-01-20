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
# Page Config
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
st.sidebar.markdown(
    """
    **AI-Driven Predictive Maintenance System**

    ✔ Failure Probability Prediction  
    ✔ Root Cause Analysis (Explainable AI)  
    ✔ Real Industrial Sensor Data  

    **Technology Stack**
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
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)
explainer = shap.TreeExplainer(model)

# -------------------------------
# Machine Ideal Defaults & Reasons
# -------------------------------
machine_defaults = {
    "CNC Milling": {"Type":"M","Air_temp":300,"Process_temp":310,"RPM":4000,"Torque":60,"Tool_wear":50,
                    "Reason":"High speed and multi-point cutting can cause chipping or flank wear"},
    "Drilling Machine": {"Type":"M","Air_temp":295,"Process_temp":305,"RPM":3000,"Torque":80,"Tool_wear":40,
                         "Reason":"High thrust force + poor cooling may lead to edge and corner wear"},
    "Grinding Machine": {"Type":"H","Air_temp":300,"Process_temp":320,"RPM":5000,"Torque":20,"Tool_wear":30,
                         "Reason":"Abrasive grain fracture leads to abrasive wear"},
    "Tapping Machine": {"Type":"L","Air_temp":290,"Process_temp":300,"RPM":1000,"Torque":150,"Tool_wear":60,
                        "Reason":"High torque causes friction wear and breakage"},
    "Broaching Machine": {"Type":"L","Air_temp":285,"Process_temp":295,"RPM":500,"Torque":180,"Tool_wear":70,
                          "Reason":"Continuous cutting load causes progressive wear"},
    "Shaping Machine": {"Type":"L","Air_temp":280,"Process_temp":290,"RPM":800,"Torque":120,"Tool_wear":50,
                        "Reason":"Interrupted cutting leads to edge wear"},
    "Slotting Machine": {"Type":"M","Air_temp":285,"Process_temp":295,"RPM":900,"Torque":100,"Tool_wear":45,
                         "Reason":"Vertical cutting force causes flank wear"},
    "Sawing Machine": {"Type":"M","Air_temp":290,"Process_temp":300,"RPM":2000,"Torque":70,"Tool_wear":40,
                       "Reason":"Tooth friction and vibration leads to tooth wear"},
    "Induction Motor": {"Type":"H","Air_temp":310,"Process_temp":320,"RPM":1500,"Torque":60,"Tool_wear":0,
                        "Reason":"Electrical and mechanical stress can lead to motor failure"}
}

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(
            """
            ### 🚀 System Features
            - Predicts **machine failure probability**
            - Performs **root cause analysis**
            - Supports **manual input for multiple machines**
            - Shows ideal vs actual readings
            """
        )
    with col2:
        st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")
    st.divider()
    st.markdown(
        """
        ### 🏭 Why Predictive Maintenance?
        - Reduced downtime  
        - Lower maintenance cost  
        - Improved machine life  
        - Safer operations  
        """
    )

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine_name = st.selectbox("Select Machine Under Test", list(machine_defaults.keys()))

    # Load defaults for selected machine
    defaults = machine_defaults[machine_name]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        machine_type = st.selectbox("Machine Class", ["L","M","H"], index=["L","M","H"].index(defaults["Type"]), key="machine_type")
        air_temp = st.number_input("Air Temperature (K)", 250, 400, value=defaults["Air_temp"], key="air_temp")
    with col2:
        process_temp = st.number_input("Process Temperature (K)", 250, 400, value=defaults["Process_temp"], key="process_temp")
        rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=defaults["RPM"], key="rpm")
    with col3:
        torque = st.number_input("Torque (Nm)", 0, 200, value=defaults["Torque"], key="torque")
        tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=defaults["Tool_wear"], key="tool_wear")
    with col4:
        st.text(f"⚙️ Reason of Failure:\n{defaults['Reason']}")

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
        st.subheader(f"📈 Prediction Result for `{machine_name}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Machine Status
        if prob < 0.1:
            capability = "💚 Running at Full Capability"
        elif prob < 0.5:
            capability = "🟡 Running at Partial Capability"
        else:
            capability = "🔴 Failure Likely / Needs Attention"
        st.metric("Machine Status", capability)

        # Root Cause Analysis
        st.divider()
        st.subheader("🧠 Root Cause Analysis")
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values,list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)
        st.info(f"**Main Reason for Failure:** {defaults['Reason']}")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown(
        """
        ### 🔹 Model
        **LightGBM Classifier**
        - Gradient boosting decision trees
        - High accuracy on industrial data

        ### 🔹 Explainability
        **SHAP (XAI)**
        - Explains predictions
        - Identifies key factors causing failure

        ### 🔹 Dataset
        - AI4I 2020 Predictive Maintenance Dataset
        - 10,000 real-world industrial samples
        """
    )
    st.success("Explainable AI-based predictive maintenance system ready for real-world use.")
