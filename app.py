# =========================================================
# AI-Driven Predictive Maintenance with Root Cause Analysis
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
    ✔ Machining-Focused Industrial Data  

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
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    auc_score = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy (ROC-AUC)", f"{auc_score:.3f}")
    col2.metric("Target Machines", "CNC / Milling / Lathe")
    col3.metric("Monitoring Focus", "RPM • Torque • Tool Wear")

    st.divider()

    st.markdown(
        """
        ### 🚀 System Overview
        This system is designed for **machining-based industrial equipment**
        where **rotational speed, torque load, and tool wear** are the
        dominant indicators of failure.

        ### 🏭 Supported Use-Cases
        - CNC Milling Machines  
        - CNC Lathe / Turning Machines  
        - Drilling & Machining Centers  
        """
    )

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Failure Prediction & Machine Capability Analysis")

    st.info("Define the machine under test and enter sensor values.")

    with st.form("manual_form"):
        c0, c1, c2, c3 = st.columns(4)

        with c0:
            machine_id = st.text_input("Machine ID", "CNC-ML-01")
            process_type = st.selectbox(
                "Process Type",
                ["CNC Milling", "CNC Lathe", "Drilling / Machining"]
            )

        with c1:
            machine_type = st.selectbox("Machine Class", ["L", "M", "H"])
            air_temp = st.number_input("Air Temp (K)", 250.0, 400.0, 300.0)

        with c2:
            process_temp = st.number_input("Process Temp (K)", 250.0, 400.0, 310.0)
            speed = st.number_input("Spindle Speed (RPM)", 100, 5000, 1500)

        with c3:
            torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0)
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, 100)

        submit = st.form_submit_button("🔍 Run Prediction")

    if submit:
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": process_temp,
            "Rotational_speed__rpm_": speed,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]

        # -------------------------------
        # STATUS & CAPABILITY
        # -------------------------------
        if prob < 0.2:
            status = "🟢 Full Operational Capability"
        elif prob < 0.5:
            status = "🟡 Reduced Capability – Monitor Closely"
        else:
            status = "🔴 Critical – Maintenance Required"

        st.subheader(f"🔧 Machine Under Test: {machine_id}")
        st.write(f"**Process:** {process_type}")

        k1, k2, k3 = st.columns(3)
        k1.metric("Failure Probability", f"{prob*100:.2f}%")
        k2.metric("Machine Status", status)
        k3.metric("Tool Wear Level", f"{tool_wear} min")

        # -------------------------------
        # ROOT CAUSE ANALYSIS
        # -------------------------------
        st.divider()
        st.subheader("🧠 Root Cause Analysis")

        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)

        st.bar_chart(impact)

        # -------------------------------
        # MAINTENANCE RECOMMENDATION
        # -------------------------------
        primary_cause = impact.index[0]

        if primary_cause == "Tool_wear__min_":
            recommendation = "Replace or inspect cutting tool immediately."
        elif primary_cause == "Torque__Nm_":
            recommendation = "Reduce feed rate and inspect spindle load."
        elif primary_cause == "Rotational_speed__rpm_":
            recommendation = "Check spindle speed calibration."
        else:
            recommendation = "Inspect machine operating conditions."

        st.warning(f"🛠 **Recommended Action:** {recommendation}")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model & Project Information")

    st.markdown(
        """
        ### 🔹 Model
        LightGBM Classifier optimized for tabular industrial data.

        ### 🔹 Explainability
        SHAP-based local explanations for each prediction.

        ### 🔹 Dataset
        AI4I 2020 Predictive Maintenance Dataset  
        Focused on torque, speed, temperature, and tool wear.

        ### 🔹 Project Scope
        Designed for CNC, milling, and lathe-type machines.
        """
    )

    st.success("Industrial-grade, explainable predictive maintenance system.")
