# =========================================================
# AI-Driven Predictive Maintenance with Root Cause Analysis
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

    # ✅ FIX ADDED (LightGBM-safe column names)
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
        X,
        y,
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

# -------------------------------
# SHAP Explainer
# -------------------------------
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 🚀 System Features
            - Predicts **machine failure probability**
            - Performs **root cause analysis**
            - Uses **real industrial sensor data**
            - Supports **manual input**
            """
        )

    with col2:
        st.metric(
            label="Model ROC-AUC Score",
            value=f"{auc_score:.3f}"
        )

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
# MANUAL PREDICTION + ROOT CAUSE
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Failure Prediction & Root Cause Analysis")

    st.info("Enter machine parameters to predict failure and identify root causes.")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
            air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)

        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0)
            speed = st.number_input("Rotational Speed (rpm)", 100, 5000, 1500)

        with col3:
            torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0)
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, 100)

        submit = st.form_submit_button("🔍 Predict")

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
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("📈 Prediction Result")

        st.progress(float(prob))
        st.metric("Failure Probability", f"{prob * 100:.2f}%")

        status = "⚠️ Failure Likely" if prediction == 1 else "✅ Normal Operation"
        st.metric("Prediction Status", status)

        # -------------------------------
        # ROOT CAUSE ANALYSIS
        # -------------------------------
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

        st.info(
            f"**Primary Root Cause:** `{impact.index[0]}` "
            "has the highest impact on failure risk."
        )

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
        - High accuracy on tabular data

        ### 🔹 Explainability
        **SHAP (Explainable AI)**
        - Explains individual predictions
        - Identifies failure root causes

        ### 🔹 Dataset
        - AI4I 2020 Predictive Maintenance Dataset
        - 10,000 industrial samples
        """
    )

    st.success("Explainable AI-based predictive maintenance system ready for real-world use.")
