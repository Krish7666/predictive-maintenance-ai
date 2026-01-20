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
    **Project:**  
    AI-Driven Predictive Maintenance  
    with Failure Probability  
    & Root Cause Analysis  

    **Tech Stack:**  
    - Machine Learning  
    - LightGBM  
    - SHAP (Explainable AI)  
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
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model(df):
    features = [
        'Type',
        'Air_temperature__K_',
        'Process_temperature__K_',
        'Rotational_speed__rpm_',
        'Torque__Nm_',
        'Tool_wear__min_'
    ]

    X = df[features]
    y = df['Machine_failure']

    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

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

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, le, auc, features

model, le, auc_score, feature_names = train_model(df)

# -------------------------------
# SHAP Explainer (Safe Initialization)
# -------------------------------
explainer = None
try:
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 🚀 System Capabilities
            - Predicts **machine failure probability**
            - Performs **root cause analysis**
            - Uses **real industrial sensor data**
            - Supports **manual real-time input**
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
        - Minimizes unexpected downtime  
        - Reduces maintenance costs  
        - Extends machine lifespan  
        - Improves operational safety  
        """
    )

# =========================================================
# MANUAL PREDICTION + ROOT CAUSE ANALYSIS
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Manual Failure Prediction & Root Cause Analysis")

    st.info("Enter machine sensor values to predict failure probability and identify root causes.")

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

        submit = st.form_submit_button("🔍 Predict & Analyze")

    if submit:
        # Input dataframe
        input_df = pd.DataFrame([{
            'Type': machine_type,
            'Air_temperature__K_': air_temp,
            'Process_temperature__K_': process_temp,
            'Rotational_speed__rpm_': speed,
            'Torque__Nm_': torque,
            'Tool_wear__min_': tool_wear
        }])

        input_df['Type'] = le.transform(input_df['Type'])

        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("📈 Prediction Result")

        st.progress(float(prob))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Failure Probability", f"{prob*100:.2f}%")

        with col2:
            status = "⚠️ Failure Likely" if prediction == 1 else "✅ Normal Operation"
            st.metric("Prediction Status", status)

        # -------------------------------
        # ROOT CAUSE ANALYSIS (SAFE)
        # -------------------------------
        st.divider()
        st.subheader("🧠 Root Cause Analysis")

        if explainer is not None:
            shap_values = explainer.shap_values(input_df)

            shap_df = pd.DataFrame(
                shap_values[1],
                columns=feature_names
            )

            impact = shap_df.iloc[0].abs().sort_values(ascending=False)
            st.markdown("### 🔍 Feature Impact (SHAP Explainability)")
        else:
            importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

            impact = importance
            st.markdown("### 🔍 Feature Impact (Model-Based Importance)")

        st.bar_chart(impact)

        top_feature = impact.index[0]
        st.info(
            f"**Primary Root Cause:** `{top_feature}` has the highest contribution "
            "towards the predicted failure risk."
        )

# =========================================================
# MODEL INFO PAGE
# =========================================================
if menu == "Model Info":
    st.title("📚 Model & Dataset Information")

    st.markdown(
        """
        ### 📌 Machine Learning Model
        **LightGBM Classifier**
        - Gradient boosting decision trees
        - High accuracy on tabular industrial data
        - Fast training & prediction

        ### 🧠 Explainability
        - SHAP for local explainability (when available)
        - Feature importance as fallback for reliability

        ### 📊 Dataset
        - AI4I 2020 Predictive Maintenance Dataset
        - 10,000 real industrial samples
        - Sensor-based machine parameters

        ### 🎯 Objective
        - Predict machine failure
        - Provide actionable root cause insights
        """
    )

    st.success("This system demonstrates real-world, reliable, explainable AI for predictive maintenance.")
