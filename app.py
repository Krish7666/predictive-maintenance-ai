# =========================================================
# AI-Driven Predictive Maintenance with Root Cause Analysis
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# =========================================================
# SIDEBAR
# =========================================================
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

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")

    # Clean column names safely
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('[^A-Za-z0-9]+', '_', regex=True)
    )

    return df


df = load_data()

# =========================================================
# TRAIN MODEL
# =========================================================
@st.cache_data
def train_model(df):

    # Explicit, verified AI4I feature list
    feature_columns = [
        "Type",
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min"
    ]

    target_column = "Machine_failure"

    X = df[feature_columns].copy()
    y = df[target_column]

    # Encode categorical feature
    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Model evaluation
    auc_score = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    return model, encoder, auc_score, feature_columns


model, encoder, auc_score, feature_columns = train_model(df)

# SHAP explainer (tree-based, perfect for LightGBM)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":

    st.title("🔧 AI-Driven Predictive Maintenance")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 🚀 System Overview
            This system predicts **machine failure probability**
            and explains **why the failure may occur** using
            **Explainable AI (XAI)** techniques.

            **Key Capabilities**
            - Failure probability estimation  
            - Root cause identification  
            - Real-time manual input  
            - Industrial-grade ML model  
            """
        )

    with col2:
        st.metric(
            "Model ROC-AUC Score",
            f"{auc_score:.3f}"
        )

    st.divider()

    st.markdown(
        """
        ### 🏭 Why Predictive Maintenance?
        - Prevents unplanned downtime  
        - Reduces maintenance costs  
        - Extends machine lifespan  
        - Improves operational safety  
        """
    )

# =========================================================
# MANUAL PREDICTION + ROOT CAUSE ANALYSIS
# =========================================================
if menu == "Manual Prediction":

    st.title("📊 Failure Prediction & Root Cause Analysis")

    st.info(
        "Enter real-time machine sensor values to predict "
        "failure probability and identify root causes."
    )

    with st.form("prediction_form"):

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

        # Prepare input
        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air_temperature_K": air_temp,
            "Process_temperature_K": process_temp,
            "Rotational_speed_rpm": speed,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear
        }])

        input_data["Type"] = encoder.transform(input_data["Type"])

        # Prediction
        failure_prob = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📈 Prediction Result")

        st.progress(float(failure_prob))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Failure Probability", f"{failure_prob * 100:.2f}%")

        with col2:
            st.metric(
                "Machine Status",
                "⚠️ Failure Likely" if prediction else "✅ Normal Operation"
            )

        # =================================================
        # ROOT CAUSE ANALYSIS (SHAP)
        # =================================================
        st.divider()
        st.subheader("🧠 Root Cause Analysis (Explainable AI)")

        shap_values = explainer.shap_values(input_data)[1]

        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_columns
        )

        impact = shap_df.iloc[0].abs().sort_values(ascending=False)

        st.markdown("### 🔍 Feature Impact on Prediction")
        st.bar_chart(impact)

        top_feature = impact.index[0]
        st.info(
            f"**Primary Root Cause:** `{top_feature}` "
            "has the strongest influence on the predicted failure."
        )

# =========================================================
# MODEL INFO PAGE
# =========================================================
if menu == "Model Info":

    st.title("📚 Model & Dataset Information")

    st.markdown(
        """
        ### 🤖 Machine Learning Model
        **LightGBM Classifier**
        - Gradient Boosted Decision Trees  
        - High accuracy on structured industrial data  
        - Fast and scalable  

        ### 🧠 Explainability
        **SHAP (SHapley Additive Explanations)**
        - Explains individual predictions  
        - Identifies root causes  
        - Trusted XAI method in industry  

        ### 📊 Dataset
        - AI4I 2020 Predictive Maintenance Dataset  
        - 10,000 samples  
        - Real sensor-based measurements  

        ### 🎯 Objective
        - Predict machine failure  
        - Provide actionable maintenance insights  
        """
    )

    st.success(
        "This application demonstrates an industry-ready, "
        "explainable AI system for predictive maintenance."
    )
