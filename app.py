# =========================================================
# AI-Driven Predictive Maintenance with Root Cause Analysis
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**

✔ Failure Probability Prediction  
✔ Root Cause Analysis (XAI)  
✔ Real Industrial Sensor Data  

**Technology Stack**
- LightGBM  
- SHAP (Explainable AI)  
- Streamlit  
""")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"]
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ai4i2020.csv")

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model(df):

    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    target = "Machine failure"

    X = df[feature_columns].copy()
    y = df[target]

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

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, encoder, auc, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":

    st.title("🔧 AI-Driven Predictive Maintenance System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🚀 What this system does
        - Predicts **machine failure probability**
        - Uses **real industrial sensor data**
        - Identifies **root causes using Explainable AI**
        - Helps prevent **unexpected breakdowns**
        """)

    with col2:
        st.metric(
            label="Model ROC-AUC Score",
            value=f"{auc_score:.3f}"
        )

# =========================================================
# MANUAL PREDICTION
# =========================================================
elif menu == "Manual Prediction":

    st.title("📊 Manual Failure Prediction")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", 290.0, 330.0, 300.0)

        with col2:
            process_temp = st.number_input("Process temperature [K]", 300.0, 360.0, 310.0)
            speed = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)

        with col3:
            torque = st.number_input("Torque [Nm]", 10.0, 100.0, 40.0)
            tool_wear = st.number_input("Tool wear [min]", 0, 300, 50)

        submit = st.form_submit_button("🔍 Predict")

    if submit:

        input_df = pd.DataFrame([{
            "Type": encoder.transform([machine_type])[0],
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("📈 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ Failure Likely — Risk: {probability*100:.2f}%")
        else:
            st.success(f"✅ Normal Operation — Risk: {probability*100:.2f}%")

        # -------------------------------
        # ROOT CAUSE ANALYSIS
        # -------------------------------
        st.subheader("🧠 Root Cause Analysis")

        shap_values = explainer.shap_values(input_df)

        # Safe SHAP handling
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_columns
        )

        impact = shap_df.iloc[0].abs().sort_values(ascending=False)

        st.write("Top contributing parameters:")
        st.dataframe(impact.head(5))

        fig, ax = plt.subplots()
        impact.head(5).plot(kind="barh", ax=ax)
        ax.set_title("Top Root Causes")
        ax.set_xlabel("Impact on Failure Prediction")
        plt.gca().invert_yaxis()

        st.pyplot(fig)

# =========================================================
# MODEL INFO
# =========================================================
elif menu == "Model Info":

    st.title("📚 Model Information")

    st.markdown("""
    ### 🔍 Machine Learning Model
    **LightGBM Classifier**
    - Gradient Boosting based
    - Fast & scalable
    - Ideal for industrial sensor data

    ### 🧠 Explainable AI
    **SHAP (SHapley Additive Explanations)**
    - Explains individual predictions
    - Identifies root cause of failure

    ### 📊 Dataset
    - AI4I 2020 Predictive Maintenance Dataset
    - 10,000 industrial samples
    - Realistic failure scenarios
    """)

    st.success("System ready for academic and industrial demonstration ✅")
