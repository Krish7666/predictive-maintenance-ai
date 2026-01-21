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
st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**

✔ Failure Probability Prediction  
✔ Root Cause Analysis (Explainable AI)  
✔ Real Industrial Sensor Data  

**Technology Stack**
- LightGBM  
- SHAP (Explainable AI)  
- Streamlit  
""")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav_radio"
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")

    # LightGBM-safe column names
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

    # Ensure numeric columns
    numeric_cols = ["Air_temperature__K_", "Process_temperature__K_",
                    "Rotational_speed__rpm_", "Torque__Nm_", "Tool_wear__min_"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)  # remove bad rows

    return df

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model(df):
    FEATURES = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]

    X = df[FEATURES].copy()
    y = df["Machine_failure"]

    # Encode Type
    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"].astype(str))

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LightGBM Model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ROC-AUC Score
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, encoder, auc_score, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# =========================================================
# Ideal values per machine for manual input
# =========================================================
IDEAL_VALUES = {
    "CNC Milling": {
        "Type": "M", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 4000, "Torque__Nm_": 50, "Tool_wear__min_": 50
    },
    "Drilling Machine": {
        "Type": "H", "Air_temperature__K_": 300, "Process_temperature__K_": 320,
        "Rotational_speed__rpm_": 3000, "Torque__Nm_": 70, "Tool_wear__min_": 40
    },
    "Grinding Machine": {
        "Type": "L", "Air_temperature__K_": 300, "Process_temperature__K_": 300,
        "Rotational_speed__rpm_": 5000, "Torque__Nm_": 20, "Tool_wear__min_": 30
    },
    "Tapping Machine": {
        "Type": "H", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 500, "Torque__Nm_": 100, "Tool_wear__min_": 60
    },
    "Broaching Machine": {
        "Type": "H", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 100, "Torque__Nm_": 120, "Tool_wear__min_": 80
    },
    "Shaping Machine": {
        "Type": "M", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 200, "Torque__Nm_": 90, "Tool_wear__min_": 50
    },
    "Slotting Machine": {
        "Type": "M", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 200, "Torque__Nm_": 60, "Tool_wear__min_": 40
    },
    "Sawing Machine": {
        "Type": "M", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 2500, "Torque__Nm_": 50, "Tool_wear__min_": 30
    },
    "Induction Motor": {
        "Type": "L", "Air_temperature__K_": 300, "Process_temperature__K_": 310,
        "Rotational_speed__rpm_": 1500, "Torque__Nm_": 40, "Tool_wear__min_": 20
    }
}

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI Predictive Maintenance System")
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
        st.metric(label="Model ROC-AUC Score", value=f"{auc_score:.3f}")

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
    st.info("Select machine under test. Default values are ideal; modify if needed.")

    # Machine selector
    machine_name = st.selectbox("Select Machine", list(IDEAL_VALUES.keys()))

    # Get ideal defaults
    defaults = IDEAL_VALUES[machine_name]

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            machine_id = st.text_input("Machine ID / Name", "Machine_1")
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=["L","M","H"].index(defaults["Type"]))
            air_temp = st.number_input("Air Temperature (K)", 250, 400, float(defaults["Air_temperature__K_"]))

        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250, 400, float(defaults["Process_temperature__K_"]))
            speed = st.number_input("Rotational Speed (RPM)", 100, 5000, float(defaults["Rotational_speed__rpm_"]))

        with col3:
            torque = st.number_input("Torque (Nm)", 0, 200, float(defaults["Torque__Nm_"]))
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, float(defaults["Tool_wear__min_"]))

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

        # Encode type
        input_df["Type"] = encoder.transform(input_df["Type"].astype(str))

        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader(f"📈 Prediction Result for `{machine_id}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Machine capability
        if prob < 0.1:
            status = "💚 Running at Full Capability"
        elif prob < 0.5:
            status = "🟡 Running at Partial Capability"
        else:
            status = "🔴 Failure Likely / Needs Attention"
        st.metric("Machine Status", status)

        # SHAP Root Cause
        st.divider()
        st.subheader("🧠 Root Cause Analysis")

        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=FEATURES)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)

        # Detailed cause
        cause_text = ""
        for feature, val in impact.items():
            cause_text += f"- **{feature}** affects failure probability significantly.\n"
        st.info(f"**Failure Cause Analysis:**\n{cause_text}")

# =========================================================
# MODEL INFO
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
    - Identifies key root causes  

    ### 🔹 Dataset
    - AI4I 2020 Predictive Maintenance Dataset  
    - ~10,000 industrial samples
    """)
    st.success("Explainable AI-based predictive maintenance system ready for real-world use.")
