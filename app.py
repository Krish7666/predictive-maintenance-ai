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
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Predictive Maintenance AI")
st.sidebar.markdown(
    """
**AI-Driven Predictive Maintenance System**

- Failure Probability Prediction
- Root Cause Analysis (Explainable AI)
- Real Industrial Sensor Data

**Tech Stack**
- LightGBM
- SHAP (XAI)
- Streamlit
"""
)

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
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
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
# Machine Defaults
# -------------------------------
machine_defaults = {
    "CNC Milling": {"Type": "M", "Air_temp": 300, "Process_temp": 320, "RPM": 4000, "Torque": 60, "Tool_wear": 100, "Reason": "High RPM + Multi-point cutting causes chipping and flank wear"},
    "Drilling Machine": {"Type": "H", "Air_temp": 310, "Process_temp": 330, "RPM": 2000, "Torque": 80, "Tool_wear": 120, "Reason": "High thrust force + poor cooling leads to corner and edge wear"},
    "Grinding Machine": {"Type": "L", "Air_temp": 300, "Process_temp": 300, "RPM": 5000, "Torque": 30, "Tool_wear": 50, "Reason": "Abrasive grain fracture causes abrasive wear"},
    "Tapping Machine": {"Type": "H", "Air_temp": 305, "Process_temp": 315, "RPM": 500, "Torque": 120, "Tool_wear": 80, "Reason": "High torque leads to adhesive wear and breakage"},
    "Broaching Machine": {"Type": "H", "Air_temp": 300, "Process_temp": 310, "RPM": 300, "Torque": 150, "Tool_wear": 200, "Reason": "Continuous cutting load leads to progressive wear"},
    "Shaping Machine": {"Type": "M", "Air_temp": 300, "Process_temp": 310, "RPM": 400, "Torque": 90, "Tool_wear": 70, "Reason": "Interrupted cutting causes edge wear"},
    "Slotting Machine": {"Type": "M", "Air_temp": 300, "Process_temp": 310, "RPM": 450, "Torque": 60, "Tool_wear": 90, "Reason": "Vertical cutting force leads to flank wear"},
    "Sawing Machine": {"Type": "M", "Air_temp": 300, "Process_temp": 310, "RPM": 1500, "Torque": 50, "Tool_wear": 100, "Reason": "Tooth friction and vibration cause tooth wear"},
    "Induction Motor": {"Type": "L", "Air_temp": 310, "Process_temp": 320, "RPM": 1450, "Torque": 40, "Tool_wear": 0, "Reason": "Electrical & mechanical stress may cause motor failure"},
}

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("Predictive Maintenance AI Dashboard")
    st.markdown(
        """
### System Features
- Predicts machine failure probability
- Root cause analysis with SHAP
- Displays real industrial sensor data
- Allows manual input for simulation
"""
    )
    st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("Machine Failure Prediction & Diagnosis")
    
    machine_name = st.selectbox("Select Machine Under Test", list(machine_defaults.keys()))
    
    # Update session state if machine changes
    defaults = machine_defaults[machine_name]
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            # Update only if machine selection changed
            if st.session_state.get("selected_machine") != machine_name:
                st.session_state[key] = value
    st.session_state["selected_machine"] = machine_name

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        machine_type = st.selectbox("Machine Class", ["L","M","H"], index=["L","M","H"].index(st.session_state["Type"]))
        air_temp = st.number_input("Air Temperature (K)", 250, 400, value=st.session_state["Air_temp"])
    with col2:
        process_temp = st.number_input("Process Temperature (K)", 250, 400, value=st.session_state["Process_temp"])
        rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=st.session_state["RPM"])
    with col3:
        torque = st.number_input("Torque (Nm)", 0, 200, value=st.session_state["Torque"])
        tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=st.session_state["Tool_wear"])
    
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
        st.subheader(f"Prediction for {machine_name}")
        st.metric("Failure Probability", f"{prob*100:.2f}%")
        
        if prob < 0.1:
            status = "Running at Full Capability"
        elif prob < 0.5:
            status = "Running at Partial Capability"
        else:
            status = "Failure Likely / Needs Attention"
        st.metric("Machine Status", status)
        
        st.divider()
        st.subheader("Root Cause Analysis")
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)
        
        st.info(f"**Main Reason for Failure:** {defaults['Reason']}")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("Model Information")
    st.markdown(
        """
### Model
- LightGBM Classifier
- Gradient boosting decision trees
- High accuracy on industrial tabular data

### Explainability
- SHAP (Explainable AI)
- Shows feature contribution to failure

### Dataset
- AI4I 2020 Predictive Maintenance
- 10,000 industrial samples
"""
    )
