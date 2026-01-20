# =========================================================
# AI-Driven Predictive Maintenance Dashboard
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
st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**  

- Failure Probability Prediction  
- Root Cause Analysis (Explainable AI)  
- Real Industrial Sensor Data  

**Tech Stack**: LightGBM, SHAP, Streamlit
""")

menu = st.sidebar.radio("Navigation", ["Home", "Manual Prediction", "Model Info"])

# -------------------------------
# Machine Ideal Values + Failure Reasons
# -------------------------------
machine_defaults = {
    "CNC Milling": {"Type": "M", "Air_temp": 300, "Process_temp": 310, "RPM": 3000, "Torque": 50, "Tool_wear": 50,
                    "Reason": "High RPM + Multi-point cutting → Tool wear"},
    "Drilling Machine": {"Type": "H", "Air_temp": 300, "Process_temp": 300, "RPM": 2000, "Torque": 80, "Tool_wear": 60,
                         "Reason": "High thrust force + poor cooling → Edge/Corner wear"},
    "Grinding Machine": {"Type": "H", "Air_temp": 310, "Process_temp": 320, "RPM": 4000, "Torque": 20, "Tool_wear": 40,
                         "Reason": "Abrasive grain fracture → Abrasive wear"},
    "Tapping Machine": {"Type": "L", "Air_temp": 290, "Process_temp": 300, "RPM": 500, "Torque": 100, "Tool_wear": 70,
                        "Reason": "High torque → Adhesive wear / Breakage"},
    "Broaching Machine": {"Type": "L", "Air_temp": 295, "Process_temp": 305, "RPM": 200, "Torque": 120, "Tool_wear": 100,
                          "Reason": "Continuous cutting load → Progressive wear"},
    "Shaping Machine": {"Type": "L", "Air_temp": 295, "Process_temp": 305, "RPM": 400, "Torque": 80, "Tool_wear": 60,
                        "Reason": "Interrupted cutting → Edge wear"},
    "Slotting Machine": {"Type": "M", "Air_temp": 300, "Process_temp": 310, "RPM": 500, "Torque": 70, "Tool_wear": 55,
                         "Reason": "Vertical cutting force → Flank wear"},
    "Sawing Machine": {"Type": "M", "Air_temp": 300, "Process_temp": 305, "RPM": 1500, "Torque": 50, "Tool_wear": 40,
                       "Reason": "Tooth friction & vibration → Tooth wear"},
    "Induction Motor": {"Type": "H", "Air_temp": 320, "Process_temp": 330, "RPM": 4000, "Torque": 40, "Tool_wear": 20,
                        "Reason": "High load → Insulation / bearing wear"}
}

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

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")
    st.markdown(f"**Model ROC-AUC:** {auc_score:.3f}")
    st.markdown("""
    ### System Features
    - Predicts machine failure probability
    - Performs root cause analysis
    - Ideal readings per machine
    - User-modifiable inputs
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine_selected = st.selectbox("Select Machine Under Test", list(machine_defaults.keys()))

    # Load default values for the selected machine
    defaults = machine_defaults[machine_selected]

    # --- Input Form ---
    with st.form("manual_form"):
        machine_id = st.text_input("Machine ID / Name", f"{machine_selected}_01")
        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Machine Class", ["L","M","H"], index=["L","M","H"].index(defaults["Type"]), key="type")
            air_temp = st.number_input("Air Temperature (K)", 250, 400, value=defaults["Air_temp"], key="air_temp")
        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250, 400, value=defaults["Process_temp"], key="process_temp")
            rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=defaults["RPM"], key="rpm")
        with col3:
            torque = st.number_input("Torque (Nm)", 0, 200, value=defaults["Torque"], key="torque")
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=defaults["Tool_wear"], key="tool_wear")
        submit = st.form_submit_button("Predict")

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

        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        # Machine Status
        if prob < 0.1:
            capability = "Running at Full Capability"
        elif prob < 0.5:
            capability = "Partial Capability"
        else:
            capability = "Failure Likely / Needs Attention"

        st.subheader(f"Prediction Result for `{machine_id}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")
        st.metric("Machine Status", capability)

        # Root Cause Analysis
        st.subheader("Root Cause Analysis")
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)
        st.info(f"**Primary Root Cause:** `{impact.index[0]}`\n**Reason:** {defaults['Reason']}")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown(f"**ROC-AUC:** {auc_score:.3f}")
    st.markdown("""
    - Model: LightGBM Classifier
    - Explainability: SHAP (XAI)
    - Dataset: AI4I 2020 Predictive Maintenance
    - Purpose: Predict machine failure + root cause analysis
    """)
