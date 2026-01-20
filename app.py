# =========================================================
# AI-Driven Predictive Maintenance with Dynamic Root Cause
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
st.sidebar.title("Predictive Maintenance AI")
st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**

- Failure Probability Prediction
- Dynamic Root Cause Analysis
- Real Industrial Sensor Data

**Technology Stack**
- LightGBM
- SHAP
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
    features = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]

    X = df[features].copy()
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
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    return model, encoder, auc_score, features

model, encoder, auc_score, feature_columns = train_model(df)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# -------------------------------
# Machine Failure Knowledge Base
# -------------------------------
machine_failure_reason = {
    "CNC Lathe Machine": "High RPM generates heat → flank and crater wear",
    "CNC Milling Machine": "Multi-point high-speed cutting → chipping and flank wear",
    "Drilling Machine": "High thrust or RPM → corner/edge wear",
    "Grinding Machine": "High spindle speed → abrasive grain fracture",
    "Tapping Machine": "High torque → adhesive wear or breakage",
    "Broaching Machine": "Continuous load → progressive tool wear",
    "Shaping Machine": "Interrupted cutting → edge wear",
    "Slotting Machine": "Vertical cutting force → flank wear",
    "Sawing Machine": "Tooth friction & vibration → tooth wear",
    "Conveyor Belt Motor": "Overload or misalignment → bearing wear/overheating",
    "Induction Motor": "Voltage imbalance/thermal overload → insulation failure",
    "Pump Motor": "Cavitation/load → bearing and shaft wear",
    "Fan/Blower Motor": "Imbalance/dust → vibration-induced bearing failure",
    "Compressor Motor": "High pressure load → winding degradation"
}

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("AI-Driven Predictive Maintenance System")
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("""
        ### System Features
        - Predicts machine failure probability
        - Dynamic root cause analysis
        - Uses real industrial sensor data
        - Supports manual input
        """)
    with col2:
        st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")

    st.divider()
    st.markdown("""
    ### Why Predictive Maintenance?
    - Reduce downtime
    - Lower maintenance cost
    - Extend machine life
    - Improve operational safety
    """)

# =========================================================
# MANUAL PREDICTION + DYNAMIC ROOT CAUSE
# =========================================================
if menu == "Manual Prediction":
    st.title("Failure Prediction & Machine Status")

    st.info("Specify machine, enter parameters, get failure probability and dynamic root cause.")

    with st.form("manual_form"):
        col0, col1, col2, col3 = st.columns([1,1,1,1])
        with col0:
            machine_name = st.selectbox("Select Machine", list(machine_failure_reason.keys()))
        with col1:
            machine_type = st.selectbox("Machine Type", ["L","M","H"])
            air_temp = st.number_input("Air Temperature (K)", 250.0,400.0,300.0)
        with col2:
            process_temp = st.number_input("Process Temperature (K)", 250.0,400.0,310.0)
            rpm = st.number_input("Rotational Speed (rpm)", 100,5000,1500)
        with col3:
            torque = st.number_input("Torque (Nm)", 0.0,200.0,40.0)
            tool_wear = st.number_input("Tool Wear (min)", 0,500,100)

        submit = st.form_submit_button("Predict & Analyze")

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

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader(f"Prediction Result for `{machine_name}`")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # Determine Machine Status
        if prob < 0.1:
            status = "Running at Full Capability"
        elif prob < 0.5:
            status = "Partial Capability"
        else:
            status = "Failure Likely / Needs Attention"
        st.metric("Machine Status", status)

        # -------------------------------
        # Dynamic Root Cause Analysis
        # -------------------------------
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values,list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)

        # Factor-based warning
        factor_reason = ""
        if rpm > 0.9*5000:  # example threshold
            factor_reason += "High RPM; "
        if torque > 150:
            factor_reason += "High Torque; "
        if tool_wear > 300:
            factor_reason += "Excessive Tool Wear; "

        machine_specific_reason = machine_failure_reason.get(machine_name,"Operational stress may cause failure.")

        st.divider()
        st.subheader("Root Cause Analysis")
        st.bar_chart(impact)
        st.markdown(f"""
**Critical Feature(s):** {impact.index[0]}  
**Factor-based Reason(s):** {factor_reason if factor_reason else 'Normal range'}  
**Machine-specific Reason:** {machine_specific_reason}
""")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("Model Information")
    st.markdown("""
### Model
- LightGBM Classifier
- Gradient boosting trees, accurate for tabular sensor data

### Explainability
- SHAP (Explainable AI)
- Shows per-feature contribution to predicted failure

### Dataset
- AI4I 2020 Predictive Maintenance Dataset
- 10,000 samples with industrial sensor readings
""")
    st.success("Dynamic, machine-aware predictive maintenance system ready for industrial use.")
