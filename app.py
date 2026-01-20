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
    st.markdown(f"**Model ROC-AUC:** {auc_score:.3f}")# =========================================================
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
st.sidebar.markdown("""
**AI-Driven Predictive Maintenance System**

✔ Failure Probability Prediction  
✔ Root Cause Analysis (Explainable AI)  
✔ Ideal Sensor Values per Machine  

**Technology Stack**  
- LightGBM  
- SHAP (XAI)  
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
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)

    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, encoder, auc_score, feature_columns

model, encoder, auc_score, feature_columns = train_model(df)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# -------------------------------
# Machine Defaults
# -------------------------------
machine_defaults = {
    "CNC Lathe Machine": {"Air_temp": 300, "Process_temp": 310, "RPM": 2500, "Torque": 70, "Tool_wear": 50},
    "CNC Milling Machine": {"Air_temp": 300, "Process_temp": 320, "RPM": 4000, "Torque": 80, "Tool_wear": 60},
    "Drilling Machine": {"Air_temp": 310, "Process_temp": 330, "RPM": 1500, "Torque": 100, "Tool_wear": 60},
    "Grinding Machine": {"Air_temp": 305, "Process_temp": 315, "RPM": 4000, "Torque": 50, "Tool_wear": 40},
    "Tapping Machine": {"Air_temp": 300, "Process_temp": 310, "RPM": 500, "Torque": 150, "Tool_wear": 70},
    "Broaching Machine": {"Air_temp": 295, "Process_temp": 305, "RPM": 200, "Torque": 180, "Tool_wear": 80},
    "Shaping Machine": {"Air_temp": 300, "Process_temp": 310, "RPM": 300, "Torque": 120, "Tool_wear": 60},
    "Slotting Machine": {"Air_temp": 300, "Process_temp": 310, "RPM": 250, "Torque": 90, "Tool_wear": 55},
    "Sawing Machine": {"Air_temp": 300, "Process_temp": 310, "RPM": 1200, "Torque": 70, "Tool_wear": 50},
    "Induction Motor": {"Air_temp": 300, "Process_temp": 320, "RPM": 4000, "Torque": 40, "Tool_wear": 100}
}

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 System Features
        - Predicts **machine failure probability**
        - Performs **root cause analysis**
        - Loads **ideal sensor values per machine**
        - Users can override values
        """)
    with col2:
        st.metric("Model ROC-AUC Score", f"{auc_score:.3f}")

    st.divider()
    st.markdown("""
    ### 🏭 Why Predictive Maintenance?
    - Reduced downtime  
    - Lower maintenance cost  
    - Improved machine life  
    - Safer operations  
    """)

# =========================================================
# MANUAL PREDICTION PAGE
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")
    st.info("Select a machine, check its ideal values, adjust if needed, and predict failure.")

    # Machine Selection
    machine_selected = st.selectbox("Select Machine Under Test", list(machine_defaults.keys()))

    # Initialize session_state for ideal values
    if "last_machine" not in st.session_state or st.session_state.last_machine != machine_selected:
        defaults = machine_defaults[machine_selected]
        st.session_state.air_temp = defaults["Air_temp"]
        st.session_state.process_temp = defaults["Process_temp"]
        st.session_state.rpm = defaults["RPM"]
        st.session_state.torque = defaults["Torque"]
        st.session_state.tool_wear = defaults["Tool_wear"]
        st.session_state.last_machine = machine_selected

    # Inputs bound to session_state
    air_temp = st.number_input("Air Temperature (K)", 250, 400, value=st.session_state.air_temp, key="air_temp")
    process_temp = st.number_input("Process Temperature (K)", 250, 400, value=st.session_state.process_temp, key="process_temp")
    rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, value=st.session_state.rpm, key="rpm")
    torque = st.number_input("Torque (Nm)", 0, 200, value=st.session_state.torque, key="torque")
    tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=st.session_state.tool_wear, key="tool_wear")

    # Predict Button
    if st.button("🔍 Predict Failure"):
        input_df = pd.DataFrame([{
            "Type": machine_selected[0],  # Use first letter for 'Type' mapping
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

        # Result
        st.divider()
        st.subheader(f"📈 Prediction Result for `{machine_selected}`")
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
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_df = pd.DataFrame(shap_array, columns=feature_columns)
        impact = shap_df.iloc[0].abs().sort_values(ascending=False)
        st.bar_chart(impact)

        # Provide human-readable reason for failure
        reason_mapping = {
            "RPM": "High RPM → excessive heat → tool wear",
            "Torque": "High torque → friction & load stress",
            "Tool_wear__min_": "Excessive tool wear reduces efficiency",
            "Air_temperature__K_": "High ambient temperature may affect machine",
            "Process_temperature__K_": "High process temperature → thermal stress"
        }
        top_cause = impact.index[0]
        st.info(f"**Primary Root Cause:** `{top_cause}` — {reason_mapping.get(top_cause, 'Check sensor values & load')}")


# =========================================================
# MODEL INFO PAGE
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
    - Identifies root causes of failure

    ### 🔹 Dataset
    - AI4I 2020 Predictive Maintenance Dataset
    - 10,000 industrial samples
    """)
    st.success("Explainable AI-based predictive maintenance system ready for real-world use.")

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
