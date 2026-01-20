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

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"]
)

# -------------------------------
# Machine Failure Knowledge Base
# -------------------------------
MACHINE_FAILURE_REASON = {
    "CNC Lathe Machine":
        "High RPM generates excessive heat at the cutting edge, "
        "leading to flank and crater wear.",

    "CNC Milling Machine":
        "High-speed multi-point cutting causes vibration and tool chipping.",

    "Drilling Machine":
        "High thrust force and poor chip evacuation cause corner wear.",

    "Grinding Machine":
        "Very high speed causes abrasive grain fracture and surface burn.",

    "Tapping Machine":
        "Very high torque results in friction, adhesive wear, or tap breakage.",

    "Broaching Machine":
        "Continuous cutting load causes progressive tool wear.",

    "Shaping Machine":
        "Interrupted cutting creates impact stress leading to edge wear.",

    "Slotting Machine":
        "Vertical cutting forces cause flank wear.",

    "Sawing Machine":
        "Tooth vibration and friction lead to progressive tooth wear.",

    "Conveyor Belt Motor":
        "Overloading and misalignment increase torque, causing motor overheating.",

    "Induction Motor":
        "Thermal overload and voltage imbalance damage windings."
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
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, encoder, auc, features

model, encoder, auc_score, feature_columns = train_model(df)

explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🚀 Capabilities
        - Predicts **machine failure probability**
        - Explains **why failure happens**
        - Machine-aware diagnostics
        - Industrial sensor based
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

    st.divider()

    st.markdown("""
    ### 🏭 Targeted Industrial Assets
    CNC Machines • Motors • Rotating Equipment  
    Torque-RPM-Tool Wear focused systems
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            machine_name = st.selectbox(
                "Machine Under Test",
                list(MACHINE_FAILURE_REASON.keys())
            )
            machine_type = st.selectbox("Machine Class", ["L", "M", "H"])

        with col2:
            air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)
            process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0)

        with col3:
            rpm = st.number_input("Rotational Speed (RPM)", 100, 5000, 1500)
            torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0)
            tool_wear = st.number_input("Tool Wear (min)", 0, 500, 100)

        submit = st.form_submit_button("🔍 Analyze Machine")

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
        pred = model.predict(input_df)[0]

        st.divider()
        st.subheader("📈 Prediction Result")

        st.metric("Failure Probability", f"{prob*100:.2f}%")

        status = "🟢 Normal" if pred == 0 else "🔴 Failure Likely"
        st.metric("Machine Status", status)

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

        st.warning(f"""
        **Primary Cause:** {impact.index[0]}

        **Machine-Specific Failure Reason:**  
        {MACHINE_FAILURE_REASON[machine_name]}
        """)

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP (Explainable AI)  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    Designed for torque-RPM-tool wear driven industrial machines.
    """)

    st.success("Industrial-ready explainable predictive maintenance system.")
