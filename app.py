# =========================================================
# AI-Driven Predictive Maintenance Dashboard (STABLE)
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar (UNIQUE KEY FIXED)
# ---------------------------------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="main_navigation"   # 🔥 FIX
)

# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Model training
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Machine ideal values (REALISTIC)
# ---------------------------------------------------------
MACHINE_PROFILES = {
    "CNC Milling":       dict(rpm=3000, torque=80, tool=60),
    "Drilling Machine": dict(rpm=1500, torque=120, tool=70),
    "Grinding Machine": dict(rpm=4500, torque=40, tool=40),
    "Tapping Machine":  dict(rpm=600,  torque=160, tool=90),
    "Broaching Machine":dict(rpm=200,  torque=180, tool=110),
    "Shaping Machine":  dict(rpm=300,  torque=130, tool=75),
    "Slotting Machine": dict(rpm=250,  torque=110, tool=70),
    "Sawing Machine":   dict(rpm=1200, torque=90,  tool=65),
    "Induction Motor":  dict(rpm=4000, torque=40,  tool=0)
}

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if menu == "Home":
    st.title("AI-Driven Predictive Maintenance")
    st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    st.markdown("""
    **This system predicts failures for industrial rotating machines
    using torque, RPM, temperature, and tool wear signals.**
    """)

# ---------------------------------------------------------
# MANUAL PREDICTION (SESSION STATE FIXED)
# ---------------------------------------------------------
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction")

    machine = st.selectbox(
        "Select Machine",
        list(MACHINE_PROFILES.keys()),
        key="machine_selector"
    )

    # RESET VALUES WHEN MACHINE CHANGES
    if "active_machine" not in st.session_state or st.session_state.active_machine != machine:
        st.session_state.active_machine = machine
        st.session_state.rpm = MACHINE_PROFILES[machine]["rpm"]
        st.session_state.torque = MACHINE_PROFILES[machine]["torque"]
        st.session_state.tool = MACHINE_PROFILES[machine]["tool"]
        st.session_state.air = 300
        st.session_state.process = 310

    col1, col2, col3 = st.columns(3)

    with col1:
        air = st.number_input("Air Temp (K)", 250, 400, key="air")
        process = st.number_input("Process Temp (K)", 250, 400, key="process")

    with col2:
        rpm = st.number_input("RPM", 100, 5000, key="rpm")
        torque = st.number_input("Torque (Nm)", 0, 200, key="torque")

    with col3:
        tool = st.number_input("Tool Wear (min)", 0, 500, key="tool")

    if st.button("Predict Failure", key="predict_btn"):
        X = pd.DataFrame([{
            "Type": "L",
            "Air_temperature__K_": air,
            "Process_temperature__K_": process,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool
        }])
        X["Type"] = encoder.transform(X["Type"])

        prob = model.predict_proba(X)[0][1]

        st.subheader("Result")
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.15:
            st.success("Machine running at full capability")
        elif prob < 0.5:
            st.warning("Performance degradation detected")
        else:
            st.error("High failure risk — maintenance required")

        # Root cause
        shap_vals = explainer.shap_values(X)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = abs(pd.Series(shap_array[0], index=feature_columns)).sort_values(ascending=False)

        st.subheader("Root Cause")
        st.write(f"Primary contributor: **{impact.index[0]}**")

# ---------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------
if menu == "Model Info":
    st.title("Model Details")
    st.markdown("""
    **Algorithm:** LightGBM  
    **Dataset:** AI4I 2020  
    **Explainability:** SHAP  
    **Machines supported:** CNC, Drilling, Grinding, Motors, etc.
    """)
