# =========================================================
# CNC Milling Predictive Maintenance (Stable Version)
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="CNC Milling Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔧 CNC Milling AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="menu"
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
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

    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    return model, encoder, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 CNC Milling Predictive Maintenance System")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **System Capabilities**
        - Predict CNC milling machine failure probability
        - Analyze impact of RPM, Torque & Tool Wear
        - Explain failures using engineering + SHAP
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

    st.divider()

    st.markdown("""
    This system is **specifically designed for CNC milling machines**  
    operating under rotating cutting conditions.
    """)

# =========================================================
# MANUAL PREDICTION (CNC MILLING)
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 CNC Milling Failure Prediction")

    st.info("Default values represent **healthy CNC milling operation**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            min_value=500,
            max_value=5000,
            value=2500,
            step=50
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=10.0,
            max_value=150.0,
            value=45.0,
            step=1.0
        )

    with col3:
        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=500,
            value=60,
            step=5
        )

    air_temp = st.slider("Air Temperature (K)", 270, 330, 300)
    proc_temp = st.slider("Process Temperature (K)", 290, 380, 315)

    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Type": "M",  # Milling = Medium
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.25:
            st.success("🟢 Machine Operating Normally")
        elif prob < 0.6:
            st.warning("🟡 Machine Condition Degrading")
        else:
            st.error("🔴 Failure Likely – Maintenance Required")

        # ---------------- SHAP ANALYSIS ----------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        impact = pd.Series(
            shap_array[0],
            index=FEATURES
        ).abs().sort_values(ascending=False)

        st.subheader("🧠 Root Cause Analysis")
        st.bar_chart(impact)

        main_cause = impact.index[0]

        if "Rotational_speed" in main_cause:
            st.info(
                "High spindle speed is increasing cutting temperature and vibration, "
                "accelerating flank and crater wear in the milling tool."
            )
        elif "Torque" in main_cause:
            st.info(
                "Elevated torque indicates excessive cutting load, leading to mechanical "
                "stress on the spindle and tool holder."
            )
        elif "Tool_wear" in main_cause:
            st.info(
                "Tool wear has crossed its efficient operating range, causing unstable "
                "cutting conditions and higher failure risk."
            )
        else:
            st.info(
                "Failure risk is influenced by combined thermal and mechanical factors."
            )

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Machine Focus:** CNC Milling Machine  
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    Designed for **industrial CNC machining environments**.
    """)
