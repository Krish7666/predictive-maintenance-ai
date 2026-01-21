# =========================================================
# AI-Based Predictive Maintenance for Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
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
# Train LightGBM Model
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
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, encoder, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ Induction Motor AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Motor Health Prediction", "Model Info"]
)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("⚙️ Induction Motor Predictive Maintenance")

    st.metric("Model ROC-AUC", f"{auc_score:.3f}")

    st.markdown("""
    ### System Capabilities
    - Predict **induction motor failure probability**
    - Detect early degradation trends
    - Explain failure causes using AI
    - Suitable for pumps, conveyors, fans, blowers
    """)

# =========================================================
# MOTOR PREDICTION
# =========================================================
if menu == "Motor Health Prediction":
    st.title("📊 Induction Motor Health Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Motor Speed (RPM)",
            min_value=500.0,
            max_value=3000.0,
            value=1450.0,
            step=10.0
        )

    with col2:
        torque = st.number_input(
            "Load Torque (Nm)",
            min_value=5.0,
            max_value=120.0,
            value=35.0,
            step=1.0
        )

    with col3:
        wear = st.number_input(
            "Operational Wear (min)",
            min_value=0.0,
            max_value=250.0,
            value=40.0,
            step=5.0
        )

    air_temp = st.slider(
        "Ambient Temperature (K)",
        min_value=270,
        max_value=330,
        value=300
    )

    proc_temp = st.slider(
        "Motor Operating Temperature (K)",
        min_value=290,
        max_value=380,
        value=315
    )

    if st.button("🔍 Predict Motor Health"):
        input_df = pd.DataFrame([{
            "Type": "M",
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.25:
            st.success("🟢 Motor operating under healthy conditions")
        elif prob < 0.6:
            st.warning("🟡 Motor showing degradation signs")
        else:
            st.error("🔴 High risk of motor failure")

        # ---------------- Diagnosis ----------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        impact = (
            pd.Series(abs(shap_array[0]), index=FEATURES)
            .sort_values(ascending=False)
        )

        main_factor = impact.index[0]

        st.subheader("🧠 Failure Cause Analysis")

        if "Torque" in main_factor:
            st.info(
                "Excessive load torque is stressing the motor shaft and bearings. "
                "Prolonged overload can cause bearing fatigue and increased vibration."
            )
        elif "Rotational_speed" in main_factor:
            st.info(
                "High rotational speed increases centrifugal forces and vibration levels, "
                "accelerating bearing wear and insulation aging."
            )
        elif "Process_temperature" in main_factor:
            st.info(
                "Elevated operating temperature indicates thermal stress on motor windings, "
                "which can degrade insulation and reduce motor lifespan."
            )
        elif "Tool_wear" in main_factor:
            st.info(
                "High operational wear reflects prolonged usage under load, "
                "leading to mechanical fatigue and efficiency loss."
            )
        else:
            st.info(
                "Failure risk arises from combined mechanical and thermal stress conditions."
            )

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Details")

    st.markdown("""
    **Algorithm:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    **Optimized for:**  
    - Induction motors  
    - Rotating industrial machinery  
    - Load & temperature driven failures  
    """)
