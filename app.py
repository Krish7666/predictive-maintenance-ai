# =========================================================
# AI-Driven Predictive Maintenance (Induction Motor Focus)
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

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance – Induction Motor",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Manual Diagnosis", "Model Info"],
    key="nav"
)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train model
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
    return model, encoder, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# OVERVIEW
# =========================================================
if menu == "Overview":
    st.title("⚙️ Induction Motor Predictive Maintenance")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### System Scope
        This system predicts failures in **induction-motor-driven machinery** such as:
        - Conveyor belt motors
        - Pumps & compressors
        - Fans & blowers
        - Gearbox-driven systems

        The model learns failure behavior from **load, speed, temperature and wear**.
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

    st.divider()

    st.markdown("""
    ### Engineering Logic Used
    - Torque ↑ ⇒ RPM ↓ (load increase)
    - High torque ⇒ thermal & mechanical stress
    - Excessive wear ⇒ efficiency loss
    - Combined effects drive failure probability
    """)

# =========================================================
# MANUAL DIAGNOSIS
# =========================================================
if menu == "Manual Diagnosis":
    st.title("📊 Induction Motor Failure Diagnosis")

    st.info("Enter **current operating conditions** of the induction motor.")

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            value=1450.0,
            step=10.0
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            value=35.0,
            step=1.0
        )

    with col3:
        tool_wear = st.number_input(
            "Operational Wear Index",
            value=20.0,
            step=1.0
        )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0,
        step=1.0
    )

    proc_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0,
        step=1.0
    )

    if st.button("🔍 Run Diagnosis"):
        input_df = pd.DataFrame([{
            "Type": "M",
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.25:
            st.success("🟢 Motor operating within safe limits")
        elif prob < 0.6:
            st.warning("🟡 Motor under increasing load stress")
        else:
            st.error("🔴 High risk of failure detected")

        # -------------------------------------------------
        # SHAP ROOT CAUSE GRAPH
        # -------------------------------------------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        impact = pd.Series(
            shap_array[0],
            index=FEATURES
        ).sort_values(key=abs)

        st.subheader("🧠 Failure Contribution Analysis")

        fig1, ax1 = plt.subplots()
        impact.plot(kind="barh", ax=ax1)
        ax1.set_xlabel("Impact on Failure Risk")
        ax1.set_title("Root Cause Contribution")
        st.pyplot(fig1)

        # -------------------------------------------------
        # TORQUE vs RPM PHYSICS GRAPH
        # -------------------------------------------------
        st.subheader("⚙️ Torque–Speed Relationship")

        rpm_range = np.linspace(300, 3000, 60)
        torque_curve = (rpm * torque) / rpm_range  # inverse relation

        fig2, ax2 = plt.subplots()
        ax2.plot(rpm_range, torque_curve, label="Load Curve")
        ax2.scatter(rpm, torque, color="red", label="Current Point")

        ax2.set_xlabel("RPM")
        ax2.set_ylabel("Torque (Nm)")
        ax2.set_title("Induction Motor Load Behavior")
        ax2.legend()

        st.pyplot(fig2)

        # -------------------------------------------------
        # DIAGNOSIS TEXT
        # -------------------------------------------------
        main_factor = impact.index[-1]

        st.subheader("🛠 Engineering Diagnosis")

        if "Torque" in main_factor:
            st.write(
                "Failure risk is primarily driven by **high mechanical load**. "
                "Increased torque demand reduces speed and raises current, "
                "leading to overheating and insulation stress."
            )
        elif "Rotational_speed" in main_factor:
            st.write(
                "Operating speed deviation is the dominant factor. "
                "Speed reduction under load indicates possible overload or bearing drag."
            )
        elif "Tool_wear" in main_factor:
            st.write(
                "Wear accumulation is degrading efficiency, increasing friction "
                "and energy losses in the motor–load system."
            )
        else:
            st.write(
                "Failure risk arises from combined thermal and mechanical stress conditions."
            )

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Algorithm:** LightGBM Gradient Boosting  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    ### Optimized For
    - Induction motors  
    - Rotating machinery  
    - Load-dependent failure analysis  

    This model is **not tool-specific** and is best suited for **motor-driven systems**.
    """)

    st.success("System stable, interpretable, and industry-aligned.")
