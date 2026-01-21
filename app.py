# =========================================================
# 🔧 AI-Based Predictive Maintenance for Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ------------------ MACHINE IDEAL PROFILE ------------------
INDUCTION_MOTOR_PROFILE = {
    "rpm": 1450,
    "torque": 35,
    "tool_wear": 20,
    "air_temp": 300,
    "proc_temp": 310
}

# ------------------ SIDEBAR ------------------
st.sidebar.title("🔧 Induction Motor Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav_menu"
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # Already existing dataset
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    df = df[df["Type"] == "M"]  # Only induction motors
    return df

df = load_data()

# ------------------ TRAIN MODEL ------------------
@st.cache_data
def train_model():
    FEATURES = [
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]
    X = df[FEATURES]
    y = df["Machine_failure"]

    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    # ROC-AUC for reference
    auc_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
    explainer = shap.TreeExplainer(model)
    return model, auc_score, FEATURES, explainer

model, auc_score, FEATURES, explainer = train_model()

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 Induction Motor Predictive Maintenance")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Capabilities**
        - Failure probability prediction
        - Rule-based safety monitoring
        - Maintenance recommendations
        - Motor health scoring
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Induction Motor Manual Testing")

    col1, col2 = st.columns(2)
    with col1:
        rpm = st.number_input("Rotational Speed (RPM)", value=float(INDUCTION_MOTOR_PROFILE["rpm"]), step=10)
        torque = st.number_input("Torque (Nm)", value=float(INDUCTION_MOTOR_PROFILE["torque"]), step=1.0)
        tool_wear = st.number_input("Tool Wear (min)", value=float(INDUCTION_MOTOR_PROFILE["tool_wear"]), step=1.0)

    with col2:
        air_temp = st.number_input("Air Temperature (K)", value=float(INDUCTION_MOTOR_PROFILE["air_temp"]), step=1.0)
        proc_temp = st.number_input("Process Temperature (K)", value=float(INDUCTION_MOTOR_PROFILE["proc_temp"]), step=1.0)

    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        # ------------------ ML Prediction ------------------
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        # ------------------ Status ------------------
        status = (
            "🟢 Normal Operation" if prob < 0.25
            else "🟡 Degrading Condition" if prob < 0.6
            else "🔴 Failure Likely"
        )
        st.subheader(status)

        # ------------------ Rule-Based Safety ------------------
        critical_flags = []

        if proc_temp > 400:
            critical_flags.append("⚠️ Process temperature extremely high! Risk of severe thermal damage.")
        if air_temp > 360:
            critical_flags.append("⚠️ Air temperature too high! Cooling efficiency compromised.")
        if rpm > 1800:
            critical_flags.append("⚠️ Motor overspeed! Bearing & rotor stress likely.")
        if torque > 70:
            critical_flags.append("⚠️ Excessive torque! Mechanical overload possible.")

        if critical_flags:
            st.error("🚨 Critical Operating Condition Detected")
            for msg in critical_flags:
                st.write(msg)

        # ------------------ Failure Diagnosis ------------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_array[0], index=FEATURES).abs().sort_values(ascending=False)
        main = impact.index[0]

        st.subheader("🧠 Failure Diagnosis")
        if "rpm" in main.lower():
            st.info("High RPM is increasing dynamic and thermal stress, accelerating wear and vibration fatigue.")
        elif "torque" in main.lower():
            st.info("Excessive torque load causes mechanical stress and risk of drivetrain failure.")
        elif "wear" in main.lower():
            st.info("Tool wear is high; friction and poor performance are increasing failure probability.")
        elif "temperature" in main.lower():
            st.info("Extreme temperatures are causing thermal degradation and potential failure.")
        else:
            st.info("Combined operational parameters are influencing the failure risk.")

        # ------------------ Maintenance Recommendation ------------------
        st.subheader("🛠️ Maintenance Recommendations")
        if prob < 0.25:
            st.success("Motor is healthy. Routine monitoring is sufficient.")
        elif prob < 0.6:
            st.warning("Consider preventive maintenance soon. Inspect bearings, lubrication, and cooling.")
        else:
            st.error("Immediate maintenance required! Stop operation and inspect critical components.")

        # ------------------ Motor Health Score ------------------
        health_score = max(0, 100 - prob*100)
        st.subheader("💓 Motor Health Score")
        st.progress(int(health_score))

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown("""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 (Induction Motors subset)

    The system predicts failure probability, highlights extreme operating conditions,
    and provides maintenance recommendations and a motor health score.
    """)
