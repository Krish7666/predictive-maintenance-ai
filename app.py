# =========================================================
# AI-Based Predictive Maintenance – Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Predictive Maintenance – Induction Motors",
    page_icon="🔧",
    layout="wide"
)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🔧 Predictive Maintenance AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="navigation"
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV (Optional)",
    type=["csv"]
)

# =========================================================
# LOAD AI4I 2020 DATASET
# =========================================================
@st.cache_data
def load_base_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

base_df = load_base_data()

# =========================================================
# TRAIN LIGHTGBM MODEL (ONCE)
# =========================================================
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

model, encoder, auc_score, FEATURES = train_model(base_df)

# =========================================================
# HOME PAGE
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Based Predictive Maintenance")
    st.subheader("Focused on Induction Motor Driven Machinery")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Supported Equipment**
        - Induction Motors  
        - Conveyor Belt Motors  
        - Pumps  
        - Fans & Blowers  
        - Gearbox-driven Systems  

        **Core Parameters**
        - RPM
        - Torque
        - Temperature
        - Wear
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")
        st.info(
            "Model trained on AI4I 2020 dataset "
            "and designed for real industrial usage."
        )

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Induction Motor Failure Prediction")

    st.markdown("### Manual Sensor Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            value=1450.0
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            value=35.0
        )

    with col3:
        tool_wear = st.number_input(
            "Tool Wear (min)",
            value=20.0
        )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    proc_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    st.divider()

    # ---------------- WHAT-IF LOAD LOGIC ----------------
    load_ratio = torque / max(rpm, 1)

    st.markdown("### 🧪 What-If Load Simulation")
    st.write(f"**Load Stress Index:** `{load_ratio:.3f}`")

    if load_ratio > 0.05:
        st.warning("High load detected → RPM drop & overheating risk")

    # ---------------- PREDICTION BUTTON ----------------
    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Type": encoder.transform(["M"])[0],
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": proc_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()

        st.metric(
            "Failure Probability",
            f"{prob*100:.2f}%"
        )

        health_score = max(0, 100 - prob*100)

        st.metric(
            "Motor Health Score",
            f"{health_score:.1f} / 100"
        )

        # ---------------- STATUS ----------------
        if prob < 0.25:
            st.success("🟢 Normal Operation")
        elif prob < 0.6:
            st.warning("🟡 Degrading Condition")
        else:
            st.error("🔴 Failure Likely")

        # ---------------- MAINTENANCE CARD ----------------
        st.markdown("### 🛠 Maintenance Recommendation")

        if torque > 50:
            st.info(
                "High torque load detected. "
                "Check mechanical alignment, bearing friction, and gearbox load."
            )
        elif rpm > 1600:
            st.info(
                "Motor operating above rated speed. "
                "Inspect cooling and vibration levels."
            )
        elif tool_wear > 80:
            st.info(
                "Wear indicators high. "
                "Schedule preventive maintenance."
            )
        else:
            st.info(
                "Operating parameters within safe range. "
                "Continue routine monitoring."
            )

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📘 Model Details")

    st.markdown("""
    **Algorithm:** LightGBM Classifier  
    **Dataset:** AI4I 2020 Predictive Maintenance  
    **Target:** Machine Failure  

    **Why Induction Motors?**
    - Dominant industrial prime mover  
    - Failure patterns strongly correlate with RPM & Torque  
    - AI4I dataset aligns with motor-driven systems  

    **Designed for Hackathons & Industry Demos**
    """)
