# =========================================================
# AI-Driven Predictive Maintenance (Stable Version)
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="main_menu"
)

# -------------------------------
# Machine Ideal Profiles
# -------------------------------
MACHINE_PROFILES = {
    "CNC Milling":        {"rpm": 2500, "torque": 50, "wear": 60},
    "Drilling Machine":  {"rpm": 1800, "torque": 70, "wear": 50},
    "Grinding Machine":  {"rpm": 3500, "torque": 20, "wear": 40},
    "Tapping Machine":   {"rpm": 800,  "torque": 90, "wear": 45},
    "Broaching Machine": {"rpm": 300,  "torque": 120,"wear": 70},
    "Shaping Machine":   {"rpm": 500,  "torque": 80, "wear": 60},
    "Slotting Machine":  {"rpm": 600,  "torque": 75, "wear": 55},
    "Sawing Machine":    {"rpm": 1200, "torque": 60, "wear": 50},
    "Induction Motor":   {"rpm": 1450, "torque": 40, "wear": 30},
}

# -------------------------------
# Load + Prepare Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type"] = df["Type"].map(type_map)

    features = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]

    X = df[features].astype(float)
    y = df["Machine_failure"].astype(int)

    return X, y, features

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model():
    X, y, features = load_data()

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
    explainer = shap.TreeExplainer(model)

    return model, explainer, auc, features

model, explainer, auc_score, feature_columns = train_model()

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **What this system does**
        - Predicts machine failure probability
        - Explains *why* failure may occur
        - Uses real industrial sensor data
        - Starts from ideal operating conditions
        """)

    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")

    machine = st.selectbox(
        "Machine Under Test",
        list(MACHINE_PROFILES.keys()),
        key="machine_select"
    )

    defaults = MACHINE_PROFILES[machine]

    col1, col2, col3 = st.columns(3)

    with col1:
        machine_class = st.selectbox("Machine Class", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)

    with col2:
        process_temp = st.number_input("Process Temperature (K)", 250.0, 450.0, 310.0)
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            100, 6000,
            value=defaults["rpm"],
            key=f"rpm_{machine}"
        )

    with col3:
        torque = st.number_input(
            "Torque (Nm)",
            0.0, 200.0,
            value=float(defaults["torque"]),
            key=f"torque_{machine}"
        )
        wear = st.number_input(
            "Tool Wear (min)",
            0, 500,
            value=defaults["wear"],
            key=f"wear_{machine}"
        )

    if st.button("🔍 Predict"):
        type_map = {"L": 0, "M": 1, "H": 2}

        input_df = pd.DataFrame([{
            "Type": type_map[machine_class],
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": process_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": wear
        }])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        if prob < 0.15:
            st.success("🟢 Operating Normally")
        elif prob < 0.5:
            st.warning("🟡 Degrading – Monitor Closely")
        else:
            st.error("🔴 High Failure Risk")

        # -------------------------------
        # Root Cause (SHAP)
        # -------------------------------
        shap_values = explainer.shap_values(input_df)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        impact = pd.Series(
            abs(shap_array[0]),
            index=feature_columns
        ).sort_values(ascending=False)

        st.subheader("🧠 Root Cause Contribution")
        st.bar_chart(impact)

        st.info(f"Primary contributor: **{impact.index[0]}**")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")

    st.markdown("""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  

    The model learns failure patterns from:
    - RPM
    - Torque
    - Tool Wear
    - Temperature conditions
    """)

    st.success("System ready for industrial-grade predictive maintenance.")
