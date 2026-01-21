# =========================================================
# AI-Driven Predictive Maintenance with Diagnosis
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
# Ideal machine profiles (ALL FLOATS)
# ---------------------------------------------------------
MACHINE_PROFILES = {
    "CNC Milling":      {"rpm": 2500.0, "torque": 45.0, "wear": 60.0},
    "Drilling Machine": {"rpm": 1800.0, "torque": 70.0, "wear": 50.0},
    "Grinding Machine": {"rpm": 3200.0, "torque": 20.0, "wear": 40.0},
    "Tapping Machine":  {"rpm": 600.0,  "torque": 95.0, "wear": 30.0},
    "Broaching Machine":{"rpm": 300.0,  "torque": 110.0,"wear": 40.0},
    "Shaping Machine":  {"rpm": 450.0,  "torque": 80.0, "wear": 50.0},
    "Slotting Machine": {"rpm": 500.0,  "torque": 75.0, "wear": 50.0},
    "Sawing Machine":   {"rpm": 1200.0, "torque": 55.0, "wear": 60.0},
    "Induction Motor":  {"rpm": 1450.0, "torque": 35.0, "wear": 20.0}
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"]
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

    le = LabelEncoder()
    X["Type"] = le.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, le, auc, FEATURES

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance")
    st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    st.markdown("""
    This system predicts **machine failure probability**
    using **LightGBM + SHAP** based on:
    - RPM
    - Torque
    - Tool Wear
    - Air & Process Temperature
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Manual Machine Testing")

    machine = st.selectbox("Select Machine", list(MACHINE_PROFILES.keys()))
    profile = MACHINE_PROFILES[machine]

    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input(
            "Rotational Speed (RPM)",
            min_value=0.0,
            max_value=5000.0,
            value=profile["rpm"],
            step=10.0
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=150.0,
            value=profile["torque"],
            step=1.0
        )

    with col3:
        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0.0,
            max_value=500.0,
            value=profile["wear"],
            step=1.0
        )

    air_temp = st.slider("Air Temperature (K)", 270.0, 330.0, 300.0)
    proc_temp = st.slider("Process Temperature (K)", 290.0, 380.0, 310.0)

    if st.button("🔍 Run Prediction"):
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

        st.metric("Failure Probability", f"{prob*100:.2f}%")

        status = (
            "🟢 Normal Operation" if prob < 0.25
            else "🟡 Degrading Condition" if prob < 0.6
            else "🔴 Failure Likely"
        )
        st.subheader(status)

        # SHAP diagnosis
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_array[0], index=FEATURES).abs().sort_values(ascending=False)

        st.subheader("🧠 Root Cause Analysis")
        st.bar_chart(impact)

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Info")
    st.markdown("""
    **Algorithm:** LightGBM  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020  
    **Purpose:** Industrial Predictive Maintenance
    """)
