# =========================================================
# Hackathon-Ready AI Predictive Maintenance for Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="⚡",
    layout="wide"
)

# ---------------- Ideal Motor Profile ----------------
INDUCTION_MOTOR_PROFILE = {
    "rpm": 1450.0,
    "torque": 35.0,
    "tool_wear": 20.0,
    "air_temp": 300.0,
    "process_temp": 310.0
}

# ---------------- Sidebar ----------------
st.sidebar.title("⚡ Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "CSV / Batch Prediction", "Model Info"],
    key="nav_menu"
)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------- Train Model ----------------
@st.cache_data
def train_model():
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
    explainer = shap.TreeExplainer(model)

    return model, le, auc, FEATURES, explainer

model, encoder, auc_score, FEATURES, explainer = train_model()

# ---------------- Helper Function ----------------
def interpret_prediction(prob):
    if prob > 0.6:
        return "🔴 Failure Likely", max(0, 100 - prob * 100)
    elif prob > 0.25:
        return "🟡 Degrading Condition", max(0, 100 - prob * 100)
    else:
        return "🟢 Normal Operation", max(0, 100 - prob * 100)

# ---------------- Home ----------------
if menu == "Home":
    st.title("⚡ AI-Driven Predictive Maintenance (Induction Motors)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Project Capabilities**
        - Failure probability prediction  
        - Torque–RPM inverse relation  
        - What-if load simulation  
        - Maintenance recommendations  
        - Motor health score  
        - SHAP explainability  
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")

# ---------------- Manual Prediction ----------------
if menu == "Manual Prediction":
    st.title("📊 Manual Prediction & What-If Simulation")

    col1, col2, col3 = st.columns(3)

    torque = col1.number_input("Torque (Nm)", 0.0, 200.0, 35.0)
    rpm = col2.number_input("Rotational Speed (RPM)", 0.0, 5000.0, 1450.0)
    tool_wear = col3.number_input("Tool Wear (min)", 0.0, 500.0, 20.0)

    air_temp = st.number_input("Air Temperature (K)", 0.0, 1000.0, 300.0)
    process_temp = st.number_input("Process Temperature (K)", 0.0, 2000.0, 310.0)

    if st.button("🔍 Predict Failure"):
        input_df = pd.DataFrame([{
            "Type": "M",
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": process_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])

        input_df["Type"] = encoder.transform(input_df["Type"].astype(str))
        prob = model.predict_proba(input_df[FEATURES])[0][1]
        status, health = interpret_prediction(prob)

        st.metric("Failure Probability", f"{prob*100:.2f}%")
        st.subheader(status)
        st.progress(int(health))

# ---------------- CSV / Batch Prediction ----------------
if menu == "CSV / Batch Prediction":
    st.title("📂 CSV / Batch Failure Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)

        missing_cols = set(FEATURES) - set(batch_df.columns)
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        batch_df = batch_df[FEATURES].copy()
        batch_df["Type"] = encoder.transform(batch_df["Type"].astype(str))

        probs = model.predict_proba(batch_df[FEATURES])[:, 1]
        statuses, health_scores = [], []

        for p in probs:
            s, h = interpret_prediction(p)
            statuses.append(s)
            health_scores.append(h)

        batch_df["Failure_Probability"] = probs
        batch_df["Failure_Status"] = statuses
        batch_df["Health_Score"] = health_scores

        st.subheader("📊 Batch Prediction Results")
        st.dataframe(batch_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            batch_df.to_csv(index=False),
            "predictive_maintenance_batch_results.csv",
            "text/csv"
        )

# ---------------- Model Info ----------------
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown(f"""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  
    **ROC-AUC:** {auc_score:.3f}
    """)
