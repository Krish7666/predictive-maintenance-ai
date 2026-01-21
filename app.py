# =========================================================
# AI-Driven Predictive Maintenance for Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

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
menu = st.sidebar.radio("Navigation", ["Home", "Manual Prediction", "Model Info"], key="nav_menu")

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # make sure this file exists
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

# ---------------- Home ----------------
if menu == "Home":
    st.title("⚡ AI-Driven Predictive Maintenance (Induction Motors)")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Capabilities**
        - Failure probability prediction  
        - Torque-RPM inverse relation  
        - What-if load simulation  
        - Maintenance recommendations  
        - SHAP explainability  
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    st.divider()
    st.markdown("Predict failures in **industrial induction motors** using operational data.")

# ---------------- Manual Prediction ----------------
if menu == "Manual Prediction":
    st.title("📊 Manual Prediction & What-If Simulation")

    st.subheader("Induction Motor Inputs")
    col1, col2, col3 = st.columns(3)

    # Torque input
    torque = col1.number_input(
        "Torque (Nm)",
        min_value=0.0, max_value=200.0,
        value=float(INDUCTION_MOTOR_PROFILE["torque"]),
        step=1.0
    )

    # RPM input: inversely related to torque
    rpm_default = max(0.0, INDUCTION_MOTOR_PROFILE["rpm"] - (torque - INDUCTION_MOTOR_PROFILE["torque"])*10)
    rpm = col2.number_input(
        "Rotational Speed (RPM)",
        min_value=0.0, max_value=5000.0,
        value=float(rpm_default),
        step=10.0
    )

    # Tool wear
    tool_wear = col3.number_input(
        "Tool Wear (min)",
        min_value=0.0, max_value=500.0,
        value=float(INDUCTION_MOTOR_PROFILE["tool_wear"]),
        step=1.0
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=0.0, max_value=1000.0,
        value=float(INDUCTION_MOTOR_PROFILE["air_temp"]),
        step=1.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=0.0, max_value=2000.0,
        value=float(INDUCTION_MOTOR_PROFILE["process_temp"]),
        step=1.0
    )

    # Optional file upload
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if st.button("🔍 Predict Failure"):
        input_df = pd.DataFrame([{
            "Type": "M",  # Induction motor
            "Air_temperature__K_": air_temp,
            "Process_temperature__K_": process_temp,
            "Rotational_speed__rpm_": rpm,
            "Torque__Nm_": torque,
            "Tool_wear__min_": tool_wear
        }])
        input_df["Type"] = encoder.transform(input_df["Type"])
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.metric("Failure Probability", f"{prob*100:.2f}%")
        status = (
            "🟢 Normal Operation" if prob < 0.25
            else "🟡 Degrading Condition" if prob < 0.6
            else "🔴 Failure Likely"
        )
        st.subheader(status)

        # ---------------- Diagnosis ----------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_array[0], index=FEATURES).abs().sort_values(ascending=False)
        main = impact.index[0]

        st.subheader("🧠 Failure Diagnosis")
        if "rpm" in main.lower():
            st.info("High RPM can increase thermal and dynamic stress, accelerating wear and vibration fatigue.")
        elif "torque" in main.lower():
            st.info("High torque places mechanical load on drivetrain, increasing component stress and failure risk.")
        elif "wear" in main.lower():
            st.info("Excessive tool wear causes friction, poor cutting, and heat generation.")
        else:
            st.info("Failure risk is driven by combined thermal and mechanical loading conditions.")

        # ---------------- What-If Load Simulation ----------------
        st.subheader("⚡ What-If Load Simulation")
        sim_torque = st.slider("Simulate Torque Increase", 0.0, 200.0, float(torque), 1.0)
        sim_rpm = max(0.0, rpm - (sim_torque - torque)*10)
        st.metric("Simulated RPM due to Torque change", f"{sim_rpm:.2f}")

        # ---------------- Maintenance Recommendation ----------------
        st.subheader("🛠 Maintenance Recommendations")
        if prob > 0.6:
            st.success("🔧 Immediate inspection & preventive maintenance required.")
        elif prob > 0.25:
            st.info("⚙️ Schedule routine maintenance soon.")
        else:
            st.success("✅ Motor operating normally. Continue standard monitoring.")

        # ---------------- Motor Health Score ----------------
        health_score = max(0, 100 - prob*100)
        st.subheader("💚 Motor Health Score")
        st.progress(int(health_score))

# ---------------- Model Info ----------------
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown(f"""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  
    **Focus:** Induction motors only  

    **Model ROC-AUC:** {auc_score:.3f}
    """)
