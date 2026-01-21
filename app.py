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
# Ideal machine profiles
# ---------------------------------------------------------
MACHINE_PROFILES = {
    "CNC Milling":      {"type":"M","rpm":2500.0,"torque":45.0,"wear":60.0,"air_temp":300.0,"proc_temp":310.0},
    "Drilling Machine": {"type":"M","rpm":1800.0,"torque":70.0,"wear":50.0,"air_temp":300.0,"proc_temp":310.0},
    "Grinding Machine": {"type":"M","rpm":3200.0,"torque":20.0,"wear":40.0,"air_temp":300.0,"proc_temp":310.0},
    "Tapping Machine":  {"type":"M","rpm":600.0,"torque":95.0,"wear":30.0,"air_temp":300.0,"proc_temp":310.0},
    "Broaching Machine":{"type":"M","rpm":300.0,"torque":110.0,"wear":40.0,"air_temp":300.0,"proc_temp":310.0},
    "Shaping Machine":  {"type":"M","rpm":450.0,"torque":80.0,"wear":50.0,"air_temp":300.0,"proc_temp":310.0},
    "Slotting Machine": {"type":"M","rpm":500.0,"torque":75.0,"wear":50.0,"air_temp":300.0,"proc_temp":310.0},
    "Sawing Machine":   {"type":"M","rpm":1200.0,"torque":55.0,"wear":60.0,"air_temp":300.0,"proc_temp":310.0},
    "Induction Motor":  {"type":"M","rpm":1450.0,"torque":35.0,"wear":20.0,"air_temp":300.0,"proc_temp":310.0}
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔧 Predictive Maintenance AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Model Info"],
    key="nav_menu"
)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # make sure this file exists
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df

df = load_data()

# ---------------------------------------------------------
# Train model
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
    return model, le, auc, features

model, encoder, auc_score, FEATURES = train_model(df)
explainer = shap.TreeExplainer(model)

# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.title("🔧 AI-Driven Predictive Maintenance")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### System Capabilities
        - Failure probability prediction  
        - Multi-machine support  
        - Root cause diagnosis (SHAP)  
        - Ideal operating profiles loaded automatically  
        """)
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    st.divider()
    st.markdown("""
    Predict failures in **rotating & cutting machinery** using real sensor features:  
    **RPM, Torque, Tool Wear, Air & Process Temperature**
    """)

# =========================================================
# MANUAL PREDICTION
# =========================================================
if menu == "Manual Prediction":
    st.title("📊 Machine Failure Prediction & Diagnosis")
    machine = st.selectbox("Machine Under Test", list(MACHINE_PROFILES.keys()), key="machine_select")
    profile = MACHINE_PROFILES[machine]

    col1, col2, col3 = st.columns(3)
    with col1:
        rpm = st.number_input("Rotational Speed (RPM)", 0, 5000, value=float(profile["rpm"]), step=10, key="rpm")
    with col2:
        torque = st.number_input("Torque (Nm)", 0.0, 150.0, value=float(profile["torque"]), step=1.0, key="torque")
    with col3:
        tool_wear = st.number_input("Tool Wear (min)", 0, 500, value=float(profile["wear"]), step=1, key="wear")

    air_temp = st.slider("Air Temperature (K)", 270, 330, float(profile["air_temp"]))
    proc_temp = st.slider("Process Temperature (K)", 290, 380, float(profile["proc_temp"]))

    if st.button("🔍 Run Prediction"):
        input_df = pd.DataFrame([{
            "Type": profile["type"],
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
        status = (
            "🟢 Running at Full Capability" if prob < 0.25
            else "🟡 Degrading / Partial Capability" if prob < 0.6
            else "🔴 Failure Likely / Needs Attention"
        )
        st.subheader(status)

        # ---------------- Diagnosis ----------------
        shap_vals = explainer.shap_values(input_df)
        shap_array = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        impact = pd.Series(shap_array[0], index=FEATURES).abs().sort_values(ascending=False)
        main = impact.index[0]

        st.subheader("🧠 Failure Diagnosis")
        st.bar_chart(impact)

        # More descriptive root cause
        if "rpm" in main.lower():
            st.info(
                f"{machine}: High rotational speed increases heat, vibration, and dynamic stress, accelerating wear and potential failure."
            )
        elif "torque" in main.lower():
            st.info(
                f"{machine}: Excessive torque imposes high mechanical stress, increasing the likelihood of mechanical or drive failures."
            )
        elif "wear" in main.lower():
            st.info(
                f"{machine}: Tool wear has exceeded safe limits, causing poor cutting efficiency, heat build-up, and risk of fracture."
            )
        else:
            st.info(f"{machine}: Combined thermal and mechanical stress is driving failure risk.")

# =========================================================
# MODEL INFO
# =========================================================
if menu == "Model Info":
    st.title("📚 Model Information")
    st.markdown(f"""
    **Model:** LightGBM Classifier  
    **Explainability:** SHAP  
    **Dataset:** AI4I 2020 Predictive Maintenance  
    **Features Used:** {', '.join(FEATURES)}
    """)
    st.success("Explainable AI for predictive maintenance of multiple industrial machines.")
