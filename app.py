import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="AI Predictive Maintenance – Induction Motors", layout="wide")

# ---------------------------------
# SIDEBAR – DATA UPLOAD
# ---------------------------------
st.sidebar.title("📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Predictive Maintenance CSV",
    type=["csv"]
)

# ---------------------------------
# LOAD DATA SAFELY
# ---------------------------------
@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)

    # ✅ Fallback synthetic dataset (NO CRASH)
    np.random.seed(42)
    return pd.DataFrame({
        "Air temperature [K]": np.random.normal(300, 5, 1000),
        "Process temperature [K]": np.random.normal(310, 5, 1000),
        "Rotational speed [rpm]": np.random.normal(1500, 200, 1000),
        "Torque [Nm]": np.random.normal(40, 10, 1000),
        "Tool wear [min]": np.random.randint(0, 250, 1000),
        "Machine failure": np.random.binomial(1, 0.15, 1000)
    })

df = load_data(uploaded_file)

# ---------------------------------
# TRAIN MODEL (NO FEATURE ERRORS)
# ---------------------------------
@st.cache_resource
def train_model(data):
    FEATURES = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = data[FEATURES].values
    y = data["Machine failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc

model, auc_score = train_model(df)

# ---------------------------------
# RUL FUNCTION (INDUCTION MOTOR LOGIC)
# ---------------------------------
def calculate_rul(failure_prob, torque, rpm):
    base = (1 - failure_prob) * 1200
    load_factor = torque / max(rpm, 1)
    degradation = load_factor * 150
    return max(50, int(base - degradation))

# ---------------------------------
# NAVIGATION
# ---------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Manual Input", "RUL & What-If", "Maintenance Advice"]
)

# ---------------------------------
# OVERVIEW
# ---------------------------------
if page == "Overview":
    st.title("🔧 AI-Based Predictive Maintenance (Induction Motors)")
    st.metric("Model ROC-AUC", round(auc_score, 3))

    st.markdown("""
    **Optimized For**
    - Induction Motors
    - Pumps, Fans, Conveyors
    - Rotating Machinery

    **Key Features**
    - Failure Probability
    - Remaining Useful Life (RUL)
    - Load-Based What-If Simulation
    """)

# ---------------------------------
# MANUAL INPUT
# ---------------------------------
if page == "Manual Input":
    st.title("⚙️ Motor Operating Conditions")

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)
        torque = st.number_input("Torque (Nm)", value=40.0)

    with col2:
        rpm_raw = st.number_input("Rotational Speed (RPM)", value=1500.0)
        tool_wear = st.number_input("Tool Wear (min)", value=120.0)

    # ✅ Physical relation: Torque ↑ → RPM ↓
    rpm = max(300, rpm_raw - (torque * 1.8))

    input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])

    if st.button("🔮 Predict Failure"):
        prob = model.predict_proba(input_data)[0][1]
        st.metric("Failure Probability", f"{prob*100:.2f}%")

        st.session_state.update({
            "prob": prob,
            "rpm": rpm,
            "torque": torque
        })

# ---------------------------------
# RUL + WHAT-IF
# ---------------------------------
if page == "RUL & What-If":
    st.title("🕒 Remaining Useful Life")

    if "prob" not in st.session_state:
        st.warning("⚠️ Run prediction first")
    else:
        rul = calculate_rul(
            st.session_state["prob"],
            st.session_state["torque"],
            st.session_state["rpm"]
        )

        st.metric("Estimated RUL", f"{rul} hours")

        st.subheader("🔁 What-If Load Simulation")
        st.write(f"+10% Load → ~{int(rul*0.7)} hrs")
        st.write(f"+20% Load → ~{int(rul*0.5)} hrs")

# ---------------------------------
# MAINTENANCE ADVICE
# ---------------------------------
if page == "Maintenance Advice":
    st.title("🧠 Maintenance Recommendation")

    if "prob" not in st.session_state:
        st.warning("⚠️ Predict failure first")
    else:
        p = st.session_state["prob"]

        if p < 0.3:
            st.success("✅ Normal operation. Routine monitoring recommended.")
        elif p < 0.6:
            st.warning("⚠️ Medium risk. Inspect bearings & lubrication.")
        else:
            st.error("🚨 High failure risk. Schedule immediate maintenance.")
