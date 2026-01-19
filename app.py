import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Predictive Maintenance AI", layout="centered")

st.title("🔧 AI-Driven Predictive Maintenance")
st.subheader("Failure Probability & Root Cause Analysis")

# =====================================
# LOAD & TRAIN MODEL
# =====================================
@st.cache_resource
def train_model():
   df = pd.read_csv("ai4i2020.csv")

    # Clean column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    # Encode categorical
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

    X = df.drop('Machine_failure', axis=1)
    y = df['Machine_failure']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, le

model, le = train_model()

st.success("✅ Model trained successfully using historical data")

# =====================================
# USER INPUT SECTION
# =====================================
st.header("📥 Enter Live Machine Parameters")

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.slider("Air Temperature (K)", 290, 320, 300)
process_temp = st.slider("Process Temperature (K)", 300, 330, 310)
rpm = st.slider("Rotational Speed (RPM)", 1000, 3000, 1800)
torque = st.slider("Torque (Nm)", 10, 100, 60)
tool_wear = st.slider("Tool Wear (min)", 0, 300, 150)

# =====================================
# PREDICTION
# =====================================
if st.button("🔍 Predict Machine Health"):

    input_data = pd.DataFrame([{
        'UDI': 99999,
        'Product_ID': 999,
        'Type': le.transform([machine_type])[0],
        'Air_temperature_K': air_temp,
        'Process_temperature_K': process_temp,
        'Rotational_speed_rpm': rpm,
        'Torque_Nm': torque,
        'Tool_wear_min': tool_wear,
        'TWF': 0,
        'HDF': 0,
        'PWF': 0,
        'OSF': 0,
        'RNF': 0
    }])

    failure_prob = model.predict_proba(input_data)[0][1]

    # Health Status
    if failure_prob < 0.3:
        status = "🟢 NORMAL"
    elif failure_prob < 0.7:
        status = "🟠 WARNING"
    else:
        status = "🔴 CRITICAL"

    # Root Cause Analysis
    root_causes = []

    if process_temp > 310:
        root_causes.append("High Process Temperature")

    if torque > 65:
        root_causes.append("Excessive Load / Torque")

    if tool_wear > 200:
        root_causes.append("Tool Wear Limit Reached")

    # =====================================
    # DISPLAY OUTPUT
    # =====================================
    st.subheader("📊 Prediction Result")
    st.metric("Failure Probability", f"{failure_prob*100:.2f}%")
    st.write("### Machine Status:", status)

    st.subheader("🧠 Root Cause Analysis")
    if root_causes:
        for cause in root_causes:
            st.warning(cause)
    else:
        st.success("No abnormal patterns detected")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("AI-Driven Predictive Maintenance | Hackathon Demo")
