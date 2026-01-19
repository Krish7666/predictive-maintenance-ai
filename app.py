# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

st.set_page_config(page_title="AI-Driven Predictive Maintenance", layout="wide")

st.title("🔧 AI-Driven Predictive Maintenance")
st.subheader("Predict Machine Failure Probability & Analyze Root Causes")

# ====== Upload or Load Dataset ======
uploaded_file = st.file_uploader("Upload CSV dataset (ai4i2020.csv)", type="csv")

use_fake = st.checkbox("Use fake data for testing (optional)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_fake:
    st.info("Generating fake dataset...")
    np.random.seed(42)
    n_samples = st.number_input("Number of fake samples", min_value=5, max_value=1000, value=10)
    df = pd.DataFrame({
        "UDI": range(1, n_samples+1),
        "Product_ID": ["M"+str(i) for i in range(n_samples)],
        "Type": np.random.choice(["L","M","H"], n_samples),
        "Air_temperature_[K]": np.random.uniform(295, 305, n_samples),
        "Process_temperature_[K]": np.random.uniform(305, 315, n_samples),
        "Rotational_speed_[rpm]": np.random.randint(1200, 2500, n_samples),
        "Torque_[Nm]": np.random.uniform(10, 70, n_samples),
        "Tool_wear_[min]": np.random.randint(0, 250, n_samples),
        "Machine_failure": np.random.choice([0,1], n_samples, p=[0.95,0.05]),
        "TWF": np.random.choice([0,1], n_samples, p=[0.95,0.05]),
        "HDF": np.random.choice([0,1], n_samples, p=[0.95,0.05]),
        "PWF": np.random.choice([0,1], n_samples, p=[0.95,0.05]),
        "OSF": np.random.choice([0,1], n_samples, p=[0.95,0.05]),
        "RNF": np.random.choice([0,1], n_samples, p=[0.99,0.01])
    })
else:
    st.warning("Please upload a CSV dataset or select fake data.")
    st.stop()

st.success(f"Dataset Loaded! Shape: {df.shape}")

# ===== Clean column names =====
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# ===== Preprocessing =====
target_col = "Machine_failure"
X = df.drop([target_col, 'Product_ID', 'UDI'], axis=1)
y = df[target_col]

# Handle categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== Train LightGBM Model =====
st.info("Training LightGBM model... This may take a few seconds.")
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# ===== Predictions =====
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

st.subheader("✅ Model Evaluation")
st.text(classification_report(y_test, y_pred))
st.write("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# ===== Predict for New Data =====
st.subheader("Predict Failure for New Inputs")
with st.form(key="predict_form"):
    type_input = st.selectbox("Type", ["L","M","H"])
    air_temp = st.number_input("Air temperature [K]", 295, 315, 300)
    process_temp = st.number_input("Process temperature [K]", 305, 315, 310)
    rpm = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
    torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool wear [min]", 0, 300, 100)
    TWF = st.selectbox("TWF", [0,1])
    HDF = st.selectbox("HDF", [0,1])
    PWF = st.selectbox("PWF", [0,1])
    OSF = st.selectbox("OSF", [0,1])
    RNF = st.selectbox("RNF", [0,1])
    submit_btn = st.form_submit_button("Predict Failure Probability")

if submit_btn:
    new_data = pd.DataFrame({
        "Type": [LabelEncoder().fit(["L","M","H"]).transform([type_input])[0]],
        "Air_temperature_[K]": [air_temp],
        "Process_temperature_[K]": [process_temp],
        "Rotational_speed_[rpm]": [rpm],
        "Torque_[Nm]": [torque],
        "Tool_wear_[min]": [tool_wear],
        "TWF": [TWF],
        "HDF": [HDF],
        "PWF": [PWF],
        "OSF": [OSF],
        "RNF": [RNF]
    })
    pred_prob = clf.predict_proba(new_data)[:,1][0]
    st.write(f"🔴 Predicted Failure Probability: **{pred_prob*100:.2f}%**")

    # Simple root cause hint
    st.write("⚡ Likely Root Causes Based on Feature Values:")
    causes = []
    if tool_wear > 200: causes.append("High Tool Wear")
    if torque > 60: causes.append("Excess Torque")
    if air_temp > 303: causes.append("High Air Temperature")
    if process_temp > 312: causes.append("High Process Temperature")
    if rpm > 2000: causes.append("High Rotational Speed")
    if not causes:
        causes.append("Normal conditions")
    st.write(", ".join(causes))
