import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Induction Motor Predictive Maintenance", layout="wide")

st.title("🔧 AI-Based Predictive Maintenance (Induction Motors)")

# -----------------------------
# Load & Prepare Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")

    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    FEATURES = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[FEATURES]
    y = df["Machine failure"]

    return X, y, FEATURES

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model():
    X, y, FEATURES = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc, FEATURES

model, auc_score, FEATURES = train_model()

st.success(f"Model Loaded | ROC-AUC: {auc_score:.3f}")

# -----------------------------
# Manual Input Section
# -----------------------------
st.subheader("🧪 Manual Testing (Induction Motor)")

col1, col2 = st.columns(2)

with col1:
    load_class = st.selectbox(
        "Operational Load Class (Dataset Proxy)",
        options=["Low", "Medium", "High"],
        index=1
    )

    load_map = {"Low": 0, "Medium": 1, "High": 2}
    type_val = load_map[load_class]

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0,
        step=1.0
    )

    tool_wear = st.number_input(
        "Tool Wear / Runtime Wear (min)",
        value=100.0,
        step=10.0
    )

with col2:
    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0,
        step=1.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0,
        step=1.0
    )

    # Torque ↑ → RPM ↓ (Induction motor behavior)
    rpm = max(500.0, 1500.0 - (torque * 3))

    st.info(f"Estimated RPM (Auto-adjusted): {rpm:.0f}")

# -----------------------------
# Prediction
# -----------------------------
input_data = pd.DataFrame([[
    type_val,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
]], columns=FEATURES)

failure_prob = model.predict_proba(input_data)[0][1]

st.subheader("📊 Prediction Result")

if failure_prob < 0.3:
    st.success(f"Low Failure Risk — {failure_prob*100:.1f}%")
elif failure_prob < 0.6:
    st.warning(f"Moderate Failure Risk — {failure_prob*100:.1f}%")
else:
    st.error(f"High Failure Risk — {failure_prob*100:.1f}%")

# -----------------------------
# Graph (Torque vs RPM)
# -----------------------------
st.subheader("📉 Torque vs RPM Relationship (Induction Motor)")

torque_range = np.linspace(0, 100, 50)
rpm_curve = 1500 - torque_range * 3
rpm_curve = np.clip(rpm_curve, 500, None)

fig, ax = plt.subplots()
ax.plot(torque_range, rpm_curve)
ax.scatter(torque, rpm, s=80)

ax.set_xlabel("Torque (Nm)")
ax.set_ylabel("RPM")
ax.set_title("Inverse Torque–Speed Characteristic")

st.pyplot(fig)

# -----------------------------
# Cause Analysis
# -----------------------------
st.subheader("🧠 Cause Analysis")

st.write("""
The predicted failure probability is influenced by **load severity**, **thermal stress**, 
and **mechanical wear**.  

Higher torque increases current draw in induction motors, leading to **thermal rise** and 
speed drop. If this condition persists alongside elevated temperatures or wear, the risk of 
bearing damage, insulation breakdown, or overload-related failure increases significantly.
""")
