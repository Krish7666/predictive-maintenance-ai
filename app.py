import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(
    page_title="AI Predictive Maintenance – Induction Motors",
    layout="wide"
)

st.title("🔧 AI-Based Predictive Maintenance (Induction Motors)")

# =====================================================
# Load & Clean Dataset (CRITICAL FIX)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")

    # ---- FIX: sanitize column names for LightGBM ----
    df.columns = (
        df.columns
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_")
    )

    # Encode load type (proxy for operating load)
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    FEATURES = [
        "Type",
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min"
    ]

    X = df[FEATURES]
    y = df["Machine_failure"]

    return X, y, FEATURES

# =====================================================
# Train LightGBM Model
# =====================================================
@st.cache_resource
def train_model():
    X, y, FEATURES = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    return model, auc, FEATURES

model, auc_score, FEATURES = train_model()

st.success(f"Model trained successfully | ROC-AUC: {auc_score:.3f}")

# =====================================================
# Manual Input (Induction Motor)
# =====================================================
st.subheader("🧪 Manual Failure Prediction")

col1, col2 = st.columns(2)

with col1:
    load_class = st.selectbox(
        "Operational Load Class",
        ["Low", "Medium", "High"],
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
        step=5.0
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

    # ---- Induction motor physics ----
    rpm = max(500.0, 1500.0 - torque * 3)

    st.info(f"Estimated Motor Speed: **{rpm:.0f} RPM**")

# =====================================================
# Prediction
# =====================================================
input_df = pd.DataFrame([[
    type_val,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
]], columns=FEATURES)

failure_prob = model.predict_proba(input_df)[0][1]

st.subheader("📊 Prediction Result")

if failure_prob < 0.3:
    st.success(f"🟢 Low Risk | Failure Probability: {failure_prob*100:.1f}%")
elif failure_prob < 0.6:
    st.warning(f"🟡 Moderate Risk | Failure Probability: {failure_prob*100:.1f}%")
else:
    st.error(f"🔴 High Risk | Failure Probability: {failure_prob*100:.1f}%")

# =====================================================
# Graph: Torque vs RPM
# =====================================================
st.subheader("📉 Torque–Speed Characteristic (Induction Motor)")

torque_range = np.linspace(0, 100, 50)
rpm_curve = np.clip(1500 - torque_range * 3, 500, None)

fig, ax = plt.subplots()
ax.plot(torque_range, rpm_curve)
ax.scatter(torque, rpm, s=80)

ax.set_xlabel("Torque (Nm)")
ax.set_ylabel("RPM")
ax.set_title("Inverse Torque–Speed Relationship")

st.pyplot(fig)

# =====================================================
# Cause Analysis
# =====================================================
st.subheader("🧠 Cause Analysis")

st.write("""
In induction motors, **increased torque demand causes speed reduction**
due to higher slip and current draw.

When elevated torque is combined with:
- high operating temperatures  
- increasing runtime wear  

the motor experiences **thermal stress, insulation aging, and bearing load**,
significantly increasing the probability of failure.
""")
