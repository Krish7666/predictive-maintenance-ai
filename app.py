import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import lightgbm as lgb
import shap

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    layout="wide"
)

st.title("🔧 AI-Based Predictive Maintenance System")
st.write("Predict machine failure and identify root causes using Explainable AI")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    return df

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -------------------- DATA PREPROCESSING --------------------
df = df.drop(columns=["UDI", "Product ID"])

le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- MODEL TRAINING --------------------
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{acc * 100:.2f}%")

# -------------------- SHAP EXPLAINER --------------------
explainer = shap.TreeExplainer(model)

# -------------------- USER INPUT --------------------
st.subheader("🧪 Machine Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    type_input = st.selectbox("Product Type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature [K]", 290.0, 320.0, 300.0)
    process_temp = st.number_input("Process temperature [K]", 300.0, 350.0, 310.0)

with col2:
    speed = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
    torque = st.number_input("Torque [Nm]", 10.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool wear [min]", 0, 300, 50)

with col3:
    st.info("Fill realistic machine values for accurate prediction")

type_encoded = le.transform([type_input])[0]

input_df = pd.DataFrame([[
    type_encoded,
    air_temp,
    process_temp,
    speed,
    torque,
    tool_wear
]], columns=feature_names)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Machine Failure"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Machine Failure Likely (Risk: {probability * 100:.2f}%)")
    else:
        st.success(f"✅ Machine Healthy (Risk: {probability * 100:.2f}%)")

    # -------------------- ROOT CAUSE ANALYSIS --------------------
    st.subheader("🧠 Root Cause Analysis (Explainable AI)")

    shap_values = explainer.shap_values(input_df)

    # SAFE handling of SHAP outputs
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    shap_df = pd.DataFrame(
        shap_array,
        columns=feature_names
    )

    impact = shap_df.iloc[0].abs().sort_values(ascending=False)

    st.write("Top contributing parameters:")
    st.dataframe(impact.head(5))

    # -------------------- SHAP BAR PLOT --------------------
    fig, ax = plt.subplots()
    impact.head(5).plot(kind="barh", ax=ax)
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Top 5 Root Causes")
    plt.gca().invert_yaxis()

    st.pyplot(fig)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Developed for academic & industrial predictive maintenance use")
