# ===========================================
# AI-Driven Predictive Maintenance App
# Failure Probability & Root Cause Analysis
# Streamlit-ready for Hackathon
# ===========================================

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

st.title("🔧 AI-Driven Predictive Maintenance")
st.write("Predict Machine Failure Probability & Analyze Root Causes")

# ======== Load Dataset ========
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # CSV should be in repo root
    # Clean column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

df = load_data()
st.write(f"Dataset Loaded! Shape: {df.shape}")
st.dataframe(df.head())

# ======== Fake Data Generator (Optional) ========
def generate_fake_data(n=5):
    np.random.seed(42)
    fake_df = pd.DataFrame({
        'UDI': np.arange(1, n+1),
        'Product_ID': [f"M{i}" for i in range(n)],
        'Type': np.random.choice(['L','M','H'], n),
        'Air_temperature__K_': np.random.uniform(295, 305, n),
        'Process_temperature__K_': np.random.uniform(305, 315, n),
        'Rotational_speed__rpm_': np.random.randint(1200, 2500, n),
        'Torque__Nm_': np.random.uniform(5, 70, n),
        'Tool_wear__min_': np.random.randint(0, 250, n),
        'TWF': np.random.choice([0,1], n, p=[0.95,0.05]),
        'HDF': np.random.choice([0,1], n, p=[0.95,0.05]),
        'PWF': np.random.choice([0,1], n, p=[0.95,0.05]),
        'OSF': np.random.choice([0,1], n, p=[0.95,0.05]),
        'RNF': np.random.choice([0,1], n, p=[0.995,0.005])
    })
    return fake_df

st.sidebar.header("Options")
use_fake = st.sidebar.checkbox("Use fake test data for prediction?", value=False)
n_fake = st.sidebar.number_input("Number of fake samples", min_value=1, max_value=20, value=5)

# ======== Preprocessing ========
target_col = 'Machine_failure'

X = df.drop([target_col, 'Product_ID', 'UDI'], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======== Train Model ========
@st.cache_resource
def train_model():
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)
    return clf

st.info("Training LightGBM model...")
clf = train_model()
st.success("Model trained successfully!")

# ======== Evaluation ========
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

st.subheader("✅ Model Evaluation")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# ======== Prediction on New Data ========
st.subheader("🔮 Predict Machine Failure Probability")

if use_fake:
    input_df = generate_fake_data(n_fake)
    st.write("Using fake generated data:")
    st.dataframe(input_df)
else:
    st.write("Using historical dataset test set:")
    input_df = X_test.copy()
    input_df[target_col] = y_test
    st.dataframe(input_df.head())

# Predict probability of failure
predict_cols = [c for c in input_df.columns if c != target_col]
pred_probs = clf.predict_proba(input_df[predict_cols])[:,1]
input_df['Failure_Probability'] = pred_probs

st.write("Predicted Failure Probability:")
st.dataframe(input_df[['Failure_Probability'] + predict_cols])

st.info("🎉 App is ready for deployment! Use the sidebar to toggle fake data.")
