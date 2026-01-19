# =====================================
# AI-Driven Predictive Maintenance App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")
st.title("🔧 AI-Driven Predictive Maintenance")
st.subheader("Predict Machine Failure Probability & Analyze Root Causes")

# -------------------------------
# 1. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # Make sure CSV is in the repo
    # Clean column names for LightGBM
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

df = load_data()
st.success(f"Historic Dataset Loaded! Shape: {df.shape}")

# -------------------------------
# 2. Manual Data Entry (User Input)
# -------------------------------
st.subheader("Enter Data Manually (Multiple Rows Supported)")

num_rows = st.number_input("Number of entries to add:", min_value=1, max_value=5, value=1)
manual_entries = []

for i in range(num_rows):
    st.markdown(f"**Entry {i+1}**")
    Type = st.selectbox(f"Type (Entry {i+1})", ['L', 'M', 'H'], key=f"type_{i}")
    Air_temp = st.number_input(f"Air temperature [K] (Entry {i+1})", 250.0, 400.0, 300.0, key=f"air_{i}")
    Process_temp = st.number_input(f"Process temperature [K] (Entry {i+1})", 250.0, 400.0, 310.0, key=f"process_{i}")
    Speed = st.number_input(f"Rotational speed [rpm] (Entry {i+1})", 100, 5000, 1500, key=f"speed_{i}")
    Torque = st.number_input(f"Torque [Nm] (Entry {i+1})", 0.0, 200.0, 40.0, key=f"torque_{i}")
    Tool_wear = st.number_input(f"Tool wear [min] (Entry {i+1})", 0, 500, 100, key=f"tool_{i}")
    
    manual_entries.append({
        'Type': Type,
        'Air_temperature__K_': Air_temp,
        'Process_temperature__K_': Process_temp,
        'Rotational_speed__rpm_': Speed,
        'Torque__Nm_': Torque,
        'Tool_wear__min_': Tool_wear
    })

manual_data = pd.DataFrame(manual_entries)

# -------------------------------
# 3. Preprocessing
# -------------------------------
X = df[['Type', 'Air_temperature__K_', 'Process_temperature__K_', 
        'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_']]
y = df['Machine_failure']

# Encode 'Type'
le = LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])

# -------------------------------
# 4. Train LightGBM Model
# -------------------------------
@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1)
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    st.subheader("Model Evaluation on Historic Data")
    st.text(classification_report(y_test, y_pred))
    st.write("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    return clf, le

clf, le = train_model(X, y)

# -------------------------------
# 5. Predict on Manual Data
# -------------------------------
if st.button("Predict Manual Data"):
    if not manual_data.empty:
        manual_X = manual_data.copy()
        manual_X['Type'] = le.transform(manual_X['Type'])
        manual_pred_prob = clf.predict_proba(manual_X)[:,1]
        manual_pred_class = clf.predict(manual_X)
        result_df = manual_data.copy()
        result_df['Failure_Probability'] = manual_pred_prob
        result_df['Predicted_Class'] = manual_pred_class
        st.subheader("Prediction Results for Manual Data")
        st.dataframe(result_df)

# -------------------------------
# 6. Footer / Notes
# -------------------------------
st.info("This app predicts machine failure probability and shows results based on both historic dataset and manual input by the user.")
