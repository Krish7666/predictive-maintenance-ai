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

st.title("🔧 AI-Driven Predictive Maintenance")
st.subheader("Predict Machine Failure Probability & Analyze Root Causes")

# -------------------------------
# 1. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")  # CSV must be in repo
    # Clean column names for LightGBM
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

df = load_data()
st.success(f"Dataset Loaded! Shape: {df.shape}")

# -------------------------------
# 2. User Manual Input
# -------------------------------
st.subheader("Or Enter Data Manually")
num_rows = st.number_input("Number of entries to add:", min_value=1, max_value=10, value=1)

manual_data = pd.DataFrame({
    'Type': ['L']*num_rows,
    'Air_temperature__K_': [300.0]*num_rows,
    'Process_temperature__K_': [310.0]*num_rows,
    'Rotational_speed__rpm_': [1500]*num_rows,
    'Torque__Nm_': [40.0]*num_rows,
    'Tool_wear__min_': [100]*num_rows
})

manual_data = st.experimental_data_editor(manual_data)

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
# 4. Train LightGBM
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
    if manual_data is not None and not manual_data.empty:
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
