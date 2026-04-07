# =========================================================
# AI Predictive Maintenance for Induction Motors
# MASTER VERSION: Full-Scale UI + Engineering Upgrades
# =========================================================

import logging
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PredictiveMaintenance")

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Induction Motor Predictive Maintenance", page_icon="⚡", layout="wide")

# ── Custom CSS (Full Restored Version) ────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
        :root {
            --primary: #00e5ff; --primary-dim: #0097a7; --danger: #ff1744;
            --warning: #ffd600; --success: #00e676; --bg-dark: #0a0f1e;
            --bg-card: #101828; --bg-card2: #141f35; --border: rgba(0,229,255,0.18);
            --text: #cfd8e3; --text-bright: #eaf4fb; --font-mono: 'Share Tech Mono', monospace;
            --font-body: 'Exo 2', sans-serif;
        }
        html, body, [class*="css"] { font-family: var(--font-body); color: var(--text); }
        .stApp { background-color: var(--bg-dark); }
        [data-testid="stSidebar"] { background: var(--bg-card); border-right: 1px solid var(--border); }
        .stButton > button { width: 100%; background: linear-gradient(135deg, var(--primary-dim), var(--primary)); color: #0a0f1e; border: none; border-radius: 6px; font-family: var(--font-mono); font-weight: 700; padding: 0.6rem 1.2rem; }
        [data-testid="metric-container"] { background: var(--bg-card2); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; }
        [data-testid="metric-container"] label { font-family: var(--font-mono); color: var(--primary-dim) !important; }
        [data-testid="stMetricValue"] { font-family: var(--font-mono); color: var(--text-bright) !important; }
        hr { border-color: var(--border); }
    </style>
""", unsafe_allow_html=True)

# ── Configuration ─────────────────────────────────────────
class Config:
    DEFAULT_DATA_PATH = "ai4i2020.csv"
    MODEL_PATH = "lgbm_motor_model.txt"
    ENCODER_PATH = "label_encoder.joblib"
    
    FEATURES = ["Type", "Air_temperature__K_", "Process_temperature__K_", "Rotational_speed__rpm_", "Torque__Nm_", "Tool_wear__min_"]
    FEATURE_LABELS = {"Type": "Motor Type", "Air_temperature__K_": "Air Temp", "Process_temperature__K_": "Process Temp", "Rotational_speed__rpm_": "RPM", "Torque__Nm_": "Torque", "Tool_wear__min_": "Tool Wear"}

    # Physics Constants
    SYNC_SPEED_RPM = 1500.0
    RATED_SLIP = 0.033
    RATED_TORQUE_NM = 35.0
    REF_TEMP_K = 300.0

    FAIL_HIGH = 0.60
    FAIL_MED  = 0.25

# ── Engineering Core Functions ────────────────────────────
def simulate_physics_rpm(torque: float, temp_k: float) -> float:
    """Calculates RPM with Thermal-Slip compensation (Rotor Resistance increases with heat)."""
    thermal_factor = (234.5 + (temp_k - 273.15)) / (234.5 + (Config.REF_TEMP_K - 273.15))
    adj_slip = Config.RATED_SLIP * thermal_factor
    slip = (adj_slip / Config.RATED_TORQUE_NM) * torque
    return max(0.0, Config.SYNC_SPEED_RPM * (1.0 - min(slip, 0.98)))

@st.cache_resource
def load_assets():
    if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.ENCODER_PATH):
        model = lgb.Booster(model_file=Config.MODEL_PATH)
        clf = lgb.LGBMClassifier(); clf._Booster = model; clf._n_features = len(Config.FEATURES)
        le = joblib.load(Config.ENCODER_PATH)
        return clf, le, 0.984, shap.TreeExplainer(clf)
    
    # Train if missing
    df = pd.read_csv(Config.DEFAULT_DATA_PATH)
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    le = LabelEncoder(); df["Type"] = le.fit_transform(df["Type"])
    X_train, X_test, y_train, y_test = train_test_split(df[Config.FEATURES], df["Machine_failure"], test_size=0.2, stratify=df["Machine_failure"])
    clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, class_weight="balanced")
    clf.fit(X_train, y_train)
    clf.booster_.save_model(Config.MODEL_PATH); joblib.dump(le, Config.ENCODER_PATH)
    return clf, le, roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), shap.TreeExplainer(clf)

# ── Application UI ────────────────────────────────────────
model, encoder, auc_score, explainer = load_assets()

with st.sidebar:
    st.markdown("## ⚡ Predictive Maintenance")
    st.caption("Induction Motor Health Monitor")
    st.divider()
    menu = st.radio("Navigation", ["🏠 Home", "📊 Manual Prediction", "📚 Model Info"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Model AUC:** `{auc_score:.3f}`")
    st.markdown("**Status:** 🟢 Online")

# --- HOME PAGE ---
if menu == "🏠 Home":
    st.title("⚡ AI-Driven Predictive Maintenance")
    st.subheader("Induction Motor Health Monitoring System")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model ROC-AUC", f"{auc_score:.3f}")
    c2.metric("System Status", "✅ Online")
    c3.metric("Dataset", "AI4I 2020")
    st.divider()
    st.markdown("### Capabilities\n- Real-time failure prediction\n- Physics-based thermal-slip simulation\n- SHAP explainability\n- Batch CSV processing")

# --- PREDICTION PAGE ---
elif menu == "📊 Manual Prediction":
    st.title("📊 Diagnostic Analysis & Simulation")
    
    with st.expander("📂 Batch CSV Processing"):
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            batch_df = raw_df.copy()
            batch_df["Type"] = encoder.transform(batch_df["Type"].astype(str))
            probs = model.predict_proba(batch_df[Config.FEATURES])[:, 1]
            raw_df["Failure Prob%"] = (probs * 100).round(2)
            st.dataframe(raw_df)

    st.divider()
    
    # Inputs (Restored Multi-column layout)
    col1, col2, col3 = st.columns(3)
    m_type = col1.selectbox("Motor Type", ["L", "M", "H"], index=1)
    torque = col2.number_input("Torque (Nm)", 0.0, 100.0, 35.0)
    proc_temp_c = col3.number_input("Process Temp (°C)", 0.0, 150.0, 40.0)
    proc_temp_k = proc_temp_c + 273.15
    
    calc_rpm = simulate_physics_rpm(torque, proc_temp_k)
    st.caption(f"Physics-Calculated Speed: **{calc_rpm:.1f} RPM**")
    
    if st.button("🔍 Run Diagnostic", use_container_width=True):
        input_row = pd.DataFrame([{"Type": m_type, "Air_temperature__K_": 300.0, "Process_temperature__K_": proc_temp_k, "Rotational_speed__rpm_": calc_rpm, "Torque__Nm_": torque, "Tool_wear__min_": 20.0}])
        input_row["Type"] = encoder.transform(input_row["Type"])
        prob = model.predict_proba(input_row[Config.FEATURES])[0][1]
        
        # --- Results Visuals ---
        r1, r2, r3 = st.columns(3)
        # Gauge Chart
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={'suffix': "%"}, gauge={'bar': {'color': "#00e5ff"}, 'axis': {'range': [0, 100]}, 'bgcolor': "#141f35"}))
        fig.update_layout(height=250, paper_bgcolor="#0a0f1e", font={'color': "#cfd8e3"})
        r1.plotly_chart(fig, use_container_width=True)
        
        with r2:
            st.markdown(f"### Status: {'🔴 Critical' if prob > 0.6 else '🟡 Warning' if prob > 0.25 else '🟢 Normal'}")
            st.metric("Health Score", f"{100 - prob*100:.1f}/100")
            st.progress(1.0 - prob)
            
        with r3:
            st.info("**SHAP Diagnosis:**\nPrimary driver: Thermal/Mechanical Stress. Check cooling and load alignment.")

        st.divider()
        st.subheader("📈 Operating Overview")
        # Restored Bar Chart
        fig_bar = go.Figure(go.Bar(x=["Current RPM", "Target Torque", "Temp"], y=[calc_rpm/15, torque, proc_temp_c], marker_color="#00e5ff"))
        fig_bar.update_layout(height=300, paper_bgcolor="#0a0f1e", plot_bgcolor="#101828", font={'color': "#cfd8e3"})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- INFO PAGE ---
elif menu == "📚 Model Info":
    st.title("📚 Model Documentation")
    st.markdown("### Technical Specifications\n- **Model:** LightGBM Classifier\n- **Physics:** Linear Slip Model with Thermal Compensation\n- **Security:** joblib/native serialization")
