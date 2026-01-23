# =========================================================
# Refactored AI Predictive Maintenance for Induction Motors
# =========================================================

import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import pickle

# ---------------- Configuration ----------------
class Config:
    """Centralized configuration for the application"""
    # File paths
    DEFAULT_DATA_PATH = "ai4i2020.csv"
    MODEL_CACHE_PATH = "model_cache.pkl"
    
    # Model parameters
    MODEL_PARAMS = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42
    }
    
    # Features
    FEATURES = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_"
    ]
    
    # Feature display names (clean names for UI)
    FEATURE_DISPLAY_NAMES = {
        "Type": "Type",
        "Air_temperature__K_": "Air Temperature (K)",
        "Process_temperature__K_": "Process Temperature (K)",
        "Rotational_speed__rpm_": "Rotational Speed (RPM)",
        "Torque__Nm_": "Torque (Nm)",
        "Tool_wear__min_": "Tool Wear (min)"
    }
    
    # Thresholds
    FAILURE_THRESHOLD_HIGH = 0.6  # Failure likely
    FAILURE_THRESHOLD_MEDIUM = 0.25  # Degrading condition
    
    CRITICAL_LIMITS = {
        "process_temp": 400.0,  # K
        "air_temp": 360.0,  # K
        "rpm": 1800.0,
        "torque": 70.0  # Nm
    }
    
    # Physical constraints for validation
    VALID_RANGES = {
        "torque": (0.0, 200.0),
        "rpm": (0.0, 5000.0),
        "tool_wear": (0.0, 500.0),
        "air_temp": (250.0, 400.0),
        "process_temp": (250.0, 450.0)
    }
    
    # CSV upload limits
    MAX_CSV_SIZE_MB = 10
    MAX_CSV_ROWS = 10000

# Ideal motor profile
INDUCTION_MOTOR_PROFILE = {
    "rpm": 1450.0,
    "torque": 35.0,
    "tool_wear": 20.0,
    "air_temp": 300.0,
    "process_temp": 310.0
}

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="⚡",
    layout="wide"
)

# ---------------- Validation Functions ----------------
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_input_values(torque: float, rpm: float, tool_wear: float, 
                         air_temp: float, process_temp: float) -> None:
    """
    Validate input values against physical constraints.
    
    Args:
        torque: Torque value in Nm
        rpm: Rotational speed in RPM
        tool_wear: Tool wear in minutes
        air_temp: Air temperature in Kelvin
        process_temp: Process temperature in Kelvin
        
    Raises:
        ValidationError: If any input is outside valid range
    """
    validations = {
        "Torque": (torque, Config.VALID_RANGES["torque"]),
        "RPM": (rpm, Config.VALID_RANGES["rpm"]),
        "Tool Wear": (tool_wear, Config.VALID_RANGES["tool_wear"]),
        "Air Temperature": (air_temp, Config.VALID_RANGES["air_temp"]),
        "Process Temperature": (process_temp, Config.VALID_RANGES["process_temp"])
    }
    
    for name, (value, (min_val, max_val)) in validations.items():
        if not min_val <= value <= max_val:
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}. Got: {value}"
            )
    
    # Additional logical validation
    if process_temp < air_temp:
        raise ValidationError(
            f"Process temperature ({process_temp}K) cannot be lower than air temperature ({air_temp}K)"
        )

def validate_csv_upload(uploaded_file) -> pd.DataFrame:
    """
    Validate and load uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError: If CSV is invalid
    """
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > Config.MAX_CSV_SIZE_MB:
        raise ValidationError(
            f"File size ({file_size_mb:.1f}MB) exceeds limit of {Config.MAX_CSV_SIZE_MB}MB"
        )
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV file: {str(e)}")
    
    # Check row count
    if len(df) > Config.MAX_CSV_ROWS:
        raise ValidationError(
            f"CSV has {len(df)} rows, exceeds limit of {Config.MAX_CSV_ROWS}"
        )
    
    # Check required columns
    missing_cols = set(Config.FEATURES) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Validate data types and ranges
    for col in Config.FEATURES:
        if col == "Type":
            if not df[col].isin(['L', 'M', 'H']).all():
                raise ValidationError(f"Column '{col}' contains invalid values. Must be L, M, or H")
        else:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column '{col}' must contain numeric values")
            
            # Check for reasonable ranges (loose check for batch data)
            if df[col].min() < 0:
                raise ValidationError(f"Column '{col}' contains negative values")
    
    return df

# ---------------- Data Loading ----------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    
    Returns:
        Preprocessed DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not os.path.exists(Config.DEFAULT_DATA_PATH):
        raise FileNotFoundError(
            f"Data file '{Config.DEFAULT_DATA_PATH}' not found. "
            f"Please ensure the file exists in the application directory."
        )
    
    try:
        df = pd.read_csv(Config.DEFAULT_DATA_PATH)
        df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

# ---------------- Model Training ----------------
@st.cache_data
def train_model() -> Tuple[lgb.LGBMClassifier, LabelEncoder, float, shap.TreeExplainer]:
    """
    Train the predictive maintenance model.
    
    Returns:
        Tuple of (model, encoder, auc_score, explainer)
    """
    df = load_data()
    
    X = df[Config.FEATURES].copy()
    y = df["Machine_failure"]
    
    le = LabelEncoder()
    X["Type"] = le.fit_transform(X["Type"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = lgb.LGBMClassifier(**Config.MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    explainer = shap.TreeExplainer(model)
    
    return model, le, auc, explainer

# ---------------- Prediction Functions ----------------
def preprocess_input(torque: float, rpm: float, tool_wear: float,
                    air_temp: float, process_temp: float,
                    encoder: LabelEncoder, motor_type: str = "M") -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        torque: Torque in Nm
        rpm: Rotational speed in RPM
        tool_wear: Tool wear in minutes
        air_temp: Air temperature in K
        process_temp: Process temperature in K
        encoder: Fitted label encoder
        motor_type: Motor type (L, M, or H)
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    input_df = pd.DataFrame([{
        "Type": motor_type,
        "Air_temperature__K_": air_temp,
        "Process_temperature__K_": process_temp,
        "Rotational_speed__rpm_": rpm,
        "Torque__Nm_": torque,
        "Tool_wear__min_": tool_wear
    }])
    
    input_df["Type"] = encoder.transform(input_df["Type"].astype(str))
    return input_df

def predict_failure(input_df: pd.DataFrame, model: lgb.LGBMClassifier) -> float:
    """
    Predict failure probability.
    
    Args:
        input_df: Preprocessed input DataFrame
        model: Trained model
        
    Returns:
        Failure probability (0-1)
    """
    return model.predict_proba(input_df[Config.FEATURES])[0][1]

def get_failure_status(probability: float) -> Tuple[str, str]:
    """
    Get failure status based on probability.
    
    Args:
        probability: Failure probability (0-1)
        
    Returns:
        Tuple of (status_emoji, status_text)
    """
    if probability >= Config.FAILURE_THRESHOLD_HIGH:
        return "🔴", "Failure Likely"
    elif probability >= Config.FAILURE_THRESHOLD_MEDIUM:
        return "🟡", "Degrading Condition"
    else:
        return "🟢", "Normal Operation"

def get_critical_alerts(torque: float, rpm: float, air_temp: float, 
                       process_temp: float) -> list:
    """
    Check for critical operating conditions.
    
    Returns:
        List of alert messages
    """
    alerts = []
    
    if process_temp > Config.CRITICAL_LIMITS["process_temp"]:
        alerts.append("⚠️ Process temperature extremely high! Risk of severe thermal damage.")
    if air_temp > Config.CRITICAL_LIMITS["air_temp"]:
        alerts.append("⚠️ Air temperature too high! Cooling efficiency compromised.")
    if rpm > Config.CRITICAL_LIMITS["rpm"]:
        alerts.append("⚠️ Motor overspeed! Bearing & rotor stress likely.")
    if torque > Config.CRITICAL_LIMITS["torque"]:
        alerts.append("⚠️ Excessive torque! Mechanical overload possible.")
    
    return alerts

def get_diagnosis(shap_values, features: list) -> str:
    """
    Get failure diagnosis based on SHAP values.
    
    Args:
        shap_values: SHAP values array
        features: List of feature names
        
    Returns:
        Diagnosis message
    """
    shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
    impact = pd.Series(shap_array[0], index=features).abs().sort_values(ascending=False)
    main_factor = impact.index[0].lower()
    
    diagnoses = {
        "rpm": "High RPM can increase thermal & dynamic stress, accelerating wear and vibration fatigue.",
        "torque": "High torque places mechanical load on drivetrain, increasing component stress & failure risk.",
        "wear": "Excessive tool wear causes friction, poor cutting, and heat generation.",
    }
    
    for key, message in diagnoses.items():
        if key in main_factor:
            return message
    
    return "Failure risk is driven by combined thermal & mechanical loading conditions."

def get_maintenance_recommendation(probability: float) -> str:
    """
    Get maintenance recommendation based on failure probability.
    
    Args:
        probability: Failure probability (0-1)
        
    Returns:
        Recommendation message
    """
    if probability > Config.FAILURE_THRESHOLD_HIGH:
        return "🔧 Immediate inspection & preventive maintenance required."
    elif probability > Config.FAILURE_THRESHOLD_MEDIUM:
        return "⚙️ Schedule routine maintenance soon."
    else:
        return "✅ Motor operating normally. Continue standard monitoring."

def simulate_torque_impact(base_rpm: float, current_torque: float, 
                          new_torque: float) -> float:
    """
    Simulate RPM change due to torque increase (inverse relationship).
    
    Args:
        base_rpm: Base rotational speed
        current_torque: Current torque value
        new_torque: New torque value
        
    Returns:
        Simulated RPM
    """
    return max(0.0, base_rpm - (new_torque - current_torque) * 10)

# ---------------- Visualization ----------------
def create_parameters_chart(rpm: float, torque: float, tool_wear: float,
                           sim_rpm: float, air_temp: float, 
                           process_temp: float) -> go.Figure:
    """
    Create interactive bar chart of motor parameters.
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    parameters = [
        ("RPM", rpm, 'royalblue'),
        ("Torque", torque, 'firebrick'),
        ("Tool Wear", tool_wear, 'darkgreen'),
        ("Simulated RPM", sim_rpm, 'orange'),
        ("Air Temp", air_temp, 'skyblue'),
        ("Process Temp", process_temp, 'crimson')
    ]
    
    for name, value, color in parameters:
        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[value],
            marker_color=color,
            text=[f"{value:.1f}"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Motor Operating Parameters & Simulation",
        barmode='group',
        height=500,
        template='plotly_white',
        yaxis_title="Value",
        xaxis_title="Parameter"
    )
    
    return fig

# ---------------- Batch Processing ----------------
def process_batch_predictions(df: pd.DataFrame, model: lgb.LGBMClassifier,
                             encoder: LabelEncoder) -> pd.DataFrame:
    """
    Process batch predictions from CSV.
    
    Args:
        df: Input DataFrame
        model: Trained model
        encoder: Label encoder
        
    Returns:
        DataFrame with predictions
    """
    df_copy = df.copy()
    df_copy["Type"] = encoder.transform(df_copy["Type"].astype(str))
    
    batch_probs = model.predict_proba(df_copy[Config.FEATURES])[:, 1]
    df["Failure_Probability"] = batch_probs
    df["Failure_Status"] = df["Failure_Probability"].apply(
        lambda x: "🔴 Failure Likely" if x > Config.FAILURE_THRESHOLD_HIGH else
                  "🟡 Degrading" if x > Config.FAILURE_THRESHOLD_MEDIUM else
                  "🟢 Normal"
    )
    
    return df

# ============================================================
# Main Application
# ============================================================

# Initialize app
try:
    model, encoder, auc_score, explainer = train_model()
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.info("Please upload the dataset file 'ai4i2020.csv' to the application directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error initializing application: {str(e)}")
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.title("⚡ Predictive Maintenance AI")
menu = st.sidebar.radio("Navigation", ["Home", "Manual Prediction", "Model Info"], key="nav_menu")

# ---------------- Home Page ----------------
if menu == "Home":
    st.title("⚡ AI-Driven Predictive Maintenance (Induction Motors)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Project Capabilities**
        - Failure probability prediction with validated inputs
        - Torque-RPM inverse relation modeling
        - What-if load simulation
        - Maintenance recommendations & motor health score
        - SHAP explainability for diagnosis
        - Batch CSV processing with safety checks
        """)
        st.markdown("💡 Demo Scenarios: Use manual inputs to see real-time predictions and recommendations.")
    with col2:
        st.metric("Model ROC-AUC", f"{auc_score:.3f}")
    
    st.divider()

# ---------------- Manual Prediction ----------------
elif menu == "Manual Prediction":
    st.title("📊 Manual Prediction & What-If Simulation")
    st.subheader("Induction Motor Inputs")
    
    # CSV Upload Section
    uploaded_file = st.file_uploader(
        "📂 Upload CSV for batch prediction (Max 10MB, 10,000 rows)",
        type=["csv"]
    )
    
    # Manual Input Section
    col1, col2, col3 = st.columns(3)
    
    torque = col1.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=float(INDUCTION_MOTOR_PROFILE["torque"]),
        step=1.0,
        help="Valid range: 0-200 Nm"
    )
    
    rpm_default = max(0.0, INDUCTION_MOTOR_PROFILE["rpm"] - 
                     (torque - INDUCTION_MOTOR_PROFILE["torque"]) * 10)
    rpm = col2.number_input(
        "Rotational Speed (RPM)",
        min_value=0.0,
        max_value=5000.0,
        value=float(rpm_default),
        step=10.0,
        help="Valid range: 0-5000 RPM"
    )
    
    tool_wear = col3.number_input(
        "Tool Wear (min)",
        min_value=0.0,
        max_value=500.0,
        value=float(INDUCTION_MOTOR_PROFILE["tool_wear"]),
        step=1.0,
        help="Valid range: 0-500 minutes"
    )
    
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=float(INDUCTION_MOTOR_PROFILE["air_temp"]),
        step=1.0,
        help="Valid range: 250-400 K"
    )
    
    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=450.0,
        value=float(INDUCTION_MOTOR_PROFILE["process_temp"]),
        step=1.0,
        help="Valid range: 250-450 K (must be >= Air Temperature)"
    )
    
    # Prediction Button
    if st.button("🔍 Predict Failure"):
        
        # ================= CSV BATCH PREDICTION =================
        if uploaded_file is not None:
            try:
                batch_df = validate_csv_upload(uploaded_file)
                result_df = process_batch_predictions(batch_df, model, encoder)
                
                st.subheader("📂 Batch Prediction Results")
                st.dataframe(result_df)
                
                st.download_button(
                    "⬇️ Download Results",
                    result_df.to_csv(index=False),
                    file_name="predictive_maintenance_results.csv",
                    mime="text/csv"
                )
                st.success(f"✅ Batch prediction completed for {len(result_df)} records")
                
            except ValidationError as e:
                st.error(f"❌ Validation Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error processing CSV: {str(e)}")
            
            st.stop()
        
        # ================= MANUAL PREDICTION =================
        try:
            # Validate inputs
            validate_input_values(torque, rpm, tool_wear, air_temp, process_temp)
            
            # Preprocess and predict
            input_df = preprocess_input(torque, rpm, tool_wear, air_temp, 
                                       process_temp, encoder)
            prob = predict_failure(input_df, model)
            
            # Display Results
            st.subheader("⚡ Prediction Outcome")
            st.metric("Failure Probability", f"{prob*100:.2f}%")
            
            emoji, status = get_failure_status(prob)
            st.subheader(f"{emoji} {status}")
            
            # Diagnosis
            st.subheader("🧠 Failure Diagnosis")
            shap_vals = explainer.shap_values(input_df[Config.FEATURES])
            diagnosis = get_diagnosis(shap_vals, Config.FEATURES)
            st.info(diagnosis)
            
            # Critical Alerts
            alerts = get_critical_alerts(torque, rpm, air_temp, process_temp)
            if alerts:
                st.subheader("🚨 Rule-Based Safety Alerts")
                st.error("Critical Operating Condition Detected")
                for msg in alerts:
                    st.write(msg)
            
            # What-If Simulation
            st.subheader("⚡ What-If Load Simulation")
            sim_torque = st.slider(
                "Simulate Torque Increase",
                0.0, 200.0,
                float(torque),
                1.0
            )
            sim_rpm = simulate_torque_impact(rpm, torque, sim_torque)
            st.metric("Simulated RPM due to Torque change", f"{sim_rpm:.2f}")
            
            # Visualization
            st.subheader("📈 Motor Operating Parameters Overview")
            fig = create_parameters_chart(rpm, torque, tool_wear, sim_rpm, 
                                         air_temp, process_temp)
            st.plotly_chart(fig, use_container_width=True)
            
            # Maintenance Recommendation
            st.subheader("🛠 Maintenance Recommendations")
            recommendation = get_maintenance_recommendation(prob)
            if prob > Config.FAILURE_THRESHOLD_HIGH:
                st.error(recommendation)
            elif prob > Config.FAILURE_THRESHOLD_MEDIUM:
                st.warning(recommendation)
            else:
                st.success(recommendation)
            
            # Health Score
            health_score = max(0, 100 - prob * 100)
            st.subheader("💚 Motor Health Score")
            st.progress(int(health_score))
            st.caption(f"Health Score: {health_score:.1f}/100")
            
        except ValidationError as e:
            st.error(f"❌ Validation Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# ---------------- Model Info ----------------
elif menu == "Model Info":
    st.title("📚 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Model Details
        **Algorithm:** LightGBM Classifier  
        **Explainability:** SHAP (SHapley Additive exPlanations)  
        **Dataset:** AI4I 2020 Predictive Maintenance  
        **Focus:** Induction motors only  
        
        **Performance**  
        - ROC-AUC Score: {auc_score:.3f}
        - Training/Test Split: 80/20
        - Stratified sampling
        """)
    
    with col2:
        st.markdown("""
        ### Features Used
        - Motor Type (L/M/H)
        - Air Temperature (K)
        - Process Temperature (K)
        - Rotational Speed (RPM)
        - Torque (Nm)
        - Tool Wear (minutes)
        
        ### Safety Features
        - Input validation
        - CSV size limits
        - Critical threshold alerts
        - Physical constraint checks
        """)
    
    st.divider()
    
    st.markdown("""
    ### Failure Thresholds
    - **High Risk (Red):** ≥ 60% probability
    - **Medium Risk (Yellow):** 25-60% probability
    - **Low Risk (Green):** < 25% probability
    
    ### Critical Operating Limits
    - Process Temperature: > 400 K
    - Air Temperature: > 360 K
    - Rotational Speed: > 1800 RPM
    - Torque: > 70 Nm
    """)
