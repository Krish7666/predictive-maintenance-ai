# =========================================================
# AI Predictive Maintenance for Induction Motors
# Refactored & Improved — Production Grade
# =========================================================

import logging
import os
import pickle
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("PredictiveMaintenance")

# ── Page Config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="Induction Motor Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Import distinctive fonts */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

        /* Root theme */
        :root {
            --primary: #00e5ff;
            --primary-dim: #0097a7;
            --danger: #ff1744;
            --warning: #ffd600;
            --success: #00e676;
            --bg-dark: #0a0f1e;
            --bg-card: #101828;
            --bg-card2: #141f35;
            --border: rgba(0,229,255,0.18);
            --text: #cfd8e3;
            --text-bright: #eaf4fb;
            --font-mono: 'Share Tech Mono', monospace;
            --font-body: 'Exo 2', sans-serif;
        }

        /* App-wide */
        html, body, [class*="css"] {
            font-family: var(--font-body);
            color: var(--text);
        }
        .stApp { background-color: var(--bg-dark); }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-card);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .stRadio label {
            font-family: var(--font-mono);
            letter-spacing: 0.05em;
            color: var(--text-bright);
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, var(--primary-dim), var(--primary));
            color: #0a0f1e;
            border: none;
            border-radius: 6px;
            font-family: var(--font-mono);
            font-weight: 700;
            letter-spacing: 0.08em;
            padding: 0.6rem 1.2rem;
            transition: opacity 0.2s ease, transform 0.1s ease;
        }
        .stButton > button:hover  { opacity: 0.88; transform: translateY(-1px); }
        .stButton > button:active { transform: translateY(0); }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }
        [data-testid="metric-container"] label {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            color: var(--primary-dim) !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-family: var(--font-mono);
            color: var(--text-bright) !important;
        }

        /* Inputs */
        .stNumberInput input, .stSlider {
            font-family: var(--font-mono);
        }

        /* Divider */
        hr { border-color: var(--border); }

        /* Info / success / warning / error banners */
        .stAlert { border-radius: 8px; font-family: var(--font-body); }

        /* Section headers */
        h1, h2, h3 {
            font-family: var(--font-body);
            font-weight: 700;
            color: var(--text-bright);
        }
        h1 { letter-spacing: -0.01em; }

        /* Dataframe */
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

        /* Progress bar */
        .stProgress > div > div { background-color: var(--primary); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Configuration ─────────────────────────────────────────
class Config:
    """Single source of truth for all app constants."""

    # Paths
    DEFAULT_DATA_PATH = "ai4i2020.csv"
    MODEL_CACHE_PATH = "model_cache.pkl"

    # LightGBM parameters
    MODEL_PARAMS: dict = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "class_weight": "balanced",   # handles class imbalance automatically
        "random_state": 42,
        "verbose": -1,
    }

    # Feature columns (must match CSV after sanitisation)
    FEATURES: list = [
        "Type",
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_",
    ]

    # Feature → human-readable label mapping for diagnosis
    FEATURE_LABELS: dict = {
        "Type":                      "motor type",
        "Air_temperature__K_":       "air temperature",
        "Process_temperature__K_":   "process temperature",
        "Rotational_speed__rpm_":    "rotational speed (RPM)",
        "Torque__Nm_":               "torque",
        "Tool_wear__min_":           "tool wear",
    }

    # Failure thresholds
    FAILURE_THRESHOLD_HIGH: float = 0.60
    FAILURE_THRESHOLD_MEDIUM: float = 0.25

    # Rule-based safety limits
    CRITICAL_LIMITS: dict = {
        "process_temp": 400.0,   # K
        "air_temp":     360.0,   # K
        "rpm":         1800.0,
        "torque":        70.0,   # Nm
    }

    # Physical valid ranges for manual input validation
    VALID_RANGES: dict = {
        "torque":       (0.0,  200.0),
        "rpm":          (0.0, 5000.0),
        "tool_wear":    (0.0,  500.0),
        "air_temp":   (250.0,  400.0),
        "process_temp":(250.0, 450.0),
    }

    # CSV upload limits
    MAX_CSV_SIZE_MB: int  = 10
    MAX_CSV_ROWS:    int  = 10_000

    # Induction motor — rated / ideal operating point
    MOTOR_RATED: dict = {
        "rpm":          1450.0,
        "torque":         35.0,   # Nm
        "tool_wear":      20.0,   # min
        "air_temp":      300.0,   # K
        "process_temp":  310.0,   # K
    }

    # Induction motor physical constants used for slip-based simulation
    # Synchronous speed for a 4-pole motor at 50 Hz = 1500 RPM
    SYNC_SPEED_RPM:   float = 1500.0
    RATED_SLIP:       float = 0.033        # (1500-1450)/1500
    RATED_TORQUE_NM:  float = 35.0        # Nm at rated slip


# ── Custom Exception ───────────────────────────────────────
class ValidationError(Exception):
    """Raised when user-supplied data fails a validation check."""


# ── Validation Helpers ────────────────────────────────────
def validate_input_values(
    torque: float,
    rpm: float,
    tool_wear: float,
    air_temp: float,
    process_temp: float,
) -> None:
    """
    Validate manual inputs against physical constraints.
    Raises ValidationError with a descriptive message on failure.
    """
    checks = {
        "Torque (Nm)":              (torque,       Config.VALID_RANGES["torque"]),
        "Rotational Speed (RPM)":   (rpm,          Config.VALID_RANGES["rpm"]),
        "Tool Wear (min)":          (tool_wear,    Config.VALID_RANGES["tool_wear"]),
        "Air Temperature (K)":      (air_temp,     Config.VALID_RANGES["air_temp"]),
        "Process Temperature (K)":  (process_temp, Config.VALID_RANGES["process_temp"]),
    }

    for label, (value, (lo, hi)) in checks.items():
        if not lo <= value <= hi:
            raise ValidationError(
                f"{label} must be between {lo} and {hi}. Received: {value}"
            )

    if process_temp < air_temp:
        raise ValidationError(
            f"Process temperature ({process_temp} K) cannot be lower than "
            f"air temperature ({air_temp} K) — thermodynamic constraint violated."
        )


def validate_csv_upload(uploaded_file) -> pd.DataFrame:
    """
    Validate an uploaded CSV file and return a clean DataFrame.
    Raises ValidationError on any issue.
    """
    # Size check
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > Config.MAX_CSV_SIZE_MB:
        raise ValidationError(
            f"File size {size_mb:.1f} MB exceeds the {Config.MAX_CSV_SIZE_MB} MB limit."
        )

    # Parse
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValidationError(f"Could not parse CSV: {exc}") from exc

    # Row count
    if len(df) > Config.MAX_CSV_ROWS:
        raise ValidationError(
            f"CSV contains {len(df):,} rows — limit is {Config.MAX_CSV_ROWS:,}."
        )

    # Column presence
    missing = set(Config.FEATURES) - set(df.columns)
    if missing:
        raise ValidationError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    # Per-column type/value checks
    for col in Config.FEATURES:
        if col == "Type":
            if not df[col].isin(["L", "M", "H"]).all():
                bad = df.loc[~df[col].isin(["L", "M", "H"]), col].unique().tolist()
                raise ValidationError(
                    f"Column 'Type' contains invalid values {bad}. Allowed: L, M, H."
                )
        else:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column '{col}' must be numeric.")
            if df[col].isnull().any():
                raise ValidationError(f"Column '{col}' contains missing values.")
            if (df[col] < 0).any():
                raise ValidationError(f"Column '{col}' contains negative values.")

    return df


# ── Data Loading ──────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load the AI4I 2020 dataset and sanitise column names."""
    if not os.path.exists(Config.DEFAULT_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset '{Config.DEFAULT_DATA_PATH}' not found in the working directory."
        )
    try:
        df = pd.read_csv(Config.DEFAULT_DATA_PATH)
        df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        logger.info("Dataset loaded: %d rows, %d cols", *df.shape)
        return df
    except Exception as exc:
        raise RuntimeError(f"Error reading dataset: {exc}") from exc


# ── Model — Training & Caching ────────────────────────────
def _train_fresh(
    df: pd.DataFrame,
) -> Tuple[lgb.LGBMClassifier, LabelEncoder, float, shap.TreeExplainer]:
    """Train LightGBM classifier from scratch and return all artefacts."""
    X = df[Config.FEATURES].copy()
    y = df["Machine_failure"]

    le = LabelEncoder()
    X["Type"] = le.fit_transform(X["Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = lgb.LGBMClassifier(**Config.MODEL_PARAMS)
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    explainer = shap.TreeExplainer(clf)

    logger.info("Model trained — ROC-AUC: %.4f", auc)
    return clf, le, auc, explainer


@st.cache_resource(show_spinner="Initialising model…")
def get_model() -> Tuple[lgb.LGBMClassifier, LabelEncoder, float, shap.TreeExplainer]:
    """
    Return trained model artefacts.
    Loads from disk cache when available; trains and caches otherwise.
    Uses st.cache_resource so artefacts are shared across sessions without
    re-training on every Streamlit rerun.
    """
    cache_path = Config.MODEL_CACHE_PATH

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                artefacts = pickle.load(fh)
            logger.info("Model loaded from cache: %s", cache_path)
            return artefacts
        except Exception as exc:
            logger.warning("Cache load failed (%s) — retraining.", exc)

    df = load_data()
    artefacts = _train_fresh(df)

    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(artefacts, fh)
        logger.info("Model cached to: %s", cache_path)
    except Exception as exc:
        logger.warning("Could not write model cache: %s", exc)

    return artefacts


# ── Preprocessing & Prediction ───────────────────────────
def preprocess_single(
    torque: float,
    rpm: float,
    tool_wear: float,
    air_temp: float,
    process_temp: float,
    motor_type: str,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Build a one-row DataFrame ready for inference."""
    row = {
        "Type":                     motor_type,
        "Air_temperature__K_":      air_temp,
        "Process_temperature__K_":  process_temp,
        "Rotational_speed__rpm_":   rpm,
        "Torque__Nm_":              torque,
        "Tool_wear__min_":          tool_wear,
    }
    df = pd.DataFrame([row])
    df["Type"] = encoder.transform(df["Type"].astype(str))
    return df[Config.FEATURES]


def predict_failure_prob(input_df: pd.DataFrame, clf: lgb.LGBMClassifier) -> float:
    """Return failure probability (class=1) for a preprocessed input row."""
    return float(clf.predict_proba(input_df)[0][1])


def failure_status(prob: float) -> Tuple[str, str]:
    """Map probability to (emoji, label) status."""
    if prob >= Config.FAILURE_THRESHOLD_HIGH:
        return "🔴", "Failure Likely"
    if prob >= Config.FAILURE_THRESHOLD_MEDIUM:
        return "🟡", "Degrading Condition"
    return "🟢", "Normal Operation"


def critical_alerts(
    torque: float, rpm: float, air_temp: float, process_temp: float
) -> list:
    """Return list of critical alert strings for out-of-safe-limit values."""
    lim = Config.CRITICAL_LIMITS
    alerts = []
    if process_temp > lim["process_temp"]:
        alerts.append(
            f"⚠️ Process temperature {process_temp:.1f} K > {lim['process_temp']} K — "
            "severe thermal damage risk."
        )
    if air_temp > lim["air_temp"]:
        alerts.append(
            f"⚠️ Air temperature {air_temp:.1f} K > {lim['air_temp']} K — "
            "cooling efficiency compromised."
        )
    if rpm > lim["rpm"]:
        alerts.append(
            f"⚠️ Speed {rpm:.0f} RPM > {lim['rpm']:.0f} RPM — "
            "bearing & rotor stress likely."
        )
    if torque > lim["torque"]:
        alerts.append(
            f"⚠️ Torque {torque:.1f} Nm > {lim['torque']} Nm — "
            "mechanical overload possible."
        )
    return alerts


# ── SHAP-based Diagnosis ──────────────────────────────────
def get_diagnosis(
    shap_explainer: shap.TreeExplainer,
    input_df: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Compute SHAP values and return (top_feature_label, diagnosis_message).
    Compatible with both old (list) and new (Explanation object) SHAP APIs.
    """
    explanation = shap_explainer(input_df)

    # Robustly extract values regardless of SHAP version
    if hasattr(explanation, "values"):
        vals = explanation.values          # shape: (1, n_features) or (1, n_features, n_classes)
        if vals.ndim == 3:
            vals = vals[:, :, 1]          # take class-1 SHAP values for binary classification
        shap_row = vals[0]
    else:
        # Legacy list output: [shap_class0, shap_class1]
        shap_row = explanation[1][0] if isinstance(explanation, list) else explanation[0]

    impact = pd.Series(np.abs(shap_row), index=Config.FEATURES).sort_values(ascending=False)
    top_feature = impact.index[0]
    top_label   = Config.FEATURE_LABELS.get(top_feature, top_feature)

    # Map feature → domain-specific diagnosis
    diagnoses = {
        "Rotational_speed__rpm_": (
            "High rotational speed elevates centrifugal forces on the rotor and "
            "bearings, accelerates thermal fatigue, and amplifies vibration-induced wear."
        ),
        "Torque__Nm_": (
            "Elevated torque increases mechanical stress on the drivetrain, shaft, and "
            "windings. Sustained overload accelerates insulation breakdown and fatigue cracking."
        ),
        "Tool_wear__min_": (
            "Excessive tool wear increases cutting resistance and heat generation, "
            "raising motor load current and thermal stress beyond design limits."
        ),
        "Process_temperature__K_": (
            "High process temperature degrades winding insulation and lubrication "
            "viscosity, directly shortening bearing and coil life."
        ),
        "Air_temperature__K_": (
            "Elevated ambient temperature reduces the motor's cooling capacity, "
            "causing internal temperatures to approach thermal trip limits sooner."
        ),
        "Type": (
            "The motor load classification is the primary differentiating factor. "
            "Ensure the motor type rating matches the actual duty cycle."
        ),
    }

    message = diagnoses.get(
        top_feature,
        "Failure risk is driven by combined thermal and mechanical loading conditions.",
    )
    return top_label, message


def maintenance_recommendation(prob: float) -> Tuple[str, str]:
    """Return (level, recommendation_text) based on failure probability."""
    if prob > Config.FAILURE_THRESHOLD_HIGH:
        return "error", "🔧 Immediate inspection & preventive maintenance required. Do not defer."
    if prob > Config.FAILURE_THRESHOLD_MEDIUM:
        return "warning", "⚙️ Schedule routine maintenance within the next maintenance window."
    return "success", "✅ Motor operating normally. Continue standard monitoring intervals."


# ── Physics-Based What-If Simulation ─────────────────────
def simulate_slip_rpm(new_torque_nm: float) -> float:
    """
    Estimate steady-state rotor speed under a new torque load using the
    linear region of the induction motor torque-slip characteristic:

        slip ≈ (rated_slip / rated_torque) × new_torque     [for T << T_max]
        rpm  = sync_speed × (1 - slip)

    This is physically grounded — unlike the arbitrary ×10 factor.
    Clamps to [0, sync_speed) and warns the user if outside the linear region.
    """
    # Avoid division by zero guard
    if Config.RATED_TORQUE_NM == 0:
        return Config.SYNC_SPEED_RPM

    slip = (Config.RATED_SLIP / Config.RATED_TORQUE_NM) * new_torque_nm
    slip = min(slip, 0.99)                         # clamp: rotor can't exceed sync speed
    simulated_rpm = Config.SYNC_SPEED_RPM * (1.0 - slip)
    return max(0.0, simulated_rpm)


# ── Batch Processing ──────────────────────────────────────
def process_batch(
    df: pd.DataFrame,
    clf: lgb.LGBMClassifier,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """
    Add failure probability and status columns to a copy of the input DataFrame.
    Never mutates the original.
    """
    result = df.copy()
    encoded = result["Type"].map(
        {orig: enc for orig, enc in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    )
    X = result[Config.FEATURES].copy()
    X["Type"] = encoded

    probs = clf.predict_proba(X[Config.FEATURES])[:, 1]
    result["Failure_Probability_%"] = (probs * 100).round(2)
    result["Status"] = pd.cut(
        probs,
        bins=[-1, Config.FAILURE_THRESHOLD_MEDIUM, Config.FAILURE_THRESHOLD_HIGH, 2],
        labels=["🟢 Normal", "🟡 Degrading", "🔴 Failure Likely"],
    )
    return result


# ── Visualisation ─────────────────────────────────────────
def build_gauge(prob: float) -> go.Figure:
    """Failure probability gauge chart."""
    color = (
        "#ff1744" if prob >= Config.FAILURE_THRESHOLD_HIGH
        else "#ffd600" if prob >= Config.FAILURE_THRESHOLD_MEDIUM
        else "#00e676"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 32, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cfd8e3"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#141f35",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 25],  "color": "#0d2b1a"},
                    {"range": [25, 60], "color": "#2b2000"},
                    {"range": [60, 100],"color": "#3b0010"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.75,
                    "value": round(prob * 100, 1),
                },
            },
            title={"text": "Failure Probability", "font": {"color": "#cfd8e3", "size": 14}},
        )
    )
    fig.update_layout(
        height=260,
        paper_bgcolor="#0a0f1e",
        font_color="#cfd8e3",
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


def build_parameters_chart(
    rpm: float,
    torque: float,
    tool_wear: float,
    sim_rpm: float,
    air_temp: float,
    process_temp: float,
) -> go.Figure:
    """Grouped bar chart comparing current vs simulated parameters."""
    params = ["RPM", "Torque (Nm)", "Tool Wear (min)", "Air Temp (K)", "Process Temp (K)"]
    current_vals = [rpm, torque, tool_wear, air_temp, process_temp]
    sim_vals     = [sim_rpm, torque, tool_wear, air_temp, process_temp]  # only RPM changes

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current",
        x=params,
        y=current_vals,
        marker_color="#00e5ff",
        text=[f"{v:.1f}" for v in current_vals],
        textposition="outside",
        textfont=dict(size=11, color="#cfd8e3"),
    ))
    fig.add_trace(go.Bar(
        name="Simulated",
        x=params,
        y=sim_vals,
        marker_color="#ff6d00",
        text=[f"{v:.1f}" for v in sim_vals],
        textposition="outside",
        textfont=dict(size=11, color="#cfd8e3"),
    ))
    fig.update_layout(
        title=dict(text="Operating Parameters — Current vs Simulated", font_color="#eaf4fb"),
        barmode="group",
        height=420,
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#101828",
        font_color="#cfd8e3",
        legend=dict(bgcolor="#141f35", bordercolor="#1e3a5f", borderwidth=1),
        yaxis=dict(gridcolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def build_shap_bar(
    shap_explainer: shap.TreeExplainer,
    input_df: pd.DataFrame,
) -> go.Figure:
    """Horizontal bar chart of SHAP feature importances."""
    explanation = shap_explainer(input_df)

    if hasattr(explanation, "values"):
        vals = explanation.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        shap_row = vals[0]
    else:
        shap_row = explanation[1][0] if isinstance(explanation, list) else explanation[0]

    labels = [Config.FEATURE_LABELS.get(f, f) for f in Config.FEATURES]
    colors = ["#ff1744" if v > 0 else "#00e676" for v in shap_row]

    sorted_pairs = sorted(zip(shap_row, labels, colors), key=lambda x: abs(x[0]))
    s_vals, s_labels, s_colors = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(s_vals),
        y=list(s_labels),
        orientation="h",
        marker_color=list(s_colors),
        text=[f"{v:+.4f}" for v in s_vals],
        textposition="outside",
        textfont=dict(size=10, color="#cfd8e3"),
    ))
    fig.update_layout(
        title=dict(text="SHAP Feature Impact (red = pushes toward failure)", font_color="#eaf4fb"),
        height=320,
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#101828",
        font_color="#cfd8e3",
        xaxis=dict(title="SHAP Value", gridcolor="#1e3a5f", zeroline=True, zerolinecolor="#00e5ff"),
        yaxis=dict(gridcolor="#1e3a5f"),
        margin=dict(t=50, b=30, l=140, r=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# Application Entry Point
# ═══════════════════════════════════════════════════════════

# Initialise model (cached across reruns)
try:
    model, encoder, auc_score, explainer = get_model()
except FileNotFoundError as exc:
    st.error(f"❌ {exc}")
    st.info("Please place `ai4i2020.csv` in the same directory as this script.")
    st.stop()
except Exception as exc:
    st.error(f"❌ Application failed to initialise: {exc}")
    logger.exception("Fatal startup error")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Predictive Maintenance")
    st.caption("Induction Motor Health Monitor")
    st.divider()
    menu = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Manual Prediction", "📚 Model Info"],
        key="nav_menu",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"**Model AUC:** `{auc_score:.3f}`")
    st.markdown("**Status:** 🟢 Online")


# ── Home ──────────────────────────────────────────────────
if menu == "🏠 Home":
    st.title("⚡ AI-Driven Predictive Maintenance")
    st.subheader("Induction Motor Health Monitoring System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model ROC-AUC",   f"{auc_score:.3f}")
    col2.metric("System Status",   "✅ Online")
    col3.metric("Dataset",         "AI4I 2020")

    st.divider()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Capabilities**

- **Real-time failure prediction** with calibrated probability output
- **Physics-based what-if simulation** using the induction motor slip model
- **SHAP explainability** — understand *why* the model flags a risk
- **Rule-based safety alerts** for critical operating limit violations
- **Batch CSV processing** with full validation and downloadable results
- **Motor health score** with actionable maintenance recommendations
        """)

    with c2:
        st.markdown("""
**Failure Thresholds**

| Risk Level   | Probability     |
|:------------|:---------------|
| 🔴 High risk  | ≥ 60 %          |
| 🟡 Degrading  | 25 – 60 %       |
| 🟢 Normal     | < 25 %          |

**Safety Limits**

| Parameter       | Limit       |
|:---------------|:-----------|
| Process Temp   | > 400 K     |
| Air Temp       | > 360 K     |
| Speed          | > 1800 RPM  |
| Torque         | > 70 Nm     |
        """)

    st.divider()
    st.info(
        "💡 Navigate to **Manual Prediction** to enter motor parameters and get an instant "
        "health assessment, or upload a CSV for bulk analysis."
    )


# ── Manual Prediction ─────────────────────────────────────
elif menu == "📊 Manual Prediction":
    st.title("📊 Manual Prediction & What-If Simulation")

    # ── CSV Batch Upload ──────────────────────────────────
    with st.expander("📂 Upload CSV for batch prediction", expanded=False):
        st.caption(
            f"Accepted columns: {', '.join(Config.FEATURES)} | "
            f"Max {Config.MAX_CSV_SIZE_MB} MB / {Config.MAX_CSV_ROWS:,} rows"
        )
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.subheader("Motor Input Parameters")

    # Motor type selector — was previously hardcoded to "M"
    motor_type = st.selectbox(
        "Motor Type",
        options=["L", "M", "H"],
        index=1,
        help="L = Low load, M = Medium load, H = High load",
    )

    col1, col2, col3 = st.columns(3)

    torque = col1.number_input(
        "Torque (Nm)",
        min_value=0.0, max_value=200.0,
        value=float(Config.MOTOR_RATED["torque"]),
        step=1.0,
        help="Valid: 0 – 200 Nm",
    )

    # Default RPM derived from slip model based on current torque
    rpm_default = simulate_slip_rpm(torque)
    rpm = col2.number_input(
        "Rotational Speed (RPM)",
        min_value=0.0, max_value=5000.0,
        value=round(rpm_default, 1),
        step=10.0,
        help="Valid: 0 – 5000 RPM",
    )

    tool_wear = col3.number_input(
        "Tool Wear (min)",
        min_value=0.0, max_value=500.0,
        value=float(Config.MOTOR_RATED["tool_wear"]),
        step=1.0,
        help="Valid: 0 – 500 minutes",
    )

    col4, col5 = st.columns(2)
    air_temp = col4.number_input(
        "Air Temperature (K)",
        min_value=250.0, max_value=400.0,
        value=float(Config.MOTOR_RATED["air_temp"]),
        step=1.0,
        help="Valid: 250 – 400 K",
    )
    process_temp = col5.number_input(
        "Process Temperature (K)",
        min_value=250.0, max_value=450.0,
        value=float(Config.MOTOR_RATED["process_temp"]),
        step=1.0,
        help="Valid: 250 – 450 K (must be ≥ Air Temperature)",
    )

    st.divider()
    predict_btn = st.button("🔍 Predict Failure", use_container_width=True)

    if predict_btn:

        # ── Batch mode ────────────────────────────────────
        if uploaded_file is not None:
            try:
                batch_df = validate_csv_upload(uploaded_file)
                result_df = process_batch(batch_df, model, encoder)

                st.subheader("📂 Batch Prediction Results")

                n_normal    = (result_df["Status"] == "🟢 Normal").sum()
                n_degrading = (result_df["Status"] == "🟡 Degrading").sum()
                n_failure   = (result_df["Status"] == "🔴 Failure Likely").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Records", len(result_df))
                m2.metric("🟢 Normal",        n_normal)
                m3.metric("🟡 Degrading",     n_degrading)
                m4.metric("🔴 At Risk",        n_failure)

                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=result_df.to_csv(index=False),
                    file_name="pm_batch_results.csv",
                    mime="text/csv",
                )
                st.success(f"✅ Batch prediction complete — {len(result_df):,} records processed.")

            except ValidationError as exc:
                st.error(f"❌ Validation Error: {exc}")
                logger.warning("CSV validation failed: %s", exc)
            except Exception as exc:
                st.error(f"❌ Unexpected error processing CSV: {exc}")
                logger.exception("Batch processing error")

        # ── Manual single prediction ──────────────────────
        else:
            try:
                validate_input_values(torque, rpm, tool_wear, air_temp, process_temp)

                input_df = preprocess_single(
                    torque, rpm, tool_wear, air_temp, process_temp, motor_type, encoder
                )
                prob = predict_failure_prob(input_df, model)

                # ── Result header ─────────────────────────
                st.subheader("⚡ Prediction Outcome")
                emoji, status_label = failure_status(prob)

                r1, r2, r3 = st.columns(3)
                r1.plotly_chart(build_gauge(prob), use_container_width=True)

                with r2:
                    st.markdown(f"### {emoji} {status_label}")
                    health = max(0.0, 100.0 - prob * 100.0)
                    st.metric("Motor Health Score", f"{health:.1f} / 100")
                    st.progress(int(health) / 100)

                with r3:
                    level, rec_text = maintenance_recommendation(prob)
                    if level == "error":
                        st.error(rec_text)
                    elif level == "warning":
                        st.warning(rec_text)
                    else:
                        st.success(rec_text)

                # ── Critical Safety Alerts ────────────────
                alerts = critical_alerts(torque, rpm, air_temp, process_temp)
                if alerts:
                    st.subheader("🚨 Safety Alerts")
                    for alert in alerts:
                        st.error(alert)

                # ── SHAP Diagnosis ────────────────────────
                st.subheader("🧠 Failure Diagnosis (SHAP)")
                top_label, diag_msg = get_diagnosis(explainer, input_df)
                st.info(f"**Primary driver:** {top_label}\n\n{diag_msg}")
                st.plotly_chart(build_shap_bar(explainer, input_df), use_container_width=True)

                # ── What-If Simulation ────────────────────
                st.subheader("⚡ What-If Load Simulation")
                st.caption(
                    "Adjust the torque slider to see how rotor speed changes "
                    "using the induction motor slip model (linear region approximation)."
                )
                sim_torque = st.slider(
                    "Simulate Torque (Nm)",
                    min_value=0.0, max_value=200.0,
                    value=float(torque), step=1.0,
                )
                sim_rpm = simulate_slip_rpm(sim_torque)

                s1, s2 = st.columns(2)
                s1.metric("Simulated Torque", f"{sim_torque:.1f} Nm")
                s2.metric(
                    "Predicted Rotor Speed",
                    f"{sim_rpm:.1f} RPM",
                    delta=f"{sim_rpm - rpm:+.1f} RPM vs current",
                )

                if sim_torque > 0.5 * Config.RATED_TORQUE_NM * (1.0 / Config.RATED_SLIP):
                    st.warning(
                        "⚠️ Simulation may be outside the linear slip region — "
                        "results are indicative only at very high torque values."
                    )

                # ── Parameters Chart ──────────────────────
                st.subheader("📈 Operating Parameters Overview")
                st.plotly_chart(
                    build_parameters_chart(rpm, torque, tool_wear, sim_rpm, air_temp, process_temp),
                    use_container_width=True,
                )

            except ValidationError as exc:
                st.error(f"❌ Validation Error: {exc}")
                logger.warning("Input validation failed: %s", exc)
            except Exception as exc:
                st.error(f"❌ Unexpected prediction error: {exc}")
                logger.exception("Single prediction error")


# ── Model Info ────────────────────────────────────────────
elif menu == "📚 Model Info":
    st.title("📚 Model Information")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
**Classifier:** LightGBM (`LGBMClassifier`)  
**Explainability:** SHAP `TreeExplainer`  
**Dataset:** AI4I 2020 Predictive Maintenance  
**Focus:** Induction motors (Type L / M / H)  
**ROC-AUC (held-out test set):** `{auc_score:.4f}`

**Input Features**

| Feature | Unit | Range |
|:--------|:-----|:------|
| Motor Type | — | L / M / H |
| Air Temperature | K | 250 – 400 |
| Process Temperature | K | 250 – 450 |
| Rotational Speed | RPM | 0 – 5000 |
| Torque | Nm | 0 – 200 |
| Tool Wear | min | 0 – 500 |
        """)

    with c2:
        st.markdown(f"""
**Model Hyperparameters**

| Parameter | Value |
|:----------|:------|
| n_estimators | {Config.MODEL_PARAMS['n_estimators']} |
| learning_rate | {Config.MODEL_PARAMS['learning_rate']} |
| max_depth | {Config.MODEL_PARAMS['max_depth']} |
| num_leaves | {Config.MODEL_PARAMS['num_leaves']} |
| class_weight | balanced |

**Physics Simulation (Slip Model)**

| Constant | Value |
|:---------|:------|
| Synchronous speed | {Config.SYNC_SPEED_RPM:.0f} RPM |
| Rated slip | {Config.RATED_SLIP:.3f} ({Config.RATED_SLIP*100:.1f} %) |
| Rated torque | {Config.RATED_TORQUE_NM:.1f} Nm |

**Decision Thresholds**

| Level | Probability |
|:------|:-----------|
| 🔴 High risk | ≥ {Config.FAILURE_THRESHOLD_HIGH*100:.0f} % |
| 🟡 Degrading | {Config.FAILURE_THRESHOLD_MEDIUM*100:.0f} – {Config.FAILURE_THRESHOLD_HIGH*100:.0f} % |
| 🟢 Normal | < {Config.FAILURE_THRESHOLD_MEDIUM*100:.0f} % |
        """)

    st.divider()
    st.markdown("""
**Safety Architecture**

- Input range validation against physical motor constraints  
- Thermodynamic constraint: process temperature ≥ air temperature enforced  
- CSV uploads validated for size, row count, column presence, dtype, and value ranges  
- Rule-based safety alerts run independently of the ML model  
- SHAP values provide post-hoc explainability for every prediction  
- Disk-based model caching avoids retraining on every cold start  
- All server-side events logged via Python `logging`  
    """)
