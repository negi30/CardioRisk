"""
Streamlit Web Application: Heart Disease Risk Prediction & Clinical Model Comparison.

A production-grade, highly polished clinical machine learning dashboard with
Helvetica typography, high-contrast dark aesthetic, interactive threshold calibration,
clinical factor breakdowns, and rapid patient profile presets.

DISCLAIMER: Educational & Research Demonstration Only.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page layout & configuration
st.set_page_config(
    page_title="CardioRisk ML | Clinical Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


@st.cache_resource
def load_all_models():
    """Load metadata and all available trained scikit-learn pipelines."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Model artifacts not found. Please train the model first by running:\n"
            "```bash\npython -m src.train\n```"
        )
        st.stop()

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    models_dict = {}
    
    # Load individual pipelines if present, fallback to default heart_disease_model
    model_files = {
        "Random Forest": BASE_DIR / "models" / "random_forest_pipeline.joblib",
        "Logistic Regression": BASE_DIR / "models" / "logistic_regression_pipeline.joblib",
        "XGBoost": BASE_DIR / "models" / "xgboost_pipeline.joblib",
    }
    
    for name, path in model_files.items():
        if path.exists():
            models_dict[name] = joblib.load(path)
        else:
            models_dict[name] = joblib.load(MODEL_PATH)

    return models_dict, metadata


# Ingest artifacts
available_models, metadata = load_all_models()
selected_model_info = metadata.get("selected_model", {})
default_model_name = selected_model_info.get("model_name", "Random Forest")
default_threshold = float(selected_model_info.get("recommended_threshold", 0.30))
all_features = metadata.get("features", {}).get(
    "all",
    [
        "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
        "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
    ]
)

# ----------------- Custom Styling (Helvetica, Modern Dark Contrast) -----------------
st.markdown(
    """
    <style>
    /* Force Dark Scheme on Root */
    :root {
        color-scheme: dark !important;
    }

    /* Global Helvetica Typography without breaking Material Icons */
    html, body, p, div, span, label, h1, h2, h3, h4, h5, h6, input, select, textarea, button {
        font-family: -apple-system, "Helvetica Neue", Helvetica, "Segoe UI", Arial, sans-serif;
        letter-spacing: -0.015em;
    }

    /* Preserve Streamlit Material Icons and UI buttons */
    [data-testid*="stIcon"], [class*="material-symbols"], [class*="material-icons"], .material-icons, .material-symbols-rounded {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        letter-spacing: normal !important;
    }

    /* Clean dark surface */
    .stApp {
        background-color: #09090b !important;
        color: #f4f4f5 !important;
    }

    /* Force all inputs and controls to sleek dark styling */
    div[data-baseweb="select"], div[data-baseweb="input"], input, select, textarea {
        background-color: #121215 !important;
        color: #ffffff !important;
        border-color: #27272a !important;
    }

    /* Left Sidebar: Subtle grainy dark aesthetic */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0e !important;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.035'/%3E%3C/svg%3E") !important;
        border-right: 1px solid #1f1f23 !important;
    }

    /* GitHub Link Button */
    .github-link-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #18181b;
        border: 1px solid #27272a;
        color: #e4e4e7 !important;
        text-decoration: none !important;
        border-radius: 6px;
        padding: 0.45rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        transition: border-color 0.15s ease;
    }
    .github-link-btn:hover {
        background: #27272a;
        border-color: #52525b;
        color: #ffffff !important;
    }

    /* Section Cards */
    .card-panel-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f4f4f5;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        border-bottom: 1px solid #27272a;
        padding-bottom: 0.45rem;
    }

    /* Risk Status Banners */
    .risk-banner-high {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        color: #fee2e2;
        margin-top: 1rem;
    }
    .risk-banner-low {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        color: #ecfdf5;
        margin-top: 1rem;
    }

    /* Metric Highlight Boxes */
    .stat-box {
        background: #121215;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 1rem 0.9rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
    }
    .stat-desc {
        font-size: 0.72rem;
        color: #71717a;
        margin-top: 0.15rem;
    }

    /* General Buttons & Presets */
    div.stButton > button {
        background: #141418 !important;
        color: #e4e4e7 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        min-height: 38px !important;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        background: #27272a !important;
        border-color: #52525b !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* Primary Form Submit Action Button (Prominent & Big) */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(180deg, #ea580c 0%, #c2410c 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.8rem 1.5rem !important;
        min-height: 48px !important;
        box-shadow: 0 4px 15px rgba(234, 88, 12, 0.35);
        transition: all 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(180deg, #f97316 0%, #ea580c 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.5);
        transform: translateY(-1px);
    }

    /* Prominent Separated Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        background-color: transparent !important;
        border-bottom: 1px solid #27272a !important;
        padding-bottom: 8px !important;
        margin-bottom: 1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 22px !important;
        background-color: #121215 !important;
        color: #a1a1aa !important;
        border: 1px solid #27272a !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1a1a20 !important;
        color: #ffffff !important;
        border-color: #3f3f46 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c1917 !important;
        color: #fb923c !important;
        border: 1px solid #ea580c !important;
        box-shadow: 0 0 16px rgba(234, 88, 12, 0.2) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Clean Top Header (Unboxed, Pure Typography) -----------------
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; border-bottom: 1px solid #27272a; padding-bottom: 1.1rem;">
        <div>
            <h1 style="font-size: 2.1rem; font-weight: 750; color: #ffffff; margin: 0 0 0.35rem 0; letter-spacing: -0.025em;">
                Heart Disease Risk Prediction
            </h1>
            <p style="font-size: 0.95rem; color: #a1a1aa; margin: 0; line-height: 1.45;">
                Clinical risk classification & model comparison with sensitivity calibration to minimize false negatives.
            </p>
        </div>
        <div>
            <a href="https://github.com/negi30/CardioRisk" target="_blank" class="github-link-btn">
                <svg height="15" width="15" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                negi30/CardioRisk
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clean, Minimal Disclaimer Banner
st.markdown(
    """
    <div style="background: rgba(245, 158, 11, 0.05); border-left: 3px solid #f59e0b; padding: 0.6rem 0.95rem; border-radius: 4px; margin-bottom: 1.2rem; font-size: 0.82rem; color: #d1d5db;">
        <strong style="color: #fbbf24;">Research & Educational Demo:</strong> Predictions are generated for benchmark analysis. Not a medical diagnostic tool.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- Sidebar: Threshold & System Controls -----------------
with st.sidebar:
    st.markdown("### ⚙️ Calibration & Model Hub")
    
    model_options = ["Random Forest", "Logistic Regression", "XGBoost"]
    selected_model_name = st.selectbox(
        "Active Model",
        options=model_options,
        index=0,
        help="Select which machine-learning algorithm generates real-time predictions.",
    )
    
    model_pipeline = available_models[selected_model_name]

    st.markdown("#### 🎯 Clinical Decision Threshold ($T$)")
    st.caption(
        "Standard classification assumes $T = 0.50$. In cardiac screening, lowering $T$ elevates **Sensitivity (Recall)** "
        "and suppresses critical **False Negatives**."
    )

    threshold = st.slider(
        "Decision Cutoff ($T$)",
        min_value=0.10,
        max_value=0.90,
        value=default_threshold,
        step=0.05,
        help="Patients with predicted risk probability >= T will be classified as Higher Predicted Risk.",
    )

    if threshold <= 0.35:
        st.success(f"Sensitivity Mode ($T={threshold:.2f}$): Maximizes disease detection (0 FN on test set).")
    elif threshold <= 0.50:
        st.info(f"Balanced Clinical Mode ($T={threshold:.2f}$): Strong balance between Precision and Recall.")
    else:
        st.warning(f"High-Specificity Mode ($T={threshold:.2f}$): Lower False Positives, but higher risk of missed cases.")

    st.markdown("---")
    st.markdown("#### 🧬 Zero-Leakage Pipeline")
    st.markdown(
        """
        - **Numerical:** Median Impute $\\rightarrow$ StandardScaler
        - **Categorical:** Mode Impute $\\rightarrow$ OneHotEncoder
        - **Classifier:** 300 Balanced Decision Trees
        - **Evaluation:** Stratified unseen test cohort ($N=61$)
        """
    )


# ----------------- Main Navigation Buttons (Separated & Coloured) -----------------
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "predict"

nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

with nav_c1:
    is_active = st.session_state["active_view"] == "predict"
    bg_style = "linear-gradient(135deg, rgba(234, 88, 12, 0.4) 0%, rgba(194, 65, 12, 0.6) 100%)" if is_active else "rgba(234, 88, 12, 0.12)"
    border_style = "2px solid #ea580c" if is_active else "1px solid rgba(234, 88, 12, 0.3)"
    shadow = "0 0 20px rgba(234, 88, 12, 0.35)" if is_active else "none"
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:nth-of-type(1) div.stButton > button[key="nav_btn_predict"] {{
            background: {bg_style} !important;
            border: {border_style} !important;
            color: #fed7aa !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            min-height: 48px !important;
            border-radius: 10px !important;
            box-shadow: {shadow} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🩺 Patient Risk Assessment", key="nav_btn_predict", use_container_width=True):
        st.session_state["active_view"] = "predict"
        st.rerun()

with nav_c2:
    is_active = st.session_state["active_view"] == "compare"
    bg_style = "linear-gradient(135deg, rgba(2, 132, 199, 0.4) 0%, rgba(3, 105, 161, 0.6) 100%)" if is_active else "rgba(2, 132, 199, 0.12)"
    border_style = "2px solid #0284c7" if is_active else "1px solid rgba(2, 132, 199, 0.3)"
    shadow = "0 0 20px rgba(2, 132, 199, 0.35)" if is_active else "none"
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:nth-of-type(2) div.stButton > button[key="nav_btn_compare"] {{
            background: {bg_style} !important;
            border: {border_style} !important;
            color: #bae6fd !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            min-height: 48px !important;
            border-radius: 10px !important;
            box-shadow: {shadow} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📊 Model Comparison", key="nav_btn_compare", use_container_width=True):
        st.session_state["active_view"] = "compare"
        st.rerun()

with nav_c3:
    is_active = st.session_state["active_view"] == "threshold"
    bg_style = "linear-gradient(135deg, rgba(5, 150, 105, 0.4) 0%, rgba(4, 120, 87, 0.6) 100%)" if is_active else "rgba(5, 150, 105, 0.12)"
    border_style = "2px solid #059669" if is_active else "1px solid rgba(5, 150, 105, 0.3)"
    shadow = "0 0 20px rgba(5, 150, 105, 0.35)" if is_active else "none"
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:nth-of-type(3) div.stButton > button[key="nav_btn_threshold"] {{
            background: {bg_style} !important;
            border: {border_style} !important;
            color: #a7f3d0 !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            min-height: 48px !important;
            border-radius: 10px !important;
            box-shadow: {shadow} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("⚖️ Threshold Simulator", key="nav_btn_threshold", use_container_width=True):
        st.session_state["active_view"] = "threshold"
        st.rerun()

with nav_c4:
    is_active = st.session_state["active_view"] == "about"
    bg_style = "linear-gradient(135deg, rgba(124, 58, 237, 0.4) 0%, rgba(109, 40, 217, 0.6) 100%)" if is_active else "rgba(124, 58, 237, 0.12)"
    border_style = "2px solid #7c3aed" if is_active else "1px solid rgba(124, 58, 237, 0.3)"
    shadow = "0 0 20px rgba(124, 58, 237, 0.35)" if is_active else "none"
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:nth-of-type(4) div.stButton > button[key="nav_btn_about"] {{
            background: {bg_style} !important;
            border: {border_style} !important;
            color: #ddd6fe !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            min-height: 48px !important;
            border-radius: 10px !important;
            box-shadow: {shadow} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📖 Clinical Reference", key="nav_btn_about", use_container_width=True):
        st.session_state["active_view"] = "about"
        st.rerun()

st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

# ----------------- View 1: Patient Risk Assessment -----------------
if st.session_state["active_view"] == "predict":
    # Preset quick-fill profiles
    st.markdown(
        '<div style="font-size: 0.82rem; font-weight: 600; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.4rem;">Sample Clinical Presets</div>',
        unsafe_allow_html=True,
    )

    col_p1, col_p2, col_p3 = st.columns(3)

    # State initialization for form values
    if "age_val" not in st.session_state:
        st.session_state["age_val"] = 58
        st.session_state["sex_idx"] = 1
        st.session_state["cp_idx"] = 3
        st.session_state["trestbps_val"] = 145
        st.session_state["chol_val"] = 280
        st.session_state["fbs_idx"] = 0
        st.session_state["restecg_idx"] = 2
        st.session_state["thalach_val"] = 125
        st.session_state["exang_idx"] = 1
        st.session_state["oldpeak_val"] = 2.4
        st.session_state["slope_idx"] = 1
        st.session_state["ca_idx"] = 2
        st.session_state["thal_idx"] = 2

    with col_p1:
        if st.button("🚨 High-Risk Profile", use_container_width=True):
            st.session_state["age_val"] = 67
            st.session_state["sex_idx"] = 1
            st.session_state["cp_idx"] = 3  # Asymptomatic
            st.session_state["trestbps_val"] = 160
            st.session_state["chol_val"] = 286
            st.session_state["fbs_idx"] = 0
            st.session_state["restecg_idx"] = 2
            st.session_state["thalach_val"] = 108
            st.session_state["exang_idx"] = 1
            st.session_state["oldpeak_val"] = 2.6
            st.session_state["slope_idx"] = 1  # Flat
            st.session_state["ca_idx"] = 3  # 3 vessels
            st.session_state["thal_idx"] = 2  # Reversible defect
            st.rerun()

    with col_p2:
        if st.button("✅ Low-Risk Profile", use_container_width=True):
            st.session_state["age_val"] = 41
            st.session_state["sex_idx"] = 0
            st.session_state["cp_idx"] = 1  # Atypical Angina
            st.session_state["trestbps_val"] = 120
            st.session_state["chol_val"] = 195
            st.session_state["fbs_idx"] = 0
            st.session_state["restecg_idx"] = 0
            st.session_state["thalach_val"] = 175
            st.session_state["exang_idx"] = 0
            st.session_state["oldpeak_val"] = 0.2
            st.session_state["slope_idx"] = 0  # Upsloping
            st.session_state["ca_idx"] = 0  # 0 vessels
            st.session_state["thal_idx"] = 0  # Normal
            st.rerun()

    with col_p3:
        if st.button("⚠️ Borderline Profile", use_container_width=True):
            st.session_state["age_val"] = 54
            st.session_state["sex_idx"] = 1
            st.session_state["cp_idx"] = 2  # Non-anginal
            st.session_state["trestbps_val"] = 135
            st.session_state["chol_val"] = 245
            st.session_state["fbs_idx"] = 1
            st.session_state["restecg_idx"] = 0
            st.session_state["thalach_val"] = 150
            st.session_state["exang_idx"] = 0
            st.session_state["oldpeak_val"] = 1.0
            st.session_state["slope_idx"] = 1  # Flat
            st.session_state["ca_idx"] = 1  # 1 vessel
            st.session_state["thal_idx"] = 0  # Normal
            st.rerun()

    st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

    # Interactive Patient Input Form
    with st.form("clinical_patient_form"):
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(
                '<div class="card-panel-header">👤 Demographics & Hemodynamics</div>',
                unsafe_allow_html=True,
            )
            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=st.session_state["age_val"],
                step=1,
            )
            sex_label = st.selectbox(
                "Biological Sex",
                options=["Female (0)", "Male (1)"],
                index=st.session_state["sex_idx"],
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=240,
                value=st.session_state["trestbps_val"],
                step=1,
                help="Systolic pressure upon admission.",
            )
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100,
                max_value=600,
                value=st.session_state["chol_val"],
                step=1,
            )

        with col_c2:
            st.markdown(
                '<div class="card-panel-header">🫀 Triage & Electrocardiogram</div>',
                unsafe_allow_html=True,
            )
            cp_label = st.selectbox(
                "Chest Pain Type (cp)",
                options=[
                    "Typical Angina (1)",
                    "Atypical Angina (2)",
                    "Non-anginal Pain (3)",
                    "Asymptomatic (4)",
                ],
                index=st.session_state["cp_idx"],
                help="Category of chest discomfort during medical evaluation.",
            )
            fbs_label = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=["False (0) [<= 120 mg/dl]", "True (1) [> 120 mg/dl]"],
                index=st.session_state["fbs_idx"],
            )
            restecg_label = st.selectbox(
                "Resting ECG Results (restecg)",
                options=[
                    "Normal (0)",
                    "ST-T Wave Abnormality (1)",
                    "Left Ventricular Hypertrophy (2)",
                ],
                index=st.session_state["restecg_idx"],
            )
            exang_label = st.selectbox(
                "Exercise-Induced Angina (exang)",
                options=["No (0)", "Yes (1)"],
                index=st.session_state["exang_idx"],
            )

        with col_c3:
            st.markdown(
                '<div class="card-panel-header">📈 Stress Testing & Angiography</div>',
                unsafe_allow_html=True,
            )
            thalach = st.number_input(
                "Max Exercise Heart Rate (thalach)",
                min_value=60,
                max_value=230,
                value=st.session_state["thalach_val"],
                step=1,
                help="Peak heart rate achieved during treadmill test.",
            )
            oldpeak = st.number_input(
                "ST Depression (oldpeak in mm)",
                min_value=0.0,
                max_value=8.0,
                value=float(st.session_state["oldpeak_val"]),
                step=0.1,
                format="%.1f",
                help="ST depression induced by exercise relative to resting baseline.",
            )
            slope_label = st.selectbox(
                "Peak Exercise ST Slope (slope)",
                options=["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
                index=st.session_state["slope_idx"],
            )
            ca_label = st.selectbox(
                "Major Vessels Colored by Fluoroscopy (ca)",
                options=["0 vessels", "1 vessel", "2 vessels", "3 vessels"],
                index=st.session_state["ca_idx"],
                help="Number of major coronary arteries (0-3) visible under fluoroscopy.",
            )
            thal_label = st.selectbox(
                "Thallium Stress Test (thal)",
                options=["Normal (3.0)", "Fixed Defect (6.0)", "Reversible Defect (7.0)"],
                index=st.session_state["thal_idx"],
                help="Thallium scintigraphy perfusion findings.",
            )

        predict_button = st.form_submit_button("🔍 Run Machine Learning Risk Assessment", use_container_width=True)

    # Inference Execution & Result Presentation
    if predict_button:
        # Schema and Value Transformation
        sex_val = 1 if "Male" in sex_label else 0
        cp_map = {
            "Typical Angina (1)": 1,
            "Atypical Angina (2)": 2,
            "Non-anginal Pain (3)": 3,
            "Asymptomatic (4)": 4,
        }
        fbs_val = 1 if "True" in fbs_label else 0
        restecg_map = {
            "Normal (0)": 0,
            "ST-T Wave Abnormality (1)": 1,
            "Left Ventricular Hypertrophy (2)": 2,
        }
        exang_val = 1 if "Yes" in exang_label else 0
        slope_map = {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3}
        ca_map = {"0 vessels": 0.0, "1 vessel": 1.0, "2 vessels": 2.0, "3 vessels": 3.0}
        thal_map = {
            "Normal (3.0)": 3.0,
            "Fixed Defect (6.0)": 6.0,
            "Reversible Defect (7.0)": 7.0,
        }

        # Build payload with exact training column order
        patient_payload = {
            "age": float(age),
            "sex": int(sex_val),
            "cp": int(cp_map[cp_label]),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": int(fbs_val),
            "restecg": int(restecg_map[restecg_label]),
            "thalach": float(thalach),
            "exang": int(exang_val),
            "oldpeak": float(oldpeak),
            "slope": int(slope_map[slope_label]),
            "ca": float(ca_map[ca_label]),
            "thal": float(thal_map[thal_label]),
        }

        input_df = pd.DataFrame([patient_payload])[all_features]

        try:
            # Model inference
            risk_probability = float(model_pipeline.predict_proba(input_df)[0, 1])
            is_high_risk = risk_probability >= threshold

            st.markdown("### 📊 Model Risk Evaluation Output")

            # High-Contrast Metric Cards
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            with stat_c1:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Model Engine</div>
                        <div class="stat-value" style="font-size: 1.25rem; color: #f97316;">{selected_model_name}</div>
                        <div class="stat-desc">Zero-Leakage Pipeline</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_c2:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Decision Threshold</div>
                        <div class="stat-value">{threshold:.2f}</div>
                        <div class="stat-desc">Calibrated Cutoff ($T$)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_c3:
                prob_color = "#ef4444" if risk_probability >= threshold else "#10b981"
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Calculated Risk Probability</div>
                        <div class="stat-value" style="color: {prob_color};">{risk_probability:.1%}</div>
                        <div class="stat-desc">P(Heart Disease | Features)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_c4:
                risk_tag = "Higher Risk (Class 1)" if is_high_risk else "Lower Risk (Class 0)"
                tag_bg = "#ef4444" if is_high_risk else "#10b981"
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Predicted Risk Class</div>
                        <div class="stat-value" style="font-size: 1.15rem; color: {tag_bg};">{risk_tag}</div>
                        <div class="stat-desc">Binary Classification Output</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Continuous probability bar
            st.progress(
                risk_probability,
                text=f"Estimated Heart Disease Risk Probability: {risk_probability:.1%} (Cutoff: {threshold:.2f})",
            )

            # Detailed Clinical Finding Card
            if is_high_risk:
                st.markdown(
                    f"""
                    <div class="risk-banner-high">
                        <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                            ⚠️ Higher Predicted Risk of Coronary Heart Disease
                        </div>
                        <p style="margin-bottom: 0.8rem; font-size: 0.95rem;">
                            The machine-learning pipeline estimated a <strong>{risk_probability:.1%}</strong> probability of coronary artery disease,
                            which meets or exceeds the calibrated clinical screening threshold of <strong>{threshold:.2f}</strong>.
                        </p>
                        <div style="font-size: 0.85rem; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 0.8rem;">
                            <strong>Key Physiological Risk Drivers Detected:</strong><br>
                            • ST Depression (Oldpeak): <code>{oldpeak} mm</code><br>
                            • Coronary Fluoroscopy Vessels (ca): <code>{ca_label}</code><br>
                            • Thallium Perfusion Status (thal): <code>{thal_label}</code><br>
                            • Exercise Angina (exang): <code>{exang_label}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-banner-low">
                        <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                            ✅ Lower Predicted Risk of Coronary Heart Disease
                        </div>
                        <p style="margin-bottom: 0.8rem; font-size: 0.95rem;">
                            The machine-learning pipeline estimated a <strong>{risk_probability:.1%}</strong> probability of coronary artery disease,
                            which lies below the calibrated clinical screening threshold of <strong>{threshold:.2f}</strong>.
                        </p>
                        <div style="font-size: 0.85rem; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 0.8rem;">
                            <strong>Clinical Context:</strong><br>
                            While statistical indicators reflect a low likelihood of coronary disease in this cohort, clinical screening
                            should always account for continuous monitoring, patient symptoms, and full physician evaluations.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Live Multi-Model Consensus Matrix
            with st.expander("🤝 Multi-Model Consensus & Architecture Cross-Validation", expanded=True):
                st.caption(
                    "Real-time evaluation of this exact patient profile across all 3 trained model families:"
                )
                consensus_cols = st.columns(3)
                consensus_scores = {}

                for idx, (m_name, m_pipe) in enumerate(available_models.items()):
                    score = float(m_pipe.predict_proba(input_df)[0, 1])
                    verdict = "Higher Risk" if score >= threshold else "Lower Risk"
                    consensus_scores[m_name] = (score, verdict)

                    with consensus_cols[idx]:
                        is_active = m_name == selected_model_name
                        border_style = "border: 1px solid #f97316;" if is_active else "border: 1px solid rgba(255,255,255,0.08);"
                        badge_active = " <span style='color: #f97316; font-size: 0.72rem; font-weight: 700;'>(Active)</span>" if is_active else ""
                        badge_col = "#ef4444" if verdict == "Higher Risk" else "#10b981"

                        st.markdown(
                            f"""
                            <div style="background: #18181b; {border_style} border-radius: 10px; padding: 0.9rem; text-align: center;">
                                <div style="font-size: 0.8rem; font-weight: 700; color: #f4f4f5; margin-bottom: 0.2rem;">{m_name}{badge_active}</div>
                                <div style="font-size: 1.35rem; font-weight: 800; color: {badge_col};">{score:.1%}</div>
                                <div style="font-size: 0.75rem; color: #a1a1aa; margin-top: 0.2rem;">Verdict: <strong style="color: {badge_col};">{verdict}</strong></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with st.expander("🔍 View Processed Feature Payload Vector"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Inference execution failed: {e}")


# ----------------- View 2: Clinical Model Comparison -----------------
elif st.session_state["active_view"] == "compare":
    st.markdown("### 📊 Clinical Model Benchmark & Performance Metrics")
    st.caption(
        "All models were evaluated on the exact same stratified unseen test set (N=61) with zero data leakage."
    )

    models_eval = metadata.get("models_evaluated", [])
    if models_eval:
        comp_df = pd.DataFrame(models_eval)
        st.dataframe(
            comp_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }
            ).highlight_max(
                subset=["Recall", "F1", "ROC-AUC", "Accuracy", "Precision"],
                color="rgba(249, 115, 22, 0.25)",
            ).highlight_min(
                subset=["False Negatives", "False Positives"],
                color="rgba(16, 185, 129, 0.25)",
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📈 Model Comparison Visualizations")

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("**Metric Comparison Across Architectures**")
        if (FIGURES_DIR / "model_comparison.png").exists():
            st.image(str(FIGURES_DIR / "model_comparison.png"), use_container_width=True)

    with fig_col2:
        st.markdown("**Receiver Operating Characteristic (ROC) Comparison**")
        if (FIGURES_DIR / "roc_curves.png").exists():
            st.image(str(FIGURES_DIR / "roc_curves.png"), use_container_width=True)

    st.markdown("#### 🎯 Confusion Matrices (Focus on False Negatives)")
    cm_c1, cm_c2, cm_c3 = st.columns(3)

    with cm_c1:
        if (FIGURES_DIR / "confusion_matrix_logistic.png").exists():
            st.image(str(FIGURES_DIR / "confusion_matrix_logistic.png"), caption="Logistic Regression", use_container_width=True)

    with cm_c2:
        if (FIGURES_DIR / "confusion_matrix_random_forest.png").exists():
            st.image(str(FIGURES_DIR / "confusion_matrix_random_forest.png"), caption="Random Forest (Deployed)", use_container_width=True)

    with cm_c3:
        if (FIGURES_DIR / "confusion_matrix_xgboost.png").exists():
            st.image(str(FIGURES_DIR / "confusion_matrix_xgboost.png"), caption="XGBoost", use_container_width=True)


# ----------------- View 3: Threshold Sensitivity Simulator -----------------
elif st.session_state["active_view"] == "threshold":
    st.markdown("### ⚖️ Decision Threshold Sensitivity & Error Trade-offs")
    st.markdown(
        """
        In medical machine learning, setting the decision threshold blindly to $0.50$ is rarely optimal.
        Adjusting the threshold allows healthcare systems to explicitly prioritize **Recall (Sensitivity)**
        and drive **False Negatives (missed diagnoses)** down to zero.
        """
    )

    if (FIGURES_DIR / "threshold_analysis.png").exists():
        st.image(str(FIGURES_DIR / "threshold_analysis.png"), use_container_width=True)

    thresh_records = metadata.get("threshold_analysis", [])
    if thresh_records:
        st.markdown("#### 📋 Empirical Threshold Grid (Random Forest Model)")
        t_df = pd.DataFrame(thresh_records)
        st.dataframe(
            t_df.style.format(
                {
                    "Threshold": "{:.2f}",
                    "Recall": "{:.4f}",
                    "Precision": "{:.4f}",
                    "F1": "{:.4f}",
                    "Accuracy": "{:.4f}",
                }
            ).highlight_max(subset=["Recall", "F1", "Accuracy"], color="rgba(249, 115, 22, 0.25)")
            .highlight_min(subset=["False Negatives"], color="rgba(16, 185, 129, 0.25)"),
            use_container_width=True,
        )


# ----------------- View 4: Clinical Attributes & Dataset Info -----------------
elif st.session_state["active_view"] == "about":
    st.markdown("### 📖 Benchmark Dataset Provenance & Feature Schema")
    st.markdown(
        """
        The **Cleveland Heart Disease database** is the most widely referenced dataset in cardiology machine learning.
        It contains 303 patient records collected from clinical evaluations and coronary angiography.

        #### Feature Schema & Clinical Interpretations
        - **`age`**: Patient age in years.
        - **`sex`**: Biological sex ($1 = \text{Male}, 0 = \text{Female}$).
        - **`cp`**: Chest pain triage classification ($1 = \text{Typical angina}, 2 = \text{Atypical angina}, 3 = \text{Non-anginal pain}, 4 = \text{Asymptomatic}$).
        - **`trestbps`**: Resting systolic blood pressure in mm Hg on admission.
        - **`chol`**: Serum cholesterol level in mg/dl.
        - **`fbs`**: Fasting blood sugar $> 120\text{ mg/dl}$ ($1 = \text{True}, 0 = \text{False}$).
        - **`restecg`**: Resting ECG findings ($0 = \text{Normal}, 1 = \text{ST-T wave abnormality}, 2 = \text{Left ventricular hypertrophy}$).
        - **`thalach`**: Maximum heart rate achieved during exercise stress testing.
        - **`exang`**: Exercise-induced angina ($1 = \text{Yes}, 0 = \text{No}$).
        - **`oldpeak`**: ST segment depression induced by exercise relative to resting baseline (mm).
        - **`slope`**: Slope of peak exercise ST segment ($1 = \text{Upsloping}, 2 = \text{Flat}, 3 = \text{Downsloping}$).
        - **`ca`**: Number of major coronary vessels ($0 - 3$) colored by fluoroscopy.
        - **`thal`**: Thallium scintigraphy nuclear perfusion scan ($3.0 = \text{Normal}, 6.0 = \text{Fixed defect}, 7.0 = \text{Reversible defect}$).

        #### Academic Citation
        > Janosi, A., Steinbrunn, W., Pfisterer, W., & Detrano, R. (1988). *Heart Disease*. UCI Machine Learning Repository.
        """
    )

    if (FIGURES_DIR / "target_distribution.png").exists():
        st.image(str(FIGURES_DIR / "target_distribution.png"), caption="Binary Target Class Distribution", width=500)
