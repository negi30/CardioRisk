"""
Data loading and validation module for the UCI Heart Disease dataset.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils import setup_ssl, get_logger, DATA_DIR

logger = get_logger("data_loader")

# Feature classifications based on clinical and statistical characteristics
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "num"

# Human-readable value maps for categorical variables (used in UI and reports)
FEATURE_VALUE_MAPS: Dict[str, Dict[Any, str]] = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical Angina (1)",
        2: "Atypical Angina (2)",
        3: "Non-anginal Pain (3)",
        4: "Asymptomatic (4)",
    },
    "fbs": {0: "Fasting Blood Sugar <= 120 mg/dl (0)", 1: "Fasting Blood Sugar > 120 mg/dl (1)"},
    "restecg": {
        0: "Normal (0)",
        1: "ST-T Wave Abnormality (1)",
        2: "Left Ventricular Hypertrophy (2)",
    },
    "exang": {0: "No (0)", 1: "Yes (1)"},
    "slope": {
        1: "Upsloping (1)",
        2: "Flat (2)",
        3: "Downsloping (3)",
    },
    "thal": {
        3.0: "Normal (3.0)",
        6.0: "Fixed Defect (6.0)",
        7.0: "Reversible Defect (7.0)",
    },
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "age": "Age in years",
    "sex": "Biological sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)",
    "trestbps": "Resting blood pressure in mm Hg on hospital admission",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)",
    "thalach": "Maximum heart rate achieved during exercise test",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia status (3.0 = normal, 6.0 = fixed defect, 7.0 = reversible defect)",
}


def load_raw_dataset(cache_path: Path = DATA_DIR / "heart_disease_raw.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the UCI Cleveland Heart Disease dataset (ID=45).
    Falls back to local cache if offline.
    """
    setup_ssl()
    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        X = df[ALL_FEATURES]
        y = df[[TARGET_COLUMN]]
        return X, y

    logger.info("Fetching UCI Heart Disease dataset (ID=45) from repository...")
    try:
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features.copy()
        y = heart_disease.data.targets.copy()

        # Save local copy for offline reproducibility
        full_df = pd.concat([X, y], axis=1)
        full_df.to_csv(cache_path, index=False)
        logger.info(f"Saved dataset snapshot to {cache_path}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to fetch dataset from UCI: {e}")
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            return df[ALL_FEATURES], df[[TARGET_COLUMN]]
        raise RuntimeError(f"Could not load dataset from UCI repository or local cache: {e}")


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset and convert multi-class severity target (0-4) into binary risk target:
      0: No heart disease (absence)
      1: Heart disease present (severity 1, 2, 3, 4)
    """
    X, y = load_raw_dataset()

    # Convert target to binary: 0 -> 0, (1, 2, 3, 4) -> 1
    target_series = y[TARGET_COLUMN].apply(lambda v: 1 if v > 0 else 0).astype(int)
    target_series.name = "target"

    logger.info(
        f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features. "
        f"Binary target balance: 0={sum(target_series == 0)} ({sum(target_series == 0)/len(target_series):.1%}), "
        f"1={sum(target_series == 1)} ({sum(target_series == 1)/len(target_series):.1%})"
    )
    return X, target_series
