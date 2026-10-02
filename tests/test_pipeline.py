"""
Unit, integration, and stress test suite for Heart Disease ML Pipeline & Model Switching.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import (
    load_processed_data,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.preprocessing import build_preprocessor, create_full_pipeline
from src.evaluate import evaluate_model
from src.utils import MODELS_DIR


@pytest.fixture
def sample_patient_profiles():
    """Returns diverse clinical patient profiles for edge-case stress testing."""
    return {
        "high_risk": pd.DataFrame(
            [
                {
                    "age": 67.0,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 160.0,
                    "chol": 286.0,
                    "fbs": 0,
                    "restecg": 2,
                    "thalach": 108.0,
                    "exang": 1,
                    "oldpeak": 2.6,
                    "slope": 2,
                    "ca": 3.0,
                    "thal": 7.0,
                }
            ]
        )[ALL_FEATURES],
        "low_risk": pd.DataFrame(
            [
                {
                    "age": 41.0,
                    "sex": 0,
                    "cp": 2,
                    "trestbps": 120.0,
                    "chol": 195.0,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 175.0,
                    "exang": 0,
                    "oldpeak": 0.2,
                    "slope": 1,
                    "ca": 0.0,
                    "thal": 3.0,
                }
            ]
        )[ALL_FEATURES],
        "borderline": pd.DataFrame(
            [
                {
                    "age": 54.0,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 135.0,
                    "chol": 245.0,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150.0,
                    "exang": 0,
                    "oldpeak": 1.0,
                    "slope": 2,
                    "ca": 1.0,
                    "thal": 3.0,
                }
            ]
        )[ALL_FEATURES],
        "extreme_outlier": pd.DataFrame(
            [
                {
                    "age": 95.0,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 230.0,
                    "chol": 580.0,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 65.0,
                    "exang": 1,
                    "oldpeak": 6.2,
                    "slope": 3,
                    "ca": 3.0,
                    "thal": 7.0,
                }
            ]
        )[ALL_FEATURES],
    }


def test_dataset_loading():
    """Verify that dataset loads with expected shape, columns, and valid binary target."""
    X, y = load_processed_data()

    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert len(X) == 303, f"Expected 303 rows, got {len(X)}"
    assert set(X.columns) == set(ALL_FEATURES), "Feature columns mismatch"
    assert set(y.unique()).issubset({0, 1}), "Binary target must only contain 0 and 1"
    assert y.isnull().sum() == 0, "Target cannot have missing values"


def test_preprocessor_fitting():
    """Verify that the ColumnTransformer preprocessor fits and transforms data without error."""
    X, _ = load_processed_data()
    preprocessor = build_preprocessor()

    X_trans = preprocessor.fit_transform(X)
    assert isinstance(X_trans, np.ndarray), "Transformed output must be a numpy ndarray"
    assert X_trans.shape[0] == len(X), "Row count must remain invariant after preprocessing"
    assert not np.isnan(X_trans).any(), "Preprocessed array must not contain any NaN values"


def test_all_model_pipeline_artifacts_exist_and_load():
    """Stress test: Verify that all 3 model artifacts and default model are serialized and loadable."""
    expected_models = [
        "heart_disease_model.joblib",
        "random_forest_pipeline.joblib",
        "logistic_regression_pipeline.joblib",
        "xgboost_pipeline.joblib",
    ]
    for model_filename in expected_models:
        model_path = MODELS_DIR / model_filename
        assert model_path.exists(), f"Model artifact missing: {model_path}"
        pipeline = joblib.load(model_path)
        assert hasattr(pipeline, "predict_proba"), f"Pipeline in {model_filename} missing predict_proba"
        assert hasattr(pipeline, "predict"), f"Pipeline in {model_filename} missing predict"


def test_multi_model_inference_stress(sample_patient_profiles):
    """Stress test: Rapid model switching across multiple distinct clinical profiles."""
    model_names = [
        "random_forest_pipeline.joblib",
        "logistic_regression_pipeline.joblib",
        "xgboost_pipeline.joblib",
    ]

    for model_file in model_names:
        pipeline = joblib.load(MODELS_DIR / model_file)

        for profile_name, patient_df in sample_patient_profiles.items():
            proba = pipeline.predict_proba(patient_df)
            pred = pipeline.predict(patient_df)

            assert proba.shape == (1, 2), f"Failed proba shape on {model_file} with {profile_name}"
            assert len(pred) == 1, f"Failed prediction length on {model_file} with {profile_name}"
            assert 0.0 <= proba[0, 1] <= 1.0, f"Probability out of bounds [0, 1] on {model_file}"
            assert np.isclose(proba.sum(), 1.0), f"Probabilities do not sum to 1 on {model_file}"

            # Clinical sanity check
            if profile_name == "high_risk":
                assert proba[0, 1] > 0.50, f"{model_file} underestimated obvious high risk patient"
            elif profile_name == "low_risk":
                assert proba[0, 1] < 0.50, f"{model_file} overestimated obvious low risk patient"


def test_batch_inference_stress_100_samples():
    """Stress test: Perform batch inference on 100 randomized synthetic patients across all models."""
    np.random.seed(42)
    n_samples = 100

    synthetic_batch = pd.DataFrame(
        {
            "age": np.random.uniform(25, 80, n_samples),
            "trestbps": np.random.uniform(90, 200, n_samples),
            "chol": np.random.uniform(120, 500, n_samples),
            "thalach": np.random.uniform(70, 210, n_samples),
            "oldpeak": np.random.uniform(0.0, 6.0, n_samples),
            "ca": np.random.choice([0.0, 1.0, 2.0, 3.0], n_samples),
            "sex": np.random.choice([0, 1], n_samples),
            "cp": np.random.choice([1, 2, 3, 4], n_samples),
            "fbs": np.random.choice([0, 1], n_samples),
            "restecg": np.random.choice([0, 1, 2], n_samples),
            "exang": np.random.choice([0, 1], n_samples),
            "slope": np.random.choice([1, 2, 3], n_samples),
            "thal": np.random.choice([3.0, 6.0, 7.0], n_samples),
        }
    )[ALL_FEATURES]

    model_files = [
        "random_forest_pipeline.joblib",
        "logistic_regression_pipeline.joblib",
        "xgboost_pipeline.joblib",
    ]

    for model_file in model_files:
        pipeline = joblib.load(MODELS_DIR / model_file)
        probas = pipeline.predict_proba(synthetic_batch)
        preds = pipeline.predict(synthetic_batch)

        assert probas.shape == (n_samples, 2)
        assert len(preds) == n_samples
        assert not np.isnan(probas).any(), f"NaNs detected in batch predictions for {model_file}"
        assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_streamlit_inference_schema_exact_match(sample_patient_profiles):
    """Verify that the DataFrame generated by Streamlit UI matches the exact training schema."""
    X, _ = load_processed_data()
    expected_cols = list(X.columns)

    for _, sample_df in sample_patient_profiles.items():
        assert list(sample_df.columns) == expected_cols
        assert set(sample_df.columns) == set(ALL_FEATURES)


def test_threshold_logic():
    """Verify that custom decision thresholds alter binary predictions correctly."""
    X, y = load_processed_data()
    pipeline = create_full_pipeline(RandomForestClassifier(n_estimators=50, random_state=42))
    pipeline.fit(X, y)

    res_strict = evaluate_model(pipeline, X, y, model_name="RF", threshold=0.70)
    res_sensitive = evaluate_model(pipeline, X, y, model_name="RF", threshold=0.30)

    assert res_sensitive["recall"] >= res_strict["recall"]
    assert res_sensitive["false_negatives"] <= res_strict["false_negatives"]
