"""
Scikit-learn preprocessing pipelines and column transformations.
"""

from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical Pipeline:
      1. SimpleImputer with median strategy (handles potential missing values in ca, chol, etc.)
      2. StandardScaler (scales numerical attributes to zero mean, unit variance)

    Categorical Pipeline:
      1. SimpleImputer with most_frequent strategy (handles missing values in thal, etc.)
      2. OneHotEncoder (handle_unknown='ignore' ensures safe inference on unseen levels)
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def create_full_pipeline(classifier: Any) -> Pipeline:
    """
    Wrap the ColumnTransformer preprocessor and the classifier into a single sklearn Pipeline.
    This guarantees zero data leakage during training, cross-validation, and production inference.
    """
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
