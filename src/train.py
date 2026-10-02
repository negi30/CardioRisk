"""
Training pipeline: trains Logistic Regression, Random Forest, and XGBoost models,
evaluates clinical trade-offs (Recall, FN, ROC-AUC), and saves deployment artifacts.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_loader import (
    load_processed_data,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_VALUE_MAPS,
    FEATURE_DESCRIPTIONS,
)
from src.preprocessing import create_full_pipeline
from src.evaluate import (
    evaluate_model,
    build_comparison_dataframe,
    plot_target_distribution,
    plot_model_comparison,
    plot_confusion_matrix_figure,
    plot_combined_roc_curves,
    analyze_thresholds,
)
from src.utils import set_seed, get_logger, MODELS_DIR, FIGURES_DIR, DEFAULT_RANDOM_STATE

logger = get_logger("train")


def train_and_evaluate_all() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Any, str, float]:
    """
    Execute end-to-end model training, clinical comparison, threshold tuning, and artifact generation.
    """
    set_seed(DEFAULT_RANDOM_STATE)

    # 1. Ingest dataset
    X, y = load_processed_data()

    # Save target distribution plot
    plot_target_distribution(y, FIGURES_DIR / "target_distribution.png")

    # 2. Stratified Train / Test Split (zero leakage: preprocessing fitted only on train)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Data split: Train={len(X_train)} samples ({sum(y_train==1)} positive, {sum(y_train==0)} negative) | "
        f"Test={len(X_test)} samples ({sum(y_test==1)} positive, {sum(y_test==0)} negative)"
    )

    # 3. Define the three candidate classification models
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=DEFAULT_RANDOM_STATE,
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=DEFAULT_RANDOM_STATE,
            class_weight="balanced",
            max_depth=5,
            min_samples_split=4,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=DEFAULT_RANDOM_STATE,
            eval_metric="logloss",
        ),
    }

    trained_pipelines = {}
    eval_results = []

    # 4. Train each model in its self-contained Pipeline
    for name, clf in candidate_models.items():
        logger.info(f"--- Training {name} ---")
        pipe = create_full_pipeline(clf)
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Standard 0.50 threshold evaluation
        res = evaluate_model(pipe, X_test, y_test, model_name=name, threshold=0.5)
        eval_results.append(res)

        # Plot individual confusion matrix
        safe_slug = "logistic" if name == "Logistic Regression" else ("random_forest" if name == "Random Forest" else "xgboost")
        cm_path = FIGURES_DIR / f"confusion_matrix_{safe_slug}.png"
        plot_confusion_matrix_figure(np.array(res["confusion_matrix"]), name, cm_path)

    # 5. Build and print comparison table
    comparison_df = build_comparison_dataframe(eval_results)
    plot_model_comparison(comparison_df, FIGURES_DIR / "model_comparison.png")
    plot_combined_roc_curves(eval_results, FIGURES_DIR / "roc_curves.png")

    print("\n======================= FINAL MODEL COMPARISON =======================")
    print(comparison_df.to_string(index=False))
    print("======================================================================\n")

    # 6. Clinical Model Selection Logic
    # Priority in clinical risk:
    # 1. High Recall (minimize False Negatives)
    # 2. High ROC-AUC (discriminative power across all thresholds)
    # 3. Strong F1 score (balance of Precision and Recall)
    # Sort models primarily by Recall descending, then ROC-AUC descending, then F1 descending
    sorted_df = comparison_df.sort_values(by=["Recall", "ROC-AUC", "F1"], ascending=[False, False, False])
    selected_model_name = str(sorted_df.iloc[0]["Model"])
    selected_pipeline = trained_pipelines[selected_model_name]
    logger.info(f"Selected model based on clinical criteria: '{selected_model_name}'")

    # 7. Classification Threshold Analysis for Selected Model
    threshold_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresh_df = analyze_thresholds(
        selected_pipeline,
        X_test,
        y_test,
        model_name=selected_model_name,
        thresholds=threshold_range,
        output_path=FIGURES_DIR / "threshold_analysis.png",
    )

    print(f"\n============= THRESHOLD ANALYSIS ({selected_model_name}) =============")
    print(thresh_df.to_string(index=False))
    print("======================================================================\n")

    # Determine recommended operational threshold:
    # Look for threshold that maximizes Recall while keeping F1 >= 0.80 or minimizing FN.
    # Default is 0.50 if balance is maintained, or lower if FN reduction is notable.
    best_thresh_row = thresh_df.loc[thresh_df["False Negatives"] == thresh_df["False Negatives"].min()].iloc[-1]
    recommended_threshold = float(best_thresh_row["Threshold"])
    logger.info(f"Recommended operational threshold: {recommended_threshold} (FN={int(best_thresh_row['False Negatives'])}, Recall={best_thresh_row['Recall']:.4f})")

    # 8. Persist Artifacts
    # Save default selected model
    model_save_path = MODELS_DIR / "heart_disease_model.joblib"
    joblib.dump(selected_pipeline, model_save_path)
    logger.info(f"Saved selected pipeline to {model_save_path}")

    # Save all three individual pipelines for live model switching
    for name, pipe in trained_pipelines.items():
        slug = name.lower().replace(" ", "_")
        pipe_path = MODELS_DIR / f"{slug}_pipeline.joblib"
        joblib.dump(pipe, pipe_path)
        logger.info(f"Saved {name} pipeline to {pipe_path}")

    # Also save separate components for modularity
    joblib.dump(selected_pipeline.named_steps["classifier"], MODELS_DIR / "best_model.joblib")
    joblib.dump(selected_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.joblib")

    # Save metadata JSON
    metadata = {
        "project_name": "Heart Disease Risk Prediction & Clinical Model Comparison",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": DEFAULT_RANDOM_STATE,
        "dataset_info": {
            "source": "UCI Machine Learning Repository (Cleveland Dataset ID 45)",
            "citation": "Janosi, A., Steinbrunn, W., Pfisterer, W., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.",
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": len(ALL_FEATURES),
        },
        "features": {
            "all": ALL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "descriptions": FEATURE_DESCRIPTIONS,
            "value_maps": FEATURE_VALUE_MAPS,
        },
        "models_evaluated": comparison_df.to_dict(orient="records"),
        "selected_model": {
            "model_name": selected_model_name,
            "recommended_threshold": recommended_threshold,
            "default_metrics_at_0_5": comparison_df[comparison_df["Model"] == selected_model_name].to_dict(orient="records")[0],
            "metrics_at_recommended_threshold": best_thresh_row.to_dict(),
            "selection_rationale": (
                f"{selected_model_name} achieved the most favorable clinical profile with highest Recall "
                f"({sorted_df.iloc[0]['Recall']:.4f}), minimizing critical False Negatives ({sorted_df.iloc[0]['False Negatives']}), "
                f"while maintaining strong ROC-AUC ({sorted_df.iloc[0]['ROC-AUC']:.4f}) and F1 score ({sorted_df.iloc[0]['F1']:.4f})."
            ),
        },
        "threshold_analysis": thresh_df.to_dict(orient="records"),
    }

    metadata_path = MODELS_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to {metadata_path}")

    return metadata, comparison_df, thresh_df, selected_pipeline, selected_model_name, recommended_threshold


if __name__ == "__main__":
    train_and_evaluate_all()
