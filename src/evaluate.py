"""
Evaluation metrics, clinical trade-off analysis, confusion matrices, and ROC visualizations.
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.utils import FIGURES_DIR, get_logger

logger = get_logger("evaluate")

# Set standard plotting style for clean, publication-ready figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def evaluate_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate performance metrics with clinical focus on Recall and False Negatives.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "y_proba": y_proba.tolist(),
    }

    logger.info(
        f"[{model_name}] Recall: {recall:.4f} | False Negatives: {fn} | Precision: {precision:.4f} | "
        f"F1: {f1:.4f} | Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}"
    )

    return results


def build_comparison_dataframe(eval_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format model evaluation results into the standard comparison table.
    """
    rows = []
    for res in eval_results:
        rows.append(
            {
                "Model": res["model_name"],
                "Accuracy": round(res["accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1": round(res["f1"], 4),
                "ROC-AUC": round(res["roc_auc"], 4),
                "False Negatives": res["false_negatives"],
                "False Positives": res["false_positives"],
            }
        )
    return pd.DataFrame(rows)


def plot_target_distribution(y: pd.Series, output_path: Path = FIGURES_DIR / "target_distribution.png") -> None:
    """Plot binary target class counts and proportions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    labels = ["0: No Heart Disease", "1: Heart Disease Present"]
    colors = ["#2b5c8f", "#d9534f"]

    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count} ({count/len(y):.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("UCI Heart Disease - Binary Target Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) + 25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved target distribution figure to {output_path}")


def plot_model_comparison(results_df: pd.DataFrame, output_path: Path = FIGURES_DIR / "model_comparison.png") -> None:
    """Create a grouped bar chart comparing all models across key performance metrics."""
    metrics = ["Recall", "F1", "Precision", "ROC-AUC", "Accuracy"]
    melted = results_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    palette = ["#2b5c8f", "#e67e22", "#27ae60"]
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=palette)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
                fontweight="semibold",
            )

    plt.title("Clinical Model Performance Comparison (Emphasis on Recall & F1)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.0, 1.08)
    plt.ylabel("Score", fontsize=11)
    plt.legend(title="Model", frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison figure to {output_path}")


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Generate and save an annotated confusion matrix figure."""
    plt.figure(figsize=(6, 5))
    annot_labels = np.array(
        [
            [f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"],
            [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"],
        ]
    )
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}\n(Clinical Focus: Minimizing False Negatives)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure for {model_name} to {output_path}")


def plot_combined_roc_curves(
    eval_results: List[Dict[str, Any]],
    output_path: Path = FIGURES_DIR / "roc_curves.png",
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curves for all models on one figure."""
    plt.figure(figsize=(8, 6))
    colors = ["#2b5c8f", "#e67e22", "#27ae60"]

    for res, color in zip(eval_results, colors):
        fpr = res["fpr"]
        tpr = res["tpr"]
        auc = res["roc_auc"]
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{res['model_name']} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves figure to {output_path}")


def analyze_thresholds(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    thresholds: List[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    output_path: Path = FIGURES_DIR / "threshold_analysis.png",
) -> pd.DataFrame:
    """
    Evaluate candidate decision thresholds to assess clinical trade-offs between
    Recall (Sensitivity) and Precision (Positive Predictive Value), tracking False Negatives.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    records = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        records.append(
            {
                "Threshold": round(t, 2),
                "Recall": round(rec, 4),
                "Precision": round(prec, 4),
                "F1": round(f1, 4),
                "Accuracy": round(acc, 4),
                "False Negatives": fn,
                "False Positives": fp,
                "True Positives": tp,
                "True Negatives": tn,
            }
        )

    thresh_df = pd.DataFrame(records)

    # Plot threshold trade-offs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Metric Curves
    ax1.plot(thresh_df["Threshold"], thresh_df["Recall"], marker="o", color="#d9534f", lw=2.5, label="Recall (Sensitivity)")
    ax1.plot(thresh_df["Threshold"], thresh_df["Precision"], marker="s", color="#2b5c8f", lw=2, label="Precision")
    ax1.plot(thresh_df["Threshold"], thresh_df["F1"], marker="^", color="#27ae60", lw=2, label="F1-Score")
    ax1.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
    ax1.set_xlabel("Probability Decision Threshold", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Metric Value", fontsize=11, fontweight="semibold")
    ax1.set_title(f"{model_name}: Precision vs Recall vs F1 across Thresholds", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(frameon=True, loc="lower left")

    # Error Trade-off (False Negatives vs False Positives)
    ax2.plot(thresh_df["Threshold"], thresh_df["False Negatives"], marker="o", color="#d9534f", lw=2.5, label="False Negatives (Missed Disease)")
    ax2.plot(thresh_df["Threshold"], thresh_df["False Positives"], marker="s", color="#e67e22", lw=2, label="False Positives (False Alarms)")
    ax2.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
    ax2.set_xlabel("Probability Decision Threshold", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="semibold")
    ax2.set_title(f"{model_name}: Clinical Risk Trade-off (FN vs FP)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved threshold analysis figure to {output_path}")

    return thresh_df
