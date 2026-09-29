"""
evaluation.py
=============
Unified evaluation module for all models in the research project.

Computes:
- Accuracy, Precision, Recall, Macro F1, Weighted F1
- Per-class Precision, Recall, F1
- Confusion Matrix
- Classification Reports

Saves results in a standardized format for easy comparison.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def evaluate_model(y_true, y_pred, label_names=None, model_name="Model"):
    """
    Evaluate a model's predictions with comprehensive metrics.
    
    Args:
        y_true: true labels (can be string or integer)
        y_pred: predicted labels
        label_names: list of label names for reporting
        model_name: name of the model for display
    
    Returns:
        dict with all metrics
    """
    results = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=label_names,
                                    output_dict=True, zero_division=0)
    results["classification_report"] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()
    
    return results


def print_results(results):
    """Print evaluation results in a formatted way."""
    print(f"\n--- {results['model_name']} ---")
    print(f"  Accuracy:          {results['accuracy']:.4f}")
    print(f"  Precision (Macro): {results['precision_macro']:.4f}")
    print(f"  Recall (Macro):    {results['recall_macro']:.4f}")
    print(f"  F1 (Macro):        {results['f1_macro']:.4f}")
    print(f"  F1 (Weighted):     {results['f1_weighted']:.4f}")


def save_results(results, save_dir, model_name=None):
    """Save evaluation results to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    name = model_name or results.get("model_name", "model")
    name = name.lower().replace(" ", "_").replace("+", "_")
    
    # Save metrics JSON
    metrics_to_save = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open(os.path.join(save_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    
    # Save classification report as CSV
    if "classification_report" in results:
        report_df = pd.DataFrame(results["classification_report"]).transpose()
        report_df.to_csv(os.path.join(save_dir, f"{name}_classification_report.csv"))
    
    print(f"  ✓ Results saved to {save_dir}/{name}_*")


def save_confusion_matrix(y_true, y_pred, label_names, save_path, title="Confusion Matrix"):
    """Save confusion matrix as a heatmap image."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for better visualization
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_normalized, annot=False, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_comparison_table(all_results, save_path):
    """
    Create a final comparison table across all models.
    
    Args:
        all_results: list of result dicts from evaluate_model
        save_path: path to save the comparison CSV
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision (Macro)": r["precision_macro"],
            "Recall (Macro)": r["recall_macro"],
            "Macro F1": r["f1_macro"],
            "Weighted F1": r["f1_weighted"],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Macro F1", ascending=False)
    df.to_csv(save_path, index=False)
    
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)
    
    return df


def format_cv_results(cv_scores, model_name):
    """Format cross-validation results with mean ± std."""
    result = {"model_name": model_name}
    for metric_name, scores in cv_scores.items():
        result[f"{metric_name}_mean"] = float(np.mean(scores))
        result[f"{metric_name}_std"] = float(np.std(scores))
        result[f"{metric_name}_formatted"] = f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
    return result
