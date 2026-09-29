"""
error_analysis.py
=================
Detailed error analysis of model predictions.

Analyzes:
- Frequently confused emotion pairs
- Semantically similar emotions that get confused
- Rare emotion misclassifications
- Short/ambiguous utterance errors
- Generates error analysis CSV with examples
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_error_analysis(texts, y_true, y_pred, y_proba, label_names,
                        model_name="Model", save_dir="results/error_analysis"):
    """
    Comprehensive error analysis of model predictions.
    
    Args:
        texts: list of original text strings
        y_true: true labels (encoded integers)
        y_pred: predicted labels (encoded integers)
        y_proba: prediction probabilities [n_samples, n_classes] or None
        label_names: list of emotion class names
        model_name: name of the model
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    # 1. Build error DataFrame
    error_df = _build_error_dataframe(texts, y_true, y_pred, y_proba, 
                                       label_names, model_name)
    
    # 2. Confusion pair analysis
    confusion_pairs = _analyze_confusion_pairs(y_true, y_pred, label_names)
    
    # 3. Error examples
    error_examples = error_df[error_df["correct"] == False].copy()
    print(f"  Total errors: {len(error_examples)} / {len(error_df)}")
    print(f"  Error rate: {len(error_examples)/len(error_df)*100:.1f}%")
    
    # 4. Short utterance errors
    error_examples["word_count"] = error_examples["text"].apply(
        lambda x: len(str(x).split()))
    short_errors = error_examples[error_examples["word_count"] <= 5]
    print(f"  Short utterance errors (≤5 words): {len(short_errors)}")
    
    # 5. Save error analysis CSV
    error_csv_cols = ["text", "true_emotion", "predicted_emotion", 
                      "confidence", "model", "word_count"]
    error_examples[error_csv_cols].to_csv(
        os.path.join(save_dir, f"error_analysis_{model_name.lower().replace(' ', '_')}.csv"),
        index=False
    )
    
    # 6. Top confusion pairs
    pairs_df = pd.DataFrame(confusion_pairs[:20], 
                             columns=["True", "Predicted", "Count"])
    pairs_df.to_csv(os.path.join(save_dir, "top_confusion_pairs.csv"), index=False)
    
    print(f"\n  Top 10 confusion pairs:")
    for true_e, pred_e, count in confusion_pairs[:10]:
        print(f"    {true_e:20s} → {pred_e:20s}: {count}")
    
    # 7. Plot confusion heatmap for top confused pairs
    _plot_confusion_heatmap(y_true, y_pred, label_names, save_dir, model_name)
    
    # 8. Error statistics by emotion
    error_by_emotion = _error_rate_by_emotion(y_true, y_pred, label_names)
    error_by_emotion.to_csv(os.path.join(save_dir, "error_rate_by_emotion.csv"))
    
    print(f"\n  ✓ Error analysis saved to {save_dir}")
    
    return error_df, confusion_pairs


def _build_error_dataframe(texts, y_true, y_pred, y_proba, label_names, model_name):
    """Build a DataFrame with all predictions and error indicators."""
    data = {
        "text": texts,
        "true_emotion": [label_names[y] for y in y_true],
        "predicted_emotion": [label_names[y] for y in y_pred],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "model": model_name,
    }
    
    if y_proba is not None:
        data["confidence"] = [float(y_proba[i, y_pred[i]]) for i in range(len(y_pred))]
    else:
        data["confidence"] = [1.0 if c else 0.0 for c in data["correct"]]
    
    return pd.DataFrame(data)


def _analyze_confusion_pairs(y_true, y_pred, label_names):
    """Find the most common confusion pairs."""
    cm = confusion_matrix(y_true, y_pred)
    
    pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((label_names[i], label_names[j], int(cm[i, j])))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _plot_confusion_heatmap(y_true, y_pred, label_names, save_dir, model_name):
    """Plot a detailed confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_norm, annot=False, cmap="YlOrRd",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted Emotion", fontsize=12)
    ax.set_ylabel("True Emotion", fontsize=12)
    ax.set_title(f"Error Analysis Heatmap - {model_name}", fontsize=14)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_heatmap.png"), dpi=150)
    plt.close()


def _error_rate_by_emotion(y_true, y_pred, label_names):
    """Calculate error rate for each emotion class."""
    rows = []
    for i, emotion in enumerate(label_names):
        mask = y_true == i
        if mask.sum() > 0:
            correct = (y_pred[mask] == i).sum()
            total = mask.sum()
            error_rate = 1.0 - (correct / total)
            rows.append({
                "Emotion": emotion,
                "Total": int(total),
                "Correct": int(correct),
                "Errors": int(total - correct),
                "Error Rate": float(error_rate),
                "Accuracy": float(correct / total)
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Error Rate", ascending=False)
    return df
