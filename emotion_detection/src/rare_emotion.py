"""
rare_emotion.py
===============
Rare emotion / few-shot analysis module.

Analyzes:
1. Performance across emotion resource groups (high/medium/low)
2. Few-shot experiments: how performance changes with training data size
3. Whether BERT + Meta-Learning helps low-resource emotions
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
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from tqdm import tqdm

from src.evaluation import evaluate_model


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_per_group_performance(all_results, label_names, emotion_groups,
                                    save_dir="results/rare_emotion"):
    """
    Compare model performance across emotion resource groups.
    
    Args:
        all_results: list of result dicts (each containing classification_report)
        label_names: list of emotion class names
        emotion_groups: dict with 'high_resource', 'medium_resource', 'low_resource'
        save_dir: where to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Build emotion → group mapping
    emotion_to_group = {}
    for group_name, emotions in emotion_groups.items():
        for e in emotions:
            if isinstance(e, dict):
                emotion_to_group[e["emotion"]] = group_name
            else:
                emotion_to_group[e] = group_name
    
    group_results = []
    
    for result in all_results:
        model_name = result["model_name"]
        report = result.get("classification_report", {})
        
        for group_name in ["high_resource", "medium_resource", "low_resource"]:
            group_f1s = []
            group_precisions = []
            group_recalls = []
            
            for emotion in label_names:
                if emotion_to_group.get(emotion) == group_name and emotion in report:
                    metrics = report[emotion]
                    group_f1s.append(metrics.get("f1-score", 0))
                    group_precisions.append(metrics.get("precision", 0))
                    group_recalls.append(metrics.get("recall", 0))
            
            if group_f1s:
                group_results.append({
                    "Model": model_name,
                    "Resource Group": group_name.replace("_", " ").title(),
                    "Avg F1": float(np.mean(group_f1s)),
                    "Avg Precision": float(np.mean(group_precisions)),
                    "Avg Recall": float(np.mean(group_recalls)),
                    "Num Emotions": len(group_f1s)
                })
    
    group_df = pd.DataFrame(group_results)
    group_df.to_csv(os.path.join(save_dir, "per_group_performance.csv"), index=False)
    
    # Plot comparison
    if not group_df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = group_df["Model"].unique()
        groups = ["High Resource", "Medium Resource", "Low Resource"]
        x = np.arange(len(groups))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = group_df[group_df["Model"] == model]
            f1_values = []
            for g in groups:
                row = model_data[model_data["Resource Group"] == g]
                f1_values.append(row["Avg F1"].values[0] if len(row) > 0 else 0)
            
            ax.bar(x + i * width, f1_values, width, label=model, alpha=0.8)
        
        ax.set_xticks(x + width * len(models) / 2)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Average F1 Score")
        ax.set_title("Model Performance by Emotion Resource Group")
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "per_group_performance.png"), dpi=150)
        plt.close()
    
    print(f"  ✓ Per-group performance analysis saved to {save_dir}")
    return group_df


def run_few_shot_experiment(X_train, y_train, X_test, y_test,
                             label_names, config,
                             few_shot_sizes=None,
                             save_dir="results/rare_emotion"):
    """
    Few-shot experiment: train with limited data per emotion and observe performance.
    
    Samples are drawn per-class to ensure representation of all emotions.
    
    Args:
        X_train: training features (BERT embeddings)
        y_train: training labels
        X_test: test features
        y_test: test labels
        label_names: list of emotion names
        config: config dict
        few_shot_sizes: list of per-class sample sizes to test
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if few_shot_sizes is None:
        few_shot_sizes = config["rare_emotion"]["few_shot_sizes"]
    
    seed = config["random_seed"]
    np.random.seed(seed)
    
    # Include "Full" as one of the sizes
    all_sizes = few_shot_sizes + ["Full"]
    
    results_list = []
    
    for size in tqdm(all_sizes, desc="Few-shot experiment"):
        if size == "Full":
            X_sub = X_train
            y_sub = y_train
            actual_size = len(y_train)
        else:
            # Sample min(size, available) per class
            indices = []
            for cls in range(len(label_names)):
                cls_indices = np.where(y_train == cls)[0]
                n_sample = min(size, len(cls_indices))
                if n_sample > 0:
                    sampled = np.random.choice(cls_indices, n_sample, replace=False)
                    indices.extend(sampled)
            
            indices = np.array(indices)
            X_sub = X_train[indices]
            y_sub = y_train[indices]
            actual_size = len(indices)
        
        # Train logistic regression using config parameters
        lr_cfg = config.get("models", {}).get("logistic_regression", {})
        model = LogisticRegression(
            max_iter=lr_cfg.get("max_iter", 2000),
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_test)
        
        result = evaluate_model(y_test, y_pred, label_names,
                                 model_name=f"LR (n={actual_size})")
        
        results_list.append({
            "Size Per Class": str(size),
            "Total Samples": actual_size,
            "Accuracy": result["accuracy"],
            "Macro F1": result["f1_macro"],
            "Weighted F1": result["f1_weighted"],
        })
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(save_dir, "few_shot_results.csv"), index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_labels = [str(s) for s in results_df["Size Per Class"]]
    
    axes[0].plot(range(len(x_labels)), results_df["Macro F1"].values, 
                 "o-", color="#e74c3c", linewidth=2, markersize=8)
    axes[0].set_xticks(range(len(x_labels)))
    axes[0].set_xticklabels(x_labels)
    axes[0].set_xlabel("Samples per Class")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Few-Shot: Macro F1 vs Training Data Size")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(len(x_labels)), results_df["Accuracy"].values,
                 "s-", color="#3498db", linewidth=2, markersize=8)
    axes[1].set_xticks(range(len(x_labels)))
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel("Samples per Class")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Few-Shot: Accuracy vs Training Data Size")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "few_shot_results.png"), dpi=150)
    plt.close()
    
    print(f"  ✓ Few-shot experiment results saved to {save_dir}")
    return results_df
