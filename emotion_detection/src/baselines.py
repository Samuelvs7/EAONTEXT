"""
baselines.py
============
Classical baseline models using TF-IDF features.

Models:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Naive Bayes (MultinomialNB for TF-IDF)
4. Random Forest

Each model is trained, evaluated, and saved. Results include per-class metrics,
confusion matrices, and a comparison table.
"""

import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.evaluation import evaluate_model, print_results, save_results
from src.evaluation import save_confusion_matrix


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_baseline_models(config):
    """
    Build all baseline classifier instances.
    
    Returns:
        dict mapping model name to (sklearn model instance)
    """
    mc = config["models"]
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=mc["logistic_regression"]["max_iter"],
            C=mc["logistic_regression"]["C"],
            solver=mc["logistic_regression"]["solver"],
            class_weight=mc["logistic_regression"]["class_weight"],
            random_state=config["random_seed"],
            n_jobs=-1
        ),
        "SVM": SVC(
            C=mc["svm"]["C"],
            kernel=mc["svm"]["kernel"],
            probability=mc["svm"]["probability"],
            class_weight=mc["svm"]["class_weight"],
            max_iter=mc["svm"]["max_iter"],
            random_state=config["random_seed"]
        ),
        "Naive Bayes": MultinomialNB(
            alpha=mc["naive_bayes"]["alpha"]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_split=mc["random_forest"]["min_samples_split"],
            min_samples_leaf=mc["random_forest"]["min_samples_leaf"],
            class_weight=mc["random_forest"]["class_weight"],
            random_state=config["random_seed"],
            n_jobs=mc["random_forest"]["n_jobs"]
        ),
    }
    
    return models


def train_and_evaluate_baselines(X_train, y_train, X_test, y_test,
                                  label_names, config, save_dir="results/baselines"):
    """
    Train all baseline models and evaluate them.
    
    Args:
        X_train: TF-IDF training features
        y_train: training labels (encoded)
        X_test: TF-IDF test features
        y_test: test labels (encoded)
        label_names: list of emotion class names
        config: configuration dict
        save_dir: directory to save results
    
    Returns:
        list of result dicts, dict of trained models
    """
    os.makedirs(save_dir, exist_ok=True)
    models = build_baseline_models(config)
    
    all_results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: TF-IDF + {model_name}")
        print(f"{'='*50}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        results = evaluate_model(y_test, y_pred, label_names,
                                  model_name=f"TF-IDF + {model_name}")
        print_results(results)
        
        # Save results
        save_results(results, save_dir, model_name=f"tfidf_{model_name}")
        
        # Save confusion matrix
        save_confusion_matrix(
            y_test, y_pred, label_names,
            os.path.join(save_dir, f"tfidf_{model_name.lower().replace(' ', '_')}_cm.png"),
            title=f"TF-IDF + {model_name} - Confusion Matrix"
        )
        
        # Save trained model
        model_path = os.path.join(save_dir, f"tfidf_{model_name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)
        
        all_results.append(results)
        trained_models[model_name] = model
    
    # Save comparison CSV
    comparison_rows = []
    for r in all_results:
        comparison_rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision_Macro": r["precision_macro"],
            "Recall_Macro": r["recall_macro"],
            "Macro_F1": r["f1_macro"],
            "Weighted_F1": r["f1_weighted"],
        })
    
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(save_dir, "baseline_comparison.csv"), index=False)
    
    print(f"\n✓ All baseline results saved to {save_dir}")
    
    return all_results, trained_models
