"""
meta_learning.py
================
Reference meta-learning (stacking) model using TF-IDF features.

Base learners:
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest

Uses out-of-fold predictions to train the meta learner to prevent data leakage.
The meta learner combines base learner probability outputs as meta-features.
"""

import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.evaluation import evaluate_model, print_results, save_results
from src.evaluation import save_confusion_matrix


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_reference_base_learners(config):
    """
    Build the base learners for the reference meta-learning experiment.
    
    These are different from the baselines to show the meta-learning concept
    with diverse classifiers.
    """
    seed = config["random_seed"]
    mc = config["models"]
    
    base_learners = {
        "Logistic Regression": LogisticRegression(
            max_iter=mc["logistic_regression"]["max_iter"],
            C=mc["logistic_regression"]["C"],
            solver=mc["logistic_regression"]["solver"],
            multi_class=mc["logistic_regression"]["multi_class"],
            class_weight=mc["logistic_regression"]["class_weight"],
            random_state=seed, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=mc["decision_tree"]["max_depth"],
            min_samples_split=mc["decision_tree"]["min_samples_split"],
            class_weight=mc["decision_tree"]["class_weight"],
            random_state=seed
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=mc["knn"]["n_neighbors"],
            weights=mc["knn"]["weights"],
            metric=mc["knn"]["metric"],
            n_jobs=mc["knn"]["n_jobs"]
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
            random_state=seed,
            n_jobs=mc["random_forest"]["n_jobs"]
        ),
    }
    
    return base_learners


def generate_out_of_fold_predictions(X_train, y_train, base_learners, n_folds, 
                                      n_classes, random_seed):
    """
    Generate out-of-fold predictions for training the meta learner.
    
    IMPORTANT: This prevents data leakage by ensuring the meta learner
    never sees predictions from models trained on the same samples.
    
    For each fold:
    1. Train each base learner on the training fold
    2. Predict probabilities on the validation fold
    3. Collect these out-of-fold predictions as meta-features
    
    Args:
        X_train: training features (sparse matrix or array)
        y_train: training labels
        base_learners: dict of base learner instances
        n_folds: number of CV folds
        n_classes: number of emotion classes
        random_seed: for reproducibility
    
    Returns:
        meta_features: array of shape [n_train, n_base_learners * n_classes]
        trained_base_learners: dict of base learners trained on full training data
    """
    n_samples = X_train.shape[0]
    n_learners = len(base_learners)
    
    # Initialize meta-feature matrix
    meta_features = np.zeros((n_samples, n_learners * n_classes))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    print(f"  Generating out-of-fold predictions ({n_folds} folds, {n_learners} base learners)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold_idx + 1}/{n_folds}...")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        
        for learner_idx, (name, learner) in enumerate(base_learners.items()):
            # Clone the learner for this fold
            from sklearn.base import clone
            fold_learner = clone(learner)
            
            # Train on fold training data
            fold_learner.fit(X_fold_train, y_fold_train)
            
            # Predict probabilities on fold validation data
            if hasattr(fold_learner, "predict_proba"):
                probs = fold_learner.predict_proba(X_fold_val)
            else:
                # For models without predict_proba, use one-hot predictions
                preds = fold_learner.predict(X_fold_val)
                probs = np.zeros((len(preds), n_classes))
                probs[np.arange(len(preds)), preds] = 1.0
            
            # Handle case where fold doesn't have all classes
            if probs.shape[1] < n_classes:
                padded_probs = np.zeros((probs.shape[0], n_classes))
                padded_probs[:, :probs.shape[1]] = probs
                probs = padded_probs
            
            # Store out-of-fold predictions
            start_col = learner_idx * n_classes
            end_col = start_col + n_classes
            meta_features[val_idx, start_col:end_col] = probs
    
    # Train base learners on full training data for test-time predictions
    print("  Training base learners on full training data...")
    trained_base_learners = {}
    for name, learner in base_learners.items():
        from sklearn.base import clone
        full_learner = clone(learner)
        full_learner.fit(X_train, y_train)
        trained_base_learners[name] = full_learner
    
    return meta_features, trained_base_learners


def generate_test_meta_features(X_test, trained_base_learners, n_classes):
    """
    Generate meta-features for test/validation data.
    
    Uses base learners trained on the full training set.
    
    Args:
        X_test: test features
        trained_base_learners: dict of trained base learners
        n_classes: number of classes
    
    Returns:
        meta_features: array of shape [n_test, n_base_learners * n_classes]
    """
    n_samples = X_test.shape[0]
    n_learners = len(trained_base_learners)
    meta_features = np.zeros((n_samples, n_learners * n_classes))
    
    for learner_idx, (name, learner) in enumerate(trained_base_learners.items()):
        if hasattr(learner, "predict_proba"):
            probs = learner.predict_proba(X_test)
        else:
            preds = learner.predict(X_test)
            probs = np.zeros((len(preds), n_classes))
            probs[np.arange(len(preds)), preds] = 1.0
        
        if probs.shape[1] < n_classes:
            padded_probs = np.zeros((probs.shape[0], n_classes))
            padded_probs[:, :probs.shape[1]] = probs
            probs = padded_probs
        
        start_col = learner_idx * n_classes
        end_col = start_col + n_classes
        meta_features[:, start_col:end_col] = probs
    
    return meta_features


def train_reference_meta_learning(X_train, y_train, X_test, y_test,
                                   label_names, config,
                                   save_dir="results/reference_meta_learning"):
    """
    Train and evaluate the reference meta-learning model using TF-IDF features.
    
    Args:
        X_train, y_train: training data (TF-IDF)
        X_test, y_test: test data (TF-IDF)
        label_names: list of emotion class names
        config: configuration dict
        save_dir: where to save results
    
    Returns:
        result dict, trained meta learner, trained base learners
    """
    os.makedirs(save_dir, exist_ok=True)
    
    seed = config["random_seed"]
    n_folds = config["meta_learning"]["n_folds"]
    n_classes = len(label_names)
    
    print(f"\n{'='*60}")
    print("REFERENCE META-LEARNING (TF-IDF)")
    print(f"{'='*60}")
    
    # Build base learners
    base_learners = build_reference_base_learners(config)
    print(f"  Base learners: {list(base_learners.keys())}")
    
    # Generate out-of-fold predictions for meta-feature training
    meta_features_train, trained_base_learners = generate_out_of_fold_predictions(
        X_train, y_train, base_learners, n_folds, n_classes, seed
    )
    
    print(f"  Meta-features shape (train): {meta_features_train.shape}")
    
    # Generate test meta-features
    meta_features_test = generate_test_meta_features(
        X_test, trained_base_learners, n_classes
    )
    print(f"  Meta-features shape (test): {meta_features_test.shape}")
    
    # Train meta learner
    meta_params = config["meta_learning"]["meta_learner_params"]
    meta_learner = LogisticRegression(
        max_iter=meta_params["max_iter"],
        C=meta_params["C"],
        solver=meta_params["solver"],
        multi_class=meta_params["multi_class"],
        random_state=seed,
        n_jobs=-1
    )
    
    print("  Training meta learner...")
    meta_learner.fit(meta_features_train, y_train)
    
    # Predict
    y_pred = meta_learner.predict(meta_features_test)
    
    # Evaluate
    results = evaluate_model(y_test, y_pred, label_names,
                              model_name="TF-IDF + Meta-Learning (Reference)")
    print_results(results)
    
    # Save results
    save_results(results, save_dir)
    save_confusion_matrix(
        y_test, y_pred, label_names,
        os.path.join(save_dir, "reference_meta_learning_cm.png"),
        title="Reference Meta-Learning (TF-IDF) - Confusion Matrix"
    )
    
    # Save models
    joblib.dump(meta_learner, os.path.join(save_dir, "meta_learner.pkl"))
    for name, learner in trained_base_learners.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(learner, os.path.join(save_dir, f"base_learner_{safe_name}.pkl"))
    
    # Save meta-learning info
    info = {
        "base_learners": list(base_learners.keys()),
        "meta_learner": config["meta_learning"]["meta_learner"],
        "n_folds": n_folds,
        "meta_features_dim": meta_features_train.shape[1],
        "n_classes": n_classes,
        "out_of_fold": True,
        "leakage_prevention": "Used out-of-fold predictions for meta-feature generation"
    }
    with open(os.path.join(save_dir, "meta_learning_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Reference meta-learning results saved to {save_dir}")
    
    return results, meta_learner, trained_base_learners
