"""
hybrid_model.py
===============
Proposed Hybrid Architecture: BERT + Meta-Learning Ensembles

Architecture:
    Conversation Text
          ↓
    BERT (frozen, pretrained)
          ↓
    Sentence Embedding (mean pooling)
          ↓
    ┌──────────────────────────┐
    │ Logistic Regression      │
    │ SVM                      │
    │ Random Forest            │
    │ Naive Bayes (Gaussian)   │
    └──────────────────────────┘
          ↓
    Prediction Probabilities
          ↓
    Meta-Features
          ↓
    Meta Learner (Logistic Regression)
          ↓
    Final Emotion Prediction

CRITICAL: Leakage prevention via out-of-fold predictions.
- For training: generate out-of-fold predictions from each base learner
- For validation/test: train base learners on full training data, then predict
"""

import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from tqdm import tqdm

from src.evaluation import evaluate_model, print_results, save_results
from src.evaluation import save_confusion_matrix


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_hybrid_base_learners(config):
    """
    Build base learners for the proposed BERT + Meta-Learning architecture.
    
    Uses GaussianNB instead of MultinomialNB because BERT embeddings
    can have negative values (MultinomialNB requires non-negative features).
    """
    seed = config["random_seed"]
    mc = config["models"]
    
    base_learners = {
        "Logistic Regression": LogisticRegression(
            max_iter=mc["logistic_regression"]["max_iter"],
            C=mc["logistic_regression"]["C"],
            solver=mc["logistic_regression"]["solver"],
            class_weight=mc["logistic_regression"]["class_weight"],
            random_state=seed, n_jobs=-1
        ),
        "SVM": SVC(
            C=mc["svm"]["C"],
            kernel=mc["svm"]["kernel"],
            probability=mc["svm"]["probability"],  # Enable probability estimates
            class_weight=mc["svm"]["class_weight"],
            max_iter=mc["svm"]["max_iter"],
            random_state=seed
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
        "Naive Bayes": GaussianNB(),  # GaussianNB for continuous BERT embeddings
    }
    
    return base_learners


def generate_hybrid_out_of_fold_predictions(X_train, y_train, base_learners, 
                                              n_folds, n_classes, random_seed):
    """
    Generate out-of-fold probabilities for the meta learner training.
    
    VERY IMPORTANT (leakage prevention):
    - The meta learner is NEVER trained on predictions from models 
      that saw the same training examples.
    - Each sample's meta-features come from models trained WITHOUT that sample.
    
    Args:
        X_train: BERT embeddings for training data [n_samples, embedding_dim]
        y_train: training labels
        base_learners: dict of base learner instances
        n_folds: number of CV folds
        n_classes: number of emotion classes
        random_seed: random seed
    
    Returns:
        meta_features_train: [n_train, n_learners * n_classes]
        trained_full_base_learners: base learners trained on full train set
    """
    n_samples = X_train.shape[0]
    n_learners = len(base_learners)
    
    meta_features = np.zeros((n_samples, n_learners * n_classes))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    print(f"  Generating out-of-fold predictions ({n_folds} folds, {n_learners} base learners)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"    Fold {fold_idx + 1}/{n_folds}...")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        
        for learner_idx, (name, learner) in enumerate(base_learners.items()):
            fold_learner = clone(learner)
            fold_learner.fit(X_fold_train, y_fold_train)
            
            # Get probability predictions
            if hasattr(fold_learner, "predict_proba"):
                probs = fold_learner.predict_proba(X_fold_val)
            else:
                preds = fold_learner.predict(X_fold_val)
                probs = np.zeros((len(preds), n_classes))
                for i, p in enumerate(preds):
                    if p < n_classes:
                        probs[i, p] = 1.0
            
            # Handle class mismatch
            if probs.shape[1] < n_classes:
                padded = np.zeros((probs.shape[0], n_classes))
                # Map fold classes to global classes
                if hasattr(fold_learner, "classes_"):
                    for ci, c in enumerate(fold_learner.classes_):
                        if c < n_classes:
                            padded[:, c] = probs[:, ci]
                else:
                    padded[:, :probs.shape[1]] = probs
                probs = padded
            
            start_col = learner_idx * n_classes
            end_col = start_col + n_classes
            meta_features[val_idx, start_col:end_col] = probs
    
    # Train base learners on full training data
    print("  Training base learners on full training data...")
    trained_full = {}
    for name, learner in tqdm(base_learners.items(), desc="  Full training"):
        full_learner = clone(learner)
        full_learner.fit(X_train, y_train)
        trained_full[name] = full_learner
    
    return meta_features, trained_full


def generate_hybrid_test_meta_features(X_test, trained_base_learners, n_classes):
    """
    Generate meta-features for validation/test data.
    Uses base learners trained on the full training data.
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
            for i, p in enumerate(preds):
                if p < n_classes:
                    probs[i, p] = 1.0
        
        if probs.shape[1] < n_classes:
            padded = np.zeros((probs.shape[0], n_classes))
            if hasattr(learner, "classes_"):
                for ci, c in enumerate(learner.classes_):
                    if c < n_classes:
                        padded[:, c] = probs[:, ci]
            else:
                padded[:, :probs.shape[1]] = probs
            probs = padded
        
        start_col = learner_idx * n_classes
        end_col = start_col + n_classes
        meta_features[:, start_col:end_col] = probs
    
    return meta_features


def train_hybrid_model(X_train, y_train, X_test, y_test,
                        label_names, config,
                        X_valid=None, y_valid=None,
                        save_dir="results/hybrid"):
    """
    Train and evaluate the proposed BERT + Meta-Learning hybrid model.
    
    Args:
        X_train: BERT embeddings for training [n_train, 768]
        y_train: training labels
        X_test: BERT embeddings for test [n_test, 768]
        y_test: test labels
        label_names: list of emotion class names
        config: configuration dict
        X_valid: optional validation BERT embeddings
        y_valid: optional validation labels
        save_dir: where to save results
    
    Returns:
        result dict, HybridModel instance
    """
    os.makedirs(save_dir, exist_ok=True)
    
    seed = config["random_seed"]
    n_folds = config["meta_learning"]["n_folds"]
    n_classes = len(label_names)
    
    print(f"\n{'='*60}")
    print("PROPOSED MODEL: BERT + META-LEARNING ENSEMBLE")
    print(f"{'='*60}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Embedding dim: {X_train.shape[1]}")
    print(f"  Classes: {n_classes}")
    
    # 1. Build base learners
    base_learners = build_hybrid_base_learners(config)
    print(f"  Base learners: {list(base_learners.keys())}")
    
    # 2. Generate out-of-fold meta-features (leakage-free)
    meta_features_train, trained_base_learners = generate_hybrid_out_of_fold_predictions(
        X_train, y_train, base_learners, n_folds, n_classes, seed
    )
    print(f"  Meta-features (train): {meta_features_train.shape}")
    
    # 3. Generate test meta-features
    meta_features_test = generate_hybrid_test_meta_features(
        X_test, trained_base_learners, n_classes
    )
    print(f"  Meta-features (test): {meta_features_test.shape}")
    
    # 4. Train meta learner
    meta_params = config["meta_learning"]["meta_learner_params"]
    meta_learner = LogisticRegression(
        max_iter=meta_params["max_iter"],
        C=meta_params["C"],
        solver=meta_params["solver"],
        random_state=seed,
        n_jobs=-1
    )
    
    print("  Training meta learner on out-of-fold meta-features...")
    meta_learner.fit(meta_features_train, y_train)
    
    # 5. Predict on test
    y_pred = meta_learner.predict(meta_features_test)
    
    # 6. Evaluate
    results = evaluate_model(y_test, y_pred, label_names,
                              model_name="BERT + Meta-Learning (Proposed)")
    print_results(results)
    
    # 7. Save results
    save_results(results, save_dir, model_name="bert_meta_learning")
    save_confusion_matrix(
        y_test, y_pred, label_names,
        os.path.join(save_dir, "bert_meta_learning_cm.png"),
        title="BERT + Meta-Learning (Proposed) - Confusion Matrix"
    )
    
    # Also evaluate on validation if provided
    if X_valid is not None and y_valid is not None:
        meta_features_valid = generate_hybrid_test_meta_features(
            X_valid, trained_base_learners, n_classes
        )
        y_pred_valid = meta_learner.predict(meta_features_valid)
        valid_results = evaluate_model(y_valid, y_pred_valid, label_names,
                                        model_name="BERT + Meta-Learning (Valid)")
        print_results(valid_results)
        save_results(valid_results, save_dir, model_name="bert_meta_learning_valid")
    
    # 8. Save models
    joblib.dump(meta_learner, os.path.join(save_dir, "hybrid_meta_learner.pkl"))
    for name, learner in trained_base_learners.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(learner, os.path.join(save_dir, f"hybrid_base_{safe_name}.pkl"))
    
    # 9. Save architecture info
    arch_info = {
        "architecture": "BERT + Meta-Learning Ensemble",
        "bert_model": config["bert"]["model_name"],
        "embedding_dim": config["bert"]["embedding_dim"],
        "pooling": config["bert"]["pooling_strategy"],
        "fine_tuned": False,
        "base_learners": list(base_learners.keys()),
        "meta_learner": "LogisticRegression",
        "n_folds_oof": n_folds,
        "n_classes": n_classes,
        "meta_features_dim": meta_features_train.shape[1],
        "leakage_prevention": "Out-of-fold predictions for meta-feature generation",
        "probability_based": True,
    }
    with open(os.path.join(save_dir, "architecture_info.json"), "w") as f:
        json.dump(arch_info, f, indent=2)
    
    print(f"\n✓ Hybrid model results saved to {save_dir}")
    
    return results, meta_learner, trained_base_learners


def train_bert_base_classifiers(X_train, y_train, X_test, y_test,
                                 label_names, config,
                                 save_dir="results/bert"):
    """
    Train individual classifiers on BERT embeddings (Step 7).
    
    This is separate from the meta-learning hybrid - each classifier
    is evaluated independently on BERT features.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    seed = config["random_seed"]
    mc = config["models"]
    
    models = {
        "BERT + Logistic Regression": LogisticRegression(
            max_iter=mc["logistic_regression"]["max_iter"],
            C=mc["logistic_regression"]["C"],
            solver=mc["logistic_regression"]["solver"],
            class_weight=mc["logistic_regression"]["class_weight"],
            random_state=seed, n_jobs=-1
        ),
        "BERT + SVM": SVC(
            C=mc["svm"]["C"],
            kernel=mc["svm"]["kernel"],
            probability=mc["svm"]["probability"],
            class_weight=mc["svm"]["class_weight"],
            max_iter=mc["svm"]["max_iter"],
            random_state=seed
        ),
        "BERT + Random Forest": RandomForestClassifier(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_split=mc["random_forest"]["min_samples_split"],
            min_samples_leaf=mc["random_forest"]["min_samples_leaf"],
            class_weight=mc["random_forest"]["class_weight"],
            random_state=seed,
            n_jobs=mc["random_forest"]["n_jobs"]
        ),
        "BERT + Naive Bayes": GaussianNB(),  # GaussianNB for continuous features
    }
    
    all_results = []
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results = evaluate_model(y_test, y_pred, label_names, model_name=model_name)
        print_results(results)
        
        save_results(results, save_dir, model_name=model_name)
        
        safe_name = model_name.lower().replace(" ", "_").replace("+", "_")
        save_confusion_matrix(
            y_test, y_pred, label_names,
            os.path.join(save_dir, f"{safe_name}_cm.png"),
            title=f"{model_name} - Confusion Matrix"
        )
        
        joblib.dump(model, os.path.join(save_dir, f"{safe_name}_model.pkl"))
        all_results.append(results)
    
    # Save comparison
    comp_rows = [{"Model": r["model_name"], "Accuracy": r["accuracy"],
                  "Precision_Macro": r["precision_macro"],
                  "Recall_Macro": r["recall_macro"],
                  "Macro_F1": r["f1_macro"],
                  "Weighted_F1": r["f1_weighted"]} for r in all_results]
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(save_dir, "bert_classifiers_comparison.csv"), index=False
    )
    
    print(f"\n✓ BERT classifier results saved to {save_dir}")
    return all_results
