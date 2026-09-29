"""
evaluate.py
===========
Unified evaluation script for the entire research project.

Executes:
1. Model evaluation & final comparative table across all models
2. 5-Fold Cross-Validation evaluation
3. Rare emotion / few-shot analysis
4. Error analysis (confusion pairs, error CSV, short utterance errors)
5. Cross-domain validation (Empathetic Dialogues → EmoContext)

Run:
    python -u evaluate.py
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_config, load_raw_dataset
from src.preprocessing import handle_missing_and_invalid, ClassicalPreprocessor, BertPreprocessor, create_conversation_level_split
from src.bert_embeddings import BertEmbeddingExtractor
from src.evaluation import create_comparison_table, evaluate_model
from src.rare_emotion import analyze_per_group_performance, run_few_shot_experiment
from src.error_analysis import run_error_analysis
from src.cross_domain import run_cross_domain_evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def main():
    print("=" * 80)
    print("FINAL EVALUATION AND RESEARCH ANALYSIS PIPELINE")
    print("=" * 80)
    config = load_config("config.yaml")
    seed = config["random_seed"]
    np.random.seed(seed)
    
    # Load dataset
    datasets = load_raw_dataset(config)
    cleaned_datasets = {}
    for k, v in datasets.items():
        cleaned_datasets[k] = handle_missing_and_invalid(v, emotion_col="context", text_col="utterance")[0]
    
    splits = create_conversation_level_split(cleaned_datasets, config, save_dir="data/splits")
    
    train_df = splits["train"]
    test_df = splits["test"]
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])
    label_names = list(label_encoder.classes_)
    
    # =========================================================================
    # 1. Gather all result files and build final comparison table
    # =========================================================================
    result_files = [
        ("results/baselines/tfidf_logistic_regression_metrics.json", "TF-IDF + Logistic Regression"),
        ("results/baselines/tfidf_svm_metrics.json", "TF-IDF + SVM"),
        ("results/baselines/tfidf_naive_bayes_metrics.json", "TF-IDF + Naive Bayes"),
        ("results/baselines/tfidf_random_forest_metrics.json", "TF-IDF + Random Forest"),
        ("results/reference_meta_learning/tf-idf___meta-learning_(reference)_metrics.json", "TF-IDF + Meta-Learning"),
        ("results/bert/bert___logistic_regression_metrics.json", "BERT + Logistic Regression"),
        ("results/bert/bert___svm_metrics.json", "BERT + SVM"),
        ("results/bert/bert___random_forest_metrics.json", "BERT + Random Forest"),
        ("results/bert/bert___naive_bayes_metrics.json", "BERT + Naive Bayes"),
        ("results/hybrid/bert_meta_learning_metrics.json", "BERT + Meta-Learning (Proposed)")
    ]
    
    all_results = []
    for filepath, default_name in result_files:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                res = json.load(f)
                res["model_name"] = default_name
                all_results.append(res)
        else:
            print(f"  ⚠ Result file not found: {filepath}")
    
    if all_results:
        print("\n" + "=" * 80)
        print("STEP 9: FINAL COMPARISON TABLE")
        print("=" * 80)
        comparison_df = create_comparison_table(all_results, "results/final_model_comparison.csv")
    else:
        print("  No result files found yet. Run train_baselines.py, train_bert.py, and train_hybrid.py first.")
    
    # =========================================================================
    # 2. Load BERT features for downstream analysis
    # =========================================================================
    bert_prep = BertPreprocessor(config)
    test_texts = bert_prep.preprocess_series(test_df["utterance"]).tolist()
    train_texts = bert_prep.preprocess_series(train_df["utterance"]).tolist()
    
    extractor = BertEmbeddingExtractor(config)
    X_train_bert = extractor.extract_embeddings(train_texts, split_name="train")
    X_test_bert = extractor.extract_embeddings(test_texts, split_name="test")
    
    # =========================================================================
    # 3. 5-Fold Cross Validation (BERT + LR)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: 5-FOLD CROSS VALIDATION ANALYSIS")
    print("=" * 80)
    print("  Using BERT embeddings + Logistic Regression for CV...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_accs, cv_macro_f1s, cv_weighted_f1s = [], [], []
    
    lr_cfg = config.get("models", {}).get("logistic_regression", {})
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_bert, y_train)):
        clf = LogisticRegression(
            max_iter=lr_cfg.get("max_iter", 2000),
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(X_train_bert[t_idx], y_train[t_idx])
        preds = clf.predict(X_train_bert[v_idx])
        
        eval_res = evaluate_model(y_train[v_idx], preds, label_names)
        cv_accs.append(eval_res["accuracy"])
        cv_macro_f1s.append(eval_res["f1_macro"])
        cv_weighted_f1s.append(eval_res["f1_weighted"])
        print(f"    Fold {fold+1}: Acc={eval_res['accuracy']:.4f}, "
              f"MacroF1={eval_res['f1_macro']:.4f}")
    
    cv_summary = {
        "Mean Accuracy": f"{np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}",
        "Mean Macro F1": f"{np.mean(cv_macro_f1s):.4f} +/- {np.std(cv_macro_f1s):.4f}",
        "Mean Weighted F1": f"{np.mean(cv_weighted_f1s):.4f} +/- {np.std(cv_weighted_f1s):.4f}"
    }
    print("\n  5-Fold CV Results:")
    for k, v in cv_summary.items():
        print(f"    {k}: {v}")
    
    os.makedirs("results/cv", exist_ok=True)
    with open("results/cv/5fold_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    # =========================================================================
    # 4. Rare Emotion & Few-Shot Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 11: RARE EMOTION & FEW-SHOT ANALYSIS")
    print("=" * 80)
    
    eda_groups_path = "results/eda/rare_common_emotions.json"
    if os.path.exists(eda_groups_path):
        with open(eda_groups_path, "r") as f:
            eda_groups = json.load(f)["groups"]
        if all_results:
            analyze_per_group_performance(all_results, label_names, eda_groups,
                                          save_dir="results/rare_emotion")
    
    print("\n  Running controlled Few-Shot Experiment...")
    run_few_shot_experiment(X_train_bert, y_train, X_test_bert, y_test,
                             label_names, config, save_dir="results/rare_emotion")
    
    # =========================================================================
    # 5. Error Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 12: ERROR ANALYSIS")
    print("=" * 80)
    
    if os.path.exists("results/hybrid/hybrid_meta_learner.pkl"):
        print("  Running error analysis on Proposed Hybrid Model...")
        meta_learner = joblib.load("results/hybrid/hybrid_meta_learner.pkl")
        base_learners = {}
        for b_name in ["logistic_regression", "svm", "random_forest", "naive_bayes"]:
            b_path = f"results/hybrid/hybrid_base_{b_name}.pkl"
            if os.path.exists(b_path):
                base_learners[b_name] = joblib.load(b_path)
        
        from src.hybrid_model import generate_hybrid_test_meta_features
        meta_test = generate_hybrid_test_meta_features(X_test_bert, base_learners, len(label_names))
        y_pred = meta_learner.predict(meta_test)
        y_proba = meta_learner.predict_proba(meta_test)
        model_name_ea = "BERT Meta-Learning (Proposed)"
    else:
        print("  Hybrid model not found. Running error analysis on BERT + LR...")
        clf = LogisticRegression(
            max_iter=lr_cfg.get("max_iter", 2000),
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(X_train_bert, y_train)
        y_pred = clf.predict(X_test_bert)
        y_proba = clf.predict_proba(X_test_bert)
        model_name_ea = "BERT + LR"
    
    run_error_analysis(test_texts, y_test, y_pred, y_proba, label_names,
                        model_name=model_name_ea, save_dir="results/error_analysis")
    
    # =========================================================================
    # 6. Cross-Domain Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 13: CROSS-DOMAIN VALIDATION (EmoContext)")
    print("=" * 80)
    
    clf_cross = LogisticRegression(
        max_iter=lr_cfg.get("max_iter", 2000),
        C=lr_cfg.get("C", 1.0),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=seed,
        n_jobs=-1
    )
    clf_cross.fit(X_train_bert, y_train)
    in_domain_res = evaluate_model(y_test, clf_cross.predict(X_test_bert),
                                    label_names, model_name="BERT + LR")
    
    run_cross_domain_evaluation(clf_cross, extractor, label_encoder, in_domain_res,
                                 config, save_dir="results/cross_domain", is_bert=True)
    
    print("\n" + "=" * 80)
    print("✓ FULL EVALUATION & RESEARCH ANALYSIS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
