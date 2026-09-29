"""
train_baselines.py
==================
Main script to execute:
1. Data loading & cleaning
2. Classical ML Preprocessing (Pipeline A)
3. Data Splitting & Leakage Verification
4. Classical Baseline Models (LR, SVM, NB, RF with TF-IDF)
5. Reference Meta-Learning Model (Stacking on TF-IDF)

Run:
    python -u train_baselines.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from src.data_loader import load_config, load_raw_dataset
from src.preprocessing import handle_missing_and_invalid, ClassicalPreprocessor, create_conversation_level_split
from src.baselines import train_and_evaluate_baselines
from src.meta_learning import train_reference_meta_learning


def main():
    print("=" * 80)
    print("STEP 1: LOADING DATASET AND CONFIGURATION")
    print("=" * 80)
    config = load_config("config.yaml")
    np.random.seed(config["random_seed"])
    
    datasets = load_raw_dataset(config)
    
    # Clean datasets — raw data uses 'context' for emotion, 'utterance' for text
    cleaned_datasets = {}
    for split_name, df in datasets.items():
        df_clean, stats = handle_missing_and_invalid(df, emotion_col="context", text_col="utterance")
        cleaned_datasets[split_name] = df_clean
        print(f"  {split_name.upper()} cleaned: {stats['final_size']} samples "
              f"(dropped {stats['total_dropped']})")

    print("\n" + "=" * 80)
    print("STEP 2: CONVERSATION-LEVEL DATA SPLITTING & LEAKAGE CHECK")
    print("=" * 80)
    # create_conversation_level_split calls get_utterance_level_data() internally,
    # which renames 'context' → 'emotion' and cleans _comma_ tokens.
    splits = create_conversation_level_split(cleaned_datasets, config, save_dir="data/splits")
    
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    print(f"\n  Train utterances: {len(train_df)}")
    print(f"  Valid utterances: {len(valid_df)}")
    print(f"  Test  utterances: {len(test_df)}")
    
    print("\n" + "=" * 80)
    print("STEP 3: CLASSICAL ML PREPROCESSING & TF-IDF VECTORIZATION")
    print("=" * 80)
    preprocessor = ClassicalPreprocessor(config)
    
    print("Preprocessing training text...")
    train_text_processed = preprocessor.preprocess_series(train_df["utterance"])
    print("Preprocessing test text...")
    test_text_processed = preprocessor.preprocess_series(test_df["utterance"])
    
    print("\nFitting TF-IDF vectorizer on training data...")
    X_train = preprocessor.fit_tfidf(train_text_processed)
    X_test = preprocessor.transform_tfidf(test_text_processed)
    
    print("\nFitting Label Encoder...")
    y_train = preprocessor.fit_label_encoder(train_df["emotion"])
    y_test = preprocessor.transform_labels(test_df["emotion"])
    label_names = list(preprocessor.label_encoder.classes_)
    print(f"  Classes: {label_names}")
    
    # Save preprocessor artifacts
    preprocessor.save("models")
    
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING CLASSICAL BASELINE MODELS (TF-IDF)")
    print("=" * 80)
    baseline_results, trained_baselines = train_and_evaluate_baselines(
        X_train, y_train, X_test, y_test, label_names, config, save_dir="results/baselines"
    )
    
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING REFERENCE META-LEARNING MODEL (TF-IDF)")
    print("=" * 80)
    ref_meta_results, meta_learner, base_learners = train_reference_meta_learning(
        X_train, y_train, X_test, y_test, label_names, config, save_dir="results/reference_meta_learning"
    )
    
    print("\n" + "=" * 80)
    print("✓ ALL CLASSICAL BASELINES & REFERENCE META-LEARNING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
