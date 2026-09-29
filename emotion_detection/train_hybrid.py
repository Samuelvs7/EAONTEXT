"""
train_hybrid.py
===============
Main script for training the proposed architecture:
BERT + Meta-Learning Ensemble with out-of-fold probability meta-features.

Architecture:
Conversation Text → BERT → Sentence Embedding → Base Classifiers (LR, SVM, RF, NB)
→ Prediction Probabilities → Meta-Features → Meta Learner (LR) → Final Emotion

Run:
    python -u train_hybrid.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_config, load_raw_dataset
from src.preprocessing import handle_missing_and_invalid, BertPreprocessor, create_conversation_level_split
from src.bert_embeddings import BertEmbeddingExtractor
from src.hybrid_model import train_hybrid_model
from sklearn.preprocessing import LabelEncoder


def main():
    print("=" * 80)
    print("STEP 8: PROPOSED HYBRID MODEL (BERT + META-LEARNING ENSEMBLE)")
    print("=" * 80)
    config = load_config("config.yaml")
    np.random.seed(config["random_seed"])
    
    datasets = load_raw_dataset(config)
    
    cleaned_datasets = {}
    for split_name, df in datasets.items():
        df_clean, _ = handle_missing_and_invalid(df, emotion_col="context", text_col="utterance")
        cleaned_datasets[split_name] = df_clean
    
    splits = create_conversation_level_split(cleaned_datasets, config, save_dir="data/splits")
    
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_valid = label_encoder.transform(valid_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])
    label_names = list(label_encoder.classes_)
    
    bert_prep = BertPreprocessor(config)
    train_texts = bert_prep.preprocess_series(train_df["utterance"]).tolist()
    valid_texts = bert_prep.preprocess_series(valid_df["utterance"]).tolist()
    test_texts = bert_prep.preprocess_series(test_df["utterance"]).tolist()
    
    extractor = BertEmbeddingExtractor(config)
    X_train_bert = extractor.extract_embeddings(train_texts, split_name="train")
    X_valid_bert = extractor.extract_embeddings(valid_texts, split_name="valid")
    X_test_bert = extractor.extract_embeddings(test_texts, split_name="test")
    
    print("\nTraining Proposed Hybrid Model...")
    hybrid_results, meta_learner, base_learners = train_hybrid_model(
        X_train_bert, y_train, X_test_bert, y_test,
        label_names, config, X_valid=X_valid_bert, y_valid=y_valid,
        save_dir="results/hybrid"
    )
    
    print("\n" + "=" * 80)
    print("✓ PROPOSED BERT + META-LEARNING HYBRID MODEL COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
