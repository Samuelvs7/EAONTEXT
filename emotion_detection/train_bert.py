"""
train_bert.py
=============
Main script to execute:
1. Minimal cleaning (Pipeline B for BERT)
2. Extracting sentence-level embeddings using pretrained BERT (bert-base-uncased with mean pooling)
3. Disk caching of embeddings
4. Training individual classifiers on frozen BERT embeddings (LR, SVM, RF, Naive Bayes)

Run:
    python -u train_bert.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_config, load_raw_dataset
from src.preprocessing import handle_missing_and_invalid, BertPreprocessor, create_conversation_level_split
from src.bert_embeddings import BertEmbeddingExtractor
from src.hybrid_model import train_bert_base_classifiers
from sklearn.preprocessing import LabelEncoder


def main():
    print("=" * 80)
    print("STEP 6: BERT REPRESENTATION LEARNING & EMBEDDING EXTRACTION")
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
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_valid = label_encoder.transform(valid_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])
    label_names = list(label_encoder.classes_)
    
    # Save label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    
    # Pipeline B: BERT minimal cleaning
    bert_prep = BertPreprocessor(config)
    train_texts = bert_prep.preprocess_series(train_df["utterance"]).tolist()
    valid_texts = bert_prep.preprocess_series(valid_df["utterance"]).tolist()
    test_texts = bert_prep.preprocess_series(test_df["utterance"]).tolist()
    
    # Extract BERT Embeddings (cached to disk)
    extractor = BertEmbeddingExtractor(config)
    
    print("\nExtracting/loading BERT embeddings for Train split...")
    X_train_bert = extractor.extract_embeddings(train_texts, split_name="train")
    
    print("\nExtracting/loading BERT embeddings for Valid split...")
    X_valid_bert = extractor.extract_embeddings(valid_texts, split_name="valid")
    
    print("\nExtracting/loading BERT embeddings for Test split...")
    X_test_bert = extractor.extract_embeddings(test_texts, split_name="test")
    
    print("\n" + "=" * 80)
    print("STEP 7: BERT + INDIVIDUAL BASE CLASSIFIERS")
    print("=" * 80)
    bert_results = train_bert_base_classifiers(
        X_train_bert, y_train, X_test_bert, y_test,
        label_names, config, save_dir="results/bert"
    )
    
    print("\n" + "=" * 80)
    print("✓ BERT EMBEDDING EXTRACTION & BASE CLASSIFIER TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
