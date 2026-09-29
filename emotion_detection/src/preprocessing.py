"""
preprocessing.py
================
Two separate preprocessing pipelines for the emotion detection project.

Pipeline A (CLASSICAL ML):
    Text → cleaning → stopword removal → lemmatization → TF-IDF

Pipeline B (BERT):
    Text → minimal cleaning → BERT tokenizer → contextual embeddings
    Preserves punctuation, wording, and context for emotional information.

Handles: missing values, duplicate samples, empty text, invalid labels.
"""

import re
import string
import os
import pickle
import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def safe_word_tokenize(text):
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return text.split()


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# COMMON DATA CLEANING
# ==============================================================================

def handle_missing_and_invalid(df, emotion_col="emotion", text_col="utterance"):
    """
    Handle missing values, empty text, and invalid labels.
    
    Documents every decision:
    - Rows with missing/empty text are dropped (cannot classify empty text)
    - Rows with missing/empty emotion labels are dropped (no ground truth)
    - Duplicate (text, emotion) pairs within the same split are kept 
      (same text can appear in different conversations)
    
    Returns:
        Cleaned DataFrame, dict of cleaning statistics
    """
    stats = {}
    original_len = len(df)
    
    # 1. Missing text
    missing_text = df[text_col].isna().sum()
    stats["missing_text_dropped"] = int(missing_text)
    df = df.dropna(subset=[text_col])
    
    # 2. Empty text (whitespace only)
    empty_text = (df[text_col].str.strip() == "").sum()
    stats["empty_text_dropped"] = int(empty_text)
    df = df[df[text_col].str.strip() != ""]
    
    # 3. Missing emotion labels
    missing_emotion = df[emotion_col].isna().sum()
    stats["missing_emotion_dropped"] = int(missing_emotion)
    df = df.dropna(subset=[emotion_col])
    
    # 4. Empty emotion labels
    empty_emotion = (df[emotion_col].str.strip() == "").sum()
    stats["empty_emotion_dropped"] = int(empty_emotion)
    df = df[df[emotion_col].str.strip() != ""]
    
    # 5. Trim whitespace from emotions
    df[emotion_col] = df[emotion_col].str.strip().str.lower()
    
    # 6. Report duplicates but do NOT remove them
    # (same text can validly appear in multiple conversations)
    duplicates = df.duplicated(subset=[text_col, emotion_col]).sum()
    stats["duplicate_text_emotion_pairs"] = int(duplicates)
    
    stats["original_size"] = original_len
    stats["final_size"] = len(df)
    stats["total_dropped"] = original_len - len(df)
    
    return df, stats


# ==============================================================================
# PIPELINE A: CLASSICAL ML PREPROCESSING
# ==============================================================================

class ClassicalPreprocessor:
    """
    Classical NLP preprocessing pipeline:
    Text → cleaning → stopword removal → lemmatization → TF-IDF
    """
    
    def __init__(self, config):
        self.config = config["preprocessing"]["classical"]
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.label_encoder = None
    
    def clean_text(self, text):
        """
        Clean text for classical ML pipeline.
        - Replace _comma_ with comma
        - Lowercase
        - Remove punctuation
        - Remove extra whitespace
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Replace _comma_ token
        text = text.replace("_comma_", ",")
        
        # Lowercase
        if self.config["lowercase"]:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        
        # Remove punctuation
        if self.config["remove_punctuation"]:
            text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def tokenize_and_process(self, text):
        """
        Tokenize, remove stopwords, and lemmatize.
        """
        if not text:
            return ""
        
        tokens = safe_word_tokenize(text)
        
        # Remove stopwords
        if self.config["remove_stopwords"]:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.config["lemmatize"]:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Filter by minimum token length
        min_len = self.config.get("min_token_length", 2)
        tokens = [t for t in tokens if len(t) >= min_len]
        
        return " ".join(tokens)
    
    def preprocess_text(self, text):
        """Full classical preprocessing pipeline for a single text."""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_process(cleaned)
        return processed
    
    def preprocess_series(self, text_series):
        """Apply preprocessing to a pandas Series."""
        from tqdm import tqdm
        tqdm.pandas(desc="Classical preprocessing")
        return text_series.progress_apply(self.preprocess_text)
    
    def fit_tfidf(self, texts):
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: iterable of preprocessed text strings
        
        Returns:
            scipy sparse matrix of TF-IDF features
        """
        tfidf_config = self.config["tfidf"]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config["max_features"],
            ngram_range=tuple(tfidf_config["ngram_range"]),
            min_df=tfidf_config["min_df"],
            max_df=tfidf_config["max_df"],
            sublinear_tf=tfidf_config["sublinear_tf"]
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def transform_tfidf(self, texts):
        """Transform texts using the fitted TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts)
    
    def fit_label_encoder(self, labels):
        """Fit label encoder on emotion labels."""
        self.label_encoder = LabelEncoder()
        encoded = self.label_encoder.fit_transform(labels)
        print(f"  Label encoder fitted: {len(self.label_encoder.classes_)} classes")
        return encoded
    
    def transform_labels(self, labels):
        """Transform labels using fitted encoder."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit_label_encoder first.")
        return self.label_encoder.transform(labels)
    
    def save(self, save_dir):
        """Save the preprocessor artifacts."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.tfidf_vectorizer is not None:
            with open(os.path.join(save_dir, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.label_encoder is not None:
            with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(self.label_encoder, f)
        
        print(f"  Preprocessor saved to {save_dir}")
    
    def load(self, save_dir):
        """Load preprocessor artifacts."""
        tfidf_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        le_path = os.path.join(save_dir, "label_encoder.pkl")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                self.label_encoder = pickle.load(f)


# ==============================================================================
# PIPELINE B: BERT PREPROCESSING (minimal cleaning)
# ==============================================================================

class BertPreprocessor:
    """
    Minimal preprocessing for BERT.
    
    IMPORTANT: We do NOT aggressively clean text before BERT because:
    - Punctuation carries emotional cues (!, ?, ...)
    - Wording and phrasing contain emotional information
    - BERT's tokenizer handles most text naturally
    - Context is important for emotion detection
    
    Only: replace _comma_ tokens, strip extra whitespace.
    """
    
    def __init__(self, config):
        self.config = config["preprocessing"]["bert"]
        self.comma_token = config["dataset"]["comma_token"]
    
    def preprocess_text(self, text):
        """
        Minimal cleaning for BERT pipeline.
        Preserves emotional cues in text.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Replace _comma_ token with actual comma
        if self.config.get("replace_comma_token", True):
            text = text.replace(self.comma_token, ",")
        
        # Strip extra whitespace (but preserve single spaces)
        if self.config.get("strip_whitespace", True):
            text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def preprocess_series(self, text_series):
        """Apply minimal preprocessing to a pandas Series."""
        return text_series.apply(self.preprocess_text)


# ==============================================================================
# DATA SPLITTING (Conversation-level to prevent leakage)
# ==============================================================================

def create_conversation_level_split(datasets, config, save_dir="data/splits"):
    """
    The Empathetic Dialogues dataset already provides train/valid/test splits.
    We use these official splits to maintain comparability with other research.
    
    IMPORTANT: The official splits are already conversation-level 
    (all utterances from a conversation are in the same split), 
    which prevents data leakage.
    
    We verify this and save the split information for reproducibility.
    
    Returns:
        dict with 'train', 'valid', 'test' DataFrames (utterance-level)
    """
    from src.data_loader import get_utterance_level_data
    
    os.makedirs(save_dir, exist_ok=True)
    
    splits = {}
    split_info = {}
    
    for split_name, df in datasets.items():
        utt_df = get_utterance_level_data(df)
        splits[split_name] = utt_df
        
        split_info[split_name] = {
            "num_utterances": len(utt_df),
            "num_conversations": utt_df["conv_id"].nunique(),
            "conversation_ids": sorted(utt_df["conv_id"].unique().tolist())
        }
        
        # Save utterance-level data
        utt_df.to_csv(os.path.join(save_dir, f"{split_name}_utterances.csv"), index=False)
    
    # Verify no conversation leakage between splits
    train_convs = set(split_info["train"]["conversation_ids"])
    valid_convs = set(split_info["valid"]["conversation_ids"])
    test_convs = set(split_info["test"]["conversation_ids"])
    
    train_valid_overlap = train_convs & valid_convs
    train_test_overlap = train_convs & test_convs
    valid_test_overlap = valid_convs & test_convs
    
    leakage_report = {
        "train_valid_overlap": len(train_valid_overlap),
        "train_test_overlap": len(train_test_overlap),
        "valid_test_overlap": len(valid_test_overlap),
        "no_leakage": len(train_valid_overlap) == 0 and len(train_test_overlap) == 0 and len(valid_test_overlap) == 0
    }
    
    print("\n--- Data Split Verification ---")
    print(f"  Train conversations: {len(train_convs)}")
    print(f"  Valid conversations: {len(valid_convs)}")
    print(f"  Test conversations: {len(test_convs)}")
    print(f"  Train-Valid overlap: {leakage_report['train_valid_overlap']}")
    print(f"  Train-Test overlap: {leakage_report['train_test_overlap']}")
    print(f"  Valid-Test overlap: {leakage_report['valid_test_overlap']}")
    print(f"  ✓ No leakage: {leakage_report['no_leakage']}")
    
    # Save split info
    import json
    with open(os.path.join(save_dir, "split_info.json"), "w") as f:
        json.dump({
            "leakage_report": leakage_report,
            "train_utterances": len(splits["train"]),
            "valid_utterances": len(splits["valid"]),
            "test_utterances": len(splits["test"]),
            "train_conversations": len(train_convs),
            "valid_conversations": len(valid_convs),
            "test_conversations": len(test_convs),
            "random_seed": config["random_seed"]
        }, f, indent=2)
    
    print(f"  Split information saved to {save_dir}")
    
    return splits


if __name__ == "__main__":
    from src.data_loader import load_raw_dataset
    
    config = load_config()
    datasets = load_raw_dataset(config)
    
    # Test classical preprocessor
    print("\n--- Testing Classical Preprocessor ---")
    cp = ClassicalPreprocessor(config)
    sample_text = "I was so happy_comma_ it felt like a dream! :)"
    print(f"  Input: {sample_text}")
    print(f"  Output: {cp.preprocess_text(sample_text)}")
    
    # Test BERT preprocessor
    print("\n--- Testing BERT Preprocessor ---")
    bp = BertPreprocessor(config)
    print(f"  Input: {sample_text}")
    print(f"  Output: {bp.preprocess_text(sample_text)}")
