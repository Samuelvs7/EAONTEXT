"""
data_loader.py
==============
Loads and parses the Empathetic Dialogues dataset.

The dataset has columns: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags
- 'context' contains the emotion label for the entire conversation
- 'utterance' contains the actual text spoken by a speaker
- 'prompt' contains the situation description
- Commas in text fields are encoded as '_comma_'
- The 'tags' column may contain pipe-separated responses causing CSV parsing issues

This loader handles these quirks and produces clean DataFrames.
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
from collections import Counter


def load_config(config_path="config.yaml"):
    """Load experiment configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _read_empathetic_dialogues_file(filepath):
    """
    Read a single Empathetic Dialogues CSV file with robust parsing.
    
    The files are comma-separated but the 'tags' column can contain commas,
    causing standard CSV parsing to fail. We handle this by reading with
    error_bad_lines=False and fixing the 'tags' column.
    
    Returns:
        pd.DataFrame with columns: conv_id, utterance_idx, context, prompt,
                                   speaker_idx, utterance, selfeval, tags
    """
    # The first 7 columns are well-structured; everything after column 7
    # belongs to the 'tags' field. Read line by line for robustness.
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # Expected: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            
            # Split on commas - but we know fields 0-6 are safe
            # The issue is that 'prompt' and 'utterance' may contain _comma_ (not actual commas)
            # and 'tags' may contain actual commas
            # Strategy: split by comma, first 7 fields are fixed, rest is tags
            parts = line.split(",")
            
            if len(parts) < 7:
                continue  # Skip malformed lines
            
            conv_id = parts[0]
            utterance_idx = parts[1]
            context = parts[2]
            prompt = parts[3]
            speaker_idx = parts[4]
            utterance = parts[5]
            selfeval = parts[6]
            tags = ",".join(parts[7:]) if len(parts) > 7 else ""
            
            rows.append({
                "conv_id": conv_id,
                "utterance_idx": utterance_idx,
                "context": context,
                "prompt": prompt,
                "speaker_idx": speaker_idx,
                "utterance": utterance,
                "selfeval": selfeval,
                "tags": tags
            })
    
    df = pd.DataFrame(rows)
    
    # Convert types
    df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce")
    df["speaker_idx"] = pd.to_numeric(df["speaker_idx"], errors="coerce")
    
    return df


def load_raw_dataset(config):
    """
    Load all three splits of the Empathetic Dialogues dataset.
    
    Returns:
        dict with keys 'train', 'valid', 'test', each containing a DataFrame
    """
    raw_dir = config["dataset"]["raw_dir"]
    
    datasets = {}
    for split_name, file_key in [("train", "train_file"), 
                                  ("valid", "valid_file"), 
                                  ("test", "test_file")]:
        filepath = os.path.join(raw_dir, config["dataset"][file_key])
        print(f"Loading {split_name} from {filepath}...")
        df = _read_empathetic_dialogues_file(filepath)
        datasets[split_name] = df
        print(f"  → {len(df)} utterances loaded")
    
    return datasets


def clean_comma_tokens(text, comma_token="_comma_"):
    """Replace _comma_ tokens with actual commas."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    return text.replace(comma_token, ",")


def reconstruct_conversations(df):
    """
    Reconstruct conversations from individual utterances.
    
    Each conversation is identified by conv_id and ordered by utterance_idx.
    All utterances in a conversation share the same emotion label (context).
    
    Returns:
        DataFrame with one row per conversation containing:
        - conv_id: conversation identifier
        - emotion: the emotion label
        - prompt: the original situation description
        - utterances: list of (speaker_idx, utterance) tuples
        - full_conversation: concatenated conversation text
        - num_utterances: number of utterances in the conversation
        - first_utterance: the first speaker's utterance (situation description)
    """
    conversations = []
    
    for conv_id, group in df.groupby("conv_id"):
        group_sorted = group.sort_values("utterance_idx")
        
        emotion = group_sorted["context"].iloc[0].strip()
        prompt = clean_comma_tokens(group_sorted["prompt"].iloc[0])
        
        utterance_list = []
        for _, row in group_sorted.iterrows():
            utt = clean_comma_tokens(row["utterance"])
            spk = row["speaker_idx"]
            utterance_list.append((spk, utt))
        
        # Build full conversation text
        full_conv = " ".join([u[1] for u in utterance_list if isinstance(u[1], str)])
        
        # First utterance (speaker 0 or 1, the person describing the situation)
        first_utt = utterance_list[0][1] if utterance_list else ""
        
        conversations.append({
            "conv_id": conv_id,
            "emotion": emotion,
            "prompt": prompt,
            "utterances": utterance_list,
            "full_conversation": full_conv,
            "num_utterances": len(utterance_list),
            "first_utterance": first_utt
        })
    
    conv_df = pd.DataFrame(conversations)
    return conv_df


def get_utterance_level_data(df):
    """
    Get utterance-level data with cleaned text.
    Each row is one utterance with its emotion label.
    
    Returns:
        DataFrame with columns: conv_id, utterance_idx, emotion, utterance, speaker_idx
    """
    result = df.copy()
    result["emotion"] = result["context"].str.strip()
    result["utterance"] = result["utterance"].apply(clean_comma_tokens)
    result["prompt"] = result["prompt"].apply(clean_comma_tokens)
    
    return result[["conv_id", "utterance_idx", "emotion", "utterance", "speaker_idx", "prompt"]]


def get_dataset_statistics(datasets):
    """
    Compute comprehensive statistics about the dataset.
    
    Args:
        datasets: dict with 'train', 'valid', 'test' DataFrames (raw)
    
    Returns:
        dict with various statistics
    """
    stats = {}
    
    for split_name, df in datasets.items():
        utt_df = get_utterance_level_data(df)
        
        split_stats = {
            "num_utterances": len(df),
            "num_conversations": df["conv_id"].nunique(),
            "num_emotions": utt_df["emotion"].nunique(),
            "emotions": sorted(utt_df["emotion"].unique().tolist()),
            "emotion_distribution": utt_df["emotion"].value_counts().to_dict(),
        }
        
        # Conversation length statistics
        conv_lengths = df.groupby("conv_id").size()
        split_stats["avg_conversation_length"] = float(conv_lengths.mean())
        split_stats["median_conversation_length"] = float(conv_lengths.median())
        split_stats["min_conversation_length"] = int(conv_lengths.min())
        split_stats["max_conversation_length"] = int(conv_lengths.max())
        split_stats["std_conversation_length"] = float(conv_lengths.std())
        
        # Utterance length statistics (word count)
        word_counts = utt_df["utterance"].dropna().apply(lambda x: len(str(x).split()))
        split_stats["avg_utterance_words"] = float(word_counts.mean())
        split_stats["median_utterance_words"] = float(word_counts.median())
        split_stats["max_utterance_words"] = int(word_counts.max())
        
        stats[split_name] = split_stats
    
    # Overall statistics
    all_emotions = set()
    total_utterances = 0
    total_conversations = 0
    for split_name, split_stats in stats.items():
        all_emotions.update(split_stats["emotions"])
        total_utterances += split_stats["num_utterances"]
        total_conversations += split_stats["num_conversations"]
    
    stats["overall"] = {
        "total_utterances": total_utterances,
        "total_conversations": total_conversations,
        "total_emotions": len(all_emotions),
        "all_emotions": sorted(list(all_emotions))
    }
    
    return stats


if __name__ == "__main__":
    import json
    
    config = load_config()
    datasets = load_raw_dataset(config)
    
    print("\n" + "=" * 60)
    print("DATASET STRUCTURE")
    print("=" * 60)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Conversations: {df['conv_id'].nunique()}")
        print(f"  Emotions: {df['context'].str.strip().nunique()}")
        print(f"\n  Sample row:")
        print(f"  {df.iloc[0].to_dict()}")
    
    stats = get_dataset_statistics(datasets)
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(json.dumps(stats["overall"], indent=2))
