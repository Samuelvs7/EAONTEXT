"""
eda.py
======
Exploratory Data Analysis for the Empathetic Dialogues dataset.

Generates:
- Dataset statistics
- Emotion distribution plots
- Conversation length analysis
- Utterance length analysis
- Rare/common emotion analysis
- Class imbalance visualization
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.data_loader import load_raw_dataset, load_config, get_utterance_level_data
from src.data_loader import get_dataset_statistics, reconstruct_conversations


def run_eda(config, datasets, save_dir="results/eda"):
    """
    Run complete exploratory data analysis.
    
    Args:
        config: configuration dict
        datasets: dict with 'train', 'valid', 'test' raw DataFrames
        save_dir: directory to save EDA results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Basic statistics
    stats = get_dataset_statistics(datasets)
    
    # Save statistics
    with open(os.path.join(save_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nOverall Statistics:")
    print(f"  Total utterances: {stats['overall']['total_utterances']}")
    print(f"  Total conversations: {stats['overall']['total_conversations']}")
    print(f"  Total emotion classes: {stats['overall']['total_emotions']}")
    print(f"  Emotion classes: {stats['overall']['all_emotions']}")
    
    for split in ["train", "valid", "test"]:
        s = stats[split]
        print(f"\n{split.upper()} Split:")
        print(f"  Utterances: {s['num_utterances']}")
        print(f"  Conversations: {s['num_conversations']}")
        print(f"  Emotions: {s['num_emotions']}")
        print(f"  Avg conversation length: {s['avg_conversation_length']:.1f} utterances")
        print(f"  Avg utterance length: {s['avg_utterance_words']:.1f} words")
    
    # 2. Emotion distribution
    _plot_emotion_distribution(datasets, save_dir)
    
    # 3. Conversation length analysis
    _plot_conversation_length(datasets, save_dir)
    
    # 4. Utterance length analysis
    _plot_utterance_length(datasets, save_dir)
    
    # 5. Class imbalance analysis
    _analyze_class_imbalance(datasets, save_dir)
    
    # 6. Rare/common emotion analysis
    _analyze_rare_common_emotions(datasets, save_dir)
    
    # 7. Cross-split emotion distribution comparison
    _compare_splits(datasets, save_dir)
    
    print(f"\n✓ EDA results saved to {save_dir}")
    return stats


def _plot_emotion_distribution(datasets, save_dir):
    """Plot emotion distribution for training set."""
    train_df = get_utterance_level_data(datasets["train"])
    emotion_counts = train_df["emotion"].value_counts()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = sns.color_palette("husl", len(emotion_counts))
    bars = ax.bar(range(len(emotion_counts)), emotion_counts.values, color=colors)
    ax.set_xticks(range(len(emotion_counts)))
    ax.set_xticklabels(emotion_counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Emotion Category", fontsize=12)
    ax.set_ylabel("Number of Utterances", fontsize=12)
    ax.set_title("Emotion Distribution in Training Set (Empathetic Dialogues)", fontsize=14)
    
    # Add count labels on bars
    for bar, count in zip(bars, emotion_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "emotion_distribution_train.png"), dpi=150)
    plt.close()
    
    # Save as CSV
    emotion_counts.to_csv(os.path.join(save_dir, "emotion_distribution_train.csv"))
    print("  ✓ Emotion distribution plot saved")


def _plot_conversation_length(datasets, save_dir):
    """Analyze conversation lengths across splits."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (split_name, df) in enumerate(datasets.items()):
        conv_lengths = df.groupby("conv_id").size()
        axes[idx].hist(conv_lengths, bins=20, edgecolor="black", alpha=0.7,
                       color=["#3498db", "#2ecc71", "#e74c3c"][idx])
        axes[idx].set_xlabel("Number of Utterances per Conversation")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{split_name.upper()} - Conversation Lengths")
        axes[idx].axvline(conv_lengths.mean(), color="red", linestyle="--", 
                          label=f"Mean: {conv_lengths.mean():.1f}")
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "conversation_lengths.png"), dpi=150)
    plt.close()
    print("  ✓ Conversation length plot saved")


def _plot_utterance_length(datasets, save_dir):
    """Analyze utterance word counts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (split_name, df) in enumerate(datasets.items()):
        from src.data_loader import clean_comma_tokens
        word_counts = df["utterance"].apply(
            lambda x: len(str(clean_comma_tokens(x)).split()) if pd.notna(x) else 0
        )
        axes[idx].hist(word_counts, bins=50, edgecolor="black", alpha=0.7,
                       color=["#3498db", "#2ecc71", "#e74c3c"][idx])
        axes[idx].set_xlabel("Word Count per Utterance")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{split_name.upper()} - Utterance Lengths")
        axes[idx].axvline(word_counts.mean(), color="red", linestyle="--",
                          label=f"Mean: {word_counts.mean():.1f}")
        axes[idx].legend()
        axes[idx].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "utterance_lengths.png"), dpi=150)
    plt.close()
    print("  ✓ Utterance length plot saved")


def _analyze_class_imbalance(datasets, save_dir):
    """Quantify class imbalance."""
    train_df = get_utterance_level_data(datasets["train"])
    counts = train_df["emotion"].value_counts()
    
    imbalance_stats = {
        "most_common_emotion": counts.index[0],
        "most_common_count": int(counts.iloc[0]),
        "least_common_emotion": counts.index[-1],
        "least_common_count": int(counts.iloc[-1]),
        "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        "mean_count": float(counts.mean()),
        "std_count": float(counts.std()),
        "coefficient_of_variation": float(counts.std() / counts.mean()),
    }
    
    with open(os.path.join(save_dir, "class_imbalance.json"), "w") as f:
        json.dump(imbalance_stats, f, indent=2)
    
    print(f"  Class Imbalance Ratio: {imbalance_stats['imbalance_ratio']:.2f}")
    print(f"  Most common: {imbalance_stats['most_common_emotion']} ({imbalance_stats['most_common_count']})")
    print(f"  Least common: {imbalance_stats['least_common_emotion']} ({imbalance_stats['least_common_count']})")


def _analyze_rare_common_emotions(datasets, save_dir):
    """
    Group emotions by frequency: High-resource, Medium-resource, Low-resource.
    Uses percentile-based thresholds from the actual distribution.
    """
    train_df = get_utterance_level_data(datasets["train"])
    counts = train_df["emotion"].value_counts()
    
    q25 = counts.quantile(0.25)
    q75 = counts.quantile(0.75)
    
    groups = {
        "high_resource": [],
        "medium_resource": [],
        "low_resource": []
    }
    
    for emotion, count in counts.items():
        if count >= q75:
            groups["high_resource"].append({"emotion": emotion, "count": int(count)})
        elif count >= q25:
            groups["medium_resource"].append({"emotion": emotion, "count": int(count)})
        else:
            groups["low_resource"].append({"emotion": emotion, "count": int(count)})
    
    result = {
        "q25_threshold": float(q25),
        "q75_threshold": float(q75),
        "groups": groups,
        "high_resource_count": len(groups["high_resource"]),
        "medium_resource_count": len(groups["medium_resource"]),
        "low_resource_count": len(groups["low_resource"]),
    }
    
    with open(os.path.join(save_dir, "rare_common_emotions.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  Emotion Resource Groups (based on training frequency):")
    print(f"    Q25 threshold: {q25:.0f} utterances")
    print(f"    Q75 threshold: {q75:.0f} utterances")
    print(f"    High-resource ({len(groups['high_resource'])} emotions): "
          f"{', '.join([g['emotion'] for g in groups['high_resource']])}")
    print(f"    Medium-resource ({len(groups['medium_resource'])} emotions): "
          f"{', '.join([g['emotion'] for g in groups['medium_resource']])}")
    print(f"    Low-resource ({len(groups['low_resource'])} emotions): "
          f"{', '.join([g['emotion'] for g in groups['low_resource']])}")


def _compare_splits(datasets, save_dir):
    """Compare emotion distribution across splits."""
    all_dists = {}
    for split_name, df in datasets.items():
        utt_df = get_utterance_level_data(df)
        dist = utt_df["emotion"].value_counts(normalize=True)
        all_dists[split_name] = dist
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(all_dists).fillna(0)
    comparison.to_csv(os.path.join(save_dir, "emotion_distribution_comparison.csv"))
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(comparison))
    width = 0.25
    
    for i, split_name in enumerate(["train", "valid", "test"]):
        if split_name in comparison.columns:
            ax.bar(x + i * width, comparison[split_name], width,
                   label=split_name.upper(), alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(comparison.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Emotion Distribution Comparison Across Splits", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "split_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ Split comparison plot saved")


if __name__ == "__main__":
    config = load_config()
    datasets = load_raw_dataset(config)
    run_eda(config, datasets)
