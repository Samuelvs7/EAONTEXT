"""
cross_domain.py
===============
Cross-Domain Validation Module.

Evaluates model performance across domains:
Train on Empathetic Dialogues → Test on EmoContext dataset (or synthetic cross-domain test set if dataset unavailable).

Maintains complete separation: EmoContext is ONLY used for testing cross-domain generalization,
never mixed into the original Empathetic Dialogues training data.

Reports:
- In-domain performance vs. Cross-domain performance
- Performance degradation analysis
- Possible reasons for domain shift
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

from src.evaluation import evaluate_model, print_results, save_results


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Mapping from EmoContext 4 classes (others, happy, sad, angry) to Empathetic Dialogues 32 classes
EMOCONTEXT_TO_EMPATHETIC_MAP = {
    "happy": ["joyful", "excited", "proud", "grateful", "content", "happy", "hopeful"],
    "sad": ["sad", "lonely", "devastated", "disappointed", "sentimental", "grieving"],
    "angry": ["furious", "angry", "annoyed", "disgusted", "jealous"],
    "others": ["neutral", "surprised", "apprehensive", "afraid", "terrified", "anxious", "caring"]
}

# Reverse mapping: Empathetic Dialogues 32 classes -> EmoContext 4 coarse categories
EMPATHETIC_TO_EMOCONTEXT_MAP = {
    "joyful": "happy", "excited": "happy", "proud": "happy", "grateful": "happy",
    "content": "happy", "happy": "happy", "hopeful": "happy", "cheerful": "happy",
    "sad": "sad", "lonely": "sad", "devastated": "sad", "disappointed": "sad",
    "sentimental": "sad", "grieving": "sad", "nostalgic": "sad",
    "furious": "angry", "angry": "angry", "annoyed": "angry", "disgusted": "angry",
    "jealous": "angry", "resentful": "angry",
    "surprised": "others", "apprehensive": "others", "afraid": "others",
    "terrified": "others", "anxious": "others", "caring": "others",
    "guilty": "others", "ashamed": "others", "embarrassed": "others",
    "prepared": "others", "trusting": "others", "faithful": "others",
    "confident": "others", "impressed": "others", "anticipating": "others"
}


def load_emocontext_dataset(data_path=None):
    """
    Load EmoContext dataset if present.
    
    If the dataset file is not available locally, returns (None, None).
    Synthetic data generation is disabled to maintain scientific integrity.
    
    EmoContext has 4 emotions: happy, sad, angry, others.
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading EmoContext dataset from {data_path}...")
        df = pd.read_csv(data_path, sep="\t" if data_path.endswith(".txt") or data_path.endswith(".tsv") else ",")
        # Expected columns: turn1, turn2, turn3, label
        if "label" in df.columns:
            texts = df.apply(lambda r: f"{r.get('turn1', '')} {r.get('turn2', '')} {r.get('turn3', '')}".strip(), axis=1)
            labels = df["label"].values
            return texts.tolist(), labels.tolist()
    
    print("  Note: Official EmoContext dataset file not provided or not found.")
    print("  Per research protocol, synthetic proxy benchmark is disabled to avoid unscientific metrics.")
    return None, None


def map_emotions_to_coarse(labels, label_map=EMPATHETIC_TO_EMOCONTEXT_MAP):
    """Map 32 fine-grained Empathetic Dialogues emotions to 4 coarse EmoContext categories."""
    return [label_map.get(l.lower().strip(), "others") for l in labels]


def run_cross_domain_evaluation(model, extractor_or_vectorizer, label_encoder, 
                                in_domain_results, config,
                                save_dir="results/cross_domain",
                                is_bert=True,
                                base_learners=None,
                                data_path=None):
    """
    Evaluate in-domain trained model on cross-domain EmoContext test set.
    
    Args:
        model: trained model (e.g. meta learner or baseline)
        extractor_or_vectorizer: BertEmbeddingExtractor or TfidfVectorizer
        label_encoder: fitted LabelEncoder
        in_domain_results: dict containing in-domain metrics
        config: config dict
        save_dir: output directory
        is_bert: bool indicating whether feature extractor is BERT
        base_learners: optional dict of base learners if model is a stacking meta-learner
        data_path: path to EmoContext dataset file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN VALIDATION (Empathetic Dialogues → EmoContext)")
    print(f"{'='*60}")
    
    # 1. Load EmoContext dataset
    if data_path is None and config is not None:
        data_path = config.get("cross_domain", {}).get("emocontext_data_path", None)
    
    texts, true_coarse_labels = load_emocontext_dataset(data_path)
    if texts is None or true_coarse_labels is None:
        print("  ✓ Cross-domain evaluation skipped: Official EmoContext dataset not provided.")
        print("    (Synthetic benchmark disabled to maintain scientific integrity.)")
        skip_report = {
            "status": "SKIPPED",
            "reason": "Official EmoContext dataset not found. Synthetic benchmark generation disabled to preserve scientific validity."
        }
        with open(os.path.join(save_dir, "cross_domain_status.json"), "w") as f:
            json.dump(skip_report, f, indent=2)
        return None
    
    # 2. Extract features
    if is_bert:
        X_cross = extractor_or_vectorizer.extract_embeddings(texts, split_name="emocontext", force_recompute=False)
    else:
        X_cross = extractor_or_vectorizer.transform(texts)
    
    # 3. Predict fine-grained emotions matching model's expected feature interface
    if base_learners is not None:
        n_classes = len(label_encoder.classes_)
        if is_bert:
            from src.hybrid_model import generate_hybrid_test_meta_features
            X_input = generate_hybrid_test_meta_features(X_cross, base_learners, n_classes)
        else:
            from src.meta_learning import generate_test_meta_features
            X_input = generate_test_meta_features(X_cross, base_learners, n_classes)
        y_pred_encoded = model.predict(X_input)
    else:
        y_pred_encoded = model.predict(X_cross)
    pred_fine_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    # 4. Map predictions to coarse EmoContext labels
    pred_coarse_labels = map_emotions_to_coarse(pred_fine_labels)
    
    # 5. Evaluate cross-domain performance
    acc = float(accuracy_score(true_coarse_labels, pred_coarse_labels))
    macro_f1 = float(f1_score(true_coarse_labels, pred_coarse_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_coarse_labels, pred_coarse_labels, average="weighted", zero_division=0))
    
    cross_results = {
        "model_name": f"Cross-Domain ({in_domain_results['model_name']})",
        "accuracy": acc,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "in_domain_accuracy": in_domain_results["accuracy"],
        "in_domain_macro_f1": in_domain_results["f1_macro"],
        "accuracy_drop": in_domain_results["accuracy"] - acc,
        "macro_f1_drop": in_domain_results["f1_macro"] - macro_f1
    }
    
    print(f"  In-Domain Macro F1:    {in_domain_results['f1_macro']:.4f}")
    print(f"  Cross-Domain Macro F1: {macro_f1:.4f}")
    print(f"  Performance Degradation (F1 drop): {cross_results['macro_f1_drop']:.4f}")
    
    # Save results
    save_results(cross_results, save_dir, model_name="cross_domain_comparison")
    
    # Document degradation analysis
    degradation_report = {
        "summary": "Cross-domain generalization evaluation from Empathetic Dialogues to EmoContext.",
        "metrics": cross_results,
        "reasons_for_degradation": [
            "Domain Shift: Empathetic Dialogues context vs EmoContext short turn-based dialogues.",
            "Label Taxonomy Mismatch: Mapping 32 fine-grained emotions to 4 coarse categories.",
            "Annotation Style Difference: Crowdsourced situation prompts vs chat platform turns."
        ]
    }
    
    with open(os.path.join(save_dir, "degradation_analysis.json"), "w") as f:
        json.dump(degradation_report, f, indent=2)
    
    print(f"  ✓ Cross-domain results saved to {save_dir}")
    return cross_results
