# Emotion Detection in Conversational Text using BERT and Meta-Learning Ensembles

This repository implements a research-grade emotion detection system using the **Empathetic Dialogues** dataset (32 emotion categories), **BERT (bert-base-uncased)** contextual sentence embeddings, and **Meta-Learning Ensembles (Stacking)**.

---

## 🏗️ Architecture Overview

```
Empathetic Dialogues Dataset (32 Emotions)
                  ↓
       Data Preprocessing & Cleaning
                  ↓
       BERT (bert-base-uncased)
                  ↓
 Contextual / Sentence Embeddings (Mean Pooling)
                  ↓
  ┌──────────────────────────────────────────┐
  │ Base Classifiers:                        │
  │  1. Logistic Regression                  │
  │  2. Support Vector Machine (SVM)         │
  │  3. Random Forest                        │
  │  4. Naive Bayes (GaussianNB)             │
  └──────────────────────────────────────────┘
                  ↓
 Out-of-Fold Prediction Probabilities (Meta-Features)
                  ↓
   Meta Learner (Logistic Regression)
                  ↓
        Final Emotion Prediction
```

---

## 📁 Project Structure

```
emotion_detection/
│
├── config.yaml                    # Master experiment configuration & random seeds
├── requirements.txt               # Python package dependencies
│
├── data/
│   ├── raw/                       # Raw Empathetic Dialogues files (train, valid, test CSVs)
│   ├── processed/                 # Preprocessed clean datasets
│   └── splits/                    # Leakage-free conversation-level split details
│
├── models/                        # Saved model artifacts, TF-IDF vectorizers, label encoders
├── embeddings/                    # Cached BERT sentence embeddings (.npy)
│
├── results/
│   ├── eda/                       # EDA statistics, emotion distribution plots, imbalance metrics
│   ├── baselines/                 # TF-IDF classical ML baseline results (CSV, CM, JSON)
│   ├── reference_meta_learning/   # Stacking on TF-IDF features
│   ├── bert/                      # Frozen BERT embeddings + classical classifiers
│   ├── hybrid/                    # Proposed BERT + Meta-Learning hybrid model results
│   ├── rare_emotion/              # Performance breakdown by resource group & few-shot results
│   ├── error_analysis/            # Error CSV, confusion pair ranking, error heatmaps
│   └── cross_domain/              # Empathetic Dialogues → EmoContext cross-domain results
│
├── src/
│   ├── data_loader.py             # Robust CSV parsing & dataset loading
│   ├── preprocessing.py           # Pipelines A (Classical ML) & B (BERT minimal cleaning)
│   ├── eda.py                     # Statistical analysis & plot generation
│   ├── baselines.py               # Classical baseline trainers (LR, SVM, NB, RF)
│   ├── bert_embeddings.py         # BERT tokenizer, mean pooling, GPU/CPU auto-detection & caching
│   ├── meta_learning.py           # Reference meta-learning (stacking) module
│   ├── hybrid_model.py            # Proposed BERT + Meta-Learning architecture with OOF leakage prevention
│   ├── evaluation.py              # Centralized metrics computation & report generation
│   ├── rare_emotion.py            # Resource group grouping & few-shot experiments
│   ├── error_analysis.py          # Detailed confusion pair & error CSV generation
│   └── cross_domain.py            # Cross-domain generalization pipeline
│
├── train_baselines.py             # Run classical ML & reference stacking experiments
├── train_bert.py                  # Extract BERT embeddings & train BERT base classifiers
├── train_hybrid.py                # Train the proposed BERT + Meta-Learning Hybrid Model
└── evaluate.py                    # Run complete evaluation, 5-fold CV, error analysis, cross-domain test
```

---

## 🚀 Reproduction Guide & Commands

### 1. Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing & Classical Baselines
Runs Pipeline A (cleaning + TF-IDF) and trains Logistic Regression, SVM, Naive Bayes, Random Forest, and Reference Meta-Learning:
```bash
python train_baselines.py
```

### 3. Run BERT Representation Extraction & Base Classifiers
Runs Pipeline B (minimal cleaning for BERT), extracts mean-pooled sentence embeddings, caches them to disk, and trains baseline classifiers on BERT features:
```bash
python train_bert.py
```

### 4. Run Proposed Hybrid Model (BERT + Meta-Learning)
Trains base learners on BERT embeddings using out-of-fold cross-validation to construct leakage-free probability meta-features, then trains the meta learner:
```bash
python train_hybrid.py
```

### 5. Run Comprehensive Evaluation & Research Analysis
Generates final model comparison table, 5-fold cross validation, rare emotion breakdown, few-shot analysis, error analysis CSV, and cross-domain validation:
```bash
python evaluate.py
```

---

## 📊 Key Research Protocols

- **Leakage Prevention**: Split at conversation-level (`conv_id`) so all utterances from a single conversation stay in the same split.
- **Out-of-Fold Meta-Features**: Meta learner is trained strictly on out-of-fold probability predictions to ensure zero data leakage.
- **Minimal Cleaning for BERT**: Preserves punctuation, exclamation marks, and capitalizations that carry strong emotional signals.
- **Primary Metric**: **Macro F1** (chosen due to class imbalance among the 32 emotion categories), along with Weighted F1, Accuracy, Precision, and Recall.
