"""
bert_embeddings.py
==================
BERT representation learning module.

Extracts sentence-level embeddings from pretrained BERT (bert-base-uncased).
Uses mean pooling over the last hidden states with attention mask.

First experiment: frozen BERT embeddings → traditional ML classifiers.
Embeddings are cached to disk to avoid recomputation.

Handles GPU/CPU automatically with mixed precision for GPU.
"""

import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TextDataset(Dataset):
    """Simple dataset wrapping a list of texts for BERT encoding."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling over the last hidden states using attention mask.
    
    This is a sensible pooling strategy that:
    - Uses all token representations (not just [CLS])
    - Weights by attention mask to ignore padding tokens
    - Produces a fixed-size sentence embedding
    
    Args:
        model_output: BERT model output (last_hidden_state)
        attention_mask: attention mask tensor
    
    Returns:
        Pooled embedding tensor [batch_size, hidden_dim]
    """
    token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden_dim]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings weighted by mask, then divide by mask sum
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sum_embeddings / sum_mask


class BertEmbeddingExtractor:
    """
    Extracts sentence-level embeddings from pretrained BERT.
    
    The model is kept frozen (no fine-tuning) for the first experiment.
    Embeddings are cached to disk for efficiency.
    """
    
    def __init__(self, config):
        self.config = config["bert"]
        self.model_name = self.config["model_name"]
        self.max_length = self.config["max_length"]
        self.embedding_dim = self.config["embedding_dim"]
        self.cache_dir = self.config["cache_dir"]
        
        # Auto-detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.config.get("use_fp16", True) and self.device.type == "cuda"
        
        if self.device.type == "cuda":
            self.batch_size = self.config["batch_size"]
            print(f"  🖥️ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Using batch size: {self.batch_size}")
            if self.use_fp16:
                print(f"  Mixed precision: enabled")
        else:
            self.batch_size = self.config["cpu_batch_size"]
            print(f"  💻 CPU mode: batch size {self.batch_size}")
            self.use_fp16 = False
        
        # Load tokenizer and model
        print(f"  Loading {self.model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Freeze the model
        
        print(f"  ✓ BERT loaded on {self.device}")
    
    def extract_embeddings(self, texts, split_name="data", force_recompute=False):
        """
        Extract BERT embeddings for a list of texts.
        
        Caches results to disk. If cached embeddings exist and
        force_recompute is False, loads from cache.
        
        Args:
            texts: list of text strings
            split_name: name for caching (e.g., 'train', 'valid', 'test')
            force_recompute: if True, ignores cache
        
        Returns:
            numpy array of shape [num_texts, embedding_dim]
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"bert_embeddings_{split_name}.npy")
        
        # Check cache
        if os.path.exists(cache_path) and not force_recompute:
            print(f"  Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
            if len(embeddings) == len(texts):
                print(f"  ✓ Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
                return embeddings
            else:
                print(f"  ⚠ Cache size mismatch ({len(embeddings)} vs {len(texts)}), recomputing...")
        
        print(f"  Extracting BERT embeddings for {len(texts)} texts...")
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=0, pin_memory=True)
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"BERT [{split_name}]"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                if self.use_fp16:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(input_ids=input_ids, 
                                             attention_mask=attention_mask)
                        embeddings = mean_pooling(outputs, attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, 
                                         attention_mask=attention_mask)
                    embeddings = mean_pooling(outputs, attention_mask)
                
                all_embeddings.append(embeddings.float().cpu().numpy())
        
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        
        # Cache to disk
        np.save(cache_path, embeddings_array)
        print(f"  ✓ Embeddings saved to {cache_path}")
        print(f"  Shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def get_embedding_info(self):
        """Return info about the embedding configuration."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "pooling_strategy": "mean_pooling",
            "device": str(self.device),
            "mixed_precision": self.use_fp16,
            "batch_size": self.batch_size
        }


if __name__ == "__main__":
    config = load_config()
    
    # Quick test with sample texts
    extractor = BertEmbeddingExtractor(config)
    
    sample_texts = [
        "I feel so happy today!",
        "I'm really scared of the dark.",
        "This makes me angry.",
    ]
    
    embeddings = extractor.extract_embeddings(sample_texts, split_name="test_sample",
                                              force_recompute=True)
    print(f"\nSample embeddings shape: {embeddings.shape}")
    print(f"Embedding info: {extractor.get_embedding_info()}")
