import os
import logging
import json
import argparse
from typing import Dict, List, Optional, Any

import torch
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    set_seed
)
from datasets import Dataset, concatenate_datasets
from code.utils import fix_input_ids, nonempty, set_special_tokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenizedDatasetLoader:
    """
    Loader for tokenized datasets in JSONL format:
      each line is {"text": "...", "token_ids": [...]}.
    """
    def __init__(self, tokenized_files):
        self.tokenized_files = tokenized_files  # e.g. {lang_code: {split: path}}
    
    def load_dataset(self, lang_code, split):
        """Load dataset for a specific language and split."""
        if lang_code not in self.tokenized_files or split not in self.tokenized_files[lang_code]:
            logger.error(f"No tokenized file found for {lang_code} {split}")
            return None
        
        file_path = self.tokenized_files[lang_code][split]
        if not os.path.isfile(file_path):
            logger.error(f"File not found at {file_path}")
            return None
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append({
                        'text': item['text'],
                        'input_ids': item['token_ids']
                    })
            dataset = Dataset.from_list(data)
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {str(e)}")
            return None
    
    def load_combined_dataset(self, split):
        """Combine all languages' datasets for this split."""
        datasets = []
        for lang_code in self.tokenized_files.keys():
            ds = self.load_dataset(lang_code, split)
            if ds is not None:
                datasets.append(ds)
        if not datasets:
            logger.error(f"No datasets found for split {split}")
            return None
        
        return concatenate_datasets(datasets)


class CustomDataCollator:
    """
    Collator that pads input_ids + labels to the same length in a batch.
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        for f in features:
            cur_len = len(f["input_ids"])
            pad_needed = max_len - cur_len
            if "labels" not in f:
                f["labels"] = list(f["input_ids"])
            if pad_needed > 0:
                f["input_ids"] += [self.pad_id] * pad_needed
                f["labels"] += [self.pad_id] * pad_needed
        batch = {}
        for k in features[0].keys():
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        
        return batch


class CustomCausalLMModel:
    """
    A GPT2-like model with a user-specified base plus a custom HF tokenizer.
    """
    def __init__(self, hf_tokenizer, model_name="distilgpt2", config=None):
        """
        Args:
            hf_tokenizer: HuggingFace tokenizer
            model_name: Base model name (e.g. "distilgpt2", "gpt2", "gpt2-medium")
            config: Optional custom model configuration
        """
        self.model_name = model_name
        self.tokenizer = hf_tokenizer
        self.model = None
        self.custom_config = config
    
    def create_model(self):
        """Create and prepare the model for training."""
        # Load the config from pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = base_model.config
        
        # Apply any custom config overrides
        if self.custom_config:
            for key, value in self.custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"Override config: {key}={value}")
        
        # Set vocab_size to match tokenizer
        config.vocab_size = len(self.tokenizer)
        
        # Create new model from config
        self.model = AutoModelForCausalLM.from_config(config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {total_params} parameters.")

        return self.model

    def train(self, train_dataset, val_dataset, output_dir, training_args=None):
        """Train the model with the given datasets and arguments."""
        if self.model is None or self.tokenizer is None:
            logger.error("Model/Tokenizer not created or loaded.")
            return None

        # Prepare datasets
        logger.info(f"Processing train dataset: {len(train_dataset)} samples")
        train_dataset = train_dataset.map(fix_input_ids)
        train_dataset = train_dataset.filter(nonempty)
        logger.info(f"After filtering: {len(train_dataset)} samples")
        
        logger.info(f"Processing validation dataset: {len(val_dataset)} samples")
        val_dataset = val_dataset.map(fix_input_ids)
        val_dataset = val_dataset.filter(nonempty)
        logger.info(f"After filtering: {len(val_dataset)} samples")

        # Setup default training arguments optimized for A5000 GPU
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                fp16=True,  # Use mixed precision for faster training
                warmup_steps=100,
                logging_steps=50,
                evaluation_strategy="steps",
                eval_steps=500,
                save_steps=500,
                load_best_model_at_end=True,
                save_total_limit=2,  # Keep only the 2 best checkpoints
                logging_dir=os.path.join(output_dir, "logs"),
                gradient_accumulation_steps=2,  # Effective batch size of 16
                # Add early stopping
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                # Add proper weight decay
                weight_decay=0.01,
                # Enable more efficient attention computation
                torch_compile=torch.cuda.is_available(),  # Use torch.compile if available (PyTorch 2.0+)
            )
        
        # Set up data collator
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            logger.warning("No pad token defined in tokenizer, using 0")
            pad_id = 0

        data_collator = CustomDataCollator(pad_id)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        # Train the model
        logger.info(f"Starting training with {self.model_name} base model on ~1.6M tokens.")
        result = trainer.train()
        logger.info("Finished training.")
        
        # Save the model
        trainer.save_model(output_dir)
        # Also save the tokenizer for convenience
        self.tokenizer.save_pretrained(output_dir)
        
        return result
    
    def evaluate(self, test_dataset, output_dir):
        """Evaluate the model on a test dataset."""
        if self.model is None or self.tokenizer is None:
            logger.error("No model/Tokenizer for evaluation.")
            return None

        # Prepare test dataset
        test_dataset = test_dataset.map(fix_input_ids)
        test_dataset = test_dataset.filter(nonempty)

        # Setup evaluation arguments
        temp_dir = os.path.join(output_dir, "temp_eval")
        eval_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=8,
            fp16=True  # Use mixed precision for consistency
        )

        # Setup data collator
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            logger.warning("No pad token defined in tokenizer, using 0")
            pad_id = 0
        data_collator = CustomDataCollator(pad_id)

        # Create trainer and evaluate
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            data_collator=data_collator
        )
        
        logger.info("Evaluating model on test set...")
        metrics = trainer.evaluate(test_dataset)
        
        return metrics


def train_model_on_tokenized_data(tokenized_files, 
                                  tokenization_method, 
                                  base_tokenizer_path, 
                                  model_config=None, 
                                  output_dir=""):
    """
    Train a language model with recommended hyperparams on a single A5000 GPU.
    
    Args:
        tokenized_files: Dictionary mapping language codes to splits to file paths
        tokenization_method: String identifier for the tokenization method
        base_tokenizer_path: Path to the base tokenizer JSON file
        model_config: Optional custom model configuration
    
    Returns:
        Path to the saved model or None if training failed
    """
    logger.info(f"Training language model on '{tokenization_method}' tokenized data...")

    # 1) Load datasets
    loader = TokenizedDatasetLoader(tokenized_files)
    train_ds = loader.load_combined_dataset('train')
    val_ds = loader.load_combined_dataset('val')
    test_ds = loader.load_combined_dataset('test')
    
    if train_ds is None or val_ds is None or test_ds is None:
        logger.error("Some splits missing => skipping.")
        return None

    # 2) Create HuggingFace tokenizer from the tokenizer JSON
    try:
        from transformers import PreTrainedTokenizerFast
        hf_tok = PreTrainedTokenizerFast(tokenizer_file=base_tokenizer_path)
        set_special_tokens(hf_tok)
        
        # Sanity check for special tokens
        from code.utils import SPECIAL_TOKENS
        vocab = hf_tok.get_vocab()
        sanity_check = {k: vocab.get(k, "MISSING") for k in SPECIAL_TOKENS.values()}
        logger.info("Created tokenizer for model. Special tokens: " + str(sanity_check))
        
        missing_tokens = [k for k, v in sanity_check.items() if v == "MISSING"]
        if missing_tokens:
            logger.warning(f"Missing special tokens: {missing_tokens}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        return None

    # 3) Build model
    model_name = model_config.get("base_model", "distilgpt2") if model_config else "distilgpt2"
    logger.info(f"Using {model_name} as base model")
    
    set_seed(42)  # For reproducibility
    wrapper = CustomCausalLMModel(hf_tok, model_name=model_name, config=model_config)
    wrapper.create_model()

    # 4) Define output directory
    model_size = "distil" if "distil" in model_name else model_name.replace("gpt2", "").strip("-") or "base"
    
    # 5) Train model
    train_result = wrapper.train(train_ds, val_ds, output_dir)
    if train_result is None:
        logger.error("Training failed.")
        return None
    
    # 6) Evaluate model
    eval_metrics = wrapper.evaluate(test_ds, output_dir)
    if eval_metrics:
        eval_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Eval results: {eval_metrics}")

    return output_dir


def train_models_from_config(config_dict):
    """
    Train models based on a configuration dictionary.
    
    """
    global_settings = config_dict.get("global_settings", {})
    tokenizers = config_dict.get("tokenizers", [])
    
    if not tokenizers:
        logger.error("No tokenizers defined in configuration")
        return {}
    
    models_output_dir = global_settings.get("model_output_dir", "")
    # Train models for each tokenizer
    results = {}
    for tokenizer_config in tokenizers:
        name = tokenizer_config.get("name")
        if not name:
            logger.warning("Skipping tokenizer with no name")
            continue
            
        tokenizer_path = tokenizer_config.get("tokenizer_path")
        tokenized_files = tokenizer_config.get("tokenized_files", {})
        model_config = tokenizer_config.get("model_config", {})
        
        if not tokenizer_path or not tokenized_files:
            logger.warning(f"Skipping {name}: missing tokenizer_path or tokenized_files")
            continue
            
        # Check if model already exists
        model_size = model_config.get("base_model", "distilgpt2")
        model_size = "distil" if "distil" in model_size else model_size.replace("gpt2", "").strip("-") or "base"
        output_dir = os.path.join(models_output_dir, f"{name}_model_{model_size}")
        
        if os.path.exists(output_dir) and global_settings.get("skip_existing", False):
            logger.info(f"Model for {name} already exists, skipping (skip_existing=True)")
            results[name] = output_dir
            continue
            
        logger.info(f"Training model for tokenizer: {name}")
        model_path = train_model_on_tokenized_data(
            tokenized_files, 
            name,
            tokenizer_path,
            model_config,
            output_dir
        )
        
        if model_path:
            results[name] = model_path
            
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language models on tokenized data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.config:
        if os.path.isfile(args.config):
            results = train_models_from_config(args.config)
            logger.info(f"Training completed. Models saved at: {results}")
        else:
            logger.error(f"Config file not found: {args.config}")
    else:
        logger.info("Example usage: python model_trainer.py --config config.json")
        logger.info("Create example config: python model_trainer.py --create-example")