#!/usr/bin/env python3
"""
Model Evaluator for Multilingual Tokenization Experiments

This script evaluates language models trained with experiment_runner.py
on test sets for each language, computing perplexity scores and generating
visualizations using plotnine.
"""

import os
import argparse
import json
import logging
import math
import pandas as pd
import numpy as np
import torch
import hashlib
from typing import Dict, Tuple, Any, Optional, Union, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from plotnine import (
    ggplot, aes, geom_bar, geom_point, geom_line, geom_errorbar,
    facet_wrap, labs, theme, scale_fill_brewer, scale_color_brewer,
    theme_minimal, position_dodge, coord_flip, scale_y_continuous,
    geom_boxplot, geom_violin, save_as_pdf_pages, element_text, 
    geom_abline, scale_x_log10, scale_y_log10,
    theme_bw, stat_ecdf, geom_vline, geom_hline
)

from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast
)

from code.utils import fix_input_ids, nonempty, set_special_tokens
from code.model_trainer import CustomDataCollator

# Import and setup environment variables
from code.environment_setup import setup_environment
setup_environment()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory setup
MODEL_EVAL_JSON_NAME = "all_models_evaluation.json"


class ModelEvaluator:
    """Class for evaluating language models on tokenized test sets."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        model_name: str,
        device: Optional[str] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model directory
            tokenizer_path: Path to the tokenizer file
            model_name: Name of the model for identification
            device: Device to use for evaluation (cpu or cuda)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Evaluator for {model_name} will use device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            # Try loading with AutoTokenizer first 
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            set_special_tokens(self.tokenizer)
                
            # Make sure padding token is set
            if self.tokenizer.pad_token is None:
                logger.error("No padding token set!!")
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "<pad>"
            if self.tokenizer.pad_token_id is None:
                logger.error("No padding token ID set!!")
                    
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
            
    def compute_perplexity_batch(self, batch_data, data_collator):
        """
        Compute perplexity for a batch using the same collator as training.
        Also computes perplexity per byte for fair cross-tokenizer comparison.
        
        Args:
            batch_data: Batch of examples from the dataset
            data_collator: The same CustomDataCollator used in training
            
        Returns:
            Tuple of (token_perplexities, byte_perplexities, byte_counts)
        """
        # Convert batch from Dataset format (dict of lists) to list of dicts for the collator
        batch_for_collator = []
        original_texts = []
        for i in range(len(batch_data['input_ids'])):
            example = {
                'input_ids': batch_data['input_ids'][i]
            }
            batch_for_collator.append(example)
            # Store original text for byte-level normalization
            original_texts.append(batch_data['text'][i])
        
        # Use the data collator to process the batch exactly as in training
        collated_batch = data_collator(batch_for_collator)
        
        # Move tensors to the correct device
        for k, v in collated_batch.items():
            collated_batch[k] = v.to(self.device)
        
        token_perplexities, byte_perplexities, byte_counts, ex_ids = [], [], [], []
        
        with torch.no_grad():
            # Forward pass with the collated batch
            outputs = self.model(**collated_batch)
            
            # The loss is already calculated across the batch
            # But we need per-example loss for perplexity
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            labels = collated_batch["labels"]  # [batch_size, seq_len]
            
            # Calculate per-example perplexity
            for i in range(logits.size(0)):
                # Extract single example
                example_logits = logits[i]  # [seq_len, vocab_size]
                example_labels = labels[i]  # [seq_len]
                
                # Shift logits and labels for next-token prediction
                shift_logits = example_logits[:-1]  # Remove last prediction
                shift_labels = example_labels[1:]   # Remove first token (we're predicting next tokens)
                
                # Identify non-padding positions (where label != pad_id)
                pad_id = self.tokenizer.pad_token_id
                non_pad_mask = (shift_labels != pad_id)
                
                # Only calculate loss on non-padding tokens
                if non_pad_mask.sum() == 0:
                    # All padding, skip this example
                    continue
                
                # Get the relevant logits and labels
                filtered_logits = shift_logits[non_pad_mask]
                filtered_labels = shift_labels[non_pad_mask]
                
                # Calculate cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(filtered_logits, filtered_labels)
                
                # Convert to perplexity
                token_perplexity = torch.exp(loss).item()
                token_perplexities.append(token_perplexity)
                
                # Calculate byte-normalized perplexity
                original_text = original_texts[i]
                ex_ids.append( make_example_id(original_text) )
                byte_count = len(original_text.encode('utf-8'))
                byte_counts.append(byte_count)
                
                # The number of tokens used for this example
                token_count = non_pad_mask.sum().item()
                
                if byte_count > 0 and token_count > 0:
                    # Calculate perplexity normalized by bytes
                    # First, get the total log probability (NLL) by multiplying by token count
                    total_nll = loss.item() * token_count
                    # Get NLL per byte
                    nll_per_byte = total_nll / byte_count
                    # Convert to byte-level perplexity
                    byte_perplexity = math.exp(nll_per_byte)
                    byte_perplexities.append(byte_perplexity)
                else:
                    byte_perplexities.append(float('inf'))
        
        return token_perplexities, byte_perplexities, byte_counts, ex_ids
    
    def evaluate_test_set(
        self, 
        test_file_path: str, 
        batch_size: int = 8, 
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a tokenized test set using the same processing as training.
        Computes both token-level and byte-level perplexity metrics.
        
        Args:
            test_file_path: Path to tokenized test set JSONL file
            batch_size: Batch size for evaluation
            max_samples: Maximum number of examples to evaluate
            
        Returns:
            Dictionary with evaluation results including byte-level metrics
        """
        logger.info(f"Evaluating on {test_file_path}")
        
        # Load test data
        test_data = []
        try:
            with open(test_file_path, 'r', encoding='utf-8', errors="replace") as f:
                for i, line in enumerate(f):
                    if max_samples is not None and i >= max_samples:
                        break
                    item = json.loads(line.strip())
                    test_data.append({
                            'input_ids': item['token_ids'],
                            'text': item['text']
                        })
        except Exception as e:
            logger.error(f"Error loading test data from {test_file_path}: {str(e)}")
            return {
                "error": str(e),
                "perplexity": float('inf'),
                "byte_perplexity": float('inf'),
                "token_count": 0,
                "byte_count": 0,
                "sample_count": 0
            }
            
        if not test_data:
            logger.warning(f"No test data found in {test_file_path}")
            return {
                "perplexity": float('inf'),
                "byte_perplexity": float('inf'),
                "token_count": 0,
                "byte_count": 0,
                "sample_count": 0
            }
        
        # Apply the same preprocessing functions used in training
        # First, create a dataset for consistent processing
        import datasets
        dataset = datasets.Dataset.from_list(test_data)
        
        # Apply the same fix_input_ids function used in training
        dataset = dataset.map(fix_input_ids)
        # Filter out empty sequences just like in training
        dataset = dataset.filter(nonempty)
        
        # Check if we still have data after filtering
        if len(dataset) == 0:
            logger.warning(f"No data left after preprocessing {test_file_path}")
            return {
                "perplexity": float('inf'),
                "byte_perplexity": float('inf'),
                "token_count": 0,
                "byte_count": 0,
                "sample_count": 0
            }
        
        # Set up the same data collator used in training            
        data_collator = CustomDataCollator(self.tokenizer.pad_token_id)
        
        # Process in batches
        all_token_perplexities = []
        all_byte_perplexities = []
        total_tokens = sum(len(item["input_ids"]) for item in dataset)
        total_bytes = 0
        per_example=[]
        
        # Process batches with progress indicator
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {os.path.basename(test_file_path)}"):
            batch = dataset[i:i+batch_size]
            token_perplexities, byte_perplexities, byte_counts, ex_ids = self.compute_perplexity_batch(batch, data_collator)
            all_token_perplexities.extend(token_perplexities)
            all_byte_perplexities.extend(byte_perplexities)
            total_bytes += sum(byte_counts)
            per_example.extend(
                dict(id=i, token_pp=t, byte_pp=b)           # NEW
                for i, t, b in zip(ex_ids, token_perplexities, byte_perplexities)
            )
            
        # Calculate token-level statistics
        if all_token_perplexities:
            avg_token_perplexity = np.mean(all_token_perplexities)
            median_token_perplexity = np.median(all_token_perplexities)
            min_token_perplexity = np.min(all_token_perplexities)
            max_token_perplexity = np.max(all_token_perplexities)
            std_token_perplexity = np.std(all_token_perplexities)
        else:
            avg_token_perplexity = float('inf')
            median_token_perplexity = float('inf')
            min_token_perplexity = float('inf')
            max_token_perplexity = float('inf')
            std_token_perplexity = 0.0
        
        # Calculate byte-level statistics
        if all_byte_perplexities:
            avg_byte_perplexity = np.mean(all_byte_perplexities)
            median_byte_perplexity = np.median(all_byte_perplexities)
            min_byte_perplexity = np.min(all_byte_perplexities)
            max_byte_perplexity = np.max(all_byte_perplexities)
            std_byte_perplexity = np.std(all_byte_perplexities)
        else:
            avg_byte_perplexity = float('inf')
            median_byte_perplexity = float('inf')
            min_byte_perplexity = float('inf')
            max_byte_perplexity = float('inf')
            std_byte_perplexity = 0.0
        
        # Return results with both token and byte metrics
        return {
            "perplexity": avg_token_perplexity,
            "median_perplexity": median_token_perplexity,
            "min_perplexity": min_token_perplexity,
            "max_perplexity": max_token_perplexity,
            "std_perplexity": std_token_perplexity,
            "byte_perplexity": avg_byte_perplexity,
            "median_byte_perplexity": median_byte_perplexity,
            "min_byte_perplexity": min_byte_perplexity,
            "max_byte_perplexity": max_byte_perplexity,
            "std_byte_perplexity": std_byte_perplexity,
            "token_count": total_tokens,
            "byte_count": total_bytes,
            "sample_count": len(dataset),
            "per_example": per_example
        }


def evaluate_model_on_all_languages(
    model_config: Dict[str, Any],
    tokenized_test_files: Dict[str, Dict[str, str]],
    max_samples_per_language: Optional[int] = None,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Evaluate a model on test sets for all available languages.
    
    Args:
        model_config: Dictionary with model configuration
        tokenized_test_files: Dictionary mapping language codes to test file paths
        max_samples_per_language: Maximum number of examples to evaluate per language
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results for each language
    """
    model_name = model_config["name"]
    model_path = model_config["model_path"]
    tokenizer_path = model_config["tokenizer_path"]
    
    logger.info(f"Evaluating model: {model_name}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_name=model_name
    )
    
    # Evaluate on each language
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "languages": {},
        "aggregate": {}
    }
    
    all_token_perplexities = []
    all_byte_perplexities = []
    all_token_counts = []
    all_byte_counts = []
    language_token_weights = {}
    language_byte_weights = {}

    per_example_by_lang = {}
    for lang_code, test_path in tokenized_test_files.items():
        logger.info(f"Evaluating {model_name} on language: {lang_code}")
        
        lang_results = evaluator.evaluate_test_set(
            test_file_path=test_path,
            batch_size=batch_size,
            max_samples=max_samples_per_language
        )
        for row in lang_results["per_example"]:
            per_example_by_lang.setdefault(lang_code, {}).setdefault(row["id"], {})[
                model_name
            ] = row["token_pp"]

        results["languages"][lang_code] = lang_results
        
        if "error" not in lang_results:
            all_token_perplexities.append(lang_results["perplexity"])
            all_byte_perplexities.append(lang_results["byte_perplexity"])
            
            token_count = lang_results["token_count"]
            byte_count = lang_results["byte_count"]
            
            all_token_counts.append(token_count)
            all_byte_counts.append(byte_count)
            
            language_token_weights[lang_code] = token_count
            language_byte_weights[lang_code] = byte_count
    
    # Calculate aggregate statistics including byte-level metrics
    if all_token_perplexities:
        # Token-level statistics
        avg_token_perplexity = np.mean(all_token_perplexities)
        median_token_perplexity = np.median(all_token_perplexities)
        std_token_perplexity = np.std(all_token_perplexities)
        
        # Token-weighted average
        token_weighted_perplexity = np.average(
            all_token_perplexities, 
            weights=[language_token_weights[lang] for lang in results["languages"].keys()]
        )
        
        # Byte-level statistics
        avg_byte_perplexity = np.mean(all_byte_perplexities)
        median_byte_perplexity = np.median(all_byte_perplexities)
        std_byte_perplexity = np.std(all_byte_perplexities)
        
        # Byte-weighted average
        byte_weighted_perplexity = np.average(
            all_byte_perplexities, 
            weights=[language_byte_weights[lang] for lang in results["languages"].keys()]
        )
        
        total_tokens = sum(all_token_counts)
        total_bytes = sum(all_byte_counts)
        
        results["aggregate"] = {
            # Token-level metrics
            "perplexity": avg_token_perplexity,
            "median_perplexity": median_token_perplexity,
            "std_perplexity": std_token_perplexity,
            "weighted_perplexity": token_weighted_perplexity,
            "total_tokens": total_tokens,
            
            # Byte-level metrics
            "byte_perplexity": avg_byte_perplexity,
            "median_byte_perplexity": median_byte_perplexity,
            "std_byte_perplexity": std_byte_perplexity,
            "weighted_byte_perplexity": byte_weighted_perplexity,
            "total_bytes": total_bytes,
            
            "language_count": len(results["languages"]),
            "per_example_by_lang": per_example_by_lang

        }
    
    return results


def evaluate_all_models_on_all_languages(
    model_configs: List[Dict[str, Any]],
    tokenized_data: Dict[str, Dict[str, Dict[str, str]]],
    max_samples_per_language: Optional[int] = None,
    batch_size: int = 8,
    output_dir: str = 'evaluation_results'
) -> Dict[str, Any]:
    """
    Evaluate all models on all languages.
    
    Args:
        model_configs: List of model configurations
        tokenized_data: Dictionary mapping model names to language codes to test file paths
        max_samples_per_language: Maximum number of examples to evaluate per language
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results for all models
    """
    all_results = {}
    
    for model_config in model_configs:
        model_name = model_config["name"]
        
        # Get test files for this model
        if model_name not in tokenized_data: 
            logger.warning(f"No tokenized test data found for model {model_name}")
            continue
            
        # Extract test files
        test_files = {}
        for lang_code, splits in tokenized_data[model_name].items():
            if "test" in splits:
                test_files[lang_code] = splits["test"]
        
        if not test_files:
            logger.warning(f"No test files found for model {model_name}")
            continue
            
        # Evaluate model
        model_results = evaluate_model_on_all_languages(
            model_config=model_config,
            tokenized_test_files=test_files,
            max_samples_per_language=max_samples_per_language,
            batch_size=batch_size
        )
        
        all_results[model_name] = model_results
        
        # Save individual model results
        output_path = os.path.join(output_dir, f"{model_name}_evaluation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2)
        logger.info(f"Saved results for {model_name} to {output_path}")
    
    # Save combined results
    combined_path = os.path.join(output_dir, MODEL_EVAL_JSON_NAME)
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {combined_path}")
    
    return all_results


def create_dataframes(
    evaluation_data: Union[Dict[str, Any], str],
    combined_file: str = MODEL_EVAL_JSON_NAME,
    individual_files: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create pandas DataFrames from evaluation results or result files.
    
    Args:
        evaluation_data: Either a dictionary with evaluation results or a path to the results directory
        combined_file: Name of the combined results file (used only if evaluation_data is a directory path)
        individual_files: Whether to read individual model files (used only if evaluation_data is a directory path)
        
    Returns:
        Tuple of (per_language_df, aggregate_df)
    """
    # Check if we need to load evaluation results from files
    if isinstance(evaluation_data, str):
        # Assume it's a directory path
        results_dir = evaluation_data
        
        # Get the evaluation results using the read_evaluation_results function
        evaluation_results = read_evaluation_results(
                results_dir=results_dir,
                combined_file=combined_file,
                individual_files=individual_files
            )
    else:
        # Use the provided evaluation results directly
        evaluation_results = evaluation_data
    
    lang_data = []
    
    for model_name, model_results in evaluation_results.items():
        # Handle different result structures
        if "languages" in model_results:
            # This is an individual model result file
            languages_data = model_results["languages"]
            model_name_to_use = model_results.get("model_name", model_name)
        else:
            # This is from the combined file format
            languages_data = model_results.get("languages", {})
            model_name_to_use = model_name
        
        for lang_code, lang_results in languages_data.items():
            if "error" not in lang_results:
                lang_data.append({
                    "model": model_name_to_use,
                    "language": lang_code,
                    # Token-level metrics
                    "perplexity": lang_results.get("perplexity", float('inf')),
                    "median_perplexity": lang_results.get("median_perplexity", lang_results.get("perplexity", float('inf'))),
                    "std_perplexity": lang_results.get("std_perplexity", 0),
                    # Byte-level metrics
                    "byte_perplexity": lang_results.get("byte_perplexity", float('inf')),
                    "median_byte_perplexity": lang_results.get("median_byte_perplexity", lang_results.get("byte_perplexity", float('inf'))),
                    "std_byte_perplexity": lang_results.get("std_byte_perplexity", 0),
                    # Counts
                    "token_count": lang_results.get("token_count", 0),
                    "byte_count": lang_results.get("byte_count", 0),
                    "sample_count": lang_results.get("sample_count", 0)
                })
    
    per_language_df = pd.DataFrame(lang_data)
    
    # Create aggregate DataFrame with byte-level metrics
    agg_data = []
    
    for model_name, model_results in evaluation_results.items():
        # Handle different result structures
        if "aggregate" in model_results:
            # This is an individual model result file
            agg_results = model_results["aggregate"]
            model_name_to_use = model_results.get("model_name", model_name)
        else:
            # This is from the combined file format
            agg_results = model_results.get("aggregate", {})
            model_name_to_use = model_name
            
        if agg_results:
            agg_data.append({
                "model": model_name_to_use,
                # Token-level metrics
                "perplexity": agg_results.get("perplexity", float('inf')),
                "median_perplexity": agg_results.get("median_perplexity", agg_results.get("perplexity", float('inf'))),
                "weighted_perplexity": agg_results.get("weighted_perplexity", agg_results.get("perplexity", float('inf'))),
                "std_perplexity": agg_results.get("std_perplexity", 0),
                # Byte-level metrics
                "byte_perplexity": agg_results.get("byte_perplexity", float('inf')),
                "median_byte_perplexity": agg_results.get("median_byte_perplexity", agg_results.get("byte_perplexity", float('inf'))),
                "weighted_byte_perplexity": agg_results.get("weighted_byte_perplexity", agg_results.get("byte_perplexity", float('inf'))),
                "std_byte_perplexity": agg_results.get("std_byte_perplexity", 0),
                # Counts
                "language_count": agg_results.get("language_count", 0),
                "total_tokens": agg_results.get("total_tokens", 0),
                "total_bytes": agg_results.get("total_bytes", 0)
            })
    
    aggregate_df = pd.DataFrame(agg_data)
    
    return per_language_df, aggregate_df, evaluation_results


def build_per_example_delta_df(all_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten the nested dict structure into one tidy DataFrame:
    language | example_id | model | token_pp
    and then pivot so you can compute deltas easily.
    """
    rows = []
    for model, mres in all_results.items():
        if not "languages" in mres:
            continue
        for lang, lres in mres["languages"].items():
            for ex in lres["per_example"]:
                rows.append(
                    dict(language=lang,
                         example_id=ex["id"],
                         model=model,
                         token_perplexity=ex["token_pp"])
                )
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["language", "example_id"],
                          columns="model",
                          values="token_perplexity")


def generate_plots(
    per_language_df: pd.DataFrame, 
    aggregate_df: pd.DataFrame,
    output_dir: str =  "plots"
) -> Dict[str, str]:
    """
    Generate evaluation plots using plotnine.
    
    Args:
        per_language_df: DataFrame with per-language results
        aggregate_df: DataFrame with aggregate results
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    plot_paths = {}
    
    # 1. Aggregate perplexity comparison
    if not aggregate_df.empty:
        plot = (ggplot(aggregate_df, aes(x='model', y='perplexity', fill='model'))
                + geom_bar(stat='identity')
                + geom_errorbar(aes(ymin='perplexity-std_perplexity', 
                                   ymax='perplexity+std_perplexity'), 
                               width=0.2)
                + labs(title='Aggregate Perplexity Comparison',
                      x='Model', y='Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "aggregate_perplexity.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["aggregate_perplexity"] = plot_path
        
        # Also plot weighted perplexity
        plot = (ggplot(aggregate_df, aes(x='model', y='weighted_perplexity', fill='model'))
                + geom_bar(stat='identity')
                + labs(title='Token-Weighted Aggregate Perplexity',
                      x='Model', y='Weighted Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "weighted_perplexity.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["weighted_perplexity"] = plot_path
    
    # 2. Per-language perplexity comparison
    if not per_language_df.empty:
        # Ensure we don't have too many languages to display nicely
        languages = per_language_df['language'].unique()
        
        if len(languages) <= 20:
            # Create a single plot for all languages
            plot = (ggplot(per_language_df, aes(x='language', y='perplexity', fill='model'))
                    + geom_bar(stat='identity', position=position_dodge())
                    + labs(title='Per-Language Perplexity Comparison',
                          x='Language', y='Perplexity (lower is better)')
                    + theme_minimal()
                    + theme(axis_text_x=element_text(angle=45, hjust=1)))
            
            plot_path = os.path.join(output_dir, "per_language_perplexity.png")
            plot.save(plot_path, dpi=300, width=12, height=8)
            plot_paths["per_language_perplexity"] = plot_path
        else:
            # Create a faceted plot by model
            plot = (ggplot(per_language_df, aes(x='language', y='perplexity', fill='language'))
                    + geom_bar(stat='identity')
                    + facet_wrap('~model')
                    + labs(title='Per-Language Perplexity By Model',
                          x='Language', y='Perplexity (lower is better)')
                    + theme_minimal()
                    + theme(axis_text_x=element_text(angle=90, hjust=1), 
                           legend_position='none'))
            
            plot_path = os.path.join(output_dir, "per_language_perplexity_facet.png")
            plot.save(plot_path, dpi=300, width=15, height=10)
            plot_paths["per_language_perplexity_facet"] = plot_path
        
        # 3. Boxplot of perplexity distribution by model
        plot = (ggplot(per_language_df, aes(x='model', y='perplexity', fill='model'))
                + geom_boxplot()
                + labs(title='Perplexity Distribution by Model',
                      x='Model', y='Perplexity (lower is better)')
                + theme_minimal())
        
        plot_path = os.path.join(output_dir, "perplexity_boxplot.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["perplexity_boxplot"] = plot_path
        
        # 4. Violin plot with points
        plot = (ggplot(per_language_df, aes(x='model', y='perplexity', fill='model'))
                + geom_violin(alpha=0.7)
                + geom_point(aes(color='model'), alpha=0.5, position=position_dodge(width=0.9))
                + labs(title='Perplexity Distribution with Individual Languages',
                      x='Model', y='Perplexity (lower is better)')
                + theme_minimal())
        
        plot_path = os.path.join(output_dir, "perplexity_violin.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["perplexity_violin"] = plot_path
        
        # 5. Heat map of perplexity by model and language
        # Pivot the DataFrame
        heatmap_df = per_language_df.pivot(index='language', columns='model', values='perplexity')
        
        # Plot the heatmap using matplotlib directly (plotnine doesn't have a good heatmap)
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_df.values, cmap='viridis_r')  # reverse viridis: lower is better
        
        # Add labels
        plt.colorbar(label='Perplexity (lower is better)')
        plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha='right')
        plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
        plt.title('Perplexity Heatmap by Model and Language')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "perplexity_heatmap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        plot_paths["perplexity_heatmap"] = plot_path
    
    return plot_paths

def generate_byte_perplexity_plots(
    per_language_df: pd.DataFrame, 
    aggregate_df: pd.DataFrame,
    output_dir: str = "plots"
) -> Dict[str, str]:
    """
    Generate plots specifically for byte-level perplexity comparisons.
    """
    from plotnine import (
        ggplot, aes, geom_bar, geom_point, geom_errorbar,
        facet_wrap, labs, theme, theme_minimal, position_dodge,
        element_text, scale_y_continuous, geom_boxplot, geom_violin
    )
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    # 1. Aggregate byte perplexity comparison
    if not aggregate_df.empty:
        plot = (ggplot(aggregate_df, aes(x='model', y='byte_perplexity', fill='model'))
                + geom_bar(stat='identity')
                + geom_errorbar(aes(ymin='byte_perplexity-std_byte_perplexity', 
                                   ymax='byte_perplexity+std_byte_perplexity'), 
                               width=0.2)
                + labs(title='Aggregate Byte-Level Perplexity Comparison',
                      x='Model', y='Byte Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "aggregate_byte_perplexity.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["aggregate_byte_perplexity"] = plot_path
        
        # Also plot weighted byte perplexity
        plot = (ggplot(aggregate_df, aes(x='model', y='weighted_byte_perplexity', fill='model'))
                + geom_bar(stat='identity')
                + labs(title='Byte-Weighted Aggregate Perplexity',
                      x='Model', y='Weighted Byte Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "weighted_byte_perplexity.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["weighted_byte_perplexity"] = plot_path
        
        # 3. Comparison between token and byte perplexities
        comparison_df = pd.DataFrame({
            'model': aggregate_df['model'].tolist() + aggregate_df['model'].tolist(),
            'perplexity_type': ['Token'] * len(aggregate_df) + ['Byte'] * len(aggregate_df),
            'perplexity': aggregate_df['perplexity'].tolist() + aggregate_df['byte_perplexity'].tolist()
        })
        
        plot = (ggplot(comparison_df, aes(x='perplexity_type', y='perplexity', fill='model'))
                + geom_bar(stat='identity', position=position_dodge())
                + labs(title='Token vs Byte Perplexity Comparison',
                      x='Model', y='Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "token_vs_byte_perplexity.png")
        plot.save(plot_path, dpi=300, width=12, height=6)
        plot_paths["token_vs_byte_perplexity"] = plot_path
    
    # 2. Per-language byte perplexity comparison
    if not per_language_df.empty:
        # Create a single plot for all languages
        plot = (ggplot(per_language_df, aes(x='language', y='byte_perplexity', fill='model'))
                + geom_bar(stat='identity', position=position_dodge())
                + labs(title='Per-Language Byte Perplexity Comparison',
                      x='Language', y='Byte Perplexity (lower is better)')
                + theme_minimal()
                + theme(axis_text_x=element_text(angle=45, hjust=1)))
        
        plot_path = os.path.join(output_dir, "per_language_byte_perplexity.png")
        plot.save(plot_path, dpi=300, width=12, height=8)
        plot_paths["per_language_byte_perplexity"] = plot_path
        
        # Boxplot of byte perplexity distribution by model
        plot = (ggplot(per_language_df, aes(x='model', y='byte_perplexity', fill='model'))
                + geom_boxplot()
                + labs(title='Byte Perplexity Distribution by Model',
                      x='Model', y='Byte Perplexity (lower is better)')
                + theme_minimal())
        
        plot_path = os.path.join(output_dir, "byte_perplexity_boxplot.png")
        plot.save(plot_path, dpi=300, width=10, height=6)
        plot_paths["byte_perplexity_boxplot"] = plot_path
    
    return plot_paths

def _plot_deltas_pair(df: pd.DataFrame,
                      output_dir: str,
                      model_a: str,
                      model_b: str) -> None:
    """
    Produce three PNGs for (model_a, model_b):
      1) per‑language scatter with log axes
      2) per‑language ECDF (“win curve”)
      3) violin+box plot of Δ across languages
    """
    delta_col = f"delta_{model_a}_vs_{model_b}"
    if delta_col not in df.columns:
        df[delta_col] = df[model_a] - df[model_b]

    # ------- 1 & 2: per‑language plots ----------------------------------
    sub = df.reset_index()                    # keep language column

    # 1) aggregated scatter ---------------------------------------------
    p_scatter = (
        ggplot(sub, aes(x=model_a, y=model_b, colour="language"))
        + geom_point(alpha=.4, size=1)
        + geom_abline(intercept=0, slope=1, linetype="dashed", colour="grey")
        + scale_x_log10() + scale_y_log10()
        + labs(title=f"{model_a} vs {model_b} – per‑example perplexity",
               x=f"{model_a} perplexity",
               y=f"{model_b} perplexity",
               colour="language")
        + theme_bw()
    )
    p_scatter.save(os.path.join(output_dir,
           f"scatter_all_{model_a}_vs_{model_b}.png"), dpi=300)

    # 2) aggregated ECDF (“win curve”) -----------------------------------
    p_cdf = (
        ggplot(sub, aes(x=delta_col, colour="language"))
        + stat_ecdf(geom="step")
        + geom_vline(xintercept=0, linetype="dashed", colour="grey")
        + labs(title=f"Δ perplexity ({model_a} – {model_b}) win curve",
               x=f"Δ perplexity ({model_a} – {model_b})",
               y="cumulative proportion",
               colour="language")
        + theme_bw()
    )
    p_cdf.save(os.path.join(output_dir,
           f"cdf_all_{model_a}_vs_{model_b}.png"),dpi=300)

    # 3) violin/box – unchanged, still grouped by language --------------
    vdf = sub.copy()
    vdf["delta"] = vdf[delta_col]
    p_violin = (
        ggplot(vdf, aes(x="language", y="delta"))
        + geom_violin(fill="#A5C0D1", alpha=.7, width=.9)
        + geom_boxplot(width=.1, outlier_size=.4)
        + geom_hline(yintercept=0, linetype="dashed", colour="grey")
        + labs(title=f"Δ perplexity ({model_a} – {model_b}) by language",
               x="", y="Δ token‑level perplexity")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )
    p_violin.save(os.path.join(output_dir,
           f"violin_{model_a}_vs_{model_b}.png"), dpi=300, width=10, height=6)


def find_tokenized_test_data(
    data_dir: str = 'tokenized'
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Find all tokenized test data files.
    
    Args:
        data_dir: Directory to search for tokenized data
        
    Returns:
        Dictionary mapping tokenizer names to language codes to test file paths
    """
    tokenized_data = {}
    
    # Walk through the tokenized data directory
    for tokenizer_dir in os.listdir(data_dir):
        tokenizer_path = os.path.join(data_dir, tokenizer_dir)
        if not os.path.isdir(tokenizer_path):
            continue
            
        tokenized_data[tokenizer_dir] = {}
        
        # Find all language subdirectories
        for lang_dir in os.listdir(tokenizer_path):
            lang_path = os.path.join(tokenizer_path, lang_dir)
            if not os.path.isdir(lang_path):
                continue
                
            # Check for train, val, test files
            splits = {}
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(lang_path, f"{split}.jsonl")
                if os.path.isfile(split_path):
                    splits[split] = split_path
                    
            if splits:
                tokenized_data[tokenizer_dir][lang_dir] = splits
    
    return tokenized_data


def find_models_and_tokenizers(
    models_dir: str = "models",
    tokenizers_dir: str = "tokenizers"
) -> List[Dict[str, Any]]:
    """
    Find all trained models and their corresponding tokenizers.
    
    Args:
        models_dir: Directory containing trained models
        tokenizers_dir: Directory containing tokenizers
        
    Returns:
        List of dictionaries with model information
    """
    model_configs = []
    
    # Look for model directories
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if not os.path.isdir(model_path):
            continue
            
        # Extract model name from directory name
        model_name = item.split('_model')[0] if '_model' in item else item
        
        # Look for tokenizer file
        tokenizer_path = None
        # First check if a tokenizer.json exists in the model directory
        model_tokenizer = os.path.join(model_path, "tokenizer.json")
        if os.path.isfile(model_tokenizer):
            tokenizer_path = model_tokenizer
        else:
            # Otherwise look in the tokenizers directory
            for tokenizer_file in os.listdir(tokenizers_dir):
                if tokenizer_file.startswith(model_name) and tokenizer_file.endswith('.json'):
                    tokenizer_path = os.path.join(tokenizers_dir, tokenizer_file)
                    break
                elif 'base' in tokenizer_file and model_name in tokenizer_file:
                    tokenizer_path = os.path.join(tokenizers_dir, tokenizer_file)
                    break
        
        if tokenizer_path:
            model_configs.append({
                "name": model_name,
                "model_path": model_path,
                "tokenizer_path": tokenizer_path
            })
        else:
            logger.warning(f"Could not find tokenizer for model {model_name}")
    
    return model_configs

def read_evaluation_results(
    results_dir: str,
    combined_file: str = MODEL_EVAL_JSON_NAME,
    individual_files: bool = False
) -> Dict[str, Any]:
    """
    Read evaluation results from JSON files produced by evaluate_all_models_on_all_languages.
    
    Args:
        results_dir: Directory containing the evaluation result files
        combined_file: Name of the combined results file
        individual_files: Whether to read individual model files if combined file is not found
        
    Returns:
        Dictionary with evaluation results
    """
    # First try to read the combined file
    combined_path = os.path.join(results_dir, combined_file)
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'r', encoding='utf-8', errors="replace") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading combined results file: {str(e)}")
            if not individual_files:
                raise
    
    # If combined file not found or errored, try individual files
    if individual_files or not os.path.exists(combined_path):
        evaluation_results = {}
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(results_dir) 
                     if f.endswith('_evaluation.json') and f != combined_file]
        
        if not json_files:
            raise FileNotFoundError(f"No evaluation result files found in {results_dir}")
        
        # Read each file
        for file_name in json_files:
            try:
                file_path = os.path.join(results_dir, file_name)
                with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
                    model_results = json.load(f)
                
                # Extract model name from file name
                model_name = file_name.replace('_evaluation.json', '')
                evaluation_results[model_name] = model_results
                
            except Exception as e:
                print(f"Error reading {file_name}: {str(e)}")
        
        return evaluation_results
    
    # If we get here, no results could be loaded
    raise FileNotFoundError(f"Could not find or read evaluation results in {results_dir}")

def make_example_id(text: str) -> str:
    """
    Deterministically hash the raw text so the same sentence
    gets the same id no matter which tokenizer shuffled it.
    """
    return hashlib.md5(text.encode("utf‑8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on tokenized test sets")
    
    # Input options
    parser.add_argument("--models-dir", type=str, required=True, 
                       help="Directory containing trained models")
    parser.add_argument("--tokenizers-dir", type=str, required=True, 
                       help="Directory containing trained models")
    parser.add_argument("--tokenized-dir", type=str, required=True,
                       help="Directory containing tokenized data")
    parser.add_argument("--output-dir", type=str, default='evaluation_results', required=True,
                       help="Directory to save evaluation results")
    
    # Evaluation options
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of examples to evaluate per language")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--models", type=str, nargs='+',
                       help="Specific models to evaluate (default: all)")
    parser.add_argument("--languages", type=str, nargs='+',
                       help="Specific languages to evaluate (default: all)")
    parser.add_argument("--use-saved-results", action="store_true")
    parser.add_argument(
        "--plot-deltas",
        action="store_true",
        help="If set, create scatter/violin/CDF plots for per‑example Δ‑perplexity.",
    )
    parser.add_argument(
        "--compare-models",
        default=None,
        metavar="MODEL_A,MODEL_B",
        help="Comma‑separated model names to compare (defaults to first two).",
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not args.use_saved_results:
    
    # Find models and tokenizers
        model_configs = find_models_and_tokenizers(args.models_dir, args.tokenizers_dir)
        
        if args.models:
            # Filter models by name
            model_configs = [config for config in model_configs if config["name"] in args.models]
        
        if not model_configs:
            logger.error("No models found for evaluation")
            return 1
        
        logger.info(f"Found {len(model_configs)} models for evaluation")
        
        # Find tokenized test data
        tokenized_data = find_tokenized_test_data(args.tokenized_dir)
        
        if args.languages:
            # Filter by language
            for model_name in tokenized_data:
                tokenized_data[model_name] = {
                    lang: data for lang, data in tokenized_data[model_name].items()
                    if lang in args.languages
                }
        
        # Evaluate all models
        evaluation_results = evaluate_all_models_on_all_languages(
            model_configs=model_configs,
            tokenized_data=tokenized_data,
            max_samples_per_language=args.max_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Create DataFrames for plotting
        per_language_df, aggregate_df, _ = create_dataframes(evaluation_results)

    else:

        per_language_df, aggregate_df, evaluation_results = create_dataframes(
            args.output_dir,
            combined_file=os.path.join(args.output_dir, MODEL_EVAL_JSON_NAME)
        )
    
    # Generate plots
    generate_plots(per_language_df, aggregate_df, plots_dir)
    generate_byte_perplexity_plots(per_language_df, aggregate_df, plots_dir)
    if args.plot_deltas:
        df = build_per_example_delta_df(evaluation_results)

        if args.compare_models:
            pairs = [tuple(m.strip() for m in args.compare_models.split(","))]
        else:
            # every unordered pair of the model columns
            pairs = list(itertools.combinations(df.columns, 2))
    
        for model_a, model_b in pairs:
            _plot_deltas_pair(df.copy(), plots_dir, model_a, model_b)

    # Save DataFrames for reference
    per_language_df.to_csv(os.path.join(args.output_dir, "per_language_results.csv"), index=False)
    aggregate_df.to_csv(os.path.join(args.output_dir, "aggregate_results.csv"), index=False)
    
    # Print summary
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"Plots saved to {plots_dir}")
    
    # Print aggregate results table
    if not aggregate_df.empty:
        print("\nAggregate Results:")
        print(aggregate_df[['model', 'perplexity', 'weighted_perplexity', 'language_count']].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    exit(main())
