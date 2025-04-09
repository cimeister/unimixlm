# model_evaluator.py

import os
import logging
import json
import math
import torch
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments
)
from code.model_trainer import CustomDataCollator, TokenizedDatasetLoader
from code.utils import fix_input_ids, nonempty

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_plain_text_test_dataset(test_file, tokenizer, max_lines=None):
    """
    Load test data from a plain text file and tokenize each line using the provided HF tokenizer.
    Each non-empty line is tokenized to produce an "input_ids" field.
    Returns a HuggingFace Dataset.
    """
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            data.append({"input_ids": ids})
    return Dataset.from_list(data)


def evaluate_model(
    model_dir,
    test_file=None,
    tokenized_files=None,
    tokenizer_dir=None,
    split="test",
    max_lines=None,
    batch_size=8
):
    """
    Evaluate a model produced by model_trainer.py.
    
    Either pass a plain text test_file (for on-the-fly tokenization)
    OR pass tokenized_files (a dict: {lang: {split: file_path, ...}}).
    
    The model_dir is assumed to contain the saved model and the final re-indexed tokenizer.
    Evaluation metrics (e.g., perplexity) are computed from the loss.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    logger.info(f"Loaded model from {model_dir}")

    # Load the tokenizer: if tokenizer_dir is provided, load from it; otherwise from model_dir.
    if not tokenizer_dir:
        hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        logger.info(f"Loaded tokenizer from {model_dir}")
    else:
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_dir)
        logger.info(f"Loaded tokenizer from {tokenizer_dir}")

    # Prepare test dataset
    if tokenized_files:
        loader = TokenizedDatasetLoader(tokenized_files)
        test_dataset = loader.load_combined_dataset(split)
        if test_dataset is None:
            logger.error("Failed to load tokenized dataset.")
            return None
        logger.info(f"Loaded combined tokenized dataset for split '{split}' with {len(test_dataset)} examples.")
    elif test_file:
        test_dataset = load_plain_text_test_dataset(test_file, hf_tokenizer, max_lines=max_lines)

        logger.info(f"Loaded plain text test dataset from {test_file} with {len(test_dataset)} examples.")
    else:
        logger.error("No test file or pre-tokenized data provided.")
        return None

    test_dataset = test_dataset.map(fix_input_ids)
    test_dataset = test_dataset.filter(nonempty)

    # Set up data collator
    pad_id = hf_tokenizer.pad_token_id or 0
    logger.info(f"Pad ID: {pad_id}")
    collator = CustomDataCollator(pad_id)

    # Create evaluation arguments, and importantly disable column removal
    eval_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "temp_eval"),
        per_device_eval_batch_size=batch_size,
        fp16=True,
        report_to=None
    )
    max_id = 0
    i_ = None
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        max_id_cur = max(example["input_ids"])
        if max_id_cur > max_id:
            max_id = max_id_cur
            i_ = i

    logger.info(f"Example with largest ID {i_}: max_id={max_id}, len={len(test_dataset[i_]['input_ids'])}")
    logger.info(f"Model vocab size: {model.config.vocab_size}")
    # Explicitly set label_names so that the Trainer does not drop our 'labels' column.
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collator,
        eval_dataset=test_dataset,
    )

    logger.info("Starting evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Raw evaluation metrics: {metrics}")
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    logger.info(f"Final evaluation metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    # Example usage:
    # For pre-tokenized evaluation:
    
    tokenized_files = {}
    tokenized_file_dir = os.path.join('data', 'tokenized', 'langspec_viterbi_parallel')
    for lang_code in os.listdir(tokenized_file_dir):
        if os.path.isdir(os.path.join(tokenized_file_dir, lang_code)):
            tokenized_files[lang_code] = {
                'test': os.path.join(tokenized_file_dir, lang_code, "test.jsonl"),
                'val': os.path.join(tokenized_file_dir, lang_code, "val.jsonl")
            }

    tokenizer_dir = 'data/tokenizers/langspec_base_tokenizer.json'
    model_dir = "data/models/langspec_model_distil"
    metrics = evaluate_model(model_dir, tokenized_files=tokenized_files, tokenizer_dir=tokenizer_dir, split="test")

    print(json.dumps(metrics, indent=2))

    print("-----------------")

    tokenized_files = {}
    tokenized_file_dir = os.path.join('data', 'tokenized', 'multigram_viterbi_parallel')
    for lang_code in os.listdir(tokenized_file_dir):
        if os.path.isdir(os.path.join(tokenized_file_dir, lang_code)):
            tokenized_files[lang_code] = {
                'test': os.path.join(tokenized_file_dir, lang_code, "test.jsonl"),
                'val': os.path.join(tokenized_file_dir, lang_code, "val.jsonl")
            }

    tokenizer_dir = 'data/tokenizers/multi_base_tokenizer.json'
    model_dir = "data/models/multigram_model_distil"
    metrics = evaluate_model(model_dir, tokenized_files=tokenized_files, tokenizer_dir=tokenizer_dir, split="test")

    print(json.dumps(metrics, indent=2))

    print("-----------------")

    tokenized_files = {}
    tokenized_file_dir = os.path.join('data', 'tokenized', 'langspec_multigram_viterbi_parallel')
    for lang_code in os.listdir(tokenized_file_dir):
        if os.path.isdir(os.path.join(tokenized_file_dir, lang_code)):
            tokenized_files[lang_code] = {
                'test': os.path.join(tokenized_file_dir, lang_code, "test.jsonl"),
                'val': os.path.join(tokenized_file_dir, lang_code, "val.jsonl")
            }

    tokenizer_dir = 'data/tokenizers/multi_base_tokenizer.json'
    model_dir = "data/models/multi-langspec_model_distil"
    metrics = evaluate_model(model_dir, tokenized_files=tokenized_files, tokenizer_dir=tokenizer_dir, split="test")

    print(json.dumps(metrics, indent=2))

    print("-----------------")

