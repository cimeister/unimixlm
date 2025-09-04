import argparse, os, json, logging, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from types import SimpleNamespace
from code.utils import set_special_tokens, encode_text, load_tokenizer_from_config
from tqdm import tqdm

import numpy as np
import torch, datasets, evaluate
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback,
    IntervalStrategy,
    GPT2ForSequenceClassification,
    AutoConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_gpt_for_classification(model_path, num_labels, tokenizer, model_kwargs):
    """
    Properly initialize GPT model for classification tasks.
    """
    
    # Load config first
    config = AutoConfig.from_pretrained(model_path)
    
    # CRITICAL: Set all necessary config parameters
    config.num_labels = num_labels
    config.pad_token_id = tokenizer.pad_token_id
    
    # Load model with updated config - DON'T pass num_labels separately!
    model = GPT2ForSequenceClassification.from_pretrained(
        model_path,
        config=config,  # Config already has num_labels
        ignore_mismatched_sizes=True,
        **model_kwargs
    )
    
    # Initialize classification head properly if it's new
    if hasattr(model, 'score') and model.score.weight.requires_grad:
        # The classification head was randomly initialized
        # Use a smaller std for initialization
        torch.nn.init.normal_(model.score.weight, std=0.02)
        if model.score.bias is not None:
            torch.nn.init.zeros_(model.score.bias)
        logger.info("Initialized classification head with smaller std")
    
    return model


def diagnose_training_issues(model, tokenizer, train_dataset, args):
    """Diagnose why training might not be working."""
    
    print("\n=== DIAGNOSTIC CHECKS ===")
    
    # 1. Check model architecture
    print(f"Model type: {model.config.model_type}")
    print(f"Num labels: {model.config.num_labels}")
    print(f"Pad token ID in model config: {model.config.pad_token_id}")
    
    # 2. Check tokenizer
    print(f"\nTokenizer padding side: {tokenizer.padding_side}")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # 3. Check a batch of data
    print("\n=== SAMPLE BATCH ===")
    batch_size = 4
    batch = [train_dataset[i] for i in range(min(batch_size, len(train_dataset)))]
    
    # Check sequence format
    for i, example in enumerate(batch):
        input_ids = example['input_ids']
        
        # Handle both tensor and list inputs
        if torch.is_tensor(input_ids):
            input_ids_list = input_ids.tolist()
        else:
            input_ids_list = input_ids
        
        # Find first non-pad token
        first_non_pad = next((j for j, tid in enumerate(input_ids_list) if tid != tokenizer.pad_token_id), 0)
        # Find last non-pad token
        last_non_pad = len(input_ids_list) - 1 - next((j for j, tid in enumerate(reversed(input_ids_list)) if tid != tokenizer.pad_token_id), 0)
        
        print(f"\nExample {i}:")
        print(f"  Decoded: {tokenizer.decode(input_ids_list)}")
        print(f"  First 10 tokens: {input_ids_list[first_non_pad:first_non_pad+10]}")
        print(f"  Last 10 tokens: {input_ids_list[max(last_non_pad-9, 0):last_non_pad+1]}")
        print(f"  Label: {example.get('label', 'N/A')}")
        print(f"  Ends with EOS? {input_ids_list[last_non_pad] == tokenizer.eos_token_id}")
    
    # 4. Test forward pass
    print("\n=== FORWARD PASS TEST ===")
    try:
        model.eval()
        with torch.no_grad():
            # Handle tensor/list conversion properly
            if torch.is_tensor(batch[0]['input_ids']):
                input_ids = torch.stack([ex['input_ids'] for ex in batch])
                attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
            else:
                input_ids = torch.tensor([ex['input_ids'] for ex in batch])
                attention_mask = torch.tensor([ex['attention_mask'] for ex in batch])
            
            labels = torch.tensor([ex['label'] for ex in batch])
            
            # Move to model device
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            print(f"Loss: {outputs.loss.item():.4f}")
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Predictions: {outputs.logits.argmax(-1).tolist()}")
            print(f"True labels: {labels.tolist()}")
            
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== END DIAGNOSTICS ===\n")


class MetricLoggerCallback(TrainerCallback):
    """Callback to ensure metrics are properly logged during training."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info(f"Evaluation metrics at step {state.global_step}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log training metrics
            if state.global_step % args.logging_steps == 0:
                logger.info(f"Training metrics at step {state.global_step}:")
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.6f}")

# ---------------------------------------------------------------------------
# task registry --------------------------------------------------------------
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {}

def register(name):
    def _wrap(fn):
        TASKS[name] = fn
        return fn
    return _wrap


def _filter_langs(requested: List[str], available: List[str], task_name: str):
    # Helper: keep only languages that exist in the dataset
    ok = [lg for lg in requested if lg in available]
    miss = [lg for lg in requested if lg not in available]
    logger.info("%s | requested=%s  available=%s", task_name, requested, available)
    if miss:
        logger.warning("%s | languages not found and skipped: %s", task_name, miss)
    if not ok:
        raise ValueError(f"{task_name}: none of the requested languages found!")
    return ok


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_train_val_split(dataset, val_size=0.1, seed=42):
    """Split a dataset into training and validation sets."""
    return dataset.train_test_split(test_size=val_size, seed=seed)


def adjust_padding_side(model, tok):
    decoder_only = getattr(model.config, "model_type", "") in {
        "gpt2", "gpt_neo", "gpt_neox", "gptj", "llama"
    }
    tok.padding_side = "left" if decoder_only else "right"


@register("xnli")
def setup_xnli(tokenizer, lang_codes, args):
    """Setup XNLI task with proper encode_text usage - FIXED VERSION."""
    # Load datasets (your existing code)
    all_configs = get_dataset_config_names("xnli")
    ok = _filter_langs(lang_codes, all_configs, "XNLI")

    subsets = []
    for lg in ok:
        try:
            train_ds = load_dataset("xnli", lg, split="train")
            val_ds = load_dataset("xnli", lg, split="validation")
            test_ds = load_dataset("xnli", lg, split="test")
            
            logger.info(f"Loaded XNLI {lg} - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
            
            train_ds = train_ds.map(lambda ex: {"language": lg})
            val_ds = val_ds.map(lambda ex: {"language": lg})
            
            subsets.append((train_ds, val_ds))
        except Exception as e:
            logger.error(f"Error loading XNLI dataset for {lg}: {str(e)}")
            continue

    if not subsets:
        raise ValueError("No valid XNLI datasets could be loaded")

    train_ds = concatenate_datasets([ds[0] for ds in subsets])
    val_ds = concatenate_datasets([ds[1] for ds in subsets])
    
    logger.info(f"Combined XNLI dataset - train: {len(train_ds)}, val: {len(val_ds)}")

    # FIXED: Efficient batch encoding with manual EOS insertion
    def preprocess_function(examples):    
        # When batched=True, examples will always be a batch (list of items)
        premises = examples["premise"]
        hypotheses = examples["hypothesis"]
        
        # Batch encode premises and hypotheses separately 
        premise_encoded = encode_text(
            tokenizer,
            premises,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        hypothesis_encoded = encode_text(
            tokenizer,
            hypotheses,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        # Get the token IDs
        premise_ids_batch = premise_encoded["input_ids"]
        hypothesis_ids_batch = hypothesis_encoded["input_ids"]
        
        # Efficiently construct sequences
        input_ids_list = []
        max_premise_len = args.max_len // 2 - 2  # Reserve space for EOS tokens and some buffer
        max_hypothesis_len = args.max_len - max_premise_len - 3  # BOS + 2 EOS
        
        for premise_ids, hypothesis_ids in zip(premise_ids_batch, hypothesis_ids_batch):
            # Truncate if needed
            if len(premise_ids) > max_premise_len:
                premise_ids = premise_ids[:max_premise_len]
            if len(hypothesis_ids) > max_hypothesis_len:
                hypothesis_ids = hypothesis_ids[:max_hypothesis_len]
            
            # Construct sequence: premise <EOS> hypothesis <EOS>
            sequence = []
            sequence.extend(premise_ids)
            sequence.append(tokenizer.eos_token_id)
            sequence.extend(hypothesis_ids)
            sequence.append(tokenizer.eos_token_id)
            
            input_ids_list.append(sequence)
        
        # FIXED: Convert to the format expected by tokenizer.pad()
        # pad() expects a list of dictionaries, not a single dictionary with lists
        encoded_inputs_list = [
            {"input_ids": sequence} for sequence in input_ids_list
        ]
        
        # Use tokenizer's built-in padding
        padded = tokenizer.pad(
            encoded_inputs_list,
            padding=True,
            max_length=args.max_len,
            return_tensors=None  # Keep as lists
        )
        
        return padded
    
    # Preprocess datasets
    logger.info("Preprocessing XNLI training data...")
    train_ds = train_ds.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing train data",
        load_from_cache_file=not args.overwrite_cache
    )
    
    logger.info("Preprocessing XNLI validation data...")
    val_ds = val_ds.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing validation data",
        load_from_cache_file=not args.overwrite_cache
    )
    
    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"]
        }

    # Initialize model properly for GPT
    model = initialize_gpt_for_classification(
        args.model_path,
        num_labels=3,  # XNLI has 3 classes
        tokenizer=tokenizer
    )
    
    # Ensure padding side is correct
    adjust_padding_side(model, tokenizer)
    logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
    logger.info(f"Model pad_token_id: {model.config.pad_token_id}")
    
    # Run diagnostics
    diagnose_training_issues(model, tokenizer, train_ds, args)
    
    return train_ds, val_ds, model, compute_metrics


# ---------- 3) WikiANN NER --------------------------------------------------
@register("wikiann")
def setup_ner(tokenizer, lang_codes, args):
    """Setup WikiANN NER task with proper encode_text usage."""    
    try:
        cfgs = list(datasets.get_dataset_config_names("wikiann"))
    except Exception:
        cfgs = []
    if not cfgs or cfgs == ["default"]:
        cfgs = lang_codes
    okay = _filter_langs(lang_codes, cfgs, "WikiANN")
    
    train_subsets, val_subsets = [], []
    for lg in okay:
        try:
            ds = datasets.load_dataset("wikiann", lg)
            if "validation" in ds:
                train_ds, val_ds = ds["train"], ds["validation"]
            else:
                split = create_train_val_split(ds["train"], val_size=args.val_size, seed=args.seed)
                train_ds, val_ds = split["train"], split["test"]
            
            train_ds = train_ds.map(lambda ex: {"language": lg})
            val_ds = val_ds.map(lambda ex: {"language": lg})
            
            logger.info(f"WikiANN {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            train_subsets.append(train_ds)
            val_subsets.append(val_ds)
        except Exception as e:
            logger.error(f"Error loading WikiANN dataset for {lg}: {str(e)}")
            continue
    
    if not train_subsets or not val_subsets:
        raise ValueError("No valid WikiANN datasets could be loaded")
        
    train_ds = datasets.concatenate_datasets(train_subsets)
    val_ds = datasets.concatenate_datasets(val_subsets)
    
    logger.info(f"Combined WikiANN dataset - train: {len(train_ds)}, val: {len(val_ds)}")

    label_list = train_ds.features["ner_tags"].feature.names
    num_labels = len(label_list)

    def tokenize_and_align_labels(examples):
        """
        Tokenization with label alignment using encode_text.
        """
        # For NER, we need to handle word-level tokenization
        # Join words with spaces for tokenization
        if isinstance(examples["tokens"][0], list):
            texts = [" ".join(words) for words in examples["tokens"]]
        else:
            texts = " ".join(examples["tokens"])
        
        # Tokenize
        tokenized_inputs = encode_text(
            tokenizer,
            texts,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_attention_mask=True,
            add_special_tokens=True
        )
            
        # Handle batch vs single
        if isinstance(texts, str):
            # Single example - wrap in lists
            tokenized_inputs = {k: [v] if not isinstance(v, list) else v 
                               for k, v in tokenized_inputs.items()}
            examples_tokens = [examples["tokens"]]
            examples_tags = [examples["ner_tags"]]
            examples_language = [examples["language"]]
        else:
            examples_tokens = examples["tokens"]
            examples_tags = examples["ner_tags"]
            examples_language = examples["language"]
        
        labels = []
        for idx, (tokens, tags) in enumerate(zip(examples_tokens, examples_tags)):
            # Get offset mapping for this example
            offset_mapping = tokenized_inputs["offset_mapping"][idx]
            
            # Create char to word mapping
            char_to_word = {}
            char_idx = 0
            for word_idx, word in enumerate(tokens):
                for _ in word:
                    char_to_word[char_idx] = word_idx
                    char_idx += 1
                char_to_word[char_idx] = word_idx  # Space
                char_idx += 1
            
            # Align labels
            label_ids = []
            previous_word_idx = None
            
            for start_char, end_char in offset_mapping:
                # Special tokens have (0, 0) offset
                if start_char == 0 and end_char == 0:
                    label_ids.append(-100)
                    previous_word_idx = None
                else:
                    # Find corresponding word
                    word_idx = char_to_word.get(start_char)
                    
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx >= len(tags):
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # First token of a word
                        label_ids.append(tags[word_idx])
                        previous_word_idx = word_idx
                    else:
                        # Subsequent tokens of a word
                        if args.label_all_subtokens:
                            label_ids.append(tags[word_idx])
                        else:
                            label_ids.append(-100)
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        tokenized_inputs["language"] = examples_language
        
        # Remove offset_mapping as it's not needed for training
        tokenized_inputs.pop("offset_mapping", None)
        
        return tokenized_inputs

    logger.info("Preprocessing WikiANN training data...")
    train_ds = train_ds.map(
        tokenize_and_align_labels, 
        batched=True, 
        desc="Processing train tokens",
        load_from_cache_file=not args.overwrite_cache
    )
    
    # Set format for PyTorch
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels','language'])
    
    logger.info("Preprocessing WikiANN validation data...")
    val_ds = val_ds.map(
        tokenize_and_align_labels, 
        batched=True, 
        desc="Processing validation tokens",
        load_from_cache_file=not args.overwrite_cache
    )
    
    # Set format for PyTorch
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'language'])
    
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Get language information from the evaluation dataset
        # Note: This requires accessing the dataset during evaluation
        languages = []
        for batch in val_ds:
            if isinstance(batch['language'], torch.Tensor):
                languages.extend(batch['language'].tolist())
            else:
                languages.append(batch['language'])
        
        # Group by language
        lang_predictions = {}
        lang_labels = {}
        
        for i, (pred, label, lang) in enumerate(zip(predictions, labels, languages)):
            if lang not in lang_predictions:
                lang_predictions[lang] = []
                lang_labels[lang] = []
            
            # Remove ignored index (special tokens)
            true_pred = [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            true_label = [label_list[l] for (p, l) in zip(pred, label) if l != -100]
            
            lang_predictions[lang].append(true_pred)
            lang_labels[lang].append(true_label)
        
        # Compute metrics for each language
        results = {}
        overall_predictions = []
        overall_labels = []
        
        for lang in lang_predictions:
            lang_results = seqeval.compute(
                predictions=lang_predictions[lang], 
                references=lang_labels[lang]
            )
            
            results[f"{lang}_precision"] = lang_results["overall_precision"]
            results[f"{lang}_recall"] = lang_results["overall_recall"]
            results[f"{lang}_f1"] = lang_results["overall_f1"]
            results[f"{lang}_accuracy"] = lang_results["overall_accuracy"]
            
            # Add to overall
            overall_predictions.extend(lang_predictions[lang])
            overall_labels.extend(lang_labels[lang])
        
        # Compute overall metrics
        overall_results = seqeval.compute(
            predictions=overall_predictions, 
            references=overall_labels
        )
        
        results.update({
            "precision": overall_results["overall_precision"],
            "recall": overall_results["overall_recall"],
            "f1": overall_results["overall_f1"],
            "accuracy": overall_results["overall_accuracy"],
        })
        
        return results
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **args.model_kwargs
        )
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute_metrics

# ---------------------------------------------------------------------------
# Enhanced training configuration
# ---------------------------------------------------------------------------
def get_training_args(task, args, output_dir):
    """Get task-specific training arguments with proper logging - FIXED VERSION."""

    # Task-specific configurations
    task_configs = {
        "xnli": {
            "metric_for_best_model": "eval_f1",
            "greater_is_better": True,
            "eval_steps": 1000,
            "save_steps": 1000,
            "logging_steps": 100,
            "warmup_steps": 500,  # FIXED: Use explicit warmup_steps instead of ratio
            "learning_rate": 2e-5,  # FIXED: More conservative learning rate
            "epochs": 1  
        },
        "wikiann": {
            "metric_for_best_model": "eval_f1",
            "greater_is_better": True,
            "eval_steps": 1000,
            "save_steps": 1000,
            "logging_steps": 100,
            "warmup_steps": 500,
            "learning_rate": 2e-5,
            "epochs": args.epochs
        },
        "marc": {
            "metric_for_best_model": "eval_f1",
            "greater_is_better": True,
            "eval_steps": 2000,
            "save_steps": 2000,
            "logging_steps": 200,
            "warmup_steps": 500,
            "learning_rate": 2e-5,
            "epochs": args.epochs
        }
    }
    
    config = task_configs.get(task, {
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "warmup_steps": 500,
        "learning_rate": args.lr,
        "epochs": args.epochs
    })
    
    # Use task-specific learning rate if available
    learning_rate = config.get("learning_rate", args.lr)
    
    # FIXED: Use warmup_steps consistently instead of mixing with warmup_ratio
    warmup_steps = config.get("warmup_steps", 500)
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=config["epochs"],
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=config["logging_steps"],
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        gradient_accumulation_steps=args.grad_accum,
        optim="adamw_torch",
        report_to=["tensorboard"],
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_strategy=IntervalStrategy.STEPS,
        warmup_steps=warmup_steps,  
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        push_to_hub=False,
        remove_unused_columns=True,
        dataloader_drop_last=False,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        seed=args.seed,
        disable_tqdm=True,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",  # FIXED: Explicit scheduler type
    )
# ---------------------------------------------------------------------------
# programmatic entry‑point (used by run_experiment.py)
# ---------------------------------------------------------------------------
def finetune_from_config(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    entry = {
        'name':          'tok_x',
        'model_path':    '…',
        'tokenizer_path':'…',
        'tasks':        ['xnli','wikiann'],
        'hyper':        {'epochs':3,'batch_size':16,'lr':3e-5},
        'output_dir':   'results/…'
    }
    Returns { task -> metrics_json_path }.
    """
    args = SimpleNamespace(
        model_path   = entry["model_path"],
        model_kwargs = entry["model_kwargs"], 
        task         = entry["tasks"],
        epochs       = entry["hyper"].get("epochs", 3),  # Increased default
        batch_size   = entry["hyper"].get("batch_size", 16),
        lr           = entry["hyper"].get("lr", 2e-5),  # Slightly lower default
        max_len      = entry["hyper"].get("max_len", 512),
        output_dir   = entry["output_dir"],
        seed         = entry["hyper"].get("seed", 42),
        val_size     = entry["hyper"].get("val_size", 0.1),
        label_all_subtokens = entry["hyper"].get("label_all_subtokens", False),
        qa_stride    = entry["hyper"].get("qa_stride", 128),
        grad_accum   = entry["hyper"].get("gradient_accumulation_steps", 4),
        warmup_steps = entry["hyper"].get("warmup_steps", 0),
        early_stopping_patience = entry["hyper"].get("early_stopping_patience", 6),  # Increased
        eval_steps   = entry["hyper"].get("eval_steps", 500),  # More frequent
        logging_steps = entry["hyper"].get("logging_steps", 100),  # More frequent
        overwrite_cache = entry["hyper"].get("overwrite_cache", False),
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    logger.info(f"Configuration: {args}")

    tokenizer_info = entry["tokenizer_info"]
    
    tok = load_tokenizer_from_config(tokenizer_info)
    set_special_tokens(tok)
    lang_codes = entry["languages"]

    metrics_map = {}
    for task in args.task:
        logger.info(f"Starting task: {task}")
        
        try:
            # Get train/val datasets and model
            train_ds, val_ds, model, compute_metrics = TASKS[task](tok, lang_codes, args)
            train_ds = train_ds.shuffle(seed=args.seed)

            # Choose appropriate data collator
            if task == "wikiann":
                collator = DataCollatorForTokenClassification(
                    tok, 
                    pad_to_multiple_of=8,
                    padding=True
                )
            else:
                collator = DataCollatorWithPadding(
                    tok, 
                    pad_to_multiple_of=8,
                    padding=True
                )

            # Set up output directory
            output_dir = os.path.join(args.output_dir, task)
            os.makedirs(output_dir, exist_ok=True)

            # Get task-specific training arguments
            training_args = get_training_args(task, args, output_dir)

            # Set up trainer with callbacks
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                MetricLoggerCallback()  # Add custom metric logger
            ]
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tok,
                data_collator=collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )

            # Find latest checkpoint if any
            resume_from_checkpoint = None
            if os.path.isdir(output_dir):
                checkpoints = sorted(
                    [ckpt for ckpt in os.listdir(output_dir) if ckpt.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[1]),
                )
                if checkpoints:
                    latest = os.path.join(output_dir, checkpoints[-1])
                    state_path = os.path.join(latest, "trainer_state.json")
                    logger.info(f"Found checkpoint: {latest}")
                    if os.path.isfile(state_path):
                        with open(state_path) as f:
                            trainer_state = json.load(f)
                        epoch = trainer_state.get("epoch", 0)
                        if epoch >= args.epochs:
                            logger.info(f"Training already complete ({epoch}/{args.epochs} epochs).")
                            training_completed = True
                        else:                    
                            logger.info(f"Resuming from {epoch}/{args.epochs} epochs.")
                            resume_from_checkpoint = latest
                            training_completed = False
                    else:
                        training_completed = False
                else:
                    training_completed = False
            else:
                training_completed = False

            # Train the model if not already done
            if not training_completed:
                logger.info(f"Starting training for task: {task}")
                
                # Log initial evaluation
                logger.info("Running initial evaluation...")
                initial_metrics = trainer.evaluate()
                logger.info(f"Initial metrics: {initial_metrics}")
                
                # Train
                train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                logger.info(f"Training completed for task: {task}")
                
                # Save training history
                history_path = os.path.join(output_dir, "training_history.json")
                with open(history_path, "w") as f:
                    json.dump(trainer.state.log_history, f, indent=2)
            else:
                logger.info(f"Skipping training for task: {task} (already completed)")

            # Final evaluation
            logger.info(f"Running final evaluation for task: {task}")
            metrics = trainer.evaluate()
            
            # Save final model
            trainer.save_model(os.path.join(output_dir, "final_model"))
            tok.save_pretrained(os.path.join(output_dir, "final_model"))
            
            # Save metrics
            metrics_path = os.path.join(output_dir, f"{task}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Final metrics saved to {metrics_path}: {metrics}")
            
            metrics_map[task] = metrics_path
            
        except Exception as e:
            logger.error(f"Error processing task {task}: {str(e)}", exc_info=True)
            metrics_map[task] = None
            continue

    logger.info("All tasks completed")
    return metrics_map

