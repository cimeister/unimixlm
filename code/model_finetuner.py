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
    EarlyStoppingCallback
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ---------- 1) XNLI ---------------------------------------------------------
@register("xnli")
def setup_xnli(tokenizer, lang_codes, args):
    # 1) Which configs (languages) exist?
    all_configs = get_dataset_config_names("xnli")
    ok = _filter_langs(lang_codes, all_configs, "XNLI")

    # 2) Load & tag each language
    subsets = []
    for lg in ok:
        try:
            # Load validation set if available, otherwise use train and split
            if "validation" in load_dataset("xnli", lg):
                train_ds = load_dataset("xnli", lg, split="train")
                val_ds = load_dataset("xnli", lg, split="validation")
                logger.info(f"Loaded XNLI {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            else:
                ds_lg = load_dataset("xnli", lg, split="train")
                split = create_train_val_split(ds_lg, val_size=args.val_size, seed=args.seed)
                train_ds, val_ds = split["train"], split["test"]
                logger.info(f"Split XNLI {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            
            # Tag datasets with language
            train_ds = train_ds.map(lambda ex: {"language": lg})
            val_ds = val_ds.map(lambda ex: {"language": lg})
            
            subsets.append((train_ds, val_ds))
        except Exception as e:
            logger.error(f"Error loading XNLI dataset for {lg}: {str(e)}")
            continue

    if not subsets:
        raise ValueError("No valid XNLI datasets could be loaded")

    # 3) Merge them into separate train/val Datasets
    train_ds = concatenate_datasets([ds[0] for ds in subsets])
    val_ds = concatenate_datasets([ds[1] for ds in subsets])

    logger.info(f"Combined XNLI dataset - train: {len(train_ds)}, val: {len(val_ds)}")

    # Preprocessing function using encode_text
    def preprocess_function(examples):
        # Use encode_text for sequence pairs directly
        encodings = encode_text(
            tokenizer, 
            list(zip(examples["premise"], examples["hypothesis"])),  # List of (premise, hypothesis) pairs
            padding='max_length' if args.max_len else False,
            truncation=True, 
            max_length=args.max_len,
            return_attention_mask=True,
            return_token_type_ids=True  # Important for sequence classification
        )
        
        return encodings
    
    logger.info("Preprocessing XNLI training data...")
    train_ds = train_ds.map(
        preprocess_function,
        batched=True, 
        desc="Tokenizing train data"
    )
    
    logger.info("Preprocessing XNLI validation data...")
    val_ds = val_ds.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing validation data"
    )

    metric = evaluate.load("accuracy")
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=3
    )
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute

# ---------- 2) MARC ---------------------------------------------------------
@register("marc")
def setup_marc(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("amazon_reviews_multi")), "MARC")
    
    train_subsets, val_subsets = [], []
    for lg in okay:
        try:
            ds = datasets.load_dataset("amazon_reviews_multi", lg)
            train_ds = ds["train"]
            # Use validation set if available, otherwise split train
            if "validation" in ds:
                val_ds = ds["validation"]
            else:
                split = create_train_val_split(train_ds, val_size=args.val_size, seed=args.seed)
                train_ds, val_ds = split["train"], split["test"]
            
            logger.info(f"MARC {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            train_subsets.append(train_ds)
            val_subsets.append(val_ds)
        except Exception as e:
            logger.error(f"Error loading MARC dataset for {lg}: {str(e)}")
            continue
    
    if not train_subsets or not val_subsets:
        raise ValueError("No valid MARC datasets could be loaded")
        
    train_ds = datasets.concatenate_datasets(train_subsets)
    val_ds = datasets.concatenate_datasets(val_subsets)
    
    logger.info(f"Combined MARC dataset - train: {len(train_ds)}, val: {len(val_ds)}")

    num_labels = 5
    def preprocess(examples):
        return tokenizer(examples["review_body"], truncation=True, max_length=args.max_len)
    
    logger.info("Preprocessing MARC training data...")
    train_ds = train_ds.map(preprocess, batched=True, desc="Tokenizing train data")
    
    logger.info("Preprocessing MARC validation data...")
    val_ds = val_ds.map(preprocess, batched=True, desc="Tokenizing validation data")
    
    metric = evaluate.load("f1", average="macro")
    def compute(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=num_labels
    )
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute

# ---------- 3) WikiANN NER --------------------------------------------------
@register("wikiann")
def setup_ner(tokenizer, lang_codes, args):
    try:
        cfgs = list(datasets.get_dataset_config_names("wikiann"))
    except Exception:
        cfgs = []
    if not cfgs or cfgs == ["default"]:
        cfgs = lang_codes                      # fall back to user request
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

    def tokenize_and_align(examples):
        # 1) Plain string for each sentence
        texts = [" ".join(ws) for ws in examples["tokens"]]

        # 2) Encode – always ask for offsets
        enc = encode_text(
            tokenizer,
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_len,
            padding="max_length" if args.max_len else None,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        input_ids       = enc["input_ids"]
        attention_mask  = enc["attention_mask"]
        offsets_batch   = enc["offset_mapping"]

        labels_batch = []
        for words, off, gold in zip(examples["tokens"], offsets_batch, examples["ner_tags"]):
            # ---- build char-start index for each original word -------------
            starts = []
            pos = 0
            for w in words:
                starts.append(pos)
                pos += len(w) + 1           # +1 for space we injected

            # ---- map each sub-token offset to a word id --------------------
            wid_for_tok = []
            for s, _ in off:
                if s == 0 and _ == 0:       # special token, padding, etc.
                    wid_for_tok.append(None)
                else:
                    # linear scan is fine (sentences are short)
                    j = max(i for i, st in enumerate(starts) if st <= s)
                    wid_for_tok.append(j)

            # ---- align labels ---------------------------------------------
            aligned = []
            prev_w  = None
            for wid in wid_for_tok:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_w:
                    aligned.append(gold[wid])
                else:
                    aligned.append(gold[wid] if args.label_all_subtokens else -100)
                prev_w = wid
            labels_batch.append(aligned)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          labels_batch,
        }


    logger.info("Preprocessing WikiANN training data...")
    train_ds = train_ds.map(tokenize_and_align, batched=True, desc="Processing train tokens")
    
    logger.info("Preprocessing WikiANN validation data...")
    val_ds = val_ds.map(tokenize_and_align, batched=True, desc="Processing validation tokens")
    
    seqeval = evaluate.load("seqeval")
    def compute(p):
        predictions, labels = p
        predictions = predictions.argmax(-1)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"f1": results["overall_f1"]}
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path, num_labels=num_labels
    )
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute

# ---------- 4) MLQA (extractive QA) ----------------------------------------
@register("mlqa")
def setup_mlqa(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("mlqa")), "MLQA")
    
    train_subsets, val_subsets = [], []
    for lg in okay:
        try:
            ds = datasets.load_dataset("mlqa", lg)
            train_ds = ds["train"]
            if "validation" in ds:
                val_ds = ds["validation"]
            else:
                split = create_train_val_split(train_ds, val_size=args.val_size, seed=args.seed)
                train_ds, val_ds = split["train"], split["test"]
            
            logger.info(f"MLQA {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            train_subsets.append(train_ds)
            val_subsets.append(val_ds)
        except Exception as e:
            logger.error(f"Error loading MLQA dataset for {lg}: {str(e)}")
            continue
    
    if not train_subsets or not val_subsets:
        raise ValueError("No valid MLQA datasets could be loaded")
        
    train_ds = datasets.concatenate_datasets(train_subsets)
    val_ds = datasets.concatenate_datasets(val_subsets)
    
    logger.info(f"Combined MLQA dataset - train: {len(train_ds)}, val: {len(val_ds)}")

    def preprocess_qa(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_len,
            truncation="only_second",
            stride=args.qa_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Since one example might give us several features if it has a long context
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")
        
        # Let's label those examples
        inputs["start_positions"] = []
        inputs["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # Determine which example this feature came from
            sample_idx = sample_mapping[i]
            
            # Get the answers for this example
            answer = examples["answers"][sample_idx]
            start_char = answer["answer_start"][0] if len(answer["answer_start"]) > 0 else 0
            end_char = start_char + len(answer["text"][0]) if len(answer["text"]) > 0 else 0
            
            # Find tokens corresponding to answer
            token_start_index = 0
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1  # We went one too far
            
            token_end_index = token_start_index
            while token_end_index < len(offsets) and offsets[token_end_index][1] <= end_char:
                token_end_index += 1
            
            # If answer not fully within context, label it as start of context
            if offsets[token_start_index][0] > start_char or offsets[token_end_index-1][1] < end_char:
                inputs["start_positions"].append(0)
                inputs["end_positions"].append(0)
            else:
                inputs["start_positions"].append(token_start_index)
                inputs["end_positions"].append(token_end_index - 1)
        
        return inputs

    logger.info("Preprocessing MLQA training data...")
    train_ds = train_ds.map(
        preprocess_qa, 
        batched=True, 
        remove_columns=train_ds.column_names,
        desc="Processing train QA pairs"
    )
    
    logger.info("Preprocessing MLQA validation data...")
    val_ds = val_ds.map(
        preprocess_qa, 
        batched=True, 
        remove_columns=val_ds.column_names,
        desc="Processing validation QA pairs"
    )
    
    # Properly calculate QA metrics
    metric = evaluate.load("squad")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        start_logits, end_logits = predictions
        start_positions, end_positions = labels
        
        # Convert logits to predictions
        start_indices = start_logits.argmax(-1)
        end_indices = end_logits.argmax(-1)
        
        # For proper evaluation, we would need to process the validation set to extract the answers
        # This is a simplified metric calculation
        accuracy = np.mean(
            (start_indices == start_positions) & (end_indices == end_positions)
        )
        
        return {"accuracy": accuracy}
        
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute_metrics

# ---------- 5) PAWS‑X -------------------------------------------------------
@register("pawsx")
def setup_pawsx(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("paws-x")), "PAWS-X")
    
    train_subsets, val_subsets = [], []
    for lg in okay:
        try:
            ds = datasets.load_dataset("paws-x", lg)
            train_ds = ds["train"]
            if "validation" in ds:
                val_ds = ds["validation"]
            else:
                split = create_train_val_split(train_ds, val_size=args.val_size, seed=args.seed)
                train_ds, val_ds = split["train"], split["test"]
            
            logger.info(f"PAWS-X {lg} - train: {len(train_ds)}, val: {len(val_ds)}")
            train_subsets.append(train_ds)
            val_subsets.append(val_ds)
        except Exception as e:
            logger.error(f"Error loading PAWS-X dataset for {lg}: {str(e)}")
            continue
    
    if not train_subsets or not val_subsets:
        raise ValueError("No valid PAWS-X datasets could be loaded")
        
    train_ds = datasets.concatenate_datasets(train_subsets)
    val_ds = datasets.concatenate_datasets(val_subsets)
    
    logger.info(f"Combined PAWS-X dataset - train: {len(train_ds)}, val: {len(val_ds)}")
    
    num_labels = 2
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_len)
    
    logger.info("Preprocessing PAWS-X training data...")
    train_ds = train_ds.map(preprocess, batched=True, desc="Tokenizing train data")
    
    logger.info("Preprocessing PAWS-X validation data...")
    val_ds = val_ds.map(preprocess, batched=True, desc="Tokenizing validation data")
    
    metric = evaluate.load("accuracy")
    def compute(p):
        logits, labels = p
        return metric.compute(predictions=logits.argmax(-1), references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
    adjust_padding_side(model, tokenizer)
    return train_ds, val_ds, model, compute

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
        task         = entry["tasks"],
        epochs       = entry["hyper"].get("epochs", 2),
        batch_size   = entry["hyper"].get("batch_size", 16),
        lr           = entry["hyper"].get("lr", 3e-5),
        max_len      = entry["hyper"].get("max_len", 256),
        output_dir   = entry["output_dir"],
        seed         = entry["hyper"].get("seed", 42),
        val_size     = entry["hyper"].get("val_size", 0.1),
        label_all_subtokens = entry["hyper"].get("label_all_subtokens", False),
        qa_stride    = entry["hyper"].get("qa_stride", 128),
        grad_accum   = entry["hyper"].get("gradient_accumulation_steps", 4),
        warmup_steps = entry["hyper"].get("warmup_steps", 0),
        early_stopping_patience = entry["hyper"].get("early_stopping_patience", 3),
        eval_steps   = entry["hyper"].get("eval_steps", 2000),
        logging_steps = entry["hyper"].get("logging_steps", 2000),
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    logger.info(f"Configuration: {args}")

    tokenizer_info = entry["tokenizer_info"]
    
    tok = load_tokenizer_from_config(tokenizer_info)
    lang_codes = entry["languages"]

    metrics_map = {}
    for task in args.task:
        logger.info(f"Starting task: {task}")
        
        try:
            # Get train/val datasets and model
            train_ds, val_ds, model, compute_metrics = TASKS[task](tok, lang_codes, args)

            # Choose appropriate data collator
            if task == "wikiann":
                collator = DataCollatorForTokenClassification(tok)
            else:
                collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

            # Set up output directory
            output_dir = os.path.join(args.output_dir, task)
            os.makedirs(output_dir, exist_ok=True)

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size * 2,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=0.01,
                logging_steps=args.logging_steps,
                evaluation_strategy="steps",
                eval_steps=args.eval_steps,
                save_steps = args.eval_steps,
                gradient_accumulation_steps=args.grad_accum,
                report_to="tensorboard",
                fp16=torch.cuda.is_available(),
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss" if task == "mlqa" else None,
                greater_is_better=False if task == "mlqa" else True,
                save_strategy="steps",
                warmup_steps=args.warmup_steps,
                logging_dir=os.path.join(args.output_dir, f"{task}_logs"),
            )

            # Set up trainer with early stopping
            callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            
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
            #TODO: remove False here
            if False and os.path.isdir(output_dir):
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

            # Train the model if not already done
            if not training_completed:
                logger.info(f"Starting training for task: {task}")
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                logger.info(f"Training completed for task: {task}")
            else:
                logger.info(f"Skipping training for task: {task} (already completed)")

            # Evaluate and save metrics
            logger.info(f"Evaluating model for task: {task}")
            metrics = trainer.evaluate()
            metrics_path = os.path.join(args.output_dir, f"{task}_metrics.json")
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}: {metrics}")
            
            metrics_map[task] = metrics_path
            
        except Exception as e:
            logger.error(f"Error processing task {task}: {str(e)}", exc_info=True)
            metrics_map[task] = None
            continue

    logger.info("All tasks completed")
    return metrics_map