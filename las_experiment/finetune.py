#!/usr/bin/env python3
# Finetuning LLMs on FineWeb2 5M/10M tokens (multilingual)

import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

logging.basicConfig(level=logging.INFO)


def resolve_fineweb2_dataset(size: str) -> str:
    size = size.upper()
    if size == "5M":
        return "MultilingualUnigramLM/FineWeb2-5M"
    elif size == "10M":
        return "MultilingualUnigramLM/FineWeb2-10M"
    else:
        raise ValueError(f"Invalid FineWeb2 size: {size} (use '5M' or '10M')")


# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, dataset_cfg, encode_fn=None):
        max_seq_len = dataset_cfg.get("max_seq_len", 4096)

        self.examples = []

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        for t in texts:
            if encode_fn:
                ids = encode_fn(t)
            else:
                ids = tokenizer(t, truncation=False, padding=False)["input_ids"]

            ids = ids[:max_seq_len]
            pad_len = max_seq_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            labels = list(ids) + [-100] * pad_len
            ids = ids + [pad_id] * pad_len

            self.examples.append(
                {
                    "input_ids": ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def count_training_tokens(texts, tokenizer, encode_fn=None):
    if encode_fn:
        return sum(len(encode_fn(t)) for t in texts)
    return sum(tokenizer(t, return_tensors="pt")["input_ids"].numel() for t in texts)


def infer_lora_targets(model_type: str):
    if model_type in {"llama", "mistral", "qwen2", "gemma", "olmo", "olmo3"}:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif model_type in {"gpt2", "gpt_neo", "gpt_neox", "gptj", "bloom"}:
        return ["c_attn", "c_proj"]
    else:
        raise ValueError(
            f"Unsupported model type for LoRA target inference: {model_type}"
        )


# ---------------------- Main Fine-tune ----------------------
def run_finetune(cfg):
    if not cfg.langs:
        raise ValueError("No languages specified in cfg.langs")

    langs = cfg.langs
    dataset_size = cfg.dataset.size
    model_name = cfg.model.model_name
    las_tokenizer_path = (
        Path(cfg.las_tokenizer_path) if cfg.las_tokenizer_path else None
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ---- LAS tokenizer integration ----
    if las_tokenizer_path:
        base_tokenizer_path = las_tokenizer_path / "base.tokenizer.json"
        language_paths = {
            lang: str(las_tokenizer_path / f"langspec_soft_{lang}_Latn.tokenizer.json")
            for lang in langs
        }
        shared_las = LanguageSpecificUnigramLMTokenizer(vocab_size=len(tokenizer))
        shared_las.load(str(base_tokenizer_path), language_paths)

        def las_encode_ids(text):
            _, toks = shared_las.best_language_encode(text)
            ids = tokenizer.convert_tokens_to_ids(toks)
            if isinstance(ids, int):
                ids = [ids]
            return ids
    else:
        las_encode_ids = None

    # ---- Load FineWeb2 multilingual splits ----
    HF_DATASET_NAME = resolve_fineweb2_dataset(dataset_size)
    all_texts, lang_texts = [], {}
    for lang in langs:
        logging.info(f"Loading {HF_DATASET_NAME} :: {lang}_Latn")
        ds = load_dataset(
            HF_DATASET_NAME, f"{lang}_Latn", split="train", streaming=False
        )

        texts = [x["text"] for x in ds]
        logging.info(f"Example text [{lang}]: {texts[0][:200]}...")
        if texts:
            sample_text = texts[0]
            orig_tokens = tokenizer.tokenize(sample_text)
            orig_ids = tokenizer.convert_tokens_to_ids(orig_tokens)
            logging.info(
                f"Original tokenization [{lang}]: tokens={orig_tokens[:50]} ids={orig_ids[:50]}"
            )
            if las_tokenizer_path:
                las_lang, las_tokens = shared_las.best_language_encode(sample_text)
                las_ids = tokenizer.convert_tokens_to_ids(las_tokens)
                if isinstance(las_ids, int):
                    las_ids = [las_ids]
                logging.info(
                    f"LAS tokenization [{lang}] (best_lang={las_lang}): tokens={las_tokens[:50]} ids={las_ids[:50]}"
                )
        all_texts.extend(texts)
        lang_texts[lang] = texts

    # ---- WandB ----
    if cfg.logger.enabled:
        wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            name=f"{'-'.join(langs)}_{dataset_size}_{cfg.logger.run_name}",
            tags=cfg.logger.tags,
            notes=cfg.logger.notes,
        )

    # ---- Token accounting ----
    lang_token_counts = {
        lang: count_training_tokens(
            lang_texts[lang], tokenizer, encode_fn=las_encode_ids
        )
        for lang in langs
    }
    total_train_tokens = sum(lang_token_counts.values())
    logging.info(f"Total training tokens: {total_train_tokens}")
    for lang, count in lang_token_counts.items():
        logging.info(f"Training tokens [{lang}]: {count}")
    if cfg.logger.enabled and wandb.run:
        wandb.log(
            {
                "train/total_tokens": total_train_tokens,
                **{f"train/tokens_{lang}": c for lang, c in lang_token_counts.items()},
            }
        )

    # ---- Dataset ----
    dataset = TokenizedTextDataset(
        all_texts,
        tokenizer,
        cfg.dataset,
        encode_fn=las_encode_ids,
    )
    val_size = max(1, int(len(dataset) * float(cfg.val_fraction)))
    train_size = len(dataset) - val_size
    logging.info(f"Train dataset size: {train_size} samples")
    logging.info(f"Eval dataset size: {val_size} samples")
    generator = torch.Generator().manual_seed(getattr(cfg, "seed", 42))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    # ---- LoRA ----
    lora_targets = (
        [m.strip() for m in cfg.lora.target_modules.split(",")]
        if cfg.lora.target_modules
        else infer_lora_targets(model.config.model_type)
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # ---- Trainer ----
    training_args = TrainingArguments(
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"] if cfg.logger.enabled else [],
        bf16=True,
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    logging.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(str(output_dir))
    logging.info("Fine-tuning complete.")
