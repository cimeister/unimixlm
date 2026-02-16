#!/usr/bin/env python3
# Finetuning LLMs on FineWeb2 5M/10M tokens (multilingual)

import logging
import os
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from code.standard_unigramlm import StandardUnigramLMTokenizer
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
    def __init__(
        self,
        texts,
        tokenizer,
        dataset_cfg,
        pad_token_id,
        eos_token_id,
        encode_fn=None,
    ):
        max_seq_len = dataset_cfg.max_seq_len

        self.examples = []

        for t in texts:
            # 1. Tokenize: use LAS encode_fn if provided, otherwise the model tokenizer
            if encode_fn:
                ids = encode_fn(t)
            else:
                ids = tokenizer(t, truncation=False, padding=False)["input_ids"]

            # 2. Truncate to max context length
            ids = ids[:max_seq_len]

            # 3. Append EOS to mark end-of-sequence for samples shorter than
            #    the full context. Samples that fill the context are left as-is
            #    (the next sample boundary acts as an implicit stop).
            if len(ids) < max_seq_len:
                ids = ids + [eos_token_id]

            # 4. Pad to max_seq_len. Padded positions get attention_mask=0
            #    and labels=-100 so they are ignored in loss computation.
            pad_len = max_seq_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            labels = list(ids) + [-100] * pad_len
            ids = ids + [pad_token_id] * pad_len

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


def load_tokenizer_for_mode(
    las_mode, langs, tokenizer, model_tokenizer_path, las_tokenizer_path
):
    """Load tokenizer encode functions, pad_token_id, and eos_token_id for the given LAS mode.

    Args:
        las_mode: "original", "base", or "langspec"
        langs: list of language codes
        tokenizer: the model's AutoTokenizer (used for token lookups and original-mode encoding)
        model_tokenizer_path: Path to the model tokenizer directory
        las_tokenizer_path: Path to the LAS tokenizer directory (required for base/langspec)

    Returns:
        (encode_fn_for_lang, pad_token_id, eos_token_id)
        - encode_fn_for_lang: dict {lang: callable(text) -> List[int] or None}
        - pad_token_id: int
        - eos_token_id: int
    """
    las_tokenizer_dir = Path(las_tokenizer_path) if las_tokenizer_path else None

    if las_mode == "original":
        encode_fn_for_lang = {lang: None for lang in langs}
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

    elif las_mode == "base":
        if not las_tokenizer_dir:
            raise ValueError("las_tokenizer_path required for las_mode='base'")

        base_las = StandardUnigramLMTokenizer(
            vocab_size=tokenizer.vocab_size,
            from_base_tokenizer=str(model_tokenizer_path / "tokenizer.json"),
        )
        base_las.load(str(las_tokenizer_dir / "base_tokenizer.json"))

        pad_token_id = base_las.base_tokenizer.token_to_id(tokenizer.pad_token)
        eos_token_id = base_las.base_tokenizer.token_to_id(tokenizer.eos_token)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        def _base_encode(text):
            return base_las.encode(text)

        encode_fn_for_lang = {lang: _base_encode for lang in langs}

    elif las_mode == "langspec":
        if not las_tokenizer_dir:
            raise ValueError("las_tokenizer_path required for las_mode='langspec'")
        las = LanguageSpecificUnigramLMTokenizer(vocab_size=tokenizer.vocab_size)
        lang_paths = {
            lang: str(las_tokenizer_dir / f"langspec_{lang}_Latn.json")
            for lang in langs
        }
        las.load(str(model_tokenizer_path / "tokenizer.json"), lang_paths)

        pad_token_id = las.base_tokenizer.token_to_id(tokenizer.pad_token)
        eos_token_id = las.base_tokenizer.token_to_id(tokenizer.eos_token)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        def _make_lang_encode(lang_code):
            def _encode(text):
                return las.encode_lang(text, lang_code).ids

            return _encode

        encode_fn_for_lang = {lang: _make_lang_encode(lang) for lang in langs}

    else:
        raise ValueError(
            f"Unknown las_mode: {las_mode}. Use 'original', 'base', or 'langspec'."
        )

    if pad_token_id is None:
        raise ValueError(
            f"Could not resolve pad_token_id for las_mode='{las_mode}'. "
            f"pad_token='{tokenizer.pad_token}' not found in the active tokenizer."
        )
    if eos_token_id is None:
        raise ValueError(
            f"Could not resolve eos_token_id for las_mode='{las_mode}'. "
            f"eos_token='{tokenizer.eos_token}' not found in the active tokenizer."
        )
    logging.info(
        f"Active tokenizer ({las_mode}): pad_token_id={pad_token_id}, eos_token_id={eos_token_id}"
    )

    return encode_fn_for_lang, pad_token_id, eos_token_id


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
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.model_tokenizer_path or not os.path.isdir(cfg.model_tokenizer_path):
        raise ValueError(
            f"model_tokenizer_path must be a valid directory, got: {cfg.model_tokenizer_path}"
        )
    model_tokenizer_path = Path(cfg.model_tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ---- LAS tokenizer mode ----
    las_mode = str(cfg.las_mode)
    encode_fn_for_lang, pad_token_id, eos_token_id = load_tokenizer_for_mode(
        las_mode, langs, tokenizer, model_tokenizer_path, cfg.las_tokenizer_path
    )

    # ---- Load FineWeb2 multilingual splits ----
    HF_DATASET_NAME = resolve_fineweb2_dataset(dataset_size)
    lang_texts = {}
    for lang in langs:
        logging.info(f"Loading {HF_DATASET_NAME} :: {lang}_Latn")
        ds = load_dataset(
            HF_DATASET_NAME, f"{lang}_Latn", split="train", streaming=False
        )

        texts = [x["text"] for x in ds]
        if not texts:
            raise ValueError(
                f"No texts found for language '{lang}' in {HF_DATASET_NAME}"
            )
        logging.info(f"Example text [{lang}]: {texts[0][:200]}...")
        sample_text = texts[0]
        orig_ids = tokenizer(sample_text, truncation=False, padding=False)["input_ids"]
        logging.info(
            f"Original tokenization (no special chars) [{lang}]: {len(orig_ids)} tokens, ids={orig_ids[:30]}"
        )
        enc_fn = encode_fn_for_lang[lang]
        if enc_fn is not None:
            las_ids = enc_fn(sample_text)
            logging.info(
                f"LAS ({las_mode}) tokenization (no special chars) [{lang}]: {len(las_ids)} tokens, ids={las_ids[:30]}"
            )
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
            lang_texts[lang], tokenizer, encode_fn=encode_fn_for_lang[lang]
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
    per_lang_datasets = []
    for lang in langs:
        ds = TokenizedTextDataset(
            lang_texts[lang],
            tokenizer,
            cfg.dataset,
            encode_fn=encode_fn_for_lang[lang],
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        per_lang_datasets.append(ds)
    dataset = (
        ConcatDataset(per_lang_datasets)
        if len(per_lang_datasets) > 1
        else per_lang_datasets[0]
    )
    val_size = max(1, int(len(dataset) * float(cfg.val_fraction)))
    train_size = len(dataset) - val_size
    logging.info(f"Train dataset size: {train_size} samples")
    logging.info(f"Eval dataset size: {val_size} samples")
    generator = torch.Generator().manual_seed(cfg.seed)
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
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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
