#!/usr/bin/env python3

import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
import pycountry


def to_iso639_3(code: str) -> str:
    code = code.lower()
    entry = pycountry.languages.get(alpha_2=code)
    return entry.alpha_3


# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        preprocess_fn,
        dataset_cfg,
    ):
        max_seq_len = dataset_cfg.get("max_seq_len", 4096)
        align_to_max_seq_len = bool(dataset_cfg.get("align_to_max_seq_len", True))
        drop_last_pack = bool(dataset_cfg.get("drop_last_pack", False))
        pack_dataset = bool(dataset_cfg.get("pack_dataset", False))
        add_eos_between_samples = bool(
            dataset_cfg.get("add_eos_between_samples", False)
        )

        if pack_dataset and max_seq_len is None:
            raise ValueError("max_seq_len must be set when pack_dataset=True")

        self.examples = []
        if pack_dataset:
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = eos_id if eos_id is not None else 0

            tokenized = [
                tokenizer(preprocess_fn(t), truncation=False, padding=False)[
                    "input_ids"
                ]
                for t in texts
            ]

            packed_blocks = []
            current = []
            for ids in tokenized:
                if add_eos_between_samples and eos_id is not None:
                    if not ids or ids[-1] != eos_id:
                        ids = ids + [eos_id]

                while ids:
                    space = max_seq_len - len(current)
                    if space <= 0:
                        packed_blocks.append(current)
                        current = []
                        continue

                    if len(ids) <= space:
                        current.extend(ids)
                        ids = []
                    else:
                        current.extend(ids[:space])
                        ids = ids[space:]
                        packed_blocks.append(current)
                        current = []

            if current:
                if not drop_last_pack:
                    packed_blocks.append(current)

            for block in packed_blocks:
                if len(block) < max_seq_len:
                    pad_len = max_seq_len - len(block)
                    block = block + [pad_id] * pad_len
                    attn = [1] * (max_seq_len - pad_len) + [0] * pad_len
                else:
                    attn = [1] * max_seq_len
                self.examples.append({"input_ids": block, "attention_mask": attn})
        else:
            for t in texts:
                enc = tokenizer(
                    preprocess_fn(t),
                    truncation=bool(align_to_max_seq_len),
                    max_length=max_seq_len if align_to_max_seq_len else None,
                    padding="max_length" if align_to_max_seq_len else False,
                    return_attention_mask=True,
                )
                self.examples.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc.get("attention_mask"),
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def count_training_tokens(texts, tokenizer, preprocess_fn):
    total_tokens = 0
    for t in texts:
        enc = tokenizer(
            preprocess_fn(t), return_tensors="pt", truncation=False, padding=False
        )
        total_tokens += enc["input_ids"].numel()
    return total_tokens


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


def run_finetune(cfg):
    langs = list(cfg.langs)
    model_name = cfg.model.model_name
    las_tokenizer_path = (
        Path(cfg.las_tokenizer_path) if cfg.las_tokenizer_path else None
    )
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("./finetuned_model")
    ft_epochs = cfg.epochs
    ft_batch_size = cfg.batch_size
    dataset_cfg = cfg.dataset
    lora_cfg = cfg.lora
    lora_r = lora_cfg.r
    lora_alpha = lora_cfg.alpha
    lora_dropout = lora_cfg.dropout
    lora_target_modules = lora_cfg.target_modules
    learning_rate = cfg.learning_rate
    logger_cfg = cfg.logger if hasattr(cfg, "logger") else None
    wandb_enabled = bool(logger_cfg and logger_cfg.enabled)

    LANGUAGES = langs

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # TODO: Change to fineweb2 data paths
    CORPUS_DIR = Path("data/parallel/dev")
    corpus_paths = {lang: CORPUS_DIR / f"{lang}.txt" for lang in LANGUAGES}
    for lang, p in corpus_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Corpus {lang} not found: {p}")

    def preprocess_fn(x):
        return x if getattr(tokenizer, "byte_level", False) else x.strip().lower()

    if las_tokenizer_path is not None:
        # TODO: In the case when only a single file is provided, use it both as base and language-specific tokenizer
        base_tokenizer_path = las_tokenizer_path / "base.tokenizer.json"
        language_paths = {
            lang: str(
                las_tokenizer_path
                / f"langspec_soft_{to_iso639_3(lang)}_Latn.tokenizer.json"
            )
            for lang in LANGUAGES
        }
        if not base_tokenizer_path.exists():
            raise FileNotFoundError(
                f"Base LAS tokenizer not found: {base_tokenizer_path}"
            )
        for lang, p in language_paths.items():
            if not Path(p).exists():
                raise FileNotFoundError(f"LAS tokenizer for {lang} not found: {p}")

        shared_las = LanguageSpecificUnigramLMTokenizer(
            vocab_size=len(tokenizer), byte_level=True
        )
        shared_las.load(str(base_tokenizer_path), language_paths)
        logging.info(f"Loaded LAS tokenizer for fine-tuning from {las_tokenizer_path}.")

        def las_preprocess(text: str) -> str:
            _, best_tokens = shared_las.best_language_encode(preprocess_fn(text))
            return " ".join(best_tokens)

        model.resize_token_embeddings(len(shared_las.base_tokenizer.get_vocab()))
        logging.info(
            f"Resized LM embeddings to {len(shared_las.base_tokenizer.get_vocab())} tokens"
        )

    all_texts, lang_texts = [], {}
    for lang, path in corpus_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            tok_lines = [
                las_preprocess(l) if las_tokenizer_path is not None else l
                for l in lines
            ]
            all_texts.extend(tok_lines)
            lang_texts[lang] = tok_lines

    # configure wandb
    if wandb_enabled:
        wandb.init(
            project=str(logger_cfg.project),
            entity=str(logger_cfg.entity),
            name=logger_cfg.run_name,
            tags=list(logger_cfg.tags) if logger_cfg.tags else None,
            notes=logger_cfg.notes,
        )

    lang_token_counts = {
        lang: count_training_tokens(lang_texts[lang], tokenizer, preprocess_fn)
        for lang in LANGUAGES
    }
    total_train_tokens = sum(lang_token_counts.values())
    logging.info(f"Total training tokens: {total_train_tokens}")
    for lang, lang_tokens in lang_token_counts.items():
        logging.info(f"Training tokens for {lang}: {lang_tokens}")
    if wandb_enabled and wandb.run is not None:
        wandb.log(
            {
                "train/total_tokens": total_train_tokens,
                **{
                    f"train/tokens_{lang}": count
                    for lang, count in lang_token_counts.items()
                },
            }
        )

    dataset = TokenizedTextDataset(all_texts, tokenizer, preprocess_fn, dataset_cfg)
    logging.info(
        f"Dataset example count: {len(dataset)}, dataset first example length: {len(dataset[0]['input_ids'])}"
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if lora_target_modules:
        lora_targets = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    else:
        lora_targets = infer_lora_targets(model.config.model_type)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model_save_dir = str(
        output_dir
        / f"las_finetuned_{model_name.split('/')[-1]}_{str(langs)}_{'original-tokenizer' if las_tokenizer_path is None else 'las-tokenizer'}"
    )

    # ---- Fine-tuning ----
    training_args = TrainingArguments(
        num_train_epochs=ft_epochs,
        per_device_train_batch_size=ft_batch_size,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"] if wandb_enabled else [],
        bf16=True,
        learning_rate=learning_rate,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
    )
    logging.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(model_save_dir)
    logging.info("Fine-tuning complete.")
