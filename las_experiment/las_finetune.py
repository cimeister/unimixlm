#!/usr/bin/env python3

#Finetuning Language Models on either 5M or 10M Tokens from FineWeb2 
#languages: Filipino (Tagalog), Hausa, Igbo, Indonesian, Lithuanian, Chichewa (Nyanja), Romanian, Shona, Somali, Swahili, Yoruba, Malay (Standard)

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
    DataCollatorForLanguageModeling,
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
import pycountry


logging.basicConfig(level=logging.INFO)


# ---------------------- Utils ----------------------

def to_iso639_3(code: str) -> str:
    code = code.lower()
    entry = pycountry.languages.get(alpha_2=code)
    if entry is None:
        raise ValueError(f"Unknown ISO-639-1 code: {code}")
    return entry.alpha_3


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
                tokenizer(preprocess_fn(t), truncation=False, padding=False)["input_ids"]
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

            if current and not drop_last_pack:
                packed_blocks.append(current)

            for block in packed_blocks:
                if len(block) < max_seq_len:
                    pad_len = max_seq_len - len(block)
                    block = block + [pad_id] * pad_len
                    attn = [1] * (max_seq_len - pad_len) + [0] * pad_len
                else:
                    attn = [1] * max_seq_len

                self.examples.append(
                    {"input_ids": block, "attention_mask": attn}
                )

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


# ---------------------- Main Fine-tune ----------------------

def run_finetune(cfg):
    langs = list(cfg.langs)  # e.g. ["fil"] or ["fil", "hau", "ibo"]
    dataset_size = cfg.dataset.size  # "5M" or "10M"
    model_name = cfg.model.model_name
    las_tokenizer_path = (
        Path(cfg.las_tokenizer_path) if cfg.las_tokenizer_path else None
    )
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("./finetuned_model")
    ft_epochs = cfg.epochs
    ft_batch_size = cfg.batch_size
    dataset_cfg = cfg.dataset
    lora_cfg = cfg.lora
    learning_rate = cfg.learning_rate
    logger_cfg = cfg.logger if hasattr(cfg, "logger") else None
    wandb_enabled = bool(logger_cfg and logger_cfg.enabled)

    HF_DATASET_NAME = resolve_fineweb2_dataset(dataset_size)

    logging.info(f"Using FineWeb2 dataset: {HF_DATASET_NAME}")
    logging.info(f"Languages: {langs}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    def preprocess_fn(x):
        return x if getattr(tokenizer, "byte_level", False) else x.strip().lower()

    # ---- LAS tokenizer integration ----
    if las_tokenizer_path is not None:
        base_tokenizer_path = las_tokenizer_path / "base.tokenizer.json"
        language_paths = {
            lang: str(
                las_tokenizer_path
                / f"langspec_soft_{to_iso639_3(lang)}_Latn.tokenizer.json"
            )
            for lang in langs
        }

        shared_las = LanguageSpecificUnigramLMTokenizer(
            vocab_size=len(tokenizer), byte_level=True
        )
        shared_las.load(str(base_tokenizer_path), language_paths)

        def las_preprocess(text: str) -> str:
            _, best_tokens = shared_las.best_language_encode(preprocess_fn(text))
            return " ".join(best_tokens)

        model.resize_token_embeddings(len(shared_las.base_tokenizer.get_vocab()))
        logging.info(
            f"Resized embeddings to {len(shared_las.base_tokenizer.get_vocab())}"
        )
    else:
        las_preprocess = preprocess_fn

    # ---- Load FineWeb2 multilingual splits ----
    all_texts = []
    lang_texts = {}

    for lang in langs:
        config_name = f"{lang}_Latn"
        logging.info(f"Loading {HF_DATASET_NAME} :: {config_name}")

        ds = load_dataset(
            HF_DATASET_NAME,
            config_name,
            split="train",
            streaming=False,
        )

        texts = [las_preprocess(x["text"]) for x in ds]
        all_texts.extend(texts)
        lang_texts[lang] = texts

    # ---- WandB ----
    if wandb_enabled:
        wandb.init(
            project=str(logger_cfg.project),
            entity=str(logger_cfg.entity),
            name=logger_cfg.run_name,
            tags=list(logger_cfg.tags) if logger_cfg.tags else None,
            notes=logger_cfg.notes,
        )

    # ---- Token accounting ----
    lang_token_counts = {
        lang: count_training_tokens(lang_texts[lang], tokenizer, preprocess_fn)
        for lang in langs
    }
    total_train_tokens = sum(lang_token_counts.values())

    logging.info(f"Total training tokens: {total_train_tokens}")
    for lang, count in lang_token_counts.items():
        logging.info(f"Training tokens [{lang}]: {count}")

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

    # ---- Dataset ----
    dataset = TokenizedTextDataset(all_texts, tokenizer, preprocess_fn, dataset_cfg)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- LoRA ----
    if lora_cfg.target_modules:
        lora_targets = [m.strip() for m in lora_cfg.target_modules.split(",") if m.strip()]
    else:
        lora_targets = infer_lora_targets(model.config.model_type)

    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    model_save_dir = str(
        output_dir
        / f"las_finetuned_{model_name.split('/')[-1]}_{dataset_size}_{'-'.join(langs)}"
    )

    training_args = TrainingArguments(
        output_dir=model_save_dir,
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


# ---------------------- Example cfg ----------------------
class SimpleNamespace(dict):
    def __getattr__(self, k):
        return self[k]


if __name__ == "__main__":
    cfg = SimpleNamespace(
        langs=["fil", "hau"],  # ONE or MULTIPLE: ["fil"] or ["fil", "hau", "ibo"]
        dataset=SimpleNamespace(
            size="5M",  # "5M" or "10M"
            max_seq_len=2048,
            pack_dataset=True,
            add_eos_between_samples=True,
            align_to_max_seq_len=False,
            drop_last_pack=False,
        ),
        model=SimpleNamespace(
            model_name="gpt2",
        ),
        lora=SimpleNamespace(
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=None,
        ),
        epochs=1,
        batch_size=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        output_dir="./finetuned_model",
        las_tokenizer_path=None,  # or Path("path/to/las")
        logger=SimpleNamespace(
            enabled=False,
            project="fineweb2-las",
            entity=None,
            run_name="debug-run",
            tags=["fineweb2", "las", "lora"],
            notes=None,
        ),
    )

    run_finetune(cfg)
