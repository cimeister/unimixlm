#!/usr/bin/env python3
# Finetuning LLMs on FineWeb2 5M/10M/200M tokens (multilingual)

import logging
import math
import os
import random
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
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)


def parse_langs(raw_langs):
    """Parse langs config.

    Accepts either:
      - a dict  {lang: proportion, ...}  (e.g. ``langs: {yor: 0.2, fil: 1.0}``)
      - a list of plain strings           (e.g. ``langs: [yor, fil]``, proportion defaults to 1.0)
    """
    from omegaconf import DictConfig

    lang_codes = []
    lang_proportions = {}

    if isinstance(raw_langs, (dict, DictConfig)):
        items = [(str(k), float(v)) for k, v in raw_langs.items()]
    else:
        items = [(str(entry), 1.0) for entry in raw_langs]

    for lang, proportion in items:
        if lang in lang_proportions:
            raise ValueError(f"Duplicate language: '{lang}'")
        if proportion <= 0.0 or proportion > 1.0:
            raise ValueError(
                f"Proportion for '{lang}' must be in (0, 1], got {proportion}"
            )
        lang_codes.append(lang)
        lang_proportions[lang] = proportion
    return lang_codes, lang_proportions


def resolve_fineweb2_dataset(size: str) -> str:
    size = size.upper()
    if size == "5M":
        return "MultilingualUnigramLM/FineWeb2-5M"
    elif size == "10M":
        return "MultilingualUnigramLM/FineWeb2-10M"
    elif size == "100M":
        return "MultilingualUnigramLM/FineWeb2-100M-gpt2-toks"
    else:
        raise ValueError(f"Invalid FineWeb2 size: {size} (use '5M', '10M' or '100M')")


# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer: Tokenizer,
        dataset_cfg,
        pad_token_id,
        eos_token_id,
    ):
        max_seq_len = dataset_cfg.max_seq_len

        self.examples = []

        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False).ids

            # 2. Truncate to max context length
            ids = ids[:max_seq_len]

            # Compute bytes from the truncated tokens (not the original text)
            # so the byte count matches what the model actually trains on.
            num_bytes = len(
                tokenizer.decode(ids, skip_special_tokens=True).encode("utf-8")
            )

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
                    "num_bytes": num_bytes,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def count_training_tokens(texts, tokenizer: Tokenizer):
    return sum(len(tokenizer.encode(t, add_special_tokens=False).ids) for t in texts)


class BpBTrainer(Trainer):
    """Trainer subclass that computes bits-per-byte (BpB) during evaluation.

    BpB = total_NLL / (total_bytes * ln(2)), accumulated across all eval batches.
    This is tokenizer-agnostic, unlike per-token perplexity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bpb_total_nll = 0.0
        self._bpb_total_bytes = 0
        self._tracking_bpb = False

    def compute_loss(self, model, inputs, **kwargs):
        inputs.pop("num_bytes", None)
        return super().compute_loss(model, inputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        num_bytes = inputs.pop("num_bytes", None)
        labels = inputs.get("labels")

        loss, logits, labels_out = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )

        if self._tracking_bpb and loss is not None and labels is not None:
            # Labels are shifted inside the model: position i predicts position i+1.
            # Count non-padding loss tokens (same shift the model applies).
            num_loss_tokens = (labels[:, 1:] != -100).sum().item()
            self._bpb_total_nll += loss.item() * num_loss_tokens
            if num_bytes is not None:
                self._bpb_total_bytes += num_bytes.sum().item()

        return loss, logits, labels_out

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self._bpb_total_nll = 0.0
        self._bpb_total_bytes = 0
        self._tracking_bpb = True

        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self._tracking_bpb = False

        # Aggregate NLL and byte counts across all processes (each process
        # only accumulates its own eval shard). Uses accelerate's reduce,
        # the same infrastructure the Trainer uses for distributed metrics.
        total_nll = torch.tensor(self._bpb_total_nll, device=self.args.device)
        total_bytes = torch.tensor(
            float(self._bpb_total_bytes), device=self.args.device
        )
        total_nll = self.accelerator.reduce(total_nll, reduction="sum")
        total_bytes = self.accelerator.reduce(total_bytes, reduction="sum")
        total_nll = total_nll.item()
        total_bytes = int(total_bytes.item())

        if total_bytes > 0:
            bpb = total_nll / (total_bytes * math.log(2))
            output.metrics[f"{metric_key_prefix}_bpb"] = bpb
            logging.info(
                f"BpB ({metric_key_prefix}): {bpb:.4f} "
                f"(total_nll={total_nll:.2f}, total_bytes={total_bytes})"
            )

        return output


def load_las_tokenizer_with_eos_pad(path, model_tokenizer):
    """Load a LAS tokenizer JSON file and resolve pad/eos token IDs.

    Args:
        path: Path to a tokenizer JSON file (base or language-specific).
        model_tokenizer: The model's AutoTokenizer, used to look up pad/eos token strings.

    Returns:
        (tokenizer, pad_token_id, eos_token_id) tuple.
    """
    tok = Tokenizer.from_file(str(path))
    pad_token_id = tok.token_to_id(model_tokenizer.pad_token)
    eos_token_id = tok.token_to_id(model_tokenizer.eos_token)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return tok, pad_token_id, eos_token_id


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

    langs, lang_proportions = parse_langs(cfg.langs)
    lang_table = "\n".join(
        f"  {lang:<10} {prop:>6.0%}" for lang, prop in lang_proportions.items()
    )
    logging.info(
        f"Language config:\n{'  Language   Proportion'}\n{'  ' + '-' * 21}\n{lang_table}"
    )
    dataset_size = cfg.dataset.size
    model_name = cfg.model.model_name
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.model_tokenizer_path or not os.path.isdir(cfg.model_tokenizer_path):
        raise ValueError(
            f"model_tokenizer_path must be a valid directory, got: {cfg.model_tokenizer_path}"
        )
    model_tokenizer_path = Path(cfg.model_tokenizer_path)

    model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)

    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ---- LAS tokenizer mode ----
    las_mode = str(cfg.las_mode)
    las_dir = Path(cfg.las_tokenizer_path) if cfg.las_tokenizer_path else None

    if las_mode == "original":
        tokenizer_for_lang = {lang: model_tokenizer.backend_tokenizer for lang in langs}
        pad_token_id = model_tokenizer.pad_token_id
        eos_token_id = model_tokenizer.eos_token_id

    elif las_mode == "base":
        if not las_dir:
            raise ValueError("las_tokenizer_path required for las_mode='base'")

        base_tok, pad_token_id, eos_token_id = load_las_tokenizer_with_eos_pad(
            las_dir / "base_tokenizer.json", model_tokenizer
        )
        tokenizer_for_lang = {lang: base_tok for lang in langs}

    elif las_mode == "langspec":
        if not las_dir:
            raise ValueError("las_tokenizer_path required for las_mode='langspec'")

        # English is a special case: use the original model tokenizer
        non_eng_langs = [lang for lang in langs if lang != "eng"]

        lang_toks = {
            lang: load_las_tokenizer_with_eos_pad(
                las_dir / f"langspec_{lang}_Latn.json", model_tokenizer
            )
            for lang in non_eng_langs
        }

        tokenizer_for_lang = {lang: t[0] for lang, t in lang_toks.items()}

        if "eng" in langs:
            tokenizer_for_lang["eng"] = model_tokenizer.backend_tokenizer

        if lang_toks:
            _, pad_token_id, eos_token_id = next(iter(lang_toks.values()))
        else:
            # eng-only: fall back to model tokenizer IDs
            pad_token_id = model_tokenizer.pad_token_id
            eos_token_id = model_tokenizer.eos_token_id

    else:
        raise ValueError(
            f"Unknown las_mode: {las_mode}. Use 'original', 'base', or 'langspec'."
        )

    if pad_token_id is None:
        raise ValueError(
            f"Could not resolve pad_token_id for las_mode='{las_mode}'. "
            f"pad_token='{model_tokenizer.pad_token}' not found in the active tokenizer."
        )
    if eos_token_id is None:
        raise ValueError(
            f"Could not resolve eos_token_id for las_mode='{las_mode}'. "
            f"eos_token='{model_tokenizer.eos_token}' not found in the active tokenizer."
        )
    logging.info(
        f"Active tokenizer ({las_mode}): pad_token_id={pad_token_id}, eos_token_id={eos_token_id}"
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
        orig_ids = model_tokenizer(
            sample_text, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        logging.info(
            f"Original tokenization [{lang}]: {len(orig_ids)} tokens, ids={orig_ids[:30]}"
        )
        if las_mode != "original":
            las_ids = (
                tokenizer_for_lang[lang]
                .encode(sample_text, add_special_tokens=False)
                .ids
            )
            logging.info(
                f"LAS ({las_mode}) tokenization [{lang}]: {len(las_ids)} tokens, ids={las_ids[:30]}"
            )
        lang_texts[lang] = texts

    # ---- Subsample texts per language ----
    rng = random.Random(cfg.seed)
    for lang in langs:
        proportion = lang_proportions[lang]
        if proportion < 1.0:
            texts = lang_texts[lang]
            k = max(1, int(len(texts) * proportion))
            lang_texts[lang] = rng.sample(texts, k)
            logging.info(
                f"Subsampled {lang}: {len(texts)} -> {k} texts (proportion={proportion})"
            )

    # ---- WandB ----
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if cfg.logger.enabled and is_main_process:
        wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            name=f"{'-'.join(langs)}_{dataset_size}_{cfg.logger.run_name}",
            tags=cfg.logger.tags,
            notes=cfg.logger.notes,
        )

    # ---- Token accounting ----
    lang_token_counts = {
        lang: count_training_tokens(lang_texts[lang], tokenizer_for_lang[lang])
        for lang in langs
    }
    total_train_tokens = sum(lang_token_counts.values())
    logging.info(f"Total training tokens: {total_train_tokens}")
    for lang, count in lang_token_counts.items():
        logging.info(f"Training tokens [{lang}]: {count}")
    if cfg.logger.enabled and is_main_process and wandb.run:
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
            tokenizer_for_lang[lang],
            cfg.dataset,
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
        per_device_train_batch_size=cfg.per_device_batch_size,
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=300,
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"] if cfg.logger.enabled else [],
        bf16=True,
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )
    trainer = BpBTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    logging.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(str(output_dir))
    logging.info("Fine-tuning complete.")
