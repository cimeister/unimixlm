#!/usr/bin/env python3
# Finetuning LLMs on FineWeb2 / TheStack tokens (multilingual / code)

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
from tqdm import tqdm

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


def resolve_fineweb2_lang(lang: str, size: str):
    """(repo, config) for a FineWeb2 language slice as actually published.

    Hub layout (verified): every non-English language is its own repo
    ``FineWeb2-<lang>_Latn-<size>`` with a single default ``train`` split
    and no named config; English lives as the ``eng_Latn`` config of the
    multi-language ``FineWeb2-100M-olmo3-7b-toks`` repo. (The old
    ``FineWeb2-<size>`` single-repo-with-configs layout the code assumed
    only ever existed for a different language set.)
    """
    size = size.upper()
    if lang == "eng":
        return "MultilingualUnigramLM/FineWeb2-100M-olmo3-7b-toks", "eng_Latn"
    # The 500M slices were published with the inverse name ordering
    # (FineWeb2-500M-<lang>_Latn) versus the 100M layout
    # (FineWeb2-<lang>_Latn-100M); honour both.
    if size == "500M":
        return f"MultilingualUnigramLM/FineWeb2-500M-{lang}_Latn", None
    return f"MultilingualUnigramLM/FineWeb2-{lang}_Latn-{size}", None


def resolve_fineweb2_dataset(size: str) -> str:
    size = size.upper()
    if size == "5M":
        return "MultilingualUnigramLM/FineWeb2-5M"
    elif size == "10M":
        return "MultilingualUnigramLM/FineWeb2-10M"
    elif size == "50M":
        return "MultilingualUnigramLM/FineWeb2-50M"
    elif size == "100M":
        return "MultilingualUnigramLM/FineWeb2-100M-olmo3-7b-toks"
    elif size == "500M":
        return "MultilingualUnigramLM/FineWeb2-500M"
    else:
        raise ValueError(
            f"Invalid FineWeb2 size: {size} (use '5M', '10M', '50M', '100M', or '500M')"
        )


def resolve_code_dataset(lang: str, size: str) -> str:
    """Resolve a per-language code dataset HF repo name.

    Each language has its own repo:
      MultilingualUnigramLM/LangMap-TheStack-{lang}-{size}
    """
    size = size.upper()
    if size not in {"5M", "10M", "50M", "100M", "500M"}:
        raise ValueError(
            f"Invalid code dataset size: {size} "
            f"(use '5M', '10M', '50M', '100M', or '500M')"
        )
    return f"MultilingualUnigramLM/LangMap-TheStack-{lang}-{size}"


# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer: Tokenizer,
        dataset_cfg,
        pad_token_id,
        eos_token_id,
        encode_batch_size=10000,
    ):
        max_seq_len = dataset_cfg.max_seq_len

        self.examples = []
        self.total_tokens = 0

        for i in tqdm(range(0, len(texts), encode_batch_size), desc="Tokenizing"):
            batch = texts[i : i + encode_batch_size]
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)

            for enc in encodings:
                ids = enc.ids
                self.total_tokens += len(ids)

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
    if model_type in {
        "llama",
        "mistral",
        "qwen2",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "phi3",
        "olmo",
        "olmo3",
    }:
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

    # Precision auto-detect: bf16 on capable GPUs (A100), fp16 on older
    # cards without bf16 (Pascal/Maxwell local node), fp32 on CPU. Same code
    # is correct on every tier; no per-host editing.
    _bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    _dtype = (
        torch.bfloat16
        if _bf16
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    logging.info(f"Precision: bf16={_bf16}, model dtype={_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=_dtype,
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

    elif las_mode in ("langspec", "placebo"):
        # `placebo` = the langspec machinery with a scrambled score table
        # (same vocab + IDs, permuted unigram scores). It is the causal
        # control: it must show NO gain. Files are produced by
        # scripts/make_placebo.py as placebo_{lang}[_Latn].json next to the
        # langspec files, so only the filename prefix changes here.
        prefix = "langspec" if las_mode == "langspec" else "placebo"
        if not las_dir:
            raise ValueError(
                f"las_tokenizer_path required for las_mode='{las_mode}'"
            )

        domain = str(cfg.get("domain", "natural_language"))

        if domain == "code":
            # Code tokenizers: {prefix}_{lang}.json. English is allowed as a
            # rehearsal slice on the code path; it has no overlay tokenizer
            # so it falls back to the model tokenizer.
            non_eng_code_langs = [lang for lang in langs if lang != "eng"]
            lang_toks = {
                lang: load_las_tokenizer_with_eos_pad(
                    las_dir / f"{prefix}_{lang}.json", model_tokenizer
                )
                for lang in non_eng_code_langs
            }
            tokenizer_for_lang = {lang: t[0] for lang, t in lang_toks.items()}
            if "eng" in langs:
                tokenizer_for_lang["eng"] = model_tokenizer.backend_tokenizer
        else:
            # Natural language: {prefix}_{lang}_Latn.json with eng special case
            non_eng_langs = [lang for lang in langs if lang != "eng"]

            lang_toks = {
                lang: load_las_tokenizer_with_eos_pad(
                    las_dir / f"{prefix}_{lang}_Latn.json", model_tokenizer
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
            f"Unknown las_mode: {las_mode}. "
            f"Use 'original', 'base', 'langspec', or 'placebo'."
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

    # ---- Load dataset splits ----
    domain = str(cfg.get("domain", "natural_language"))
    lang_texts = {}
    for lang in langs:
        if domain == "code" and lang != "eng":
            hf_name = resolve_code_dataset(lang, dataset_size)
            logging.info(f"Loading {hf_name} (code)")
            ds = load_dataset(hf_name, split="train", streaming=False, data_dir="data")
        elif domain == "code" and lang == "eng":
            # English rehearsal on the code path: pull the same FineWeb2 slice
            # used by the NL path so the model retains general-text fluency.
            hf_name, cfg_name = resolve_fineweb2_lang("eng", dataset_size)
            logging.info(f"Loading {hf_name} :: eng_Latn (code-path English rehearsal)")
            ds = load_dataset(hf_name, cfg_name, split="train", streaming=False)
        else:
            hf_name, cfg_name = resolve_fineweb2_lang(lang, dataset_size)
            logging.info(f"Loading {hf_name} :: {cfg_name or 'train'}")
            ds = (
                load_dataset(hf_name, cfg_name, split="train", streaming=False)
                if cfg_name
                else load_dataset(hf_name, split="train", streaming=False)
            )

        texts = [x["text"] for x in ds]
        if not texts:
            raise ValueError(
                f"No texts found for language '{lang}' in {hf_name}"
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

    # ---- Dataset + token accounting (single pass) ----
    per_lang_datasets = []
    lang_token_counts = {}
    for lang in langs:
        ds = TokenizedTextDataset(
            lang_texts[lang],
            tokenizer_for_lang[lang],
            cfg.dataset,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        lang_token_counts[lang] = ds.total_tokens
        per_lang_datasets.append(ds)
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
    lora_kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    # save_embeddings=true => register embed/LM-head as modules_to_save so
    # PEFT actually serialises them. Without this, unfreeze_embeddings only
    # sets requires_grad and trainer.save_model() (PEFT) drops the trained
    # embeddings — the adapter-only artifact bug. Gated + absent-by-default
    # so existing runs are byte-for-byte unchanged.
    if cfg.get("save_embeddings", False):
        lora_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    # ---- Optionally unfreeze embedding + LM head alongside LoRA ----
    if cfg.get("unfreeze_embeddings", False):
        unfrozen = []
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                unfrozen.append(name)
        logging.info(
            f"Unfroze {len(unfrozen)} embedding/LM-head parameters alongside LoRA: "
            f"{unfrozen}"
        )

    # ---- Distributed (DDP) awareness ----
    # Under torchrun WORLD_SIZE>1. Hold the GLOBAL effective batch
    # invariant by shrinking grad-accum by world_size, so DDP results
    # stay comparable to single-GPU AND the checkpoint cadence (computed
    # from the global batch below) is independent of world_size — i.e.
    # num_checkpoints still lands evenly over the token budget.
    _world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    _grad_accum = int(cfg.gradient_accumulation_steps)
    if _world_size > 1:
        _grad_accum = max(1, _grad_accum // _world_size)
        logging.info(
            f"DDP world_size={_world_size}: grad_accum "
            f"{int(cfg.gradient_accumulation_steps)} -> {_grad_accum} "
            f"(global effective batch held constant)"
        )
    _global_batch = int(cfg.per_device_batch_size) * _grad_accum * _world_size

    # ---- Checkpoint cadence ----
    # num_checkpoints == 0  -> legacy: one checkpoint per epoch.
    # num_checkpoints  > 0  -> save_strategy="steps" with save_steps spaced
    # to land roughly that many evenly across the run. For a single-epoch
    # 500M-token run, num_checkpoints=5 ≈ one checkpoint per ~100M tokens.
    # The step estimate assumes single-process training (one `python
    # run_finetune.py` per GPU, as the LangMAP launchers do); under DDP the
    # spacing would widen by world_size but the run stays correct.
    num_checkpoints = int(cfg.get("num_checkpoints", 0))
    if num_checkpoints > 0:
        import math as _math

        eff_batch = _global_batch  # true global batch (DDP-aware)
        steps_per_epoch = max(1, _math.ceil(train_size / eff_batch))
        total_steps = steps_per_epoch * int(cfg.epochs)
        save_steps = max(1, total_steps // num_checkpoints)
        save_kwargs = dict(
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=None,  # keep every milestone checkpoint
        )
        logging.info(
            f"Checkpointing: {num_checkpoints} requested -> save_steps="
            f"{save_steps} (~{total_steps} total steps, {steps_per_epoch}/epoch, "
            f"eff_batch={eff_batch})"
        )
    else:
        save_kwargs = dict(save_strategy="epoch")
        logging.info("Checkpointing: legacy save_strategy=epoch")

    # ---- Trainer ----
    training_args = TrainingArguments(
        # MUST set output_dir: without it HF Trainer defaults to a
        # CWD-relative "trainer_output/", so intermediate checkpoints
        # (a) land on the tiny 20G volume, and (b) collide across
        # concurrent cells sharing the repo CWD. Pin it to the unique
        # per-cell dir so checkpoints are isolated and on /dev/shm
        # where the pusher uploads them.
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        remove_unused_columns=False,
        eval_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"] if cfg.logger.enabled else [],
        bf16=_bf16,
        fp16=(torch.cuda.is_available() and not _bf16),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        eval_steps=int(cfg.get("eval_steps", 300)),
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=_grad_accum,
        ddp_find_unused_parameters=(_world_size > 1),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        seed=cfg.seed,
        **save_kwargs,
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
