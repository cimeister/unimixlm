#!/usr/bin/env python3
# Finetuning LLMs on FineWeb2 5M/10M/200M tokens (multilingual)

import json
import logging
import math
import os
import random
import subprocess
import time
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
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


MODEL_ALIASES = {
    "allenai/Olmo-3-1025-7B": "olmo3-7b",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "google/gemma-3-1b-pt": "gemma3-1b",
    "google/gemma-2-2b": "gemma2-2b",
}
CODE_LANGS = {"python", "javascript", "java", "go"}
# Morphologically rich languages. P4d (F10) anchors on "set D" = fin/hun (Uralic),
# tur (Turkic), swh (Bantu), kor (Koreanic), arb (Semitic-templatic). Keep this
# set authoritative for the `domain:morph` WandB tag; `_domain_for` checks
# CODE_LANGS first, then MORPH_LANGS, so single-lang runs in set D resolve to
# `morph` (not `latin`).
MORPH_LANGS = {"tur", "fin", "hun", "swh", "kor", "arb"}
# Latin-script low-resource adaptations (F1/F5/F7 historical). `swh` used to be
# here but is now classified `morph` per the P4d typology decision; that's
# intentional and may differ from prior runs' tags.
LATIN_LANGS = {"afr", "ron", "ind", "fil", "cat", "por", "dan"}


def _base_alias(model_path):
    return MODEL_ALIASES.get(model_path, model_path.split("/")[-1].lower())


def _domain_for(langs):
    s = set(langs)
    if s & CODE_LANGS: return "code"
    if s & MORPH_LANGS: return "morph"
    if s & LATIN_LANGS: return "latin"
    return None


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return None


def build_descriptive_tags(cfg, langs, model_path, adapt_mode):
    """Descriptive tag set required by `memory/feedback_wandb_tagging.md`.

    Every WandB run emitted by this pipeline should carry these. The tags are
    additive to any `cfg.logger.tags` the user configured; never remove.

    `cfg.section` is optional. When set (e.g. `"§8"`), emits `section:§8`.
    We deliberately do NOT try to infer section from (base, lang, adapt) here
    because the mapping is not unique (§3 vs §5 both fit frozen OLMo-3 code).
    """
    base = _base_alias(model_path)
    tok_map = {"original": "bpe", "base": "base-unigram",
               "pathpiece": "pathpiece", "langspec": "langspec"}
    tags = {
        f"base:{base}",
        f"lang:{'+'.join(langs)}",
        f"tok:{tok_map.get(cfg.las_mode, cfg.las_mode)}",
        f"adapt:{adapt_mode}",
        f"lr:{cfg.learning_rate}",
        f"epochs:ep{cfg.epochs}",
    }
    if cfg.use_lora and getattr(cfg.lora, "r", None) is not None:
        tags.add(f"rank:r{cfg.lora.r}")
    dom = _domain_for(langs)
    if dom: tags.add(f"domain:{dom}")
    section = getattr(cfg, "section", None)
    if section:
        tags.add(f"section:{section}")
    return tags


def write_run_metadata(output_dir, *, cfg, model_path, adapt_mode, is_tied,
                       langs, lang_proportions, resolved_lang_paths,
                       lang_doc_counts_raw, lang_doc_counts_kept,
                       lang_tokens_budget, lang_token_counts,
                       lang_pretrunc_counts, size_kind, dataset_size,
                       heldout_manifest_path=None, wandb_run=None,
                       status="started", extra=None,
                       lang_repetition_info=None,
                       observation_steps=None, eval_token_budgets=None,
                       allow_repetition=False):
    """Write `run_metadata.json` — canonical provenance + actual-tokens record.

    Lives alongside the checkpoint. Future eval scripts and the results
    markdown read this file, not WandB (which loses nested config).

    Called twice per run: once with `status="started"` **before** `trainer.train`
    (so provenance exists even if training crashes), and once more with
    `status="completed"` + `extra={"final_metrics": ...}` afterward.
    An outer except clause calls it with `status="crashed"` + the exception.
    """
    from omegaconf import OmegaConf
    try:
        cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_snapshot = None
    total_post = sum(lang_token_counts.values())
    total_pre = sum(lang_pretrunc_counts.values())
    meta = {
        "schema_version": 2,
        "status": status,
        "output_dir": str(output_dir),
        "git_commit": _git_commit(),
        "tokenizer": {
            "las_mode": cfg.las_mode,
            "las_tokenizer_path": getattr(cfg, "las_tokenizer_path", None),
            "model_tokenizer_path": getattr(cfg, "model_tokenizer_path", None),
        },
        "model": {
            "name": model_path,
            "base_alias": _base_alias(model_path),
            "tie_word_embeddings": is_tied,
            "adaptation_mode": adapt_mode,
            "lora_rank": getattr(cfg.lora, "r", None) if cfg.use_lora else None,
            "unfreeze_embeds": bool(getattr(cfg, "unfreeze_embeddings", False)),
            "reinit_embeddings": bool(getattr(cfg, "reinit_embeddings", False)),
            "true_tie": bool(getattr(cfg, "true_tie", False)),
        },
        "data_sources": [
            {
                "lang": lang,
                "resolved_path": resolved_lang_paths.get(lang),
                "proportion": lang_proportions.get(lang, 1.0),
                "docs_raw": lang_doc_counts_raw.get(lang),
                "docs_kept": lang_doc_counts_kept.get(lang),
                "size_kind": size_kind,
                "size_spec": dataset_size,
                "target_tokens": lang_tokens_budget.get(lang),
            }
            for lang in langs
        ],
        "tokens": {
            "per_lang_posttrunc": dict(lang_token_counts),
            "per_lang_pretrunc": dict(lang_pretrunc_counts),
            "total_posttrunc_unique": total_post,
            "total_pretrunc_unique": total_pre,
            "epochs": int(cfg.epochs),
            "total_posttrunc_trained": int(total_post * cfg.epochs),
        },
        "eval_split": {
            "strategy": "heldout-manifest" if heldout_manifest_path else "val-fraction",
            "val_fraction": getattr(cfg, "val_fraction", None),
            "heldout_manifest": heldout_manifest_path,
            "seed": int(cfg.seed),
        },
        "training": {
            "learning_rate": cfg.learning_rate,
            "per_device_batch_size": cfg.per_device_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_seq_len": cfg.dataset.max_seq_len,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "seed": int(cfg.seed),
            "resume_policy": str(getattr(cfg, "resume_policy", "never")),
            "allow_repetition": bool(allow_repetition),
            "eval_token_budgets": list(eval_token_budgets) if eval_token_budgets else None,
            "observation_steps": list(observation_steps) if observation_steps else None,
            "per_lang_repetition": dict(lang_repetition_info) if lang_repetition_info else None,
            "schedule_fingerprint": _compute_schedule_fingerprint(cfg, eval_token_budgets),
        },
        "wandb": {
            "run_id": wandb_run.id if wandb_run is not None else None,
            "project": wandb_run.project if wandb_run is not None else None,
            "entity": wandb_run.entity if wandb_run is not None else None,
            "run_name": wandb_run.name if wandb_run is not None else None,
        },
        "cfg_snapshot": cfg_snapshot,
    }
    if extra:
        meta.update(extra)
    out = Path(output_dir) / "run_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp + os.replace so a concurrent reader or a crash
    # mid-write never sees a partial file.
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, out)
    return out, meta


def _resolve_embed_modules(model):
    """Return (input_emb_name, output_emb_name, is_tied) using the model's own
    get_input_embeddings()/get_output_embeddings() API (works for GPT-2's `wte`,
    OLMo/Llama/Qwen's `embed_tokens`, Gemma's `embed_tokens`, etc.).

    Matches modules by object identity, not substring — so silent misses on
    models with non-standard names (e.g. GPT-2's `wte`) are impossible.
    """
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    is_tied = (
        input_emb is not None
        and output_emb is not None
        and getattr(input_emb, "weight", None) is getattr(output_emb, "weight", None)
    )
    input_name = output_name = None
    for name, module in model.named_modules():
        if module is input_emb:
            input_name = name
        if module is output_emb:
            output_name = name
    if input_name is None:
        raise RuntimeError(
            f"Could not locate input embedding module on model "
            f"{type(model).__name__}. get_input_embeddings() returned "
            f"{type(input_emb).__name__ if input_emb is not None else None}."
        )
    return input_name, output_name, is_tied


def resolve_fineweb2_dataset(size: str) -> str:
    raise RuntimeError(
        "resolve_fineweb2_dataset is retired: it encoded the old single-repo "
        "hub layout (FineWeb2-100M-gpt2-toks) which no longer matches what is "
        "published. Use resolve_fineweb2_lang(lang, size) instead."
    )


def resolve_fineweb2_lang(lang: str, size: str):
    """(repo, config) for a FineWeb2 language slice as actually published.

    Hub layout (verified 2026-08-13): every non-English language is its own
    repo ``FineWeb2-<lang>_Latn-<size>`` with a single default ``train`` split
    and no named config; English lives as the ``eng_Latn`` config of the
    multi-language ``FineWeb2-100M-olmo3-7b-toks`` repo. Ported verbatim from
    the suchir-code-tok branch so reruns load the exact repos its runs used.
    """
    size = size.upper()
    if lang == "eng":
        if size != "100M":
            raise ValueError(
                f"resolve_fineweb2_lang: only a 100M eng slice is published "
                f"(FineWeb2-100M-olmo3-7b-toks); refusing to pair size={size!r} "
                f"non-eng data with a 100M eng slice."
            )
        return "MultilingualUnigramLM/FineWeb2-100M-olmo3-7b-toks", "eng_Latn"
    if size == "500M":
        return f"MultilingualUnigramLM/FineWeb2-500M-{lang}_Latn", None
    return f"MultilingualUnigramLM/FineWeb2-{lang}_Latn-{size}", None


def _parse_size_spec(size) -> tuple[str, int | None]:
    """Classify a `dataset.size` value.

    Returns (kind, tokens) where kind is one of:
      - "hf-hub"  : use the HF-hub name resolver (size ∈ {"5M","10M","100M"})
      - "local"   : use the full local_dir (size == "local")
      - "target"  : use the first N tokens of local_dir (size == "target:<N>"
                    where <N> is a token count, with optional K/M/B suffix)
    Unrecognized values raise ValueError.
    """
    if size is None:
        raise ValueError("dataset.size is required")
    s = str(size).strip()
    if s.lower() == "local":
        return ("local", None)
    if s.lower().startswith("target:"):
        tail = s.split(":", 1)[1].strip().upper()
        mult = 1
        if tail.endswith("K"):
            mult, tail = 1_000, tail[:-1]
        elif tail.endswith("M"):
            mult, tail = 1_000_000, tail[:-1]
        elif tail.endswith("B"):
            mult, tail = 1_000_000_000, tail[:-1]
        try:
            n = int(float(tail) * mult)
        except Exception as e:
            raise ValueError(f"Invalid token budget in size='{s}': {e}")
        if n <= 0:
            raise ValueError(f"Token budget must be positive, got {n}")
        return ("target", n)
    # Otherwise assume HF-hub size string
    return ("hf-hub", None)


def _cap_texts_to_token_budget(
    texts, tokenizer, target_tokens, max_seq_len, allow_repetition=False
):
    """Truncate a list of texts at the doc boundary that first reaches
    `target_tokens` *post-truncation* tokens, using the provided tokenizer.

    Returns a dict with keys:
      - texts:              the kept doc list (possibly shorter than input)
      - unique_tokens:      total tokens available in the corpus (capped at target
                            if cap-within-first-pass path is taken)
      - unique_docs:        number of unique docs available up to the cap
      - target_tokens:      the requested budget (unchanged)
      - allow_repetition:   whether repetition was permitted for this call

    When the corpus has at least `target_tokens` available, behavior matches the
    prior contract: cap at the first doc boundary hitting target.

    When the corpus has fewer than `target_tokens` and `allow_repetition=False`,
    raise (unchanged fail-loud behavior).

    When the corpus has fewer than `target_tokens` and `allow_repetition=True`,
    return the full deduped doc list with `unique_tokens < target_tokens`.
    The Trainer is expected to consume this dataset multiple times via
    `max_steps` rather than materializing repetitions in memory (which would
    tokenize the same docs many times and blow up RAM for short-corpus langs).
    The caller is responsible for sizing `max_steps` to hit `target_tokens`.
    """
    kept = []
    total = 0
    # Stream in chunks to avoid a full encode when the budget is small.
    CHUNK = 2048
    for i in range(0, len(texts), CHUNK):
        batch = texts[i : i + CHUNK]
        encs = tokenizer.encode_batch(batch, add_special_tokens=False)
        for text, enc in zip(batch, encs):
            n = min(len(enc.ids), max_seq_len)
            kept.append(text)
            total += n
            if total >= target_tokens:
                return {
                    "texts": kept,
                    "unique_tokens": total,
                    "unique_docs": len(kept),
                    "target_tokens": target_tokens,
                    "allow_repetition": allow_repetition,
                }
    # Corpus exhausted without hitting target.
    if not allow_repetition:
        raise RuntimeError(
            f"Corpus is too small to hit token budget: "
            f"requested {target_tokens:,}, available {total:,} tokens from "
            f"{len(texts):,} docs. Either point `local_dir` at a larger corpus, "
            f"lower `dataset.size=target:<N>`, or set `allow_repetition=true` "
            f"to repeat the corpus at training time."
        )
    if total == 0:
        raise RuntimeError(
            f"Corpus has zero tokens; cannot repeat. Input texts: {len(texts):,}"
        )
    return {
        "texts": kept,
        "unique_tokens": total,
        "unique_docs": len(kept),
        "target_tokens": target_tokens,
        "allow_repetition": allow_repetition,
    }


def _compute_schedule_fingerprint(cfg, eval_token_budgets_raw):
    """Collect the config fields that determine the LR-schedule shape.

    The HF Trainer rebuilds its cosine scheduler against `max_steps` on every
    resume, then fast-forwards by the saved optimizer step count. If any field
    below differs between submissions, the resumed run's remaining schedule
    shape diverges from the original — silently. Bug 8 (2026-04-14).

    This fingerprint is logged to `run_metadata.json` (`training.schedule_fingerprint`)
    and asserted against on `resume_policy=auto` before training starts.
    """
    return {
        "per_device_batch_size": int(cfg.per_device_batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "max_seq_len": int(cfg.dataset.max_seq_len),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "learning_rate": float(cfg.learning_rate),
        "epochs": int(cfg.epochs),
        "eval_token_budgets": (
            list(int(t) for t in eval_token_budgets_raw)
            if eval_token_budgets_raw else None
        ),
    }


def _assert_schedule_fingerprint_matches(output_dir, cfg, eval_token_budgets_raw):
    """Raise on bug-8 config drift when resuming.

    On resume, read the prior `run_metadata.json` from `output_dir` and assert
    that every schedule-affecting config field matches what we're about to
    hand to the Trainer. If any differs, raise a specific error naming the
    mismatches — never silently proceed.

    If no prior metadata exists (first resume of an older run), log a warning
    and allow the resume — we can't assert what we can't read.
    """
    meta_path = Path(output_dir) / "run_metadata.json"
    if not meta_path.exists():
        logging.warning(
            f"resume_policy='auto' resume but no {meta_path} to check bug-8 "
            f"fingerprint against. Proceeding — first-time resume of a legacy run."
        )
        return

    try:
        with open(meta_path) as f:
            prior_meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(
            f"Could not read prior run_metadata.json for bug-8 fingerprint check: {e}. "
            f"Proceeding — fingerprint verification skipped."
        )
        return

    prior_training = prior_meta.get("training", {}) or {}
    prior_fp = prior_training.get("schedule_fingerprint")
    current_fp = _compute_schedule_fingerprint(cfg, eval_token_budgets_raw)

    if prior_fp is None:
        # Legacy metadata predating bug-8 fix. We can reconstruct most of the
        # fingerprint from the older top-level fields; world_size is not
        # recorded and we can't verify it.
        prior_fp = {
            "per_device_batch_size": prior_training.get("per_device_batch_size"),
            "gradient_accumulation_steps": prior_training.get("gradient_accumulation_steps"),
            "max_seq_len": prior_training.get("max_seq_len"),
            "world_size": None,
            "learning_rate": prior_training.get("learning_rate"),
            "epochs": (prior_meta.get("tokens", {}) or {}).get("epochs"),
            "eval_token_budgets": prior_training.get("eval_token_budgets"),
        }
        # If world_size cannot be verified, we allow the resume but warn loudly.
        current_for_compare = dict(current_fp)
        current_for_compare["world_size"] = None  # opt out of comparison
        logging.warning(
            "Prior run_metadata.json has no schedule_fingerprint "
            "(pre-bug-8-fix). world_size check skipped; other fields compared."
        )
    else:
        current_for_compare = current_fp

    mismatches = {
        k: {"prior": prior_fp.get(k), "current": current_for_compare.get(k)}
        for k in current_for_compare
        if prior_fp.get(k) != current_for_compare.get(k)
    }
    if mismatches:
        raise RuntimeError(
            f"resume_policy='auto' but schedule-affecting config has changed "
            f"since the prior submission (bug 8: silent LR schedule drift).\n"
            f"Mismatches: {json.dumps(mismatches, indent=2)}\n"
            f"HF Trainer rebuilds the cosine scheduler against the NEW max_steps "
            f"on resume, then fast-forwards by the saved optimizer step count — "
            f"so a change in any of the above fields causes the LR at resume to "
            f"jump and the remaining schedule shape to differ silently.\n"
            f"To proceed, either (a) match the prior config exactly, "
            f"(b) delete the checkpoints in {output_dir} and start fresh with "
            f"resume_policy='never', or (c) set resume_policy='never' with full "
            f"awareness that you are starting over."
        )
    logging.info(
        f"bug-8 fingerprint OK: schedule-affecting config matches prior "
        f"submission ({current_fp})."
    )


class TokenBudgetCheckpointCallback(TrainerCallback):
    """Force a checkpoint save at a specific set of global_step values.

    Observation points for the Qwen3 three-way comparison (P4 plan item) are
    specified in tokens, not steps. At launch time the caller converts each
    token budget to a step count using the actual effective batch × seq_len,
    then instantiates this callback with the resulting step list. The callback
    sets `control.should_save = True` on exactly those steps; HF Trainer then
    writes `checkpoint-<step>/` as usual.
    """

    def __init__(self, observation_steps):
        self.observation_steps = sorted({int(s) for s in observation_steps if int(s) > 0})
        self._fired = set()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.observation_steps and step not in self._fired:
            control.should_save = True
            self._fired.add(step)
        return control


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
        self.total_tokens = 0        # post-truncation — what the model actually sees
        self.pretrunc_tokens = 0     # pre-truncation — diagnostic only; can inflate if many long docs get clipped

        for i in tqdm(range(0, len(texts), encode_batch_size), desc="Tokenizing"):
            batch = texts[i : i + encode_batch_size]
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)

            for enc in encodings:
                ids = enc.ids
                self.pretrunc_tokens += len(ids)

                # 2. Truncate to max context length
                ids = ids[:max_seq_len]
                self.total_tokens += len(ids)

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

    @classmethod
    def from_examples(cls, examples, total_tokens, pretrunc_tokens):
        """Construct from pre-built examples (used by streaming pipelines that
        tokenize once up-front to enforce a token budget, rather than feeding
        raw `texts` in and re-tokenizing here).

        The caller is responsible for producing examples with the same
        schema as the `__init__` path: `{input_ids, attention_mask, labels,
        num_bytes}`, already padded to `max_seq_len`, with EOS appended
        where appropriate, labels=-100 on pad positions. `total_tokens` and
        `pretrunc_tokens` are for the run_metadata record.
        """
        obj = cls.__new__(cls)
        obj.examples = list(examples)
        obj.total_tokens = int(total_tokens)
        obj.pretrunc_tokens = int(pretrunc_tokens)
        return obj

    @staticmethod
    def build_example(ids_pretrunc, tokenizer, max_seq_len, pad_token_id, eos_token_id):
        """Build a single padded example dict from a pre-tokenized id list.

        Returns (example_dict, n_pretrunc, n_posttrunc). Extracted from the
        `__init__` loop so callers (e.g. streaming builders) can use the same
        construction without allocating through the class's raw-text path.
        """
        n_pretrunc = len(ids_pretrunc)
        ids = ids_pretrunc[:max_seq_len]
        n_posttrunc = len(ids)
        num_bytes = len(
            tokenizer.decode(ids, skip_special_tokens=True).encode("utf-8")
        )
        if len(ids) < max_seq_len:
            ids = ids + [eos_token_id]
        pad_len = max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        labels = list(ids) + [-100] * pad_len
        ids = ids + [pad_token_id] * pad_len
        return (
            {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "num_bytes": num_bytes,
            },
            n_pretrunc,
            n_posttrunc,
        )


class BpBTrainer(Trainer):
    """Trainer subclass that computes bits-per-byte (BpB) during evaluation.

    BpB = total_NLL / (total_bytes * ln(2)), accumulated across all eval batches.
    This is tokenizer-agnostic, unlike per-token perplexity.
    """

    def __init__(self, *args, perf_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bpb_total_nll = 0.0
        self._bpb_total_bytes = 0
        self._tracking_bpb = False
        self._perf_config = perf_config
        self._perf_step_t0 = None
        # Training BpB accumulators (reset at each log step)
        self._train_bpb_nll = 0.0
        self._train_bpb_bytes = 0

    def log(self, logs, start_time=None):
        """Inject tokens/sec, MFU, and training BpB before dispatching to loggers."""
        if self._perf_config and self._perf_step_t0 is not None:
            elapsed = time.time() - self._perf_step_t0
            if elapsed > 0:
                pcfg = self._perf_config
                tokens_per_step = (
                    self.args.per_device_train_batch_size
                    * pcfg["seq_len"]
                    * self.args.world_size
                    * self.args.gradient_accumulation_steps
                    * self.args.logging_steps
                )
                tokens_per_sec = tokens_per_step / elapsed
                peak_flops = pcfg["peak_tflops_per_gpu"] * 1e12 * self.args.world_size
                mfu = (6 * pcfg["model_num_params"] * tokens_per_sec) / peak_flops
                logs["perf/tokens_per_sec"] = round(tokens_per_sec, 1)
                logs["perf/mfu"] = round(mfu, 4)
        self._perf_step_t0 = time.time()

        # Training BpB: accumulated since last log step
        if self._train_bpb_bytes > 0:
            train_bpb = self._train_bpb_nll / (self._train_bpb_bytes * math.log(2))
            logs["train_bpb"] = round(train_bpb, 4)
            self._train_bpb_nll = 0.0
            self._train_bpb_bytes = 0

        super().log(logs, start_time)

    def compute_loss(self, model, inputs, **kwargs):
        num_bytes = inputs.pop("num_bytes", None)
        labels = inputs.get("labels")
        loss = super().compute_loss(model, inputs, **kwargs)

        # Accumulate training BpB
        if loss is not None and labels is not None and num_bytes is not None:
            num_loss_tokens = (labels[:, 1:] != -100).sum().item()
            self._train_bpb_nll += loss.item() * num_loss_tokens
            self._train_bpb_bytes += num_bytes.sum().item()

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Load-bearing ordering: popping num_bytes HERE (before the super()
        # call) is what keeps compute_loss safe under eval. The base
        # Trainer.prediction_step internally re-dispatches to our
        # compute_loss with return_outputs=True, where super().compute_loss
        # returns a (loss, outputs) TUPLE; because num_bytes was already
        # popped, compute_loss's `num_bytes is not None` guard skips the
        # accumulation block, so tuple.item() is never attempted. Verified
        # against transformers 4.57.5 and 5.3.0 (trainer.py re-dispatch).
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
    # No silent fallbacks (2026-05-17): the previous `pad_token_id = eos_token_id`
    # on a None pad masked exactly the OLMo-vs-Qwen3 mismatch that let
    # langspec/base/pathpiece cells train with the wrong eos. If the model's
    # pad/eos strings are not in this LAS vocab, that is a config error — abort
    # naming the offending file and token, never substitute.
    if pad_token_id is None:
        raise ValueError(
            f"pad_token {model_tokenizer.pad_token!r} not found in LAS tokenizer "
            f"{path} — model_tokenizer/model mismatch or wrong tokenizer dir. "
            "Refusing to silently fall back pad→eos."
        )
    if eos_token_id is None:
        raise ValueError(
            f"eos_token {model_tokenizer.eos_token!r} not found in LAS tokenizer "
            f"{path} — model_tokenizer/model mismatch or wrong tokenizer dir."
        )
    return tok, pad_token_id, eos_token_id


def infer_lora_targets(model_type: str):
    if model_type in {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma3_text", "olmo", "olmo3"}:
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

    if not cfg.model_tokenizer_path:
        raise ValueError("model_tokenizer_path must be set")
    model_tokenizer_path = cfg.model_tokenizer_path

    model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)

    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    # ── Provenance guard (2026-05-17): model_tokenizer MUST be the tokenizer of
    # the model being trained. A Hydra-default regression silently left
    # `model_tokenizer_path` pointing at OLMo for non-original LAS modes, so
    # eos/pad were resolved from OLMo's special-token strings against the
    # model's LAS vocab (Qwen3 langspec/base/pathpiece cells trained with
    # eos=151643 instead of 151645). For LAS modes model_tokenizer is used only
    # to look up pad/eos *strings*; for `original` it IS the training tokenizer.
    # In every case it must match `cfg.model.model_name`. Fail loud on mismatch
    # — never silently train under a foreign tokenizer's special tokens.
    _model_own_tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if _model_own_tok.pad_token is None:
        _model_own_tok.pad_token = _model_own_tok.eos_token
    _mismatch = []
    if _model_own_tok.eos_token != model_tokenizer.eos_token:
        _mismatch.append(
            f"eos_token: model({model_name})={_model_own_tok.eos_token!r} "
            f"vs model_tokenizer_path({model_tokenizer_path})="
            f"{model_tokenizer.eos_token!r}"
        )
    if _model_own_tok.pad_token != model_tokenizer.pad_token:
        _mismatch.append(
            f"pad_token: model={_model_own_tok.pad_token!r} "
            f"vs model_tokenizer_path={model_tokenizer.pad_token!r}"
        )
    if len(_model_own_tok) != len(model_tokenizer):
        _mismatch.append(
            f"vocab_size: model={len(_model_own_tok)} "
            f"vs model_tokenizer_path={len(model_tokenizer)}"
        )
    if _mismatch:
        raise ValueError(
            "model_tokenizer_path does not match the model being trained "
            f"(cfg.model.model_name={model_name!r}, "
            f"cfg.model_tokenizer_path={model_tokenizer_path!r}). "
            "This silently corrupts eos/pad resolution for LAS modes — refusing "
            "to train. Set model_tokenizer_path to the model's own tokenizer. "
            "Mismatches: " + "; ".join(_mismatch)
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ---- Optional embedding re-initialization (F14b ablation) ----
    # MUST run before get_peft_model: PEFT's modules_to_save deep-copies the
    # embedding module at wrap time, so a re-init after wrapping would leave
    # the trained copy (or the tie) inconsistent with the base weights.
    if bool(getattr(cfg, "reinit_embeddings", False)):
        if not getattr(cfg, "unfreeze_embeddings", False):
            raise ValueError(
                "reinit_embeddings=true requires unfreeze_embeddings=true — "
                "re-initialized embeddings that are frozen can never recover."
            )
        emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        tied = (
            out_emb is not None
            and out_emb.weight.data_ptr() == emb.weight.data_ptr()
        )
        if out_emb is not None and not tied:
            raise ValueError(
                "reinit_embeddings currently supports only tied models "
                "(output re-init would otherwise be silently skipped)."
            )
        std = float(getattr(model.config, "initializer_range", 0.02))
        # Deterministic on CPU in fp32, seeded by cfg.seed ONLY: within a
        # (lang, seed) cell every arm gets the identical random matrix, so
        # arms still differ only in tokenizer.
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        with torch.no_grad():
            new_w = torch.empty(
                emb.weight.shape, device="cpu", dtype=torch.float32
            ).normal_(0.0, std, generator=gen)
            emb.weight.data.copy_(new_w.to(emb.weight.dtype))
        logging.info(
            f"reinit_embeddings: {tuple(emb.weight.shape)} re-initialized "
            f"N(0, {std}) with generator seed {int(cfg.seed)}; tied output "
            f"follows (data_ptr shared={tied})."
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

    elif las_mode == "pathpiece":
        if not las_dir:
            raise ValueError("las_tokenizer_path required for las_mode='pathpiece'")

        pp_tok, pad_token_id, eos_token_id = load_las_tokenizer_with_eos_pad(
            las_dir / "pathpiece_tokenizer.json", model_tokenizer
        )
        tokenizer_for_lang = {lang: pp_tok for lang in langs}

    elif las_mode in ("langspec", "placebo"):
        # "placebo" = langspec machinery with a scrambled score table
        # (scripts/make_placebo.py): same vocab, same IDs, permuted Unigram
        # scores. Control arm separating language-specific re-weighting from
        # any-segmentation-perturbation-plus-finetuning.
        if not las_dir:
            raise ValueError(f"las_tokenizer_path required for las_mode='{las_mode}'")

        file_prefix = "langspec" if las_mode == "langspec" else "placebo"
        lang_suffix = getattr(cfg.dataset, "lang_suffix", "_Latn")

        # English is a special case: use the original model tokenizer
        non_eng_langs = [lang for lang in langs if lang != "eng"]

        lang_toks = {
            lang: load_las_tokenizer_with_eos_pad(
                las_dir / f"{file_prefix}_{lang}{lang_suffix}.json", model_tokenizer
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
            f"Unknown las_mode: {las_mode}. Use 'original', 'base', 'pathpiece', 'langspec', or 'placebo'."
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

    # ---- Load splits (local parquet or HF hub) ----
    local_data_dir = getattr(cfg.dataset, "local_dir", None)
    text_column = getattr(cfg.dataset, "text_column", "text")
    lang_suffix = getattr(cfg.dataset, "lang_suffix", "_Latn")
    size_kind, target_tokens = _parse_size_spec(dataset_size)

    # Guardrail: `local` / `target:N` require local_dir; HF sizes (`5M`...) forbid it.
    if local_data_dir and size_kind == "hf-hub":
        raise ValueError(
            f"dataset.local_dir={local_data_dir} is set but dataset.size={dataset_size!r} "
            f"is an HF-hub size. When using local_dir, size must be 'local' (use full "
            f"local corpus) or 'target:<N>' (cap at N tokens, e.g. 'target:500M'). "
            f"This guardrail exists because previous runs silently ignored dataset.size "
            f"when local_dir was set."
        )
    if not local_data_dir and size_kind != "hf-hub":
        raise ValueError(
            f"dataset.size={dataset_size!r} requires dataset.local_dir to be set."
        )

    # Resolve per-lang data paths up-front so provenance is explicit on failure.
    resolved_lang_paths = {}
    lang_texts = {}
    lang_heldout_texts = {}       # held-out eval slice (local_dir path only)
    lang_heldout_manifests = {}   # manifest entries for the heldout_manifest.json
    lang_doc_counts_raw = {}
    lang_doc_counts_kept = {}
    lang_tokens_budget = {}        # unique tokens available (post-cap)
    lang_repetition_info = {}      # {lang: {unique_tokens, target_tokens, corpus_below_target}}
    allow_repetition = bool(getattr(cfg, "allow_repetition", False))
    heldout_docs = int(getattr(cfg, "heldout_docs_per_lang", 1000))
    for lang in langs:
        if local_data_dir:
            from las_experiment.held_out import get_train_texts_and_heldout
            lang_dir = Path(local_data_dir) / f"{lang}{lang_suffix}"
            if not lang_dir.is_dir() or not any(lang_dir.glob("*.parquet")):
                raise FileNotFoundError(
                    f"Expected parquet data for lang '{lang}' at {lang_dir}. "
                    f"No fallback — confirm the path before re-running."
                )
            resolved_lang_paths[lang] = str(lang_dir)
            logging.info(f"Loading local data: {lang_dir}")
            # Compute a streaming byte cap so very large corpora (hun ~100GB,
            # tur ~143GB) don't OOM by reading every parquet into RAM. The
            # downstream _cap_texts_to_token_budget call truncates further by
            # tokens — this just stops the parquet-read loop early. Apply
            # 10 bytes/token × target_tokens × 2× safety as a conservative
            # upper bound that covers UTF-8-heavy scripts (Arabic ~2.5 bytes/
            # char, Korean ~3 bytes/char) while staying well above the actual
            # bytes needed for the budget. Only applies for "target:N" size
            # (full-corpus "local" mode keeps the prior behavior).
            stream_cap_bytes = None
            if size_kind == "target":
                stream_cap_bytes = int(target_tokens) * 20
            train_texts, heldout_texts, manifest = get_train_texts_and_heldout(
                source_dir=lang_dir,
                lang=lang,
                text_column=text_column,
                seed=int(cfg.seed),
                heldout_docs=heldout_docs,
                lang_suffix=lang_suffix,
                max_train_bytes=stream_cap_bytes,
                skip_corrupted_files=bool(
                    getattr(cfg.dataset, "skip_corrupted_files", False)
                ),
            )
            texts = train_texts
            lang_heldout_texts[lang] = heldout_texts
            lang_heldout_manifests[lang] = manifest
        else:
            # Published hub layout (see resolve_fineweb2_lang): per-language
            # repos FineWeb2-<lang>_Latn-<size>; eng is the eng_Latn config of
            # the multi-language FineWeb2-100M-olmo3-7b-toks repo. This is the
            # layout the PR #21 branch runs trained on; using the same repos
            # keeps rerun data identical to those runs.
            hf_name, hf_cfg = resolve_fineweb2_lang(lang, dataset_size)
            resolved_lang_paths[lang] = f"hf://{hf_name}::{hf_cfg or 'train'}"
            logging.info(f"Loading {hf_name} :: {hf_cfg or 'train'}")
            ds = (
                load_dataset(hf_name, hf_cfg, split="train", streaming=False)
                if hf_cfg
                else load_dataset(hf_name, split="train", streaming=False)
            )
            texts = [x[text_column] for x in ds]

        if not texts:
            raise ValueError(
                f"No texts found for language '{lang}' in {resolved_lang_paths[lang]}"
            )
        lang_doc_counts_raw[lang] = len(texts) + (len(lang_heldout_texts.get(lang, [])))

        # Apply token budget if requested (only meaningful with local_dir).
        if size_kind == "target":
            max_seq_len = cfg.dataset.max_seq_len
            cap_result = _cap_texts_to_token_budget(
                texts,
                tokenizer_for_lang[lang],
                target_tokens,
                max_seq_len,
                allow_repetition=allow_repetition,
            )
            texts = cap_result["texts"]
            actual_tok = cap_result["unique_tokens"]
            actual_docs = cap_result["unique_docs"]
            corpus_below_target = actual_tok < target_tokens
            lang_tokens_budget[lang] = actual_tok
            lang_doc_counts_kept[lang] = actual_docs
            lang_repetition_info[lang] = {
                "unique_tokens": int(actual_tok),
                "unique_docs": int(actual_docs),
                "target_tokens": int(target_tokens),
                "allow_repetition": allow_repetition,
                "corpus_below_target": corpus_below_target,
            }
            if corpus_below_target:
                logging.warning(
                    f"[{lang}] corpus smaller than target: {actual_tok:,} unique tokens "
                    f"vs target {target_tokens:,}. allow_repetition=True — trainer will "
                    f"loop the dataset to hit the budget via max_steps."
                )
            else:
                logging.info(
                    f"[{lang}] token budget: capped to {actual_docs:,} docs / "
                    f"~{actual_tok:,} tokens (target={target_tokens:,})"
                )
        else:
            lang_doc_counts_kept[lang] = len(texts)
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

    # ---- Persist held-out manifest (local_dir path only) ----
    heldout_manifest_path = None
    if lang_heldout_manifests:
        from las_experiment.held_out import write_heldout_manifest
        heldout_manifest_path = str(write_heldout_manifest(
            lang_heldout_manifests, output_dir
        ))
        logging.info(
            f"Wrote held-out manifest: {heldout_manifest_path} "
            f"({sum(len(v) for v in lang_heldout_texts.values())} total heldout docs)"
        )

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
    # Classify adaptation mode once; reused for tags and run_metadata.
    if cfg.use_lora and cfg.unfreeze_embeddings:
        adapt_mode = "lora+embed"
    elif cfg.use_lora:
        adapt_mode = "frozen"
    else:
        adapt_mode = "embed-only"

    if cfg.logger.enabled and is_main_process:
        # Descriptive tag taxonomy — see memory/feedback_wandb_tagging.md
        auto_tags = list(build_descriptive_tags(
            cfg, langs, cfg.model.model_name, adapt_mode
        ))
        manual_tags = list(cfg.logger.tags) if cfg.logger.tags else []
        all_tags = sorted(set(auto_tags + manual_tags))

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", cfg.logger.project),
            entity=os.environ.get("WANDB_ENTITY", cfg.logger.entity),
            name=f"{'-'.join(langs)}_{dataset_size}_{cfg.logger.run_name}{'_uf' if cfg.unfreeze_embeddings else ''}",
            tags=all_tags,
            notes=cfg.logger.notes,
        )

    # ---- Dataset + token accounting (single pass) ----
    per_lang_datasets = []
    lang_token_counts = {}          # post-truncation, what the model actually sees
    lang_pretrunc_counts = {}       # pre-truncation, diagnostic
    for lang in langs:
        ds = TokenizedTextDataset(
            lang_texts[lang],
            tokenizer_for_lang[lang],
            cfg.dataset,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        lang_token_counts[lang] = ds.total_tokens
        lang_pretrunc_counts[lang] = ds.pretrunc_tokens
        per_lang_datasets.append(ds)
    total_train_tokens = sum(lang_token_counts.values())
    total_pretrunc_tokens = sum(lang_pretrunc_counts.values())
    logging.info(f"Total training tokens (post-trunc): {total_train_tokens}")
    logging.info(f"Total pre-truncation tokens (diagnostic): {total_pretrunc_tokens}")
    for lang in langs:
        logging.info(
            f"Training tokens [{lang}]: post={lang_token_counts[lang]} "
            f"pre={lang_pretrunc_counts[lang]}"
        )
    if cfg.logger.enabled and is_main_process and wandb.run:
        wandb.log(
            {
                "train/total_tokens": total_train_tokens,
                "train/pretrunc_tokens": total_pretrunc_tokens,
                **{f"train/tokens_{lang}": c for lang, c in lang_token_counts.items()},
                **{f"train/pretrunc_tokens_{lang}": c for lang, c in lang_pretrunc_counts.items()},
            }
        )
    # Stratified per-language train/val splits to guarantee per-language
    # representation in the eval set and enable per-language BpB reporting.
    generator = torch.Generator().manual_seed(cfg.seed)
    per_lang_train = []
    per_lang_eval = {}
    for lang, ds in zip(langs, per_lang_datasets):
        val_size = max(1, int(len(ds) * float(cfg.val_fraction)))
        train_size = len(ds) - val_size
        train_ds, eval_ds = torch.utils.data.random_split(
            ds, [train_size, val_size], generator=generator
        )
        per_lang_train.append(train_ds)
        per_lang_eval[lang] = eval_ds

    train_dataset = (
        ConcatDataset(per_lang_train)
        if len(per_lang_train) > 1
        else per_lang_train[0]
    )
    eval_dataset = (
        ConcatDataset(list(per_lang_eval.values()))
        if len(per_lang_eval) > 1
        else list(per_lang_eval.values())[0]
    )
    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Eval dataset size: {len(eval_dataset)} samples")
    for lang, ds in per_lang_eval.items():
        logging.info(f"Eval dataset [{lang}]: {len(ds)} samples")

    # ---- Adaptation strategy ----
    use_lora = getattr(cfg, "use_lora", True)

    # Capture tie_word_embeddings on the pre-PEFT model once; pass-through to
    # run_metadata rather than re-reading from the PEFT wrapper later (which
    # has unreliable forwarding for tie state).
    is_tied_weights = bool(
        getattr(model.config, "tie_word_embeddings", False)
    ) if hasattr(model, "config") else False

    if use_lora:
        # LoRA (optionally with unfrozen embeddings)
        lora_targets = (
            [m.strip() for m in cfg.lora.target_modules.split(",")]
            if cfg.lora.target_modules
            else infer_lora_targets(model.config.model_type)
        )

        unfreeze_embeds = getattr(cfg, "unfreeze_embeddings", False)
        true_tie = bool(getattr(cfg, "true_tie", False))
        modules_to_save = None
        ensure_weight_tying = False
        if unfreeze_embeds:
            input_name, output_name, is_tied = _resolve_embed_modules(model)
            # Prefer the object-identity result from the resolver (authoritative)
            is_tied_weights = bool(is_tied)
            if is_tied:
                # WARNING (measured 2026-08-14, peft 0.18.1): with only the
                # input embedding in modules_to_save, PEFT trains a deep COPY
                # for the input lookup while lm_head keeps the FROZEN original
                # Parameter — the tie is broken at wrap time and the output
                # head never trains. This is the wiring of every F10/F12/F14
                # "tie-preserving" cell (memory: modules-to-save-tie-not-
                # preserved). For a genuinely tied trainable embedding set
                # cfg.true_tie=true, which adds ensure_weight_tying=True so
                # PEFT wraps BOTH modules around ONE shared trained Parameter.
                # PEFT's ensure_weight_tying tie-detection only recognizes the
                # tied pair when modules_to_save carries the BARE module name
                # ("embed_tokens"); with the prefixed name ("model.embed_tokens")
                # it wraps the embedding but silently skips wrapping/sharing
                # the head (measured on peft 0.18.1; the data_ptr verification
                # below is the backstop). Non-true-tie keeps the full name to
                # stay byte-identical with the as-run F14 configuration.
                modules_to_save = (
                    [input_name.rsplit(".", 1)[-1]] if true_tie else [input_name]
                )
                ensure_weight_tying = true_tie
                logging.info(
                    f"Unfreezing tied embedding: {input_name} "
                    f"(true_tie={true_tie}: output head "
                    f"{'shares the trained Parameter' if true_tie else 'stays the FROZEN original'})."
                )
            else:
                if true_tie:
                    raise ValueError(
                        "true_tie=true set on an untied model — misconfigured."
                    )
                modules_to_save = [n for n in (input_name, output_name) if n]
                logging.info(f"Unfreezing untied embeddings: {modules_to_save}")
        elif true_tie:
            raise ValueError("true_tie=true requires unfreeze_embeddings=true.")

        lora_kwargs = dict(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=lora_targets,
            modules_to_save=modules_to_save,
            task_type=TaskType.CAUSAL_LM,
        )
        if ensure_weight_tying:
            lora_kwargs["ensure_weight_tying"] = True
        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)
        if ensure_weight_tying:
            # Hard verification, not trust: the trained embed copy and the
            # output head must be ONE Parameter.
            _inner = model.base_model.model
            _wrap = _inner.get_input_embeddings()
            _copy = _wrap.modules_to_save["default"].weight
            _head = _inner.get_output_embeddings()
            _head_w = (
                _head.modules_to_save["default"].weight
                if hasattr(_head, "modules_to_save") else _head.weight
            )
            if _head_w.data_ptr() != _copy.data_ptr():
                raise RuntimeError(
                    "true_tie: ensure_weight_tying did NOT share the trained "
                    "Parameter between embed copy and output head — peft "
                    "version drift? Refusing to train a mislabeled cell."
                )
            logging.info("true_tie verified: embed copy and output head share one Parameter.")
    else:
        # Embed-only: freeze all params, then unfreeze input+output embedding
        # modules. Uses object-identity resolver (handles GPT-2's `wte`, tied
        # lm_heads, etc.) rather than the substring match that silently missed
        # non-standard module names.
        input_name, output_name, is_tied = _resolve_embed_modules(model)
        is_tied_weights = bool(is_tied)
        target_names = {input_name}
        if not is_tied and output_name:
            target_names.add(output_name)
        for param in model.parameters():
            param.requires_grad = False
        trainable_count = 0
        trainable_module_names = set()
        for name, param in model.named_parameters():
            mod_name = name.rsplit(".", 1)[0]
            if mod_name in target_names:
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_module_names.add(mod_name)
        total_count = sum(p.numel() for p in model.parameters())
        logging.info(
            f"Embed-only mode: {trainable_count:,} / {total_count:,} params trainable "
            f"({100 * trainable_count / total_count:.1f}%) — "
            f"modules={sorted(trainable_module_names)} (is_tied={is_tied})"
        )
        if trainable_count == 0:
            raise RuntimeError(
                f"Embed-only mode resolved zero trainable params. "
                f"Expected module names: {sorted(target_names)}. "
                f"Available: {sorted({n.rsplit('.', 1)[0] for n, _ in model.named_parameters()})[:20]}..."
            )

    # ---- Compute observation schedule from eval_token_budgets (if provided) ----
    # Convention: tokens_per_step = per_device_batch × world_size × grad_accum × seq_len.
    # When `eval_token_budgets` is set, the last budget is the training budget
    # (overrides num_train_epochs via max_steps). The callback then forces a save
    # at each intermediate step, producing a fixed observation-point schedule
    # independent of dataset size or epoch count.
    eval_token_budgets_raw = getattr(cfg, "eval_token_budgets", None)
    observation_steps = None
    computed_max_steps = -1  # -1 means "fall back to num_train_epochs"
    if eval_token_budgets_raw:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tokens_per_step = (
            int(cfg.per_device_batch_size)
            * world_size
            * int(cfg.gradient_accumulation_steps)
            * int(cfg.dataset.max_seq_len)
        )
        if tokens_per_step <= 0:
            raise RuntimeError(
                f"Non-positive tokens_per_step: per_device={cfg.per_device_batch_size} "
                f"world={world_size} grad_accum={cfg.gradient_accumulation_steps} "
                f"seq_len={cfg.dataset.max_seq_len}"
            )
        budgets_int = [int(t) for t in eval_token_budgets_raw]
        if sorted(budgets_int) != budgets_int:
            raise ValueError(
                f"eval_token_budgets must be monotonically increasing: got {budgets_int}"
            )
        observation_steps = [max(1, math.ceil(t / tokens_per_step)) for t in budgets_int]
        computed_max_steps = observation_steps[-1]
        logging.info(
            f"eval_token_budgets enabled: tokens_per_step={tokens_per_step:,}, "
            f"observation_steps={observation_steps}, max_steps={computed_max_steps}"
        )

    # ---- Trainer ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs if computed_max_steps < 0 else 1_000_000,  # max_steps will cap
        max_steps=computed_max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        remove_unused_columns=False,
        save_strategy="no" if observation_steps else "epoch",
        eval_strategy="steps",
        eval_steps=getattr(cfg, "eval_steps", 300),
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"] if cfg.logger.enabled else [],
        bf16=True,
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=cfg.seed,
    )
    perf_config = {
        "model_num_params": sum(p.numel() for p in model.parameters()),
        "seq_len": cfg.dataset.max_seq_len,
        "peak_tflops_per_gpu": float(os.environ.get("PEAK_TFLOPS_PER_GPU", "989")),
    }
    trainer = BpBTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        perf_config=perf_config,
    )
    if observation_steps:
        trainer.add_callback(TokenBudgetCheckpointCallback(observation_steps))

    # Resume-from-checkpoint policy. `never` aborts if checkpoints exist (forces
    # explicit opt-in; prevents silent continuation of a stale run). `auto`
    # resumes from the latest (for multi-submission SLURM jobs that requeue).
    # `require` errors if no checkpoint is present (for verifying a resume
    # actually picks up state).
    resume_policy = str(getattr(cfg, "resume_policy", "never")).lower()
    if resume_policy not in {"never", "auto", "require"}:
        raise ValueError(
            f"resume_policy must be one of 'never', 'auto', 'require'; got {resume_policy!r}"
        )

    existing_ckpts = (
        sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if output_dir.exists() else []
    )
    last_ckpt = None
    if resume_policy == "never":
        if existing_ckpts:
            raise RuntimeError(
                f"resume_policy='never' and {len(existing_ckpts)} checkpoint(s) "
                f"already exist in {output_dir} (latest={existing_ckpts[-1]}). "
                f"Either delete them, move output_dir, or set resume_policy='auto'."
            )
    elif resume_policy == "auto":
        if existing_ckpts:
            last_ckpt = str(existing_ckpts[-1])
            logging.info(f"resume_policy='auto': resuming from {last_ckpt}")
            _assert_schedule_fingerprint_matches(
                output_dir, cfg, eval_token_budgets_raw
            )
    elif resume_policy == "require":
        if not existing_ckpts:
            raise RuntimeError(
                f"resume_policy='require' but no checkpoint found under {output_dir}."
            )
        last_ckpt = str(existing_ckpts[-1])
        logging.info(f"resume_policy='require': resuming from {last_ckpt}")
        _assert_schedule_fingerprint_matches(
            output_dir, cfg, eval_token_budgets_raw
        )

    # Shared kwargs for all three metadata writes (started / completed / crashed).
    # Build once so every write records the same provenance — only `status` +
    # `extra` change between writes.
    _meta_kwargs = dict(
        cfg=cfg,
        model_path=cfg.model.model_name,
        adapt_mode=adapt_mode,
        is_tied=is_tied_weights,
        langs=langs,
        lang_proportions=lang_proportions,
        resolved_lang_paths=resolved_lang_paths,
        lang_doc_counts_raw=lang_doc_counts_raw,
        lang_doc_counts_kept=lang_doc_counts_kept,
        lang_tokens_budget=lang_tokens_budget,
        lang_token_counts=lang_token_counts,
        lang_pretrunc_counts=lang_pretrunc_counts,
        size_kind=size_kind,
        dataset_size=dataset_size,
        heldout_manifest_path=heldout_manifest_path,
        lang_repetition_info=lang_repetition_info,
        observation_steps=observation_steps,
        eval_token_budgets=list(eval_token_budgets_raw) if eval_token_budgets_raw else None,
        allow_repetition=allow_repetition,
    )

    # G'1: write pre-train provenance so a crash still leaves a record.
    if is_main_process:
        try:
            meta_path, _ = write_run_metadata(
                output_dir, status="started",
                wandb_run=wandb.run if (cfg.logger.enabled and wandb.run) else None,
                **_meta_kwargs,
            )
            logging.info(f"Wrote pre-train {meta_path} (status=started)")
        except Exception as e:
            logging.exception(f"Failed to write pre-train run_metadata.json: {e}")

    logging.info("Starting fine-tuning...")
    try:
        trainer.train(resume_from_checkpoint=last_ckpt)

        # Per-language BpB evaluation
        final_metrics = {}
        if len(per_lang_eval) > 1:
            for lang, eval_ds in per_lang_eval.items():
                metrics = trainer.evaluate(
                    eval_dataset=eval_ds, metric_key_prefix=f"eval_{lang}"
                )
                bpb = metrics.get(f"eval_{lang}_bpb")
                if bpb is not None:
                    logging.info(f"Per-language BpB [{lang}]: {bpb:.4f}")
                    final_metrics[f"eval_{lang}_bpb"] = float(bpb)

        trainer.save_model(str(output_dir))

        # G'1: overwrite with status=completed + final metrics.
        if is_main_process:
            try:
                meta_path, _ = write_run_metadata(
                    output_dir, status="completed",
                    wandb_run=wandb.run if (cfg.logger.enabled and wandb.run) else None,
                    extra={"final_metrics": final_metrics},
                    **_meta_kwargs,
                )
                logging.info(f"Wrote {meta_path} (status=completed)")
                if cfg.logger.enabled and wandb.run is not None:
                    wandb.summary.update({
                        "run_metadata/total_posttrunc_unique": sum(lang_token_counts.values()),
                        "run_metadata/total_posttrunc_trained": int(sum(lang_token_counts.values()) * cfg.epochs),
                        "run_metadata/path": str(meta_path),
                        "run_metadata/status": "completed",
                    })
            except Exception as e:
                logging.exception(f"Failed to write completed run_metadata.json: {e}")

    except (KeyboardInterrupt, SystemExit):
        # User-initiated — don't write status=crashed for these.
        raise
    except Exception as train_exc:
        # G'1: preserve provenance for crashed runs. Re-raise after writing.
        if is_main_process:
            try:
                meta_path, _ = write_run_metadata(
                    output_dir, status="crashed",
                    wandb_run=wandb.run if (cfg.logger.enabled and wandb.run) else None,
                    extra={
                        "crash": {
                            "exception_type": type(train_exc).__name__,
                            "message": str(train_exc)[:2000],
                        },
                    },
                    **_meta_kwargs,
                )
                logging.info(f"Wrote {meta_path} (status=crashed)")
                if cfg.logger.enabled and wandb.run is not None:
                    wandb.summary.update({"run_metadata/status": "crashed"})
            except Exception as meta_exc:
                logging.exception(f"Failed to write crashed run_metadata.json: {meta_exc}")
        raise

    logging.info("Fine-tuning complete.")
