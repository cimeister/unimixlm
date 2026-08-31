#!/usr/bin/env python3
"""From-scratch GPT-2 training driver (multi-track, multi-set).

This is the Part C2 driver of the ★ ACTIVE plan (2026-05-19). Trains ONE
GPT-2 model per (track, set, tokenizer_type) combination — 16 runs total to
cover the full design:
    track ∈ {morph, code} × set ∈ {A, B} × tokenizer ∈ {bpe, base,
    pathpiece, langspec}.

Per (track, set):
    - Tokenizers must already exist (Part A: scripts/train_tokenizers_fromscratch.py).
    - Manifest must already exist (Part B: scripts/build_fromscratch_manifest.py).
    - Per-language LM-training token budgets come from
      `fromscratch_common.lang_token_budget(track, set)`. These are NON-UNIFORM
      (eng 50–80% of total; non-eng split per `lang_proportions`). Engcode's
      `load_training_data` only accepts a single uniform budget, so this
      driver replicates that function's logic per-language with a per-lang
      target.

Schema bridge: `run_metadata.json` is written in the schema that
`las_experiment/downstream_eval.py::_assert_checkpoint_provenance` requires
(`model.name`, `tokenizer.las_mode`, `tokenizer.las_tokenizer_path`,
`tokenizer.model_tokenizer_path == model.name`) so the existing downstream
eval pipeline accepts from-scratch checkpoints without modification.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset
from transformers import GPT2LMHeadModel, TrainingArguments

# Engcode is the source of low-level pieces we reuse. We monkey-patch its
# module-level LANGUAGES / TEXT_COL so its lang-iteration matches our track.
import scripts.train_fromscratch_engcode as ec
from scripts.train_fromscratch_engcode import (
    MemmappedEngcodeDataset,
    BpBTrainer,
    _build_memmap_cache,
    _engcode_cache_paths,
    _assert_tokenizers_consistent,
)
from code.model_trainer import (
    get_vocab_aware_config,
    get_vocab_aware_config_500m,
    assert_param_count,
)
from scripts.fromscratch_common import (
    TRACKS, SET_VOCAB, SET_TOTAL_TOKENS,
    track_langs, lm_dir, lang_token_budget, text_col,
)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── tokenizer_type ↔ tokenizer-fit filename + downstream las_mode label ──────
# train_tokenizers_fromscratch.py emits these filenames (mirror engcode).
TOK_FILE_BY_TYPE = {
    "bpe":       "bpe_tokenizer.json",
    "base":      "base_tokenizer.json",       # canonical (matches LAS dir naming)
    "pathpiece": "pathpiece_tokenizer.json",  # canonical
    # langspec is per-lang: tokenizer dir contains langspec_<lang>.json files.
}
# Downstream eval `expected_las_mode` for each training tokenizer (consumed by
# `_assert_checkpoint_provenance` and the eval_ft* / eval_ft_las branches).
LAS_MODE_BY_TYPE = {
    "bpe":       "original",   # trained-with-BPE model → native tokenizer
    "base":      "base",
    "pathpiece": "pathpiece",
    "langspec":  "langspec",
}
TARGET_PARAMS_BY_SET = {"A": 125_000_000, "A5B": 125_000_000, "B": 500_000_000}

# Per-language memmap-cache build timeout (seconds) used by the file-poll
# barrier in _load_training_data_perlang. Non-rank-0 ranks wait up to this long
# for rank-0 to finish building each language's cache. PER-LANGUAGE, not total,
# so it only needs to cover the single largest cache (SET-B English ≈ 5B tokens
# ⇒ up to a few hours). Default 8h (well inside the 12h walltime); override with
# FROMSCRATCH_CACHE_TIMEOUT_S. NOTE: new magic number introduced 2026-05-21.
CACHE_BUILD_TIMEOUT_S = int(os.environ.get("FROMSCRATCH_CACHE_TIMEOUT_S", str(8 * 3600)))


def _build_tok_paths(tokenizer_dir: Path, tokenizer_type: str,
                       langs: list[str]) -> dict:
    """Return engcode-shaped `tok_paths` for this run's tokenizer choice.

    bpe/base/pathpiece → {"train": <path>, "base": <path-to-base-for-consistency>}
    langspec           → {"base": <base path>, "per_lang": {lang: <langspec path>}}
    """
    base = tokenizer_dir / "base_tokenizer.json"
    if not base.exists():
        raise FileNotFoundError(f"Base unigram (base_tokenizer.json) missing: {base}")
    if tokenizer_type == "langspec":
        per_lang = {}
        for lang in langs:
            p = tokenizer_dir / f"langspec_{lang}.json"
            if not p.exists():
                raise FileNotFoundError(f"langspec missing for {lang}: {p}")
            per_lang[lang] = str(p)
        return {"base": str(base), "per_lang": per_lang}
    fname = TOK_FILE_BY_TYPE[tokenizer_type]
    tok_path = tokenizer_dir / fname
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer missing: {tok_path}")
    return {"train": str(tok_path), "base": str(base)}


def _select_tokenizer_for_lang(tok_paths: dict, lang: str) -> str:
    if "per_lang" in tok_paths and lang in tok_paths["per_lang"]:
        return tok_paths["per_lang"][lang]
    if "train" in tok_paths:
        return tok_paths["train"]
    return tok_paths["base"]


def _load_training_data_perlang(tok_paths: dict, manifest_path: Path,
                                  langs: list[str],
                                  per_lang_token_budget: dict[str, int],
                                  max_seq_len: int = 512,
                                  val_fraction: float = 0.005,
                                  cache_root: Path | None = None):
    """Per-language version of engcode's load_training_data. Each language
    consumes from the SAME manifest until ITS own token budget is hit
    (non-uniform). Reuses engcode's memmap-cache machinery one lang at a time
    via `_build_memmap_cache` + `MemmappedEngcodeDataset`.
    """
    import hashlib
    import time
    import torch.distributed as dist

    if cache_root is None:
        _engcode_cache_root = os.environ.get("ENGCODE_CACHE_ROOT")
        if not _engcode_cache_root:
            raise RuntimeError(
                "ENGCODE_CACHE_ROOT is not set and no cache_root was passed. "
                "Set ENGCODE_CACHE_ROOT to the scratch directory for the "
                "engcode memmap cache."
            )
        cache_root = Path(_engcode_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = dist.is_available() and dist.is_initialized()

    # Vocab-consistency: when langspec is used, sanity-assert base vs all langspec.
    if "per_lang" in tok_paths and "base" in tok_paths:
        _assert_tokenizers_consistent(
            tok_paths["base"], list(tok_paths["per_lang"].values()),
        )

    all_train, all_val = [], []
    per_lang_stats: dict[str, dict] = {}
    for lang in langs:
        target = per_lang_token_budget[lang]
        logger.info(f"[{lang}] loading from manifest, target={target:,} tokens")
        tok_path = _select_tokenizer_for_lang(tok_paths, lang)
        h = hashlib.sha1(Path(tok_path).read_bytes()).hexdigest()[:12]
        tok_sig = f"{Path(tok_path).stem}_{h}"
        bin_path, np_path, json_path = _engcode_cache_paths(
            cache_root, lang, tok_sig, max_seq_len, target,
        )
        # Rank-0 builds the cache; other ranks WAIT for it. The barrier must NOT
        # depend on dist.is_initialized() — under `torch.distributed.run` the
        # process group is not yet initialized at data-load time (HF Trainer
        # inits it later), so a dist.barrier() here is silently skipped and
        # ranks 1..N race to open a not-yet-built json_path → FileNotFoundError
        # (observed 2026-05-21, smoke job 2328436). Instead we poll the json
        # sentinel, which `_build_memmap_cache` writes LAST and atomically
        # (os.replace) — so its existence guarantees bin+npz are complete too.
        if local_rank == 0:
            if not json_path.exists():
                tok = Tokenizer.from_file(tok_path)
                pad_id = tok.token_to_id("<pad>")
                eos_id = tok.token_to_id("</s>")
                if pad_id is None or eos_id is None:
                    raise RuntimeError(
                        f"Tokenizer {tok_path} missing <pad>/</s>; refusing to "
                        f"silently substitute (re-train tokenizer with SPECIAL_TOKENS)."
                    )
                logger.info(f"[{lang}] rank-0 building memmap cache at {bin_path}")
                _build_memmap_cache(
                    manifest_path=manifest_path, lang=lang,
                    tok_path=tok_path, pad_id=pad_id, eos_id=eos_id,
                    max_seq_len=max_seq_len, target_tokens=target,
                    bin_path=bin_path, np_path=np_path, json_path=json_path,
                    max_docs_per_lang=None,
                )
            else:
                logger.info(f"[{lang}] rank-0 reusing existing cache at {bin_path}")
        else:
            # Non-zero ranks: wait (file-poll) for rank-0's atomically-written
            # json sentinel. Fail loud on timeout rather than hang forever if
            # rank-0's build crashed.
            deadline = time.time() + CACHE_BUILD_TIMEOUT_S
            waited = 0
            while not json_path.exists():
                if time.time() > deadline:
                    raise TimeoutError(
                        f"rank {local_rank}: waited {CACHE_BUILD_TIMEOUT_S}s for "
                        f"rank-0 to build {json_path} (lang={lang}); giving up. "
                        f"Likely rank-0's cache build failed — check rank-0 logs."
                    )
                time.sleep(5)
                waited += 5
                if waited % 300 == 0:
                    logger.info(f"[{lang}] rank {local_rank} still waiting for "
                                f"cache ({waited}s elapsed)")
        # If the process group IS already initialized (it isn't at this point
        # under torch.distributed.run, but be defensive), a barrier is a cheap
        # extra guard and keeps ranks aligned for the next language.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        dataset = MemmappedEngcodeDataset(bin_path, np_path, json_path)
        per_lang_stats[lang] = {
            "docs_consumed": len(dataset),
            "tokens_posttrunc": dataset.total_tokens,
            "tokens_pretrunc": dataset.pretrunc_tokens,
            "target_tokens": target,
            "tokenizer": tok_path,
        }
        val_size = max(1, int(len(dataset) * val_fraction))
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(42)
        tr, vl = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=gen,
        )
        all_train.append(tr); all_val.append(vl)
        logger.info(
            f"  {lang} dataset: {len(dataset)} samples, posttrunc="
            f"{dataset.total_tokens:,} (target {target:,})"
        )
    return ConcatDataset(all_train), ConcatDataset(all_val), per_lang_stats


def _write_run_metadata(model_dir: Path, args, vocab_size: int,
                          tok_paths: dict, manifest_path: Path,
                          per_lang_stats: dict, num_params: int,
                          status: str = "started",
                          extra: dict | None = None) -> Path:
    """Write run_metadata.json compatible with `_assert_checkpoint_provenance`.

    Schema bridge:
      model.name                       = absolute model_dir (the synthetic base)
      tokenizer.las_mode               = LAS_MODE_BY_TYPE[tokenizer_type]
      tokenizer.las_tokenizer_path     = abs tokenizer_dir
      tokenizer.model_tokenizer_path   = model.name  (eos-bug guard passes)
    """
    model_name = str(model_dir.resolve())
    meta = {
        "schema_version": 3,                # from-scratch v2 schema
        "status": status,
        "script": "scripts/train_fromscratch_v2.py",
        "track": args.track,
        "set": args.set_name,
        "tokenizer_type": args.tokenizer_type,
        "vocab_size": vocab_size,
        "target_params": TARGET_PARAMS_BY_SET[args.set_name],
        "realized_params": num_params,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "manifest_path": str(manifest_path),
        "tokenizer_dir": str(args.tokenizer_dir),
        "tokenizer_paths": tok_paths,
        "per_lang_stats": per_lang_stats,
        # ── provenance-guard-compat fields (REQUIRED by downstream_eval) ──
        "model": {
            "name": model_name,
            "n_params_target": TARGET_PARAMS_BY_SET[args.set_name],
            "n_params_realized": num_params,
        },
        "tokenizer": {
            "las_mode":             LAS_MODE_BY_TYPE[args.tokenizer_type],
            "las_tokenizer_path":   str(Path(args.tokenizer_dir).resolve()),
            "model_tokenizer_path": model_name,    # == model.name ⇒ eos-bug guard OK
        },
    }
    if extra:
        meta.update(extra)
    meta_path = model_dir / "run_metadata.json"
    tmp = meta_path.with_suffix(".json.tmp")
    json.dump(meta, open(tmp, "w"), indent=2)
    os.replace(tmp, meta_path)
    logger.info(f"Wrote {meta_path} (status={status})")
    return meta_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=list(TRACKS), required=True)
    p.add_argument("--set", dest="set_name", choices=["A", "A5B", "B"], required=True)
    p.add_argument("--tokenizer_type", required=True,
                   choices=["bpe", "base", "pathpiece", "langspec"])
    p.add_argument("--tokenizer_dir", type=str, required=True,
                   help="Output dir of train_tokenizers_fromscratch (vocab-specific).")
    p.add_argument("--manifest_path", type=str, required=True,
                   help="Output of build_fromscratch_manifest (track/set-specific).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="defaults to $SCRATCH_DIR/unimix_fromscratch_morph if unset")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0,
                   help="If >0, cap optimizer steps at this value (overrides "
                        "epochs); used for smoke tests. HF TrainingArguments "
                        "semantics: -1 means no cap (use epochs).")
    p.add_argument("--max_tokens_per_lang", type=int, default=0,
                   help="If >0, cap EACH language's LM token budget at this "
                        "value (min with the natural-distribution budget). "
                        "Used for fast smoke/login-node tests so the memmap "
                        "cache build stays small. 0 = no cap (production).")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=2000,
                   help="Checkpoint every N optimizer steps (step-based save, "
                        "NOT epoch). Required for SET-B requeue resumability; "
                        "save_total_limit=2 caps disk.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_policy", choices=["never", "auto", "require"],
                   required=True,
                   help="never=SET-A (fits one window); auto=SET-B requeue chain.")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate provenance schema + configs without training.")
    args = p.parse_args()
    if args.output_dir is None:
        scratch_dir = os.environ.get("SCRATCH_DIR")
        if not scratch_dir:
            raise RuntimeError(
                "Neither --output_dir nor SCRATCH_DIR is set; pass "
                "--output_dir or set SCRATCH_DIR to your cluster scratch root."
            )
        args.output_dir = str(Path(scratch_dir) / "unimix_fromscratch_morph")

    # ── Monkey-patch engcode's module globals for THIS track ─────────────────
    # `_stream_texts_for_lang` and the cache builder read engcode.TEXT_COL[lang];
    # any iteration over engcode.LANGUAGES must match our track.
    langs = track_langs(args.track)
    ec.LANGUAGES = langs
    ec.TEXT_COL = {l: text_col(args.track, l) for l in langs}
    # M1 (review-fix 2026-05-19): defensively delete OTHER engcode module
    # globals so any future engcode refactor that reads them from the reused
    # helpers (_build_memmap_cache, MemmappedEngcodeDataset, etc.) crashes
    # loudly instead of silently using engcode's eng+python defaults.
    for _stale in ("ENG_LM_DATA_DIR", "CODE_LM_DATA_DIR",
                    "ENG_TOK_TRAINING_DIR", "CODE_TOK_TRAINING_DIR",
                    "DEFAULT_TOKENS_PER_LANG", "CORPUS_LINES_PER_LANG"):
        if hasattr(ec, _stale):
            delattr(ec, _stale)
    logger.info(f"engcode patched: LANGUAGES={ec.LANGUAGES}, TEXT_COL={ec.TEXT_COL}")

    tokenizer_dir = Path(args.tokenizer_dir)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"--tokenizer_dir does not exist: {tokenizer_dir}")
    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"--manifest_path does not exist: {manifest_path}")

    # ── Resolve tokenizer file(s) for this tokenizer_type ───────────────────
    tok_paths = _build_tok_paths(tokenizer_dir, args.tokenizer_type, langs)
    main_tok_path = tok_paths.get("train") or tok_paths.get("base")
    main_tok = Tokenizer.from_file(main_tok_path)
    vocab_size = main_tok.get_vocab_size()
    # C3 (review-fix 2026-05-19): explicit fail-loud on missing <pad>; the
    # earlier `or 0` clause matched the Qwen3 eos-bug failure class (silent
    # collision with id 0). See memory/project_qwen3_eos_model_tokenizer_bug.
    pad_id = main_tok.token_to_id("<pad>")
    if pad_id is None:
        raise RuntimeError(
            f"Tokenizer {main_tok_path} is missing the '<pad>' special token. "
            f"Required for loss masking / batch padding; cannot silently fall "
            f"back to id=0. Re-fit the tokenizer with the SPECIAL_TOKENS set."
        )
    expected_vocab = SET_VOCAB[args.set_name]
    if vocab_size != expected_vocab:
        raise AssertionError(
            f"Tokenizer vocab {vocab_size} != SET-{args.set_name} expected "
            f"{expected_vocab}. Re-fit tokenizers at the correct vocab size."
        )

    # ── Build model config (125M for A/A5B @ 50k; 500M for B @ 100k) ────────
    if args.set_name in ("A", "A5B"):
        model_config, training_adj = get_vocab_aware_config(vocab_size, pad_id)
    else:
        model_config, training_adj = get_vocab_aware_config_500m(vocab_size, pad_id)
    target_p = TARGET_PARAMS_BY_SET[args.set_name]
    # C1 (review-fix 2026-05-21): force adamw_torch. get_vocab_aware_config sets
    # optim="adamw_8bit" for vocab<=65000 (i.e. SET-A @ 50k), but bitsandbytes
    # is NOT installed in the venv — Trainer.create_optimizer() would raise
    # AFTER the multi-hour memmap cache build, crashing all 8 SET-A runs. We do
    # NOT pip-install bnb (documented venv RECORD-corruption history + unverified
    # ARM/GH200 support); adamw_torch is fine for 125M/500M at these batch sizes.
    # Override here (driver-local) rather than in the shared model_trainer.py,
    # which other experiments depend on.
    if training_adj.get("optim") != "adamw_torch":
        logger.warning(
            f"Overriding optim {training_adj.get('optim')!r} -> 'adamw_torch' "
            f"(bitsandbytes not installed; avoids post-cache-build crash)."
        )
        training_adj["optim"] = "adamw_torch"
    model_config.n_positions = args.max_seq_len
    model_config.n_ctx = args.max_seq_len
    # M3 (review-fix 2026-05-19): build the model ONCE and assert on the
    # instantiated object (the original assert_param_count instantiated a
    # second model internally, costing ~2 GB on the 500M arch).
    model = GPT2LMHeadModel(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    _lo, _hi = int(target_p * 0.9), int(target_p * 1.1)
    if not (_lo <= num_params <= _hi):
        raise AssertionError(
            f"Param-count check failed: realized={num_params:,} not in "
            f"[{_lo:,}, {_hi:,}] (target {target_p:,} ±10%)."
        )
    logger.info(
        f"Model {num_params/1e6:.1f}M ({args.set_name}), vocab={vocab_size} "
        f"(target {target_p/1e6:.0f}M, ∈ [{_lo/1e6:.0f}M, {_hi/1e6:.0f}M])"
    )

    # ── Output dir + naming (settings encoded in path) ──────────────────────
    out_root = Path(args.output_dir)
    run_name = (f"fs_{args.track}_{args.set_name}_v{vocab_size//1000}k_"
                f"{args.tokenizer_type}_seed{args.seed}")
    model_dir = out_root / run_name
    model_dir.mkdir(parents=True, exist_ok=True)  # idempotent on all ranks

    # RANK-0-ONLY writes. Under torch.distributed.run all 4 ranks execute main();
    # the HF tokenizer save and the run_metadata write target SHARED single-file
    # paths, so doing them on every rank races (e.g. _write_run_metadata's
    # os.replace(tmp→json): one rank consumes the tmp, the others crash with
    # FileNotFoundError — observed smoke 2344885, 2026-05-23). Only rank 0 writes
    # these shared artifacts. Non-zero ranks don't need them before training
    # (the model is in-memory; the Trainer coordinates via the process group;
    # the tokenizer + metadata are eval-time artifacts).
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if _local_rank == 0:
        # ── C2 (review-fix 2026-05-19): save a HF PreTrainedTokenizerFast
        # wrapping `main_tok` to model_dir so the downstream evaluator's
        # AutoTokenizer-from-checkpoint path succeeds. Done BEFORE training so
        # even a crashed/partial run still has a tokenizer for partial-eval.
        from transformers import PreTrainedTokenizerFast as _PTTF
        _hf_tok = _PTTF(
            tokenizer_object=main_tok,
            unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
            model_max_length=args.max_seq_len,
        )
        _hf_tok.save_pretrained(str(model_dir))
        logger.info(
            f"Saved HF-wrapped main tokenizer ({args.tokenizer_type}) to {model_dir} "
            f"(vocab={vocab_size}, pad_id={pad_id})"
        )

    # ── resume_policy gate ─────────────────────────────────────────────────
    existing_ckpts = sorted(model_dir.glob("checkpoint-*"))
    if existing_ckpts and args.resume_policy == "never":
        raise RuntimeError(
            f"resume_policy=never but {len(existing_ckpts)} checkpoint dir(s) "
            f"exist in {model_dir}. Refusing to silently continue a prior run."
        )
    if not existing_ckpts and args.resume_policy == "require":
        raise RuntimeError(
            f"resume_policy=require but NO checkpoint dirs in {model_dir}."
        )

    # ── Per-lang token budgets (NON-uniform; from natural-distribution) ────
    budgets = lang_token_budget(args.track, args.set_name)
    if args.max_tokens_per_lang > 0:
        capped = {l: min(b, args.max_tokens_per_lang) for l, b in budgets.items()}
        logger.warning(
            f"SMOKE/TEST: capping each lang's budget at "
            f"{args.max_tokens_per_lang:,} tokens (was {budgets} -> {capped}). "
            f"NOT a production run — memmap cache keyed by this target."
        )
        budgets = capped
    logger.info(f"Per-lang LM token budgets ({args.track}/SET-{args.set_name}): {budgets}")

    if args.dry_run:
        # Schema-bridge dry-run: write minimal provenance metadata and exit.
        meta_path = _write_run_metadata(
            model_dir, args, vocab_size, tok_paths, manifest_path,
            per_lang_stats={l: {"target_tokens": budgets[l]} for l in langs},
            num_params=num_params, status="dry-run",
        )
        # Verify downstream_eval._assert_checkpoint_provenance accepts it.
        from las_experiment.downstream_eval import _assert_checkpoint_provenance
        _assert_checkpoint_provenance(
            ckpt_path=str(model_dir),
            base_pretrained=str(model_dir.resolve()),
            expected_las_mode=LAS_MODE_BY_TYPE[args.tokenizer_type],
            expected_las_dir=str(Path(args.tokenizer_dir).resolve())
                if args.tokenizer_type != "bpe" else None,
        )
        logger.info("Dry run: schema bridge verified; provenance guard accepts.")
        return

    # ── Load data + write pre-train metadata ────────────────────────────────
    train_ds, val_ds, per_lang_stats = _load_training_data_perlang(
        tok_paths, manifest_path, langs, budgets,
        max_seq_len=args.max_seq_len,
    )
    # RANK-0-ONLY: single shared run_metadata.json (os.replace would race across
    # ranks). Subsequent status updates (_update_status) are already is_main-gated.
    if _local_rank == 0:
        _write_run_metadata(
            model_dir, args, vocab_size, tok_paths, manifest_path,
            per_lang_stats, num_params, status="started",
        )

    # ── Training (mirrors engcode main()) ───────────────────────────────────
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        try:
            import wandb
            _wandb_entity = os.environ.get("WANDB_ENTITY")
            if not _wandb_entity:
                raise RuntimeError(
                    "WANDB_ENTITY is not set. Set it to your WandB entity "
                    "(user or team) before enabling wandb logging."
                )
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "langmap-lm-exps"),
                entity=_wandb_entity,
                name=f"fs_{run_name}",
                tags=["fromscratch", f"track:{args.track}", f"set:{args.set_name}",
                       f"tok:{args.tokenizer_type}", f"vocab:{vocab_size//1000}k",
                       f"params:{num_params//1_000_000}M", "section:§FS"],
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")

    # HF semantics: max_steps=-1 disables the cap (use epochs). Translate our
    # arg's convention (0 = no cap) to HF's (-1 = no cap).
    hf_max_steps = args.max_steps if args.max_steps > 0 else -1
    if hf_max_steps > 0:
        logger.info(
            f"SMOKE mode: capping training at max_steps={hf_max_steps} "
            f"(overrides epochs={args.epochs})"
        )
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=args.epochs,
        max_steps=hf_max_steps,
        per_device_train_batch_size=training_adj["per_device_train_batch_size"],
        gradient_accumulation_steps=training_adj["gradient_accumulation_steps"],
        learning_rate=training_adj["learning_rate"],
        lr_scheduler_type=training_adj["lr_scheduler_type"],
        warmup_steps=training_adj["warmup_steps"],
        weight_decay=training_adj["weight_decay"],
        max_grad_norm=training_adj["max_grad_norm"],
        adam_epsilon=training_adj["adam_epsilon"],
        optim=training_adj["optim"],
        bf16=True,
        eval_strategy="steps", eval_steps=args.eval_steps,
        # Step-based checkpointing (NOT epoch): SET-B (500M @ 10B tokens) will
        # not finish one epoch in a 12h window, so epoch-save would write
        # NOTHING on TIMEOUT and the resume_policy=auto requeue would restart
        # from scratch forever. Periodic step saves make the requeue chain
        # genuinely resumable; save_total_limit caps disk (latest 2 kept).
        # save_steps=2000 is a deliberate constant (≈ a few % of SET-A's run,
        # frequent enough that a TIMEOUT loses <1 window's tail of progress).
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, logging_steps=10,
        report_to=["wandb"], remove_unused_columns=False, seed=args.seed,
    )
    perf_config = {
        "model_num_params": num_params, "seq_len": args.max_seq_len,
        "peak_tflops_per_gpu": float(os.environ.get("PEAK_TFLOPS_PER_GPU", "989")),
    }
    trainer = BpBTrainer(
        model=model, train_dataset=train_ds, eval_dataset=val_ds,
        args=training_args, perf_config=perf_config,
    )

    def _update_status(status: str, extra: dict | None = None):
        try:
            meta_path = model_dir / "run_metadata.json"
            meta = json.load(open(meta_path))
            meta["status"] = status
            if extra: meta.update(extra)
            tmp = meta_path.with_suffix(".json.tmp")
            json.dump(meta, open(tmp, "w"), indent=2)
            os.replace(tmp, meta_path)
        except Exception as e:
            logger.exception(f"Failed to update status to {status}: {e}")

    try:
        resume = bool(existing_ckpts) and args.resume_policy == "auto"
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(model_dir))
        if is_main:
            _update_status("completed")
        logger.info(f"=== Training complete. Model in {model_dir} ===")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if is_main:
            _update_status("crashed", {"crash": {
                "exception_type": type(exc).__name__,
                "message": str(exc)[:2000],
            }})
        raise


if __name__ == "__main__":
    main()
