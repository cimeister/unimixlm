#!/usr/bin/env python3
"""Per-language *validation* BpB for from-scratch (F13) models — clean eval-mode.

The training-time logged ``eval_bpb`` (e.g. 1.0620 for morph-A BPE) is an
AGGREGATE over ``ConcatDataset(all_val)`` — all languages mixed in one eval
loop. This script breaks that number down PER LANGUAGE by reconstructing the
exact held-out val split that training used and evaluating each language's
slice on its own, IN EVAL MODE (dropout disabled).

IMPORTANT — why this does NOT reproduce the logged eval_bpb:
  The training-time logged eval_bpb is ~0.015 HIGHER than a clean eval-mode
  recompute. Diagnosed 2026-05-23: morph-A BPE recomputes to AGG≈1.047 in eval
  mode (dropout off) vs 1.075 in train mode (dropout on); the logged value 1.062
  sits ~halfway → the 4-GPU training-time eval did NOT fully disable dropout
  (a DDP/accelerate eval-mode propagation quirk). The contamination is uniform
  across all tokenizers (identical train/eval code + dropout=0.1), so the
  BPE-vs-LangMAP DELTAS in the headline are preserved, but the clean absolute
  validation BpB is this eval-mode recompute. The logged value is reported
  alongside FOR REFERENCE ONLY (not an oracle).

Split reconstruction (byte-identical to
``train_fromscratch_v2.py::_load_training_data_perlang``):
  per-lang memmap cache located via the same
  ``_engcode_cache_paths(cache_root, lang, tok_sig, max_seq_len, target)`` key
  (``tok_sig = f"{tok_stem}_{sha1(tok_bytes)[:12]}"``; tokenizer + target read
  from the model's run_metadata.json); then
  ``val_size = max(1, int(len*VAL_FRACTION))``,
  ``random_split(ds, [train_size, val_size],
                 generator=torch.Generator().manual_seed(SPLIT_SEED))``.

BpB convention matches BpBTrainer: BpB = Σ(loss·n_loss_tokens) / (Σ bytes · ln2),
fp32 weights under bf16 autocast (matches training's AMP forward; the bf16-vs-fp32
weight-load difference is <1e-3 here and does not affect the comparison).

Single-GPU (login-node) usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_fromscratch_val_perlang.py \
      --model_dir $SCRATCH_DIR/fs_morph_A_v50k_bpe_seed42 \
      --output_json eval_results/fromscratch_val_perlang/morph_A_bpe.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, default_data_collator

from tokenizers import Tokenizer

import scripts.train_fromscratch_engcode as ec
from scripts.train_fromscratch_engcode import (
    MemmappedEngcodeDataset,
    _engcode_cache_paths,
    _build_memmap_cache,
)
from scripts.fromscratch_common import track_langs, text_col

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# These MUST match train_fromscratch_v2.py::_load_training_data_perlang.
VAL_FRACTION = 0.005
SPLIT_SEED = 42


def _logged_eval_bpb(model_dir: Path) -> float | None:
    """Final logged aggregate eval_bpb from the latest checkpoint's
    trainer_state.json — reported for reference (dropout-contaminated, NOT an
    oracle; see module docstring)."""
    cks = sorted(glob.glob(str(model_dir / "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    if not cks:
        return None
    ts = Path(cks[-1]) / "trainer_state.json"
    if not ts.exists():
        return None
    d = json.load(open(ts))
    evs = [h["eval_bpb"] for h in d.get("log_history", []) if "eval_bpb" in h]
    return evs[-1] if evs else None


@torch.no_grad()
def _val_bpb_for_lang(model, device, ds_val, batch_size, max_seq_len):
    """Eval-mode (dropout off) BpB over a val Subset. Processes every example
    (no sampler padding/dropping). Returns (bpb, nll, n_bytes, n_examples)."""
    dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                    collate_fn=default_data_collator, num_workers=0)
    tot_nll = 0.0
    tot_bytes = 0
    for batch in dl:
        num_bytes = batch.pop("num_bytes")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**batch)
        # Same convention as BpBTrainer.prediction_step: loss is the batch-mean
        # over non-(-100) shifted positions; ×n recovers the batch NLL sum.
        n_loss_tokens = (batch["labels"][:, 1:] != -100).sum().item()
        tot_nll += out.loss.item() * n_loss_tokens
        tot_bytes += int(num_bytes.sum().item())
    bpb = tot_nll / (tot_bytes * math.log(2)) if tot_bytes > 0 else float("nan")
    return bpb, tot_nll, tot_bytes, len(ds_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="From-scratch model dir (model.safetensors + run_metadata.json).")
    ap.add_argument("--weights_dir", default=None,
                    help="DIAGNOSTIC: load weights from here instead of --model_dir "
                         "(e.g. a specific checkpoint-N). run_metadata still from --model_dir.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--langs", default=None,
                    help="Comma-separated subset of langs to eval (default: all in "
                         "run_metadata per_lang_stats). Use to skip langs whose cache "
                         "is unavailable (e.g. eng for the code track).")
    ap.add_argument("--build_missing", action="store_true",
                    help="If a per-lang memmap cache is absent, REBUILD it from the "
                         "manifest via the exact train-time machinery (_build_memmap_"
                         "cache). This is a deterministic reconstruction (byte-identical "
                         "to training), NOT a data substitution. Without this flag a "
                         "missing cache fails loud.")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--cache_root", default=None,
                    help="defaults to $ENGCODE_CACHE_ROOT if unset")
    args = ap.parse_args()
    if args.cache_root is None:
        args.cache_root = os.environ.get("ENGCODE_CACHE_ROOT")
        if not args.cache_root:
            raise RuntimeError(
                "Neither --cache_root nor ENGCODE_CACHE_ROOT is set; pass "
                "--cache_root or set ENGCODE_CACHE_ROOT to the scratch "
                "directory for the engcode memmap cache."
            )

    model_dir = Path(args.model_dir)
    meta_path = model_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {model_dir}")
    meta = json.load(open(meta_path))
    per_lang = meta.get("per_lang_stats")
    if not per_lang:
        raise RuntimeError(f"run_metadata.json has no per_lang_stats: {meta_path}")
    cache_root = Path(args.cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Memmap cache root not found: {cache_root}")

    # Mirror train_fromscratch_v2's monkey-patch: _build_memmap_cache /
    # _stream_texts_for_lang read engcode.TEXT_COL[lang]. The engcode default only
    # covers (eng, python); StarCoder javascript/java use the 'content' column and
    # aren't in the default → KeyError when --build_missing rebuilds them. Patch
    # TEXT_COL/LANGUAGES for this model's track so the rebuild reads the right column.
    track = meta.get("track")
    if track:
        _tlangs = track_langs(track)
        ec.LANGUAGES = _tlangs
        ec.TEXT_COL = {l: text_col(track, l) for l in _tlangs}

    logged = _logged_eval_bpb(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_src = args.weights_dir if args.weights_dir else str(model_dir)
    model = AutoModelForCausalLM.from_pretrained(weights_src, torch_dtype=torch.float32)
    model.to(device).eval()  # dropout OFF — the correct validation regime

    results = {
        "model_dir": str(model_dir), "weights_src": weights_src,
        "track": meta.get("track"), "set": meta.get("set"),
        "tokenizer_type": meta.get("tokenizer_type"),
        "vocab_size": meta.get("vocab_size"),
        "val_fraction": VAL_FRACTION, "split_seed": SPLIT_SEED,
        "max_seq_len": args.max_seq_len, "eval_mode": "eval (dropout off)",
        "logged_eval_bpb_reference": logged,
        "per_lang": {},
    }

    want_langs = ([l.strip() for l in args.langs.split(",") if l.strip()]
                  if args.langs else list(per_lang.keys()))
    missing = [l for l in want_langs if l not in per_lang]
    if missing:
        raise ValueError(f"--langs {missing} not in run_metadata per_lang_stats "
                         f"({list(per_lang.keys())})")
    manifest_path = Path(meta["manifest_path"])

    tot_nll = 0.0
    tot_bytes = 0
    for lang in want_langs:
        st = per_lang[lang]
        tok_path = st["tokenizer"]
        target = st["target_tokens"]
        h = hashlib.sha1(Path(tok_path).read_bytes()).hexdigest()[:12]
        tok_sig = f"{Path(tok_path).stem}_{h}"
        bin_p, np_p, json_p = _engcode_cache_paths(
            cache_root, lang, tok_sig, args.max_seq_len, target)
        if not json_p.exists():
            if not args.build_missing:
                raise FileNotFoundError(
                    f"[{lang}] memmap cache missing: {json_p}. This is the EXACT "
                    f"train-time cache; refusing to substitute (would answer a "
                    f"different question). Pass --build_missing to reconstruct it "
                    f"from {manifest_path} (deterministic, byte-identical). "
                    f"tok={tok_path} target={target}")
            tok = Tokenizer.from_file(tok_path)
            pad_id = tok.token_to_id("<pad>")
            eos_id = tok.token_to_id("</s>")
            if pad_id is None or eos_id is None:
                raise RuntimeError(
                    f"[{lang}] tokenizer {tok_path} missing <pad>/</s>; cannot rebuild.")
            logger.info(f"[{lang}] REBUILDING memmap cache (target={target:,}) at {bin_p}")
            _build_memmap_cache(
                manifest_path=manifest_path, lang=lang, tok_path=tok_path,
                pad_id=pad_id, eos_id=eos_id, max_seq_len=args.max_seq_len,
                target_tokens=target, bin_path=bin_p, np_path=np_p, json_path=json_p,
                max_docs_per_lang=None)
        ds = MemmappedEngcodeDataset(bin_p, np_p, json_p)
        val_size = max(1, int(len(ds) * VAL_FRACTION))
        train_size = len(ds) - val_size
        gen = torch.Generator().manual_seed(SPLIT_SEED)
        _, vl = random_split(ds, [train_size, val_size], generator=gen)

        bpb, nll, nb, n = _val_bpb_for_lang(
            model, device, vl, args.eval_batch_size, args.max_seq_len)
        results["per_lang"][lang] = {
            "bpb": bpb, "n_val_examples": n, "nll": nll, "n_bytes": nb,
            "target_tokens": target, "tokenizer": tok_path,
        }
        tot_nll += nll
        tot_bytes += nb
        logger.info(f"[{lang}] val bpb={bpb:.4f}  n_val={n}  bytes={nb:,}")

    agg = tot_nll / (tot_bytes * math.log(2)) if tot_bytes > 0 else float("nan")
    results["aggregate_bpb"] = agg
    results["total_val_bytes"] = tot_bytes

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 64)
    ref = f"{logged:.6f}" if logged is not None else "n/a"
    logger.info(f"eval-mode aggregate val bpb = {agg:.6f}  "
                f"(training-logged ref = {ref}, dropout-contaminated)")
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
