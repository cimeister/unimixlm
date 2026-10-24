#!/usr/bin/env python3
"""Held-out BpB evaluation — reconstructs the held-out set from the manifest
written by `las_experiment/held_out.py` and scores a checkpoint on it.

The manifest (`<checkpoint>/heldout_manifest.json`) lists the exact
`(lang, file, row_idx)` tuples reserved at training time, along with the
source directory. This script re-reads those rows from disk and computes BpB.

Contract:
  - Manifest comes from `las_experiment.held_out.write_heldout_manifest`.
  - Script does NOT re-derive the held-out set — it reads manifest exactly.
    This guarantees the eval set identity is frozen at training time, not
    re-computed (which could drift if `heldout_docs` or `seed` changes).

Usage:
  python scripts/eval_heldout_bpb.py \\
      --checkpoint $SCRATCH_DIR/ft-las-langspec_fin_500M_ep5_lr3e-04_r64_uf \\
      --base_model Qwen/Qwen3-0.6B \\
      --tokenizer_mode langspec \\
      --tokenizer_dir ./las_tokenizers/qwen3-0.6b_v151k_fineweb2sample_50klines_fin-hun-tur_softem \\
      --output_json eval_results/heldout_bpb_qwen3_fin.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from las_experiment.bpb_core import (
    compute_bpb,
    load_eval_model,
    load_las_routing_tokenizer,
    pick_tokenizer,
    tokenize_texts,
    tokenize_texts_las_routed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def load_heldout_manifest(checkpoint: Path) -> dict:
    path = checkpoint / "heldout_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No held-out manifest at {path}. The checkpoint was either "
            f"produced before Tier-1 (no manifest written) or under a config "
            f"that disabled manifest writing. No fallback — train with the "
            f"fixed pipeline and try again."
        )
    with open(path) as f:
        return json.load(f)


def load_rows_by_ids(source_dir: Path, file_name: str, row_idxs: list[int],
                     text_column: str) -> list[str]:
    """Read specific rows from a parquet file by index. The manifest uses
    integer row indices into the original on-disk parquet; we preserve those."""
    import pyarrow.parquet as pq
    path = source_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Source parquet not found: {path}")
    pf = pq.ParquetFile(str(path))
    tbl = pf.read(columns=[text_column])
    col = tbl[text_column]
    out = []
    for i in row_idxs:
        if i < 0 or i >= len(col):
            raise IndexError(
                f"Manifest references row {i} in {path} but the file has "
                f"{len(col)} rows. Manifest may be from a different version "
                f"of the corpus."
            )
        v = col[i].as_py()
        if v:
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the model output dir (contains heldout_manifest.json).")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--tokenizer_mode", required=True,
                    choices=["original", "base", "pathpiece", "langspec"])
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--lang_suffix", default=None,
                    help="Override lang_suffix. If omitted, read from the manifest "
                         "(preferred — guarantees the eval uses the same suffix as training).")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--base_only", action="store_true",
                    help="Skip loading PEFT/full-model weights and evaluate the un-fine-tuned "
                         "base_model. Manifest at --checkpoint is still used for held-out IDs.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    manifest = load_heldout_manifest(ckpt)
    tok_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None

    if args.base_only:
        # Baseline run: skip ckpt weights entirely; evaluate the un-fine-tuned
        # base_model on the manifest's held-out docs.
        logging.info(f"--base_only: loading {args.base_model} without PEFT/full-model weights")
        model, device = load_eval_model(args.base_model, peft_path=None)
        is_peft = False
    else:
        # In our P4d layout, `args.checkpoint` is the run-dir (with
        # heldout_manifest.json at the top level); the actual model weights
        # live one level deeper in a `checkpoint-N/` subdir. If we find such
        # subdirs, use the largest-step one as the weight path. Otherwise
        # treat `args.checkpoint` itself as the weight path (legacy / from-
        # scratch layout where everything is at one level).
        ckpt_for_load = ckpt
        ckpt_subdirs = sorted(
            ckpt.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if ckpt_subdirs and not (ckpt / "adapter_config.json").exists() and not (ckpt / "model.safetensors").exists():
            ckpt_for_load = ckpt_subdirs[-1]
            logging.info(f"Auto-descended to checkpoint subdir: {ckpt_for_load.name}")

        # `load_eval_model` auto-detects PEFT vs full-model save and applies
        # the tied-fix embed re-share where needed.
        is_peft = (ckpt_for_load / "adapter_config.json").exists()
        model, device = load_eval_model(args.base_model, peft_path=str(ckpt_for_load))
    model_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = model_tokenizer.pad_token_id or 0

    results = {
        "checkpoint": str(ckpt),
        "base_model": args.base_model,
        "is_peft": is_peft,
        "tokenizer_mode": args.tokenizer_mode,
        "tokenizer_dir": str(tok_dir) if tok_dir else None,
        "manifest_path": str(ckpt / "heldout_manifest.json"),
        "per_lang": {},
    }

    # For langspec mode, build a single LAS routing tokenizer loaded with EVERY
    # langspec_*.json in the dir; each text routes via best_language_encode.
    # Using the per-language tokenizer directly (oracle) bypasses routing.
    las_router = None
    if args.tokenizer_mode == "langspec":
        if tok_dir is None:
            raise ValueError("tokenizer_mode=langspec requires --tokenizer_dir.")
        las_router = load_las_routing_tokenizer(
            tok_dir, vocab_size=len(model_tokenizer),
            model_tokenizer=model_tokenizer,
        )
        logging.info(
            f"LAS routing pool: {sorted(las_router.per_lang_tok.keys())}"
        )

    for lang, per_lang_manifest in manifest["per_lang"].items():
        source_dir = Path(per_lang_manifest["source_dir"])
        text_column = per_lang_manifest["text_column"]
        # Read lang_suffix from the manifest (fallback to CLI / legacy default).
        lang_suffix = (
            args.lang_suffix
            if args.lang_suffix is not None
            else per_lang_manifest.get("lang_suffix", "_Latn")
        )

        # Reconstruct texts by grouping manifest ids by file.
        by_file: dict[str, list[int]] = {}
        for entry in per_lang_manifest["heldout_ids"]:
            by_file.setdefault(entry["file"], []).append(entry["row_idx"])

        texts: list[str] = []
        for file_name, idxs in by_file.items():
            texts.extend(load_rows_by_ids(source_dir, file_name, sorted(idxs), text_column))
        logging.info(
            f"[{lang}] reconstructed {len(texts)} held-out docs from "
            f"{len(by_file)} file(s) in {source_dir}"
        )

        if las_router is not None:
            (token_ids_list, num_bytes_list, picked_langs,
             all_routed_langs, n_dropped) = tokenize_texts_las_routed(
                texts, las_router, args.max_seq_len,
            )
            routing_dist: dict[str, int] = {}
            for pl in all_routed_langs:
                routing_dist[pl] = routing_dist.get(pl, 0) + 1
        else:
            tok = pick_tokenizer(
                args.tokenizer_mode, tok_dir, lang, lang_suffix=lang_suffix,
                model_tokenizer=model_tokenizer,
            )
            token_ids_list, num_bytes_list = tokenize_texts(texts, tok, args.max_seq_len)
            routing_dist = None

        res = compute_bpb(
            model, token_ids_list, num_bytes_list, device,
            batch_size=args.batch_size, max_seq_len=args.max_seq_len, pad_id=pad_id,
        )
        res["n_docs"] = len(texts)
        if routing_dist is not None:
            res["routing_dist"] = routing_dist
            res["n_dropped"] = n_dropped
            if routing_dist:
                top_lang = max(routing_dist, key=routing_dist.get)
                res["routing_top_lang"] = top_lang
                res["routing_top_lang_frac"] = (
                    routing_dist[top_lang] / sum(routing_dist.values())
                )
            else:
                res["routing_top_lang"] = None
                res["routing_top_lang_frac"] = 0.0
        results["per_lang"][lang] = res
        log_extra = ""
        if routing_dist is not None:
            log_extra = (
                f"  routed_top={res['routing_top_lang']}"
                f"@{res['routing_top_lang_frac']:.0%}"
                f"  dropped={n_dropped}"
            )
        logging.info(
            f"[{lang}] bpb={res['bpb']:.4f}  "
            f"tokens={res['n_tokens']:,}  bytes={res['n_bytes']:,}  "
            f"docs={res['n_docs']}{log_extra}"
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
