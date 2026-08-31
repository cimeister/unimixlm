#!/usr/bin/env python3
"""FLORES+ BpB evaluation — disjoint-by-construction held-out reference.

Evaluates a model (base OR PEFT checkpoint) on FLORES+ parallel sentences for
the requested languages. FLORES+ is Wikipedia/news-domain, disjoint from
FineWeb/StarCoder training corpora.

Data layout: `{flores_root}/{lang_code}.txt` — one sentence per line, 997
sentences per language, 210 languages available.

Usage:
  python scripts/eval_flores_bpb.py \\
      --base_model Qwen/Qwen3-0.6B \\
      --peft_path $SCRATCH_DIR/ft-las-langspec_fin_500M_ep5_lr3e-04_r64_uf \\
      --tokenizer_mode langspec \\
      --tokenizer_dir ./las_tokenizers/qwen3-0.6b_v151k_fineweb2sample_50klines_fin-hun-tur_softem \\
      --langs fin_Latn,hun_Latn,tur_Latn \\
      --output_json eval_results/flores_bpb_qwen3_lora+embed.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
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


def load_flores_texts(flores_root: Path, lang_code: str) -> list[str]:
    """Read FLORES+ plaintext for one language. `lang_code` must include the
    script suffix (e.g. `fin_Latn`, `eng_Latn`, `kor_Hang`). Empty lines are
    dropped. Raises loudly if the file is missing — no silent fallback."""
    path = flores_root / f"{lang_code}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES+ data for {lang_code!r} not found at {path}. "
            f"Check the language code (script-suffix format like 'fin_Latn') "
            f"and the --flores_root path."
        )
    with open(path, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]
    logging.info(f"Loaded {len(texts)} sentences from {path}")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True,
                    help="HF model name or path (used for native tokenizer + as PEFT base).")
    ap.add_argument("--peft_path", default=None,
                    help="Optional PEFT adapter path. If set, PEFT eval-time tie-weight workaround is applied.")
    ap.add_argument("--tokenizer_mode", required=True,
                    choices=["original", "base", "pathpiece", "langspec"])
    ap.add_argument("--tokenizer_dir", default=None,
                    help="Required unless tokenizer_mode=original.")
    ap.add_argument("--langs", required=True,
                    help="Comma-separated FLORES+ language codes with script suffix (e.g. 'fin_Latn,hun_Latn').")
    ap.add_argument("--flores_root", default=None,
                    help="FLORES+ parallel-data directory "
                         "(tokenizer_training/flores_parallel_data); "
                         "defaults to $FLORES_DATA_DIR if unset")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="Weight dtype at load. Default bfloat16 matches all other "
                         "project eval-time BpB. Use float32 to match training-logged "
                         "(fp32-weights+autocast) BpB exactly (avoids ~0.018 bf16 shift).")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    if args.flores_root is None:
        args.flores_root = os.environ.get("FLORES_DATA_DIR")
        if not args.flores_root:
            raise RuntimeError(
                "Neither --flores_root nor FLORES_DATA_DIR is set; pass "
                "--flores_root or set FLORES_DATA_DIR to the FLORES+ "
                "parallel-data directory."
            )

    flores_root = Path(args.flores_root)
    if not flores_root.is_dir():
        raise FileNotFoundError(f"FLORES+ root not found: {flores_root}")

    tok_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    model, device = load_eval_model(
        args.base_model, peft_path=args.peft_path,
        dtype=getattr(torch, args.dtype))
    model_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = model_tokenizer.pad_token_id or 0

    results = {
        "base_model": args.base_model,
        "peft_path": args.peft_path,
        "tokenizer_mode": args.tokenizer_mode,
        "tokenizer_dir": str(tok_dir) if tok_dir else None,
        "flores_root": str(flores_root),
        "per_lang": {},
    }

    # For `tokenizer_mode=langspec` we build a single LAS routing tokenizer
    # loaded with EVERY langspec_*.json in `tok_dir`, then route every text
    # through `best_language_encode_batch`. Using the per-language tokenizer
    # directly (as we did before 2026-05-14) bypasses routing — the file label
    # picks the language for free, which is an oracle eval, not LAS-at-inference.
    las_router = None
    if args.tokenizer_mode == "langspec":
        if tok_dir is None:
            raise ValueError(
                "tokenizer_mode=langspec requires --tokenizer_dir."
            )
        las_router = load_las_routing_tokenizer(
            tok_dir, vocab_size=len(model_tokenizer),
            model_tokenizer=model_tokenizer,
        )
        logging.info(
            f"LAS routing pool: {sorted(las_router.per_lang_tok.keys())}"
        )

    for lang_code in langs:
        texts = load_flores_texts(flores_root, lang_code)
        if las_router is not None:
            (token_ids_list, num_bytes_list, picked_langs,
             all_routed_langs, n_dropped) = tokenize_texts_las_routed(
                texts, las_router, args.max_seq_len,
            )
            # Routing diagnostic: count over ALL routed texts (not just the
            # subset that survived to BpB). Otherwise routing_top_lang_frac
            # is biased upward by however many texts got dropped (empty
            # segmentation / zero bytes).
            routing_dist: dict[str, int] = {}
            for pl in all_routed_langs:
                routing_dist[pl] = routing_dist.get(pl, 0) + 1
        else:
            # FLORES+ codes already include the script suffix (e.g. `fin_Latn`).
            # Pass `lang_suffix=""` so `pick_tokenizer` doesn't re-append it.
            tok = pick_tokenizer(
                args.tokenizer_mode, tok_dir, lang_code, lang_suffix="",
                model_tokenizer=model_tokenizer,
            )
            token_ids_list, num_bytes_list = tokenize_texts(texts, tok, args.max_seq_len)
            routing_dist = None

        res = compute_bpb(
            model, token_ids_list, num_bytes_list, device,
            batch_size=args.batch_size, max_seq_len=args.max_seq_len, pad_id=pad_id,
        )
        res["n_sentences"] = len(texts)
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
        results["per_lang"][lang_code] = res
        log_extra = ""
        if routing_dist is not None:
            log_extra = (
                f"  routed_top={res['routing_top_lang']}"
                f"@{res['routing_top_lang_frac']:.0%}"
                f"  dropped={n_dropped}"
            )
        logging.info(
            f"[{lang_code}] bpb={res['bpb']:.4f}  "
            f"tokens={res['n_tokens']:,}  bytes={res['n_bytes']:,}  "
            f"sents={res['n_sentences']}{log_extra}"
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
