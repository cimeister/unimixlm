#!/usr/bin/env python3
"""Build a shared-seed, NO-REPEAT manifest for from-scratch LM training,
parameterized by (track, set).

The manifest is built ONCE per (track, set) and CONSUMED by all 4 tokenizer
variants in that set, so every variant trains on the identical document
set/order. The per-language doc budget is sized via
`target_bytes_per_lang(track, set, oversample)` (default 1.5×) so the most-
compressing tokenizer (typically BPE) still satisfies its per-lang token
target without repetition; less-compressing tokenizers hit their target
sooner and stop. Mirrors scripts/train_fromscratch_engcode.build_manifest
but generalized to N languages with per-lang token budgets from
`fromscratch_common.lang_token_budget`.

Outputs:
  - <out_dir>/fromscratch_<track>_<set>_seed<seed>.jsonl   (manifest)
  - <out_dir>/fromscratch_<track>_<set>_seed<seed>.meta.json (provenance)

Manifest line format (one JSON object per line):
  {"lang": <str>, "file": <abs path>, "row_idx": <int>}
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

from scripts.fromscratch_common import (
    TRACKS,
    SET_TOTAL_TOKENS,
    SET_VOCAB,
    OVERSAMPLE_FACTOR,
    BYTES_PER_TOKEN_BY_LANG,
    track_langs,
    lm_dir,
    text_col,
    lang_token_budget,
    select_files_for_token_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def _safe_num_rows(path: str) -> int:
    """Read row count from parquet metadata only (no data read)."""
    return pq.ParquetFile(path).metadata.num_rows


def build_manifest(track: str, set_name: str, out_dir: Path,
                    seed: int = 42,
                    oversample: float = OVERSAMPLE_FACTOR) -> dict:
    """Build a no-repeat shared-seed manifest for (track, set).
    Idempotent: if both the .jsonl and .meta.json exist and parse, returns
    the existing meta dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fromscratch_{track}_{set_name}_seed{seed}"
    manifest_path = out_dir / f"{stem}.jsonl"
    meta_path = out_dir / f"{stem}.meta.json"
    if manifest_path.exists() and meta_path.exists():
        try:
            existing = json.load(open(meta_path))
            logging.info(f"Manifest exists: {manifest_path}")
            return existing
        except Exception:
            logging.warning(f"meta unreadable; rebuilding: {meta_path}")

    token_budgets = lang_token_budget(track, set_name)
    # The token-aware selector measures real tokens/doc by tokenizing a text
    # sample with the BPE tokenizer for this (track, set). It must already exist
    # (Part A tok phase). Convention: <ROOT>/<track>_tokenizers_<vocab>/, where
    # ROOT = out_dir.parent (out_dir is ROOT/manifests).
    bpe_path = (out_dir.parent / f"{track}_tokenizers_{SET_VOCAB[set_name]}"
                / "bpe_tokenizer.json")
    if not bpe_path.is_file():
        raise FileNotFoundError(
            f"BPE tokenizer required for token-aware manifest sizing not found: "
            f"{bpe_path}. Run the tokenizer-fit phase (Part A) before the "
            f"manifest phase for {track}/SET-{set_name}."
        )
    per_lang_meta: dict[str, dict] = {}
    all_entries: list[dict] = []
    seen: set[tuple[str, str, int]] = set()

    for lang in track_langs(track):
        data_dir = lm_dir(track, set_name, lang)
        tgt_tok = token_budgets[lang]
        # TOKEN-AWARE selection (2026-05-23 fix): measures real tokens/doc with
        # the BPE tokenizer, then selects files by row count to cover
        # tgt_tok × oversample docs. Replaces the file-byte selector that
        # under-provisioned hun (sized by fat non-text parquet bytes + a wrong
        # bytes/token constant).
        files, sizing = select_files_for_token_target(
            data_dir, tgt_tok, text_col(track, lang),
            str(bpe_path), oversample,
        )
        logging.info(
            f"[{track}/{set_name}/{lang}] tok_target={tgt_tok:,} "
            f"tok/doc≈{sizing['tokens_per_doc_measured']} need_docs={sizing['need_docs']:,} "
            f"files={len(files)}/{sizing['n_files_available']} "
            f"selected_rows={sizing['selected_rows']:,}"
        )
        lang_entries: list[dict] = []
        for f in files:
            n = _safe_num_rows(f)
            for row_idx in range(n):
                key = (lang, f, row_idx)
                if key in seen:
                    raise RuntimeError(
                        f"Duplicate manifest key {key} — no-repeat invariant "
                        f"violated by file enumeration."
                    )
                seen.add(key)
                lang_entries.append({"lang": lang, "file": f, "row_idx": row_idx})

        # Deterministic per-lang shuffle (sha1 of "seed:lang", stable across
        # Python versions); identical across tokenizers / reruns.
        seed_bytes = hashlib.sha1(f"{seed}:{lang}".encode()).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        rng.shuffle(lang_entries)
        logging.info(
            f"[{track}/{set_name}/{lang}] {len(lang_entries):,} entries across "
            f"{len(files)} files (need_docs={sizing['need_docs']:,})"
        )
        per_lang_meta[lang] = {
            "data_dir": data_dir,
            "files": files,
            "num_entries": len(lang_entries),
            "target_tokens": token_budgets[lang],
            "bytes_per_token_est": BYTES_PER_TOKEN_BY_LANG[lang],
            "oversample_factor": oversample,
            "sizing": sizing,   # token-aware: avg_text_bytes/doc, tok/doc, need_docs
            "shuffle_seed_lang_str": f"{seed}:{lang}",
        }
        all_entries.extend(lang_entries)

    # Write manifest atomically.
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, manifest_path)
    meta = {
        "schema_version": 1,
        "track": track,
        "set": set_name,
        "seed": seed,
        "total_tokens_target": SET_TOTAL_TOKENS[set_name],
        "total_entries": len(all_entries),
        "per_lang": per_lang_meta,
        "manifest_path": str(manifest_path),
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    logging.info(
        f"Wrote {len(all_entries):,} entries -> {manifest_path}; "
        f"meta -> {meta_path}"
    )
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=list(TRACKS), required=True)
    p.add_argument("--set", dest="set_name", choices=["A", "A5B", "B"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oversample", type=float, default=OVERSAMPLE_FACTOR,
                   help="Manifest sized to oversample × per-tokenizer target.")
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="defaults to $SCRATCH_DIR/unimix_fromscratch_morph/manifests if unset",
    )
    args = p.parse_args()
    if args.out_dir is None:
        scratch_dir = os.environ.get("SCRATCH_DIR")
        if not scratch_dir:
            raise RuntimeError(
                "Neither --out_dir nor SCRATCH_DIR is set; pass --out_dir or "
                "set SCRATCH_DIR to your cluster scratch root."
            )
        args.out_dir = str(Path(scratch_dir) / "unimix_fromscratch_morph" / "manifests")
    build_manifest(args.track, args.set_name, Path(args.out_dir),
                    seed=args.seed, oversample=args.oversample)


if __name__ == "__main__":
    main()
