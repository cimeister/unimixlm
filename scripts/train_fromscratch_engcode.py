#!/usr/bin/env python3
"""
Train GPT-2 125M from scratch on a 50/50 English+Python-code mixture.

Same architecture and training pipeline as train_fromscratch.py but with
two "languages": English (FineWeb) and Python (StarCoder).

Tokenizer types:
  bpe       — BPE trained on joint English+Python corpus
  pathpiece — Unigram base with all scores set to −1
  langspec  — Per-language (eng/python) probability re-estimation via EM

Usage:
  python scripts/train_fromscratch_engcode.py --tokenizer_type bpe --vocab_size 32000
  python scripts/train_fromscratch_engcode.py --tokenizer_type langspec --vocab_size 100000
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from torch.utils.data import ConcatDataset
from transformers import GPT2LMHeadModel, TrainingArguments

from code.model_trainer import get_vocab_aware_config
from las_experiment.finetune import BpBTrainer, TokenizedTextDataset
from las_experiment.tokenizer_training.training import (
    train_bpe_tokenizer,
    _generate_pathpiece_tokenizer,
    build_tekken_pretokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ── Data sources ──────────────────────────────────────────────────────────────
#
# Two tiers of data:
#   - TOKENIZER-TRAINING data: small sample, fine for fitting a tokenizer.
#   - MODEL-TRAINING data: full LM-scale corpus. The tokenizer-training sample
#     (2 GB across 80+ code languages, ~96M total tokens) is too small for
#     a 125M model.
#
# The tokenizer is still trained from the small sample (CORPUS_LINES_PER_LANG
# below); the model training iterates over the FULL corpora below.

def _require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Point it at {purpose}.")
    return value


# Tokenizer-training sample (small, representative)
ENG_TOK_TRAINING_DIR = _require_env(
    "ENG_TOK_TRAINING_DIR",
    "the FineWeb tokenizer-training sample directory "
    "(tokenizer_training/downstream_lm_training_data/fineweb_sample)",
)
CODE_TOK_TRAINING_DIR = _require_env(
    "CODE_TOK_TRAINING_DIR",
    "the StarCoder python tokenizer-training sample directory "
    "(tokenizer_training/downstream_lm_training_data/starcoder_sample/python)",
)

# Full LM-training corpora
ENG_LM_DATA_DIR = _require_env(
    "ENG_LM_DATA_DIR",
    "the FineWeb full LM-training directory "
    "(fineweb-1_3_0-quality_10-filterrobots/data)",
)
CODE_LM_DATA_DIR = _require_env(
    "CODE_LM_DATA_DIR",
    "the StarCoder python full LM-training directory "
    "(starcoderdata-threshold-0/python)",
)

LANGUAGES = ["eng", "python"]
TEXT_COL = {"eng": "text", "python": "content"}

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]

# ── Tokenizer-training corpus lines per language ─────────────────────────────
CORPUS_LINES_PER_LANG = 50_000

# ── Default model-training token budget (Chinchilla for 125M: 20× params) ───
DEFAULT_TOKENS_PER_LANG = 1_250_000_000   # 1.25B / lang → 2.5B total


def _assert_tokenizers_consistent(base_path: str, per_lang_paths: list[str],
                                   special_tokens: tuple[str, ...] = ("<unk>", "<s>", "</s>", "<pad>")) -> None:
    """Assert that per-language tokenizers share vocab_size and special-token
    ids with the base tokenizer. Fails loudly — silent mismatch misaligns the
    model's embedding rows with the tokenizer output."""
    base = Tokenizer.from_file(base_path)
    base_vocab = base.get_vocab_size()
    base_specials = {s: base.token_to_id(s) for s in special_tokens}

    for p in per_lang_paths:
        tok = Tokenizer.from_file(p)
        v = tok.get_vocab_size()
        if v != base_vocab:
            raise AssertionError(
                f"Vocab-size mismatch: base={base_vocab} vs {p}={v}. "
                f"The model is sized from the base; per-lang tokenizers must match."
            )
        for s, bid in base_specials.items():
            tid = tok.token_to_id(s)
            if tid != bid:
                raise AssertionError(
                    f"Special-token id mismatch for {s!r}: base={bid} vs {p}={tid}"
                )
    logging.info(
        f"Tokenizer consistency OK: vocab={base_vocab}, "
        f"specials={base_specials}, {len(per_lang_paths)} per-lang files"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Corpus preparation (for tokenizer training)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_corpora(corpus_dir: Path) -> dict[str, str]:
    """Write per-language .txt files for tokenizer training.

    Returns {lang: corpus_file_path}.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_info: dict[str, str] = {}

    for lang in LANGUAGES:
        out_file = corpus_dir / f"{lang}.txt"
        if out_file.exists():
            logging.info(f"  Corpus for {lang} already exists: {out_file}")
            corpus_info[lang] = str(out_file)
            continue

        text_col = TEXT_COL[lang]
        # Tokenizer training uses the small sample — cheap + representative.
        if lang == "eng":
            ds = load_dataset("parquet", data_dir=ENG_TOK_TRAINING_DIR, split="train")
        else:
            ds = load_dataset("parquet", data_dir=CODE_TOK_TRAINING_DIR, split="train")

        lines: list[str] = []
        for ex in ds:
            t = ex.get(text_col, "").strip()
            if t:
                lines.append(t)
            if len(lines) >= CORPUS_LINES_PER_LANG:
                break

        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logging.info(f"  Wrote {len(lines)} lines for {lang} to {out_file}")
        corpus_info[lang] = str(out_file)

    return corpus_info


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenizer training
# ─────────────────────────────────────────────────────────────────────────────
def train_tokenizers(args, corpus_info: dict[str, str]) -> dict:
    tok_dir = Path(args.output_dir) / f"tokenizers_{args.vocab_size}"
    tok_dir.mkdir(parents=True, exist_ok=True)
    corpus_files = list(corpus_info.values())

    if args.tokenizer_type == "bpe":
        tok_path = tok_dir / "bpe_tokenizer.json"
        if not tok_path.exists():
            logging.info(f"Training BPE tokenizer (vocab={args.vocab_size})...")
            train_bpe_tokenizer(
                corpus_files, str(tok_path),
                vocab_size=args.vocab_size,
                special_tokens=SPECIAL_TOKENS,
            )
        return {"train": str(tok_path)}

    # For pathpiece/langspec, train the base Unigram first
    base_path = tok_dir / "unigram_base.json"
    if not base_path.exists():
        logging.info(f"Training base Unigram tokenizer (vocab={args.vocab_size})...")
        from code.standard_unigramlm import StandardUnigramLMTokenizer

        base_tok = StandardUnigramLMTokenizer(
            vocab_size=args.vocab_size,
            em_mode="soft",
            byte_level=True,
            unk_token="<unk>",
            pretokenizer=build_tekken_pretokenizer(),
            decoder_override=decoders.ByteLevel(),
        )
        base_tok.train(corpus_files, str(base_path), corpus_info=corpus_info)

    # PathPiece: all scores → −1
    pp_path = tok_dir / "pathpiece.json"
    if not pp_path.exists():
        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

    if args.tokenizer_type == "pathpiece":
        return {"train": str(pp_path)}

    # LangSpec: per-language EM re-estimation
    if args.tokenizer_type == "langspec":
        from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

        las_tok = LanguageSpecificUnigramLMTokenizer(
            reestimation_em_mode="soft",
            unk_token="<unk>",
        )
        langspec_format = str(tok_dir / "langspec_{lang_code}.json")
        las_tok.train(
            corpus_info=corpus_info,
            base_tokenizer_path=str(base_path),
            tokenizer_path_format=langspec_format,
            load_base_if_exists=True,
            load_tokenizers_if_exists=True,
        )
        return {
            "base": str(base_path),
            "langspec_format": langspec_format,
            "per_lang": {lang: str(tok_dir / f"langspec_{lang}.json")
                         for lang in corpus_info},
        }

    raise ValueError(f"Unknown tokenizer_type: {args.tokenizer_type}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model-training data: shared-seed manifest + token-budget streaming
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(manifest_path: Path, seed: int = 42,
                   max_files_per_lang: int = 20) -> None:
    """Enumerate parquet rows from the full LM corpora into a deterministic
    shuffled manifest. Produced once and reused across ALL 6 tokenizer variants
    to guarantee each variant sees the same document identity set.

    `max_files_per_lang` caps enumeration. FineWeb v1 is 135 TB (59 parquet
    files at ~2-5 GB each containing ~300K-1M docs); scanning every row of
    every file takes hours and the output is hundreds of millions of manifest
    entries. For a 1.25B token/lang budget at ~3.5 B/token we need ~360M
    bytes of text per lang, comfortably within 20 files. The cap is the FIRST
    N files lexicographically — the shuffle then seeds into those files only.

    Manifest format (JSONL, one line per doc):
      {"lang": "eng"|"python", "file": "<abs path>", "row_idx": int}
    """
    import hashlib
    import json as _json
    import random
    import pyarrow.parquet as pq

    if manifest_path.exists():
        logging.info(f"Manifest already exists: {manifest_path}")
        return

    all_entries = []
    for lang, root in [("eng", ENG_LM_DATA_DIR), ("python", CODE_LM_DATA_DIR)]:
        files = sorted(Path(root).glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files under {root} for lang={lang}. "
                f"Confirm the full LM corpus is mounted. No fallback."
            )
        if max_files_per_lang and len(files) > max_files_per_lang:
            logging.info(
                f"[manifest] {lang}: capping to first {max_files_per_lang} / "
                f"{len(files)} parquets (sorted lexicographically). "
                f"To include more, pass --max_files_per_lang or set to 0 to disable."
            )
            files = files[:max_files_per_lang]

        lang_entries = []
        for f in files:
            pf = pq.ParquetFile(str(f))
            n = pf.metadata.num_rows
            for row_idx in range(n):
                lang_entries.append({
                    "lang": lang,
                    "file": str(f),
                    "row_idx": row_idx,
                })
        # Deterministic shuffle via stable string-hash → int (Python's
        # `Random(str)` was not stable across versions pre-3.11).
        seed_bytes = hashlib.sha1(f"{seed}:{lang}".encode()).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        rng.shuffle(lang_entries)
        logging.info(
            f"[manifest] {lang}: {len(lang_entries):,} entries across "
            f"{len(files)} files"
        )
        all_entries.extend(lang_entries)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for e in all_entries:
            f.write(_json.dumps(e) + "\n")
    os.replace(tmp, manifest_path)
    logging.info(f"[manifest] wrote {len(all_entries):,} entries to {manifest_path}")


def _stream_texts_for_lang(manifest_path: Path, lang: str):
    """Yield texts for one language from the shared-seed manifest.

    Memory design:
      1. Scan manifest once to build `per_file[file] = [row_idx, ...]` in
         manifest order AND `file_order = [file1, file2, ...]` in
         first-appearance order.
      2. Iterate files in `file_order`. For each file: read the column,
         yield the rows listed in `per_file[file]` in order, then DROP the
         column. Only one file's column is in memory at a time.

    Consistency across variants: the manifest is identical across all 6
    tokenizer variants (same seed). `per_file` and `file_order` are both
    derived deterministically from the manifest, so every variant sees the
    same sequence of documents. Token-budget cutoffs may land at different
    positions (efficient tokenizers consume less per doc), but the doc
    identity at any position is invariant.

    Trade-off vs strict manifest-order streaming: within a single file,
    rows are emitted in their manifest order (preserved per-file); across
    files, we no longer interleave — we drain each file before opening the
    next. This is fine for LM training (no ordering requirement beyond
    determinism) and makes memory usage O(one-file's-column).
    """
    import json as _json
    from collections import defaultdict
    import pyarrow.parquet as pq

    col_name = TEXT_COL[lang]
    per_file: dict[str, list[int]] = defaultdict(list)
    file_order: list[str] = []
    seen: set[str] = set()
    with open(manifest_path) as f:
        for line in f:
            entry = _json.loads(line)
            if entry["lang"] != lang:
                continue
            fp = entry["file"]
            if fp not in seen:
                seen.add(fp)
                file_order.append(fp)
            per_file[fp].append(entry["row_idx"])

    for fp in file_order:
        pf = pq.ParquetFile(fp)
        tbl = pf.read(columns=[col_name])
        col = [v.as_py() for v in tbl[col_name]]
        for row_idx in per_file[fp]:
            text = col[row_idx]
            if text:
                yield text
        # col goes out of scope at the next loop iteration — one file in RAM at a time.


# ------------------- Memmap-backed tokenized dataset -----
# A Python list of per-example dicts on every rank does not fit in memory.
# At engcode scale (3.8M eng docs × max_seq_len=512 × 3 lists of Python
# ints ≈ 150 GB per rank) it would exceed the 120 GB node. Instead:
#   1. Rank 0 tokenizes once; writes input_ids to a scratch .bin memmap and
#      {lengths, num_bytes} to a small .npz; all metadata to a .json.
#   2. All ranks (including rank 0) torch.distributed.barrier(), then mmap
#      the .bin read-only → shared kernel page cache, ~1× data in RAM total.
#   3. attention_mask and labels are reconstructed on-the-fly in __getitem__
#      — trivial per-step cost, zero storage cost.


def _engcode_cache_paths(cache_root: Path, lang: str, tok_sig: str,
                         max_seq_len: int, target_tokens: int) -> tuple[Path, Path, Path]:
    """Deterministic cache paths for one (lang, tokenizer, budget) combo.

    `tok_sig` encodes the tokenizer identity so two runs with different
    tokenizers don't collide on disk. Seq-len and target are also in the name
    because changing either invalidates the cache."""
    stem = f"{lang}__{tok_sig}__seq{max_seq_len}__tgt{target_tokens}"
    bin_path = cache_root / f"{stem}.input_ids.bin"
    np_path  = cache_root / f"{stem}.lens_bytes.npz"
    json_path = cache_root / f"{stem}.meta.json"
    return bin_path, np_path, json_path


def _build_memmap_cache(manifest_path: Path, lang: str,
                        tok_path: str, pad_id: int, eos_id: int,
                        max_seq_len: int, target_tokens: int,
                        bin_path: Path, np_path: Path, json_path: Path,
                        max_docs_per_lang: int | None = None) -> dict:
    """Rank-0-only builder. Streams docs from the manifest, tokenizes, writes
    padded input_ids to `bin_path` as a row-major int32 matrix of shape
    (n_examples, max_seq_len). Sidecar `np_path` stores (lengths, num_bytes)
    int32 arrays (lengths[i] = non-pad prefix of row i, incl. EOS if appended).

    Fails loudly if budget cannot be hit (consistent with the project
    no-fallback policy). Caller is responsible for rank gating + barrier."""
    import numpy as np
    import pyarrow.parquet as pq  # noqa: F401 (used transitively by stream fn)

    tok = Tokenizer.from_file(tok_path)
    CHUNK = 1024
    running_post = 0
    running_pre = 0
    n_examples = 0
    lengths: list[int] = []
    bytes_list: list[int] = []

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    stopped = False
    buf: list[str] = []
    with open(bin_path, "wb") as f_out:
        def _flush(texts: list[str]) -> bool:
            nonlocal running_post, running_pre, n_examples
            if not texts:
                return False
            encs = tok.encode_batch(texts, add_special_tokens=False)
            for enc in encs:
                ids = enc.ids
                n_pre = len(ids)
                ids = ids[:max_seq_len]
                n_post = len(ids)
                # Decode for byte count BEFORE padding/EOS so bytes reflect
                # what the model is trained on, matching TokenizedTextDataset.
                nb = len(tok.decode(ids, skip_special_tokens=True).encode("utf-8"))
                if len(ids) < max_seq_len:
                    ids = ids + [eos_id]
                L = len(ids)
                if L < max_seq_len:
                    ids = ids + [pad_id] * (max_seq_len - L)
                arr = np.asarray(ids, dtype=np.int32)
                arr.tofile(f_out)
                lengths.append(L)
                bytes_list.append(nb)
                n_examples += 1
                running_pre += n_pre
                running_post += n_post
                if running_post >= target_tokens:
                    return True
                if max_docs_per_lang is not None and n_examples >= max_docs_per_lang:
                    return True
            return False

        stream = _stream_texts_for_lang(manifest_path, lang)
        for t in stream:
            buf.append(t)
            if len(buf) >= CHUNK:
                if _flush(buf):
                    stopped = True
                    buf = []
                    break
                buf = []
        if not stopped and buf:
            _flush(buf)
            buf = []

    if running_post < target_tokens:
        # Corpus exhausted; delete partial cache to prevent stale-use on re-run.
        try:
            bin_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            f"[{lang}] exhausted manifest at {running_post:,} tokens "
            f"(target {target_tokens:,}). Re-run manifest generation with a "
            f"larger corpus or lower target. (Partial cache at {bin_path} removed.)"
        )

    np.savez(np_path,
             lengths=np.asarray(lengths, dtype=np.int32),
             num_bytes=np.asarray(bytes_list, dtype=np.int32))
    meta = {
        "lang": lang,
        "tok_path": tok_path,
        "max_seq_len": max_seq_len,
        "target_tokens": target_tokens,
        "n_examples": n_examples,
        "total_tokens_posttrunc": running_post,
        "total_tokens_pretrunc": running_pre,
        "pad_id": pad_id,
        "eos_id": eos_id,
    }
    # Atomic write: tmp + os.replace. This file is the cache-valid sentinel —
    # a half-written JSON would cause concurrent readers to crash.
    tmp_json = json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_json, json_path)
    return meta


class MemmappedEngcodeDataset(torch.utils.data.Dataset):
    """Read-only torch Dataset wrapping the memmap cache written by
    `_build_memmap_cache`. Returns the same example schema as
    `TokenizedTextDataset` (input_ids / attention_mask / labels / num_bytes),
    reconstructing attention_mask and labels at __getitem__ time from the
    stored `lengths` array. Padded positions get attention_mask=0 and
    labels=-100 so BpBTrainer's loss + byte-count logic is unchanged."""

    def __init__(self, bin_path: Path, np_path: Path, json_path: Path):
        import numpy as np
        with open(json_path) as f:
            meta = json.load(f)
        self.n_examples = int(meta["n_examples"])
        self.max_seq_len = int(meta["max_seq_len"])
        self.pad_id = int(meta["pad_id"])
        self.total_tokens = int(meta["total_tokens_posttrunc"])
        self.pretrunc_tokens = int(meta["total_tokens_pretrunc"])
        # Shared page cache across ranks when this file lives on scratch.
        self._ids = np.memmap(
            str(bin_path), dtype=np.int32, mode="r",
            shape=(self.n_examples, self.max_seq_len),
        )
        sidecar = np.load(str(np_path))
        self._lengths = sidecar["lengths"]
        self._num_bytes = sidecar["num_bytes"]
        # Cheap sanity: lengths fit in max_seq_len
        if self._lengths.shape[0] != self.n_examples:
            raise RuntimeError(
                f"Cache shape mismatch: meta n_examples={self.n_examples} vs "
                f"lengths.shape={self._lengths.shape} in {np_path}"
            )

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        # Convert memmap slice to a plain list (HF collator-friendly).
        # This copies max_seq_len ints — fast, keeps the memmap detached from
        # any downstream mutation.
        ids = self._ids[idx].tolist()
        L = int(self._lengths[idx])
        nb = int(self._num_bytes[idx])
        # attention_mask: 1s in [0, L), 0s in [L, max_seq_len)
        attention_mask = [1] * L + [0] * (self.max_seq_len - L)
        # labels: ids[:L] followed by -100 for pad positions
        labels = list(ids[:L]) + [-100] * (self.max_seq_len - L)
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_bytes": nb,
        }


def load_training_data(tokenizer_paths, manifest_path: Path,
                       target_tokens_per_lang: int,
                       max_seq_len: int = 512, val_fraction: float = 0.005,
                       max_docs_per_lang: int | None = None,
                       cache_root: Path | None = None):
    """Stream texts from the shared-seed manifest and cap each language at
    `target_tokens_per_lang` post-truncation tokens (as seen by its tokenizer).

    Memory-safe: rank 0 builds a scratch memmap cache per (lang, tokenizer,
    budget); all ranks mmap the cache read-only. See `MemmappedEngcodeDataset`.

    Efficient tokenizers consume fewer documents within the same token budget
    — that IS the fair comparison: same token budget, same underlying docs
    (taken in manifest order), different segmentation.
    """
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
    is_distributed = dist.is_available() and dist.is_initialized()

    all_train = []
    all_val = []

    # Vocab-consistency check: when per-language tokenizers are used, the model
    # is sized from the base tokenizer but inference uses the per-language ones.
    if "per_lang" in tokenizer_paths and "base" in tokenizer_paths:
        _assert_tokenizers_consistent(
            tokenizer_paths["base"], list(tokenizer_paths["per_lang"].values())
        )

    per_lang_stats: dict[str, dict] = {}
    for lang in LANGUAGES:
        logging.info(
            f"[{lang}] loading from manifest, target={target_tokens_per_lang:,} tokens..."
        )

        # Pick the right tokenizer for this lang
        if "per_lang" in tokenizer_paths and lang in tokenizer_paths["per_lang"]:
            tok_path = tokenizer_paths["per_lang"][lang]
        elif "train" in tokenizer_paths:
            tok_path = tokenizer_paths["train"]
        else:
            tok_path = tokenizer_paths["base"]

        # tok_sig encodes tokenizer identity — hashed from the tokenizer file
        # content so two different tokenizer files at the same stem never
        # collide in the shared cache_root. Stem is kept for readability.
        import hashlib
        h = hashlib.sha1(Path(tok_path).read_bytes()).hexdigest()[:12]
        tok_sig = f"{Path(tok_path).stem}_{h}"

        bin_path, np_path, json_path = _engcode_cache_paths(
            cache_root, lang, tok_sig, max_seq_len, target_tokens_per_lang,
        )

        # Rank-0 builds the cache; others wait. json_path is written LAST and
        # used as the cache-valid sentinel: a mid-write crash leaves bin+npz
        # present without json, which correctly triggers a rebuild.
        cache_valid = json_path.exists()
        if local_rank == 0 and not cache_valid:
            # Read pad_id / eos_id on rank 0 before building. Hard-raise if
            # either required special token is missing — silent fallback to
            # id=0 would misalign loss masks.
            tok = Tokenizer.from_file(tok_path)
            pad_id = tok.token_to_id("<pad>")
            eos_id = tok.token_to_id("</s>")
            if pad_id is None:
                raise RuntimeError(
                    f"Tokenizer {tok_path} is missing the '<pad>' special "
                    f"token. Required for loss masking; cannot silently "
                    f"substitute. Re-train the tokenizer with SPECIAL_TOKENS."
                )
            if eos_id is None:
                raise RuntimeError(
                    f"Tokenizer {tok_path} is missing the '</s>' special "
                    f"token. Required for end-of-document marking; cannot "
                    f"silently substitute."
                )
            logging.info(f"[{lang}] rank 0 building memmap cache at {bin_path}")
            meta = _build_memmap_cache(
                manifest_path=manifest_path, lang=lang,
                tok_path=tok_path, pad_id=pad_id, eos_id=eos_id,
                max_seq_len=max_seq_len, target_tokens=target_tokens_per_lang,
                bin_path=bin_path, np_path=np_path, json_path=json_path,
                max_docs_per_lang=max_docs_per_lang,
            )
            logging.info(
                f"  {lang}: {meta['n_examples']:,} docs → "
                f"{meta['total_tokens_posttrunc']:,} post-trunc / "
                f"{meta['total_tokens_pretrunc']:,} pre-trunc tokens"
            )
        elif local_rank == 0:
            logging.info(f"[{lang}] reusing existing cache at {bin_path}")
        else:
            # Non-zero ranks WAIT for rank-0's atomically-written json sentinel.
            # Must NOT rely on dist.barrier(): under `torch.distributed.run` the
            # process group is not initialized at data-load time, so the barrier
            # below is skipped and ranks 1..N would race to open a not-yet-built
            # json_path → FileNotFoundError (mirrors the
            # train_fromscratch_v2.py approach). json is written LAST + atomically.
            import time as _time
            _timeout = int(os.environ.get("FROMSCRATCH_CACHE_TIMEOUT_S", str(8 * 3600)))
            _deadline = _time.time() + _timeout
            while not json_path.exists():
                if _time.time() > _deadline:
                    raise TimeoutError(
                        f"rank {local_rank}: waited {_timeout}s for rank-0 to "
                        f"build {json_path} (lang={lang}); giving up — likely "
                        f"rank-0's cache build failed (check rank-0 logs)."
                    )
                _time.sleep(5)

        # All ranks synchronize before reading the cache (defensive: a no-op
        # when the PG isn't initialized — the file-poll above is the real gate).
        if is_distributed:
            dist.barrier()

        dataset = MemmappedEngcodeDataset(bin_path, np_path, json_path)

        per_lang_stats[lang] = {
            "docs_consumed": len(dataset),
            "tokens_posttrunc": dataset.total_tokens,
            "tokens_pretrunc": dataset.pretrunc_tokens,
        }

        val_size = max(1, int(len(dataset) * val_fraction))
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=gen,
        )
        all_train.append(train_ds)
        all_val.append(val_ds)

        logging.info(
            f"  {lang} dataset: {len(dataset)} samples, post-trunc "
            f"{dataset.total_tokens:,} / pre-trunc {dataset.pretrunc_tokens:,}"
        )

    train_dataset = ConcatDataset(all_train)
    val_dataset = ConcatDataset(all_val)
    return train_dataset, val_dataset, per_lang_stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", type=str, required=True,
                        choices=["bpe", "pathpiece", "langspec"])
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="defaults to $SCRATCH_DIR/unimix_fromscratch_engcode if unset")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--per_device_batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    # New: token-budget + shared-seed manifest args.
    parser.add_argument("--tokens_per_lang", type=int, default=DEFAULT_TOKENS_PER_LANG,
                        help="Target post-truncation tokens per language. "
                             "Default: Chinchilla (~2.5B total for 125M model).")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Shared-seed doc manifest. Default: "
                             "<output_dir>/manifests/engcode_docs_seed42.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for manifest shuffling (shared across all variants).")
    parser.add_argument("--max_files_per_lang", type=int, default=20,
                        help="Cap parquet files enumerated per lang (speeds up "
                             "manifest generation on 135 TB FineWeb). 0 disables.")
    args = parser.parse_args()
    if args.output_dir is None:
        scratch_dir = os.environ.get("SCRATCH_DIR")
        if not scratch_dir:
            raise RuntimeError(
                "Neither --output_dir nor SCRATCH_DIR is set; pass "
                "--output_dir or set SCRATCH_DIR to your cluster scratch root."
            )
        args.output_dir = str(Path(scratch_dir) / "unimix_fromscratch_engcode")

    output_dir = Path(args.output_dir)
    # Per naming policy: encode settings in path.
    tokens_label = f"{args.tokens_per_lang // 1_000_000}M" if args.tokens_per_lang < 1_000_000_000 \
        else f"{args.tokens_per_lang / 1e9:.1f}B"
    run_name = (
        f"engcode_v{args.vocab_size // 1000}k_fw1+sc0_"
        f"{tokens_label}perlang_seed{args.seed}_{args.tokenizer_type}"
    )
    model_dir = output_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path) if args.manifest_path else \
        output_dir / "manifests" / f"engcode_docs_seed{args.seed}.jsonl"

    # Step 1: Prepare corpus files
    logging.info("=== Preparing tokenizer training corpora ===")
    tok_dir = output_dir / f"tokenizers_{args.vocab_size}"
    corpus_dir = tok_dir / "corpus"
    corpus_info = prepare_corpora(corpus_dir)

    # Step 2: Train tokenizers
    logging.info(f"=== Training tokenizers: {args.tokenizer_type} vocab={args.vocab_size} ===")
    tok_paths = train_tokenizers(args, corpus_info)

    # Get actual vocab size
    main_tok_path = tok_paths.get("train") or tok_paths.get("base")
    tok = Tokenizer.from_file(main_tok_path)
    actual_vocab_size = tok.get_vocab_size()
    pad_id = tok.token_to_id("<pad>") or 0
    logging.info(f"Tokenizer vocab size: {actual_vocab_size}")

    # Step 3: Build model
    logging.info("=== Building GPT-2 model ===")
    model_config, training_adjustments = get_vocab_aware_config(actual_vocab_size, pad_id)
    model_config.n_positions = args.max_seq_len
    model_config.n_ctx = args.max_seq_len
    model = GPT2LMHeadModel(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {num_params / 1e6:.1f}M params, vocab={actual_vocab_size}")

    # Step 4: Build / load the shared-seed doc manifest (before data load).
    logging.info(f"=== Shared-seed manifest: {manifest_path} ===")
    build_manifest(
        manifest_path,
        seed=args.seed,
        max_files_per_lang=(args.max_files_per_lang or 10**9),
    )

    # Step 5: Load data (token-budgeted streaming from manifest).
    logging.info(
        f"=== Loading training data (budget={args.tokens_per_lang:,} tokens/lang) ==="
    )
    train_dataset, val_dataset, per_lang_stats = load_training_data(
        tok_paths,
        manifest_path=manifest_path,
        target_tokens_per_lang=args.tokens_per_lang,
        max_seq_len=args.max_seq_len,
    )
    logging.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Write a minimal run_metadata.json for this from-scratch run too
    import json as _json
    try:
        meta = {
            "schema_version": 2,
            "status": "started",
            "script": "scripts/train_fromscratch_engcode.py",
            "tokenizer_type": args.tokenizer_type,
            "vocab_size": actual_vocab_size,
            "seed": args.seed,
            "tokens_per_lang_budget": args.tokens_per_lang,
            "epochs": args.epochs,
            "max_seq_len": args.max_seq_len,
            "manifest_path": str(manifest_path),
            "data_sources": {
                "eng_lm_data": ENG_LM_DATA_DIR,
                "python_lm_data": CODE_LM_DATA_DIR,
                "eng_tok_training": ENG_TOK_TRAINING_DIR,
                "python_tok_training": CODE_TOK_TRAINING_DIR,
            },
            "tokenizer_paths": tok_paths,
            "per_lang_stats": per_lang_stats,
            "model_dir": str(model_dir),
        }
        meta_path = model_dir / "run_metadata.json"
        tmp = meta_path.with_suffix(".json.tmp")
        with open(tmp, "w") as _f:
            _json.dump(meta, _f, indent=2)
        os.replace(tmp, meta_path)
        logging.info(f"Wrote pre-train {meta_path} (status=started)")
    except Exception as _e:
        logging.exception(f"Failed to write pre-train run_metadata.json: {_e}")

    # Step 5: Train
    logging.info("=== Training ===")
    import wandb
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        # Descriptive tag taxonomy.
        _wandb_entity = os.environ.get("WANDB_ENTITY")
        if not _wandb_entity:
            raise RuntimeError(
                "WANDB_ENTITY is not set. Set it to your WandB entity (user "
                "or team) before enabling wandb logging."
            )
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "langmap-lm-exps"),
            entity=_wandb_entity,
            name=f"fromscratch_{run_name}",
            tags=[
                "fromscratch",
                "engcode",
                "base:gpt2-125m",
                "lang:eng+python",
                f"tok:{args.tokenizer_type}",
                "adapt:from-scratch",
                f"epochs:ep{args.epochs}",
                f"vocab:{args.vocab_size // 1000}k",
                f"tokens_per_lang:{tokens_label}",
                "domain:engcode",
                "section:§14",
            ],
        )

    lr = training_adjustments.get("learning_rate", 3e-4)
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=training_adjustments.get("warmup_steps", 3000),
        weight_decay=training_adjustments.get("weight_decay", 0.01),
        max_grad_norm=training_adjustments.get("max_grad_norm", 1.0),
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"],
        remove_unused_columns=False,
        seed=42,
    )

    perf_config = {
        "model_num_params": num_params,
        "seq_len": args.max_seq_len,
        "peak_tflops_per_gpu": float(os.environ.get("PEAK_TFLOPS_PER_GPU", "989")),
    }

    trainer = BpBTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        perf_config=perf_config,
    )

    def _update_status(status: str, extra: dict | None = None):
        try:
            meta_path = model_dir / "run_metadata.json"
            if not meta_path.exists():
                return
            meta = _json.load(open(meta_path))
            meta["status"] = status
            if extra:
                meta.update(extra)
            tmp = meta_path.with_suffix(".json.tmp")
            with open(tmp, "w") as _f:
                _json.dump(meta, _f, indent=2)
            os.replace(tmp, meta_path)
            logging.info(f"Updated {meta_path} (status={status})")
        except Exception as _e:
            logging.exception(f"Failed to update run_metadata.json status: {_e}")

    try:
        trainer.train()
        trainer.save_model(str(model_dir))
        logging.info(f"=== Training complete. Model saved to {model_dir} ===")
        if is_main:
            _update_status("completed")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as _train_exc:
        if is_main:
            _update_status("crashed", {
                "crash": {
                    "exception_type": type(_train_exc).__name__,
                    "message": str(_train_exc)[:2000],
                },
            })
        raise


if __name__ == "__main__":
    main()
