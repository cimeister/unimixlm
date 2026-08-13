"""Deterministic held-out eval split construction for a local parquet corpus.

Produces two disjoint doc sets: a train set and a held-out eval set. The
train set is what the model fine-tunes on; the held-out set is never seen
during training and is used for the held-out BpB / FLORES+ eval scripts.

Usage sketch (in finetune.py, inside the local_dir branch):

    train_texts, heldout_texts, manifest = get_train_texts_and_heldout(
        source_dir=lang_dir,
        lang=lang,
        text_column=text_column,
        seed=cfg.seed,
        heldout_docs=cfg.get("heldout_docs_per_lang", 1000),
    )

The manifest is a list of `(file_basename, row_idx)` tuples identifying the
held-out docs; persist it alongside the checkpoint so future eval runs can
reconstruct the exact eval set without re-deriving.

Strategy (in priority order):
  1. `"last-file"`: if there are ≥2 parquet files AND the last one
     (lexicographically) contains at least `heldout_docs`, use the first
     `heldout_docs` rows of it. Simple, fast, and the held-out file is
     physically separate so it's easy to audit.
  2. `"hash-mod"`: deterministic hash `sha1((lang, file, row_idx, seed)) %
     bucket == 0` across all files, where `bucket = max(2, total_docs //
     heldout_docs)`. Used for single-file corpora or when the last file is
     too small.

No silent fallback beyond those two. If neither yields enough held-out docs,
raise (per repo policy).

### Caveats

- **Hash-mod is NOT monotonic in `heldout_docs`.**  The `bucket` divisor
  depends on the requested count, so asking for 500 vs 1000 held-out docs
  produces *different* (not nested) sets even at the same seed. If you want
  reproducibility across a fixed seed, also fix `heldout_docs` to a single
  canonical value per experiment.
- **Actual held-out count can be slightly below `heldout_docs`** because
  empty-string rows are filtered. The manifest always records the actual
  count as `heldout_docs`. A deficit >1% is logged as a warning.
- Empties at indices after `heldout_docs` in the last-file strategy do NOT
  land in train — the last file is fully reserved as the held-out pool.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _row_hash(lang: str, file_name: str, row_idx: int, seed: int) -> int:
    """Cheap, stable hash that doesn't depend on Python's PYTHONHASHSEED."""
    key = f"{lang}:{file_name}:{row_idx}:{seed}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:12], 16)


def _read_column(path: Path, column: str) -> list[str]:
    try:
        pf = pq.ParquetFile(str(path))
        tbl = pf.read(columns=[column])
        return [v.as_py() for v in tbl[column]]
    except Exception as exc:
        # Surface the exact bad file path. Without this the traceback only
        # shows "pyarrow.lib.ArrowInvalid: ..." and the user has to bisect
        # which file in a 60+-file dataset is corrupted.
        raise type(exc)(
            f"Failed reading parquet {path}: {exc}"
        ) from exc


def _safe_get_num_rows(path: Path) -> int:
    try:
        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception as exc:
        raise type(exc)(
            f"Failed reading metadata for {path}: {exc}"
        ) from exc


def _warn_if_deficit(actual: int, requested: int, lang: str, strategy: str) -> None:
    """Log a warning when the empty-row filter produced >1% fewer docs than
    asked for. Never raise — manifest records truth."""
    if actual >= requested:
        return
    deficit = requested - actual
    if deficit / max(requested, 1) > 0.01:
        logging.warning(
            f"[held-out:{lang}:{strategy}] requested {requested} docs but kept "
            f"{actual} after filtering empty rows (deficit {deficit}). "
            f"Manifest records the actual count."
        )


def get_train_texts_and_heldout(
    source_dir: Path,
    lang: str,
    text_column: str,
    seed: int = 42,
    heldout_docs: int = 1000,
    min_docs_for_last_file_strategy: int | None = None,
    lang_suffix: str = "_Latn",
    max_train_bytes: int | None = None,
    skip_corrupted_files: bool = False,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return (train_texts, heldout_texts, manifest).

    manifest keys:
      strategy: "last-file" | "hash-mod"
      lang: str
      source_dir: str
      seed: int
      heldout_docs: int
      files_all: [{name, rows}]
      files_train: [name, ...]   # actual files read (may be subset under streaming cap)
      heldout_ids: [{file, row_idx}]  # one per heldout doc

    ### Streaming cap (max_train_bytes)
    When set, the "last-file" strategy stops reading additional parquet files
    once accumulated train UTF-8 bytes exceed this cap. This prevents OOM on
    huge corpora (hun ~100 GB, tur ~143 GB) where the full text list would
    exhaust host memory before the downstream token-budget cap kicks in.
    The actual files read are recorded in `manifest["files_train"]` so eval
    can recover the exact provenance.

    ### Reproducibility
    - Fix `seed` AND `heldout_docs` to reproduce the held-out set.
    - Under `max_train_bytes`, also fix that value — different caps produce
      different train slices even at the same seed.
    - Hash-mod is not monotonic in `heldout_docs`: asking for 500 vs 1000 at
      the same seed gives different (not nested) sets because `bucket` changes.
    """
    source_dir = Path(source_dir)
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {source_dir}")

    # Metadata pass. When `skip_corrupted_files=True`, drop files that fail
    # to read with a logged WARNING (used for read-only shared corpora like
    # StarCoder where a few parquets are corrupted on disk). Default behavior
    # is to fail loudly so a silent skip never happens without explicit opt-in.
    per_file_counts: list[tuple[str, int]] = []
    skipped_files: list[str] = []
    for f in files:
        try:
            per_file_counts.append((f.name, _safe_get_num_rows(f)))
        except Exception as exc:
            if skip_corrupted_files:
                logging.warning(
                    f"[held-out:{lang}] SKIPPING corrupted parquet {f}: {exc}"
                )
                skipped_files.append(f.name)
                continue
            raise
    # Filter `files` to match the kept metadata so downstream reads don't
    # re-encounter the bad ones.
    if skipped_files:
        skipped_set = set(skipped_files)
        files = [f for f in files if f.name not in skipped_set]
    total = sum(n for _, n in per_file_counts)
    if total < heldout_docs + 10:   # need at least heldout_docs + a minimal train set
        raise RuntimeError(
            f"Corpus at {source_dir} has {total} docs — too few to reserve "
            f"{heldout_docs} held-out docs and leave meaningful training data."
        )

    if min_docs_for_last_file_strategy is None:
        min_docs_for_last_file_strategy = heldout_docs

    last_file_name, last_file_count = per_file_counts[-1]
    # Strategy 1: last file — only viable if there are at least 2 files, so we
    # actually have train files left after reserving the last one.
    if len(files) >= 2 and last_file_count >= min_docs_for_last_file_strategy:
        last_rows = _read_column(files[-1], text_column)
        heldout_texts = [t for t in last_rows[:heldout_docs] if t]
        train_texts: list[str] = []
        train_bytes_running = 0
        files_train_actual: list[str] = []
        stream_capped = False
        for fp in files[:-1]:
            rows = [t for t in _read_column(fp, text_column) if t]
            train_texts.extend(rows)
            files_train_actual.append(fp.name)
            if max_train_bytes is not None:
                train_bytes_running += sum(len(t.encode("utf-8")) for t in rows)
                if train_bytes_running >= max_train_bytes:
                    stream_capped = True
                    logging.info(
                        f"[held-out:{lang}] stream-cap hit: "
                        f"{train_bytes_running:,} bytes ≥ max_train_bytes="
                        f"{max_train_bytes:,}; "
                        f"read {len(files_train_actual)}/{len(files)-1} train files"
                    )
                    break
        manifest = {
            "strategy": "last-file",
            "lang": lang,
            "source_dir": str(source_dir),
            "text_column": text_column,
            "lang_suffix": lang_suffix,
            "seed": seed,
            "heldout_docs": len(heldout_texts),
            "files_all": [{"name": n, "rows": r} for n, r in per_file_counts],
            "files_train": files_train_actual,
            "file_heldout": last_file_name,
            "heldout_ids": [
                {"file": last_file_name, "row_idx": i} for i in range(len(heldout_texts))
            ],
            "stream_capped": stream_capped,
            "max_train_bytes": max_train_bytes,
        }
        logging.info(
            f"[held-out:{lang}] strategy=last-file  "
            f"heldout={len(heldout_texts)} docs from {last_file_name}, "
            f"train={len(train_texts)} docs from {len(files_train_actual)} file(s)"
            + (" (stream-capped)" if stream_capped else "")
        )
        _warn_if_deficit(len(heldout_texts), heldout_docs, lang, "last-file")
        return train_texts, heldout_texts, manifest

    # Strategy 2: hash-mod over ALL files. bucket_size ≈ total/heldout_docs.
    bucket = max(2, total // heldout_docs)
    heldout_texts = []
    train_texts = []
    heldout_ids = []
    for f in files:
        rows = _read_column(f, text_column)
        for i, text in enumerate(rows):
            if not text:
                continue
            if _row_hash(lang, f.name, i, seed) % bucket == 0 and len(heldout_texts) < heldout_docs:
                heldout_texts.append(text)
                heldout_ids.append({"file": f.name, "row_idx": i})
            else:
                train_texts.append(text)
    if len(heldout_texts) < heldout_docs:
        raise RuntimeError(
            f"hash-mod strategy produced only {len(heldout_texts)} held-out "
            f"docs for lang {lang} (target {heldout_docs}); bucket={bucket}. "
            f"Loosen the bucket or increase corpus size."
        )
    manifest = {
        "strategy": "hash-mod",
        "lang": lang,
        "source_dir": str(source_dir),
        "text_column": text_column,
        "lang_suffix": lang_suffix,
        "seed": seed,
        "heldout_docs": len(heldout_texts),
        "files_all": [{"name": n, "rows": r} for n, r in per_file_counts],
        "bucket": bucket,
        "heldout_ids": heldout_ids,
    }
    logging.info(
        f"[held-out:{lang}] strategy=hash-mod  "
        f"heldout={len(heldout_texts)} docs (bucket={bucket}, "
        f"seed={seed}, heldout_docs={heldout_docs}), "
        f"train={len(train_texts)} docs. "
        f"NOTE: hash-mod is not monotonic in heldout_docs — different values "
        f"produce different (not nested) sets even at the same seed."
    )
    _warn_if_deficit(len(heldout_texts), heldout_docs, lang, "hash-mod")
    return train_texts, heldout_texts, manifest


def write_heldout_manifest(
    manifests_by_lang: dict[str, dict[str, Any]], output_dir: Path
) -> Path:
    """Write the combined held-out manifest next to the checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "heldout_manifest.json"
    combined = {
        "schema_version": 1,
        "per_lang": manifests_by_lang,
    }
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    return path
