#!/usr/bin/env python3
"""Build the A4 (F15) vocabulary-extension tokenizers for Qwen2.5-0.5B.

This script is the driver for
``las_experiment.vocab_extend.build_extended_tokenizer`` — it does not
implement the extension logic itself (that lives in vocab_extend.py, owned by
the same wave). It is scoped to the Qwen2.5-0.5B family only; Llama-3.2-1B A4
tokenizers are out of scope (A3/A4 are shown on Qwen
only).

For each of the 5 F15 languages (tur, fin, hun, arb, hin), this:
  1. Calls build_extended_tokenizer(model tokenizer.json, mono32k_<lang>.json,
     out=vocabext_<lang>.json, top_k=16000).
  2. Verifies the output: 5-line corpus roundtrip via ExtendedTokenizer,
     max token id < n_total_vocab, base-vocab ids unchanged, eos resolves to
     the model's eos id (151643 for Qwen2.5-0.5B).
  3. Extends las_tokenizers/f15/qwen2_5-0_5b/manifest.json with a
     "vocab_extend" section (sha256, n_appended, n_excluded_digit,
     decode_mismatch count), mirroring the "overlays"/"placebos"/"mono32k"
     sections prep_f15_tokenizers.py already writes there.

Do NOT commit any artifact under las_tokenizers/f15/. No HF Hub push.
"""

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from las_experiment.vocab_extend import (  # noqa: E402
    DEFAULT_TOP_K,
    ExtendedTokenizer,
    build_extended_tokenizer,
)

# ---------------------------------------------------------------------------
# Constants (named per project convention)
# ---------------------------------------------------------------------------
TOP_K = 16000  # A4: top-16k new per-language pieces
assert TOP_K == DEFAULT_TOP_K, (
    f"scripts/build_f15_vocabext.py TOP_K={TOP_K} != "
    f"las_experiment.vocab_extend.DEFAULT_TOP_K={DEFAULT_TOP_K}; the "
    f"two defaults have drifted apart; reconcile before running."
)
ROUNDTRIP_LINES = 5  # roundtrip 5 corpus lines
EXPECTED_EOS_ID = 151643  # Qwen2.5-0.5B <|endoftext|>, per manifest.json's existing overlays

FAMILY = "qwen2_5-0_5b"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LANG_CODES = {
    "tur": "tur_Latn",
    "fin": "fin_Latn",
    "hun": "hun_Latn",
    "arb": "arb_Arab",
    "hin": "hin_Deva",
}
LANGS = ["tur", "fin", "hun", "arb", "hin"]

FAMILY_DIR = REPO_ROOT / "las_tokenizers" / "f15" / FAMILY
MODEL_TOKENIZER_JSON = FAMILY_DIR / "model_tokenizer" / "tokenizer.json"
MANIFEST_PATH = FAMILY_DIR / "manifest.json"
_LAS_TOK_CORPORA_ROOT = os.environ.get("LAS_TOK_CORPORA_ROOT")
if not _LAS_TOK_CORPORA_ROOT:
    raise RuntimeError(
        "LAS_TOK_CORPORA_ROOT is not set. Point it at the scratch directory "
        "holding the LAS tokenizer-training corpora (parent of f15_corpora/)."
    )
CORPORA_DIR = Path(_LAS_TOK_CORPORA_ROOT) / "f15_corpora"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def first_n_nonempty_lines(path: Path, n: int) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
            if len(lines) >= n:
                break
    if len(lines) < n:
        raise RuntimeError(f"{path}: only {len(lines)} non-empty lines, need {n}")
    return lines


def verify_extended_tokenizer(lang: str, lang_code: str, out_path: Path,
                               manifest: dict, model_tok) -> dict:
    """Roundtrip + id-range + base-id + eos checks. Raises on any failure."""
    ext = ExtendedTokenizer.from_file(out_path)
    n_total = manifest["n_total_vocab"]

    # ---- eos resolves ----
    eos_id = ext.token_to_id(model_tok.eos_token)
    if eos_id != EXPECTED_EOS_ID:
        raise RuntimeError(
            f"{out_path}: eos_token {model_tok.eos_token!r} resolves to "
            f"{eos_id}, expected {EXPECTED_EOS_ID}"
        )

    # ---- base ids unchanged: probe a sample of base-vocab pieces ----
    with open(MODEL_TOKENIZER_JSON, encoding="utf-8") as f:
        base_json = json.load(f)
    base_vocab: dict[str, int] = base_json["model"]["vocab"]
    for piece, piece_id in list(base_vocab.items())[:1000]:
        if ext.token_to_id(piece) != piece_id:
            raise RuntimeError(
                f"{out_path}: base piece {piece!r} moved from id {piece_id} "
                f"to {ext.token_to_id(piece)}"
            )
    for a in base_json.get("added_tokens", []) or []:
        if ext.token_to_id(a["content"]) != int(a["id"]):
            raise RuntimeError(
                f"{out_path}: base added token {a['content']!r} moved from "
                f"id {a['id']} to {ext.token_to_id(a['content'])}"
            )

    # ---- roundtrip N corpus lines; max id < n_total_vocab ----
    corpus_path = CORPORA_DIR / f"{lang_code}.txt"
    lines = first_n_nonempty_lines(corpus_path, ROUNDTRIP_LINES)
    max_id_seen = -1
    roundtrip_exact = 0
    roundtrip_mismatches = []
    for line in lines:
        enc = ext.encode(line, add_special_tokens=False)
        ids = enc.ids
        if not ids:
            raise RuntimeError(f"{out_path}: corpus line encoded to zero tokens: {line!r}")
        local_max = max(ids)
        if local_max > max_id_seen:
            max_id_seen = local_max
        if local_max >= n_total:
            raise RuntimeError(
                f"{out_path}: encoded id {local_max} >= n_total_vocab {n_total}"
            )
        decoded = ext.decode(ids, skip_special_tokens=False)
        # The base tokenizer's normalizer is NFC (verified against
        # tokenizer.json), so `encode` normalizes the raw line before
        # tokenizing; compare the roundtrip against the NFC-normalized line,
        # not the raw one, or well-formed-but-non-NFC corpus text (combining
        # diacritics are common in Arabic/Devanagari) would read as a false
        # roundtrip failure that has nothing to do with the A4 extension.
        expected = unicodedata.normalize("NFC", line)
        if decoded == expected:
            roundtrip_exact += 1
        else:
            roundtrip_mismatches.append({"line": line[:80], "decoded": decoded[:80]})

    return {
        "eos_id": eos_id,
        "max_id_seen_in_roundtrip": max_id_seen,
        "n_total_vocab": n_total,
        "roundtrip_lines_checked": len(lines),
        "roundtrip_exact_matches": roundtrip_exact,
        "roundtrip_mismatches": roundtrip_mismatches,
        "base_ids_unchanged": True,
    }


def main() -> None:
    if not MODEL_TOKENIZER_JSON.is_file():
        raise FileNotFoundError(f"Missing model tokenizer.json: {MODEL_TOKENIZER_JSON}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing family manifest: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        family_manifest = json.load(f)
    if family_manifest.get("family") != FAMILY:
        raise ValueError(
            f"{MANIFEST_PATH}: family={family_manifest.get('family')!r}, expected {FAMILY!r}"
        )

    model_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token

    gc = git_commit()
    family_manifest.setdefault("vocab_extend", {})
    results = {}

    for lang in LANGS:
        lang_code = LANG_CODES[lang]
        lang_dir = FAMILY_DIR / lang
        mono_path = lang_dir / f"mono32k_{lang_code}.json"
        out_path = lang_dir / f"vocabext_{lang_code}.json"
        if not mono_path.is_file():
            raise FileNotFoundError(f"Missing monolingual BPE fit: {mono_path}")

        print(f"=== {lang} ({lang_code}) ===")
        manifest = build_extended_tokenizer(
            MODEL_TOKENIZER_JSON, mono_path, out_path, top_k=TOP_K
        )
        manifest["git_commit_build_script"] = gc

        v = verify_extended_tokenizer(lang, lang_code, out_path, manifest, model_tok)
        if v["roundtrip_mismatches"]:
            raise RuntimeError(
                f"{out_path}: {len(v['roundtrip_mismatches'])} roundtrip "
                f"mismatches out of {v['roundtrip_lines_checked']}: "
                f"{v['roundtrip_mismatches']}"
            )
        if v["max_id_seen_in_roundtrip"] >= v["n_total_vocab"]:
            raise RuntimeError(f"{out_path}: id-range check failed: {v}")
        if v["eos_id"] != EXPECTED_EOS_ID:
            raise RuntimeError(f"{out_path}: eos check failed: {v}")

        family_manifest["vocab_extend"][lang_code] = {
            "path": str(out_path),
            "sha256": manifest["sha256"]["extended_tokenizer_json"],
            "top_k_requested": manifest["top_k_requested"],
            "n_appended": manifest["n_appended"],
            "n_excluded_digit": manifest["n_excluded_digit"],
            "n_total_vocab": manifest["n_total_vocab"],
            "decode_mismatch": manifest["n_appended_native_bytelevel_decode_wrong"],
            "eos_id": v["eos_id"],
            "roundtrip_lines_checked": v["roundtrip_lines_checked"],
            "roundtrip_exact_matches": v["roundtrip_exact_matches"],
            "base_ids_unchanged": v["base_ids_unchanged"],
        }
        results[lang_code] = {
            "n_appended": manifest["n_appended"],
            "n_excluded_digit": manifest["n_excluded_digit"],
            "decode_mismatch": manifest["n_appended_native_bytelevel_decode_wrong"],
            "n_total_vocab": manifest["n_total_vocab"],
        }
        print(
            f"    n_appended={manifest['n_appended']} "
            f"n_excluded_digit={manifest['n_excluded_digit']} "
            f"decode_mismatch={manifest['n_appended_native_bytelevel_decode_wrong']} "
            f"n_total_vocab={manifest['n_total_vocab']} "
            f"roundtrip={v['roundtrip_exact_matches']}/{v['roundtrip_lines_checked']} "
            f"eos_id={v['eos_id']}"
        )

    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(family_manifest, f, indent=2)
    tmp.replace(MANIFEST_PATH)
    print(f"\nFamily manifest updated: {MANIFEST_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
