#!/usr/bin/env python3
"""Validate a saved LAS tokenizer directory end-to-end.

Runs the following tests:

- T1  pretokenized-form invariance. "".join(tok.encode(raw).tokens) must equal
      "".join(pretok(norm(raw))). If not, the Unigram scores were trained on a
      different substrate than inference uses.
- T3  JSON structural equality: every tokenizer in the dir agrees on
      pre_tokenizer / normalizer / decoder / added_tokens / post_processor.
- T4a score-key <-> token-string consistency. For each langspec_*.json every
      token returned by tok.encode(text).tokens must be a key in
      _extract_log_scores(tok), otherwise best_language_encode routing
      silently falls back to MIN_TOKEN_LOG_PROB on every piece.
- T4b routing picks target language for target-language text, >=80%.
- T5  Tokenizer.from_file <-> AutoTokenizer.from_pretrained agreement on
      model_tokenizer/ (so the two loaders used in finetune.py don't diverge).

Usage:
    python scripts/validate_tokenizer_dir.py <tokenizer_dir> [--sample-size N]

Non-zero exit on any failure. Every failure names the exact file + the smallest
reproducing input so you can investigate without rerunning the validator.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from tokenizers import Tokenizer

# Make `code.*` and `las_experiment.*` importable when this script is run
# directly from a checkout that isn't on PYTHONPATH.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.language_specific_unigramlm import (  # noqa: E402
    LanguageSpecificUnigramLMTokenizer,
    _extract_log_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("validate_tok")

_FLORES_DATA_DIR = os.environ.get("FLORES_DATA_DIR")
if not _FLORES_DATA_DIR:
    raise RuntimeError(
        "FLORES_DATA_DIR is not set. Point it at the FLORES+ parallel-data "
        "directory (tokenizer_training/flores_parallel_data)."
    )
FLORES_ROOT = Path(_FLORES_DATA_DIR)
DEFAULT_SAMPLE = 100
# Don't demand that every probe sample take the target language --- a
# "perfect" score isn't realistic for short sentences that overlap other
# languages. 80% is the chosen threshold.
MIN_PCT_ROUTING_CORRECT = 0.80


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_tok(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def _read_lines(path: Path, n: int, rng: random.Random) -> List[str]:
    """Return up to `n` non-empty lines sampled from `path` without loading
    the entire file when the file is large. Reads up to 4*n lines, keeps the
    non-empty subset, and truncates.
    """
    buf: List[str] = []
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for i, line in enumerate(f):
            if i >= 4 * n:
                break
            line = line.rstrip("\n")
            if line:
                buf.append(line)
    if len(buf) <= n:
        return buf
    rng.shuffle(buf)
    return buf[:n]


def _probe_source(dir_: Path, lang_key: str, n: int, rng: random.Random) -> Tuple[List[str], str]:
    """Return (lines, source_label) for lang_key, preferring FLORES+ when
    available and falling back to corpora/<lang_key>.txt inside the dir.

    `lang_key` is the stem used by langspec files (e.g. "fin_Latn", "python").
    """
    flores_path = FLORES_ROOT / f"{lang_key}.txt"
    if flores_path.exists():
        lines = _read_lines(flores_path, n, rng)
        if lines:
            return lines, f"flores:{flores_path.name}"
    corpus_path = dir_ / "corpora" / f"{lang_key}.txt"
    if corpus_path.exists():
        lines = _read_lines(corpus_path, n, rng)
        if lines:
            return lines, f"corpora:{corpus_path.name}"
    return [], "none"


def _langspec_key(path: Path) -> str:
    """"langspec_fin_Latn.json" -> "fin_Latn"; "langspec_python.json" -> "python"."""
    stem = path.stem
    assert stem.startswith("langspec_"), stem
    return stem[len("langspec_"):]


def _fail(failures: list, where: str, detail: str) -> None:
    """Accumulate failures rather than exit on first — gives full report."""
    log.error(f"[FAIL] {where}: {detail}")
    failures.append((where, detail))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def t1_pretok_form_invariance(
    tok_path: Path, probes: List[str], failures: list
) -> None:
    """Encoded tokens (concat) must equal pretokenized form (concat).

    Rationale: spm_train trained the Unigram over "".join(pretok(norm(raw))).
    HF's Unigram encode returns pieces that must tile the same string. If
    they don't, the Unigram model is scoring a different substrate at
    inference than it learned on.
    """
    tok = _load_tok(tok_path)
    pretok = tok.pre_tokenizer
    norm = tok.normalizer
    if pretok is None:
        _fail(failures, f"T1 {tok_path.name}",
              "pre_tokenizer is None — cannot verify form invariance")
        return
    for raw in probes:
        normed = norm.normalize_str(raw) if norm is not None else raw
        pretok_form = "".join(p for p, _ in pretok.pre_tokenize_str(normed))
        # add_special_tokens=False so post_processor's BOS/EOS don't pollute
        # the comparison — we're testing the *model* tokens against pretok.
        enc_form = "".join(tok.encode(raw, add_special_tokens=False).tokens)
        if enc_form != pretok_form:
            # Find first differing char for a short diff
            i = 0
            while i < min(len(pretok_form), len(enc_form)) and pretok_form[i] == enc_form[i]:
                i += 1
            _fail(
                failures,
                f"T1 {tok_path.name}",
                f"encode-form != pretok-form. "
                f"first diff at char {i}. "
                f"pretok[{max(0,i-5)}:{i+10}]={pretok_form[max(0,i-5):i+10]!r} "
                f"enc[{max(0,i-5)}:{i+10}]={enc_form[max(0,i-5):i+10]!r} "
                f"raw={raw[:80]!r}",
            )
            return
    log.info(f"[OK]  T1 {tok_path.name} ({len(probes)} probes)")


def _load_json_or_fail(path: Path, failures: list) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _fail(failures, f"read {path.name}", str(e))
        return {}


def t3_structural_equality(dir_: Path, failures: list) -> None:
    """All tokenizer.json files in the dir must agree on pre_tokenizer,
    normalizer, decoder, added_tokens, post_processor.

    Model section is exempt (Unigram vs BPE etc.). We also pull
    model_tokenizer/tokenizer.json as the ground-truth reference when
    present.
    """
    targets: List[Path] = sorted(dir_.glob("*.json"))
    ref_path = dir_ / "model_tokenizer" / "tokenizer.json"
    if ref_path.exists():
        targets = [ref_path] + targets

    if not targets:
        _fail(failures, "T3", f"no tokenizer.json files found in {dir_}")
        return

    ref = _load_json_or_fail(targets[0], failures)
    if not ref:
        return

    keys = ("pre_tokenizer", "normalizer", "decoder", "added_tokens", "post_processor")
    for path in targets[1:]:
        d = _load_json_or_fail(path, failures)
        if not d:
            continue
        for k in keys:
            if d.get(k) != ref.get(k):
                _fail(
                    failures,
                    f"T3 {path.name}",
                    f"field {k!r} differs from reference {targets[0].name}. "
                    f"ref={json.dumps(ref.get(k))[:200]!r} "
                    f"this={json.dumps(d.get(k))[:200]!r}",
                )
    log.info(f"[OK]  T3 structural equality across {len(targets)} files "
             f"(ref={targets[0].name})")


def t4a_score_key_consistency(
    tok_path: Path, probes: List[str], failures: list
) -> None:
    """Every token returned by tok.encode(text).tokens must be a key in
    _extract_log_scores(tok).

    If this fails the non-standard `best_language_encode` routing silently
    falls back to MIN_TOKEN_LOG_PROB on every piece and picks arbitrarily.
    """
    tok = _load_tok(tok_path)
    scores = _extract_log_scores(tok)
    scores_set = set(scores.keys())
    for raw in probes:
        # Mirror the fixed `best_language_encode` scoring path (uses
        # `id_to_token(id)` so UNK emissions canonicalize). Any ID whose
        # canonical token isn't a key in `scores` would still route at MIN.
        enc = tok.encode(raw)
        missing = []
        for tid in enc.ids:
            canonical = tok.id_to_token(tid)
            if canonical is None or canonical not in scores_set:
                missing.append((tid, canonical))
        if missing:
            _fail(
                failures,
                f"T4a {tok_path.name}",
                f"{len(missing)}/{len(enc.ids)} IDs map to tokens not in "
                f"_extract_log_scores keys. first 5 (id, canonical): "
                f"{missing[:5]} raw={raw[:80]!r}",
            )
            return
    log.info(f"[OK]  T4a {tok_path.name} ({len(probes)} probes)")


def t4b_routing_picks_target_lang(
    dir_: Path,
    langspec_paths: Dict[str, Path],
    probe_by_lang: Dict[str, List[str]],
    failures: list,
) -> None:
    """Load all langspec tokenizers together via the LAS wrapper and verify
    that target-language text routes to the target-language tokenizer
    >=MIN_PCT_ROUTING_CORRECT of the time.
    """
    if len(langspec_paths) < 2:
        log.info(f"[skip] T4b: only {len(langspec_paths)} langspec files "
                 f"(need >=2 for routing to be meaningful)")
        return
    base_path = dir_ / "base_tokenizer.json"
    if not base_path.exists():
        _fail(failures, "T4b", f"no base_tokenizer.json under {dir_}")
        return

    # vocab_size arg: the wrapper doesn't strictly require accuracy here, but
    # we pass the real size to avoid warnings.
    ref_tok = _load_tok(base_path)
    las = LanguageSpecificUnigramLMTokenizer(vocab_size=ref_tok.get_vocab_size())
    las.load(
        str(base_path),
        {lang: str(p) for lang, p in langspec_paths.items()},
    )

    for target_lang, texts in probe_by_lang.items():
        if not texts:
            continue
        picked, _encs = las.best_language_encode_batch(texts)
        pct = sum(1 for p in picked if p == target_lang) / len(picked)
        if pct < MIN_PCT_ROUTING_CORRECT:
            dist = Counter(picked)
            _fail(
                failures,
                f"T4b {dir_.name}/{target_lang}",
                f"routing picked {target_lang} for only {pct:.0%} of "
                f"{len(texts)} target-lang probes (threshold "
                f"{MIN_PCT_ROUTING_CORRECT:.0%}). distribution: {dict(dist)}",
            )
        else:
            log.info(f"[OK]  T4b {target_lang}: {pct:.0%} routed correctly "
                     f"({len(texts)} probes)")


def t5_loader_equivalence(
    dir_: Path, probes: List[str], failures: list
) -> None:
    """tokenizers.Tokenizer.from_file(model_tokenizer/tokenizer.json) must
    agree with transformers.AutoTokenizer.from_pretrained(model_tokenizer/)
    on real text — because finetune.py uses both loader stacks for different
    roles.
    """
    mt_dir = dir_ / "model_tokenizer"
    mt_json = mt_dir / "tokenizer.json"
    if not mt_json.exists():
        log.info(f"[skip] T5: no model_tokenizer/tokenizer.json under {dir_}")
        return
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        _fail(failures, "T5", f"cannot import transformers.AutoTokenizer: {e}")
        return

    tk = _load_tok(mt_json)
    at = AutoTokenizer.from_pretrained(str(mt_dir))
    for raw in probes:
        # Test both modes — finetune.py call sites exist in both, and the two
        # loader stacks must agree for any value of add_special_tokens.
        for mode in (False, True):
            a = tk.encode(raw, add_special_tokens=mode).ids
            b = at.encode(raw, add_special_tokens=mode)
            if a == b:
                continue
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            _fail(
                failures,
                f"T5 {mt_dir.name} (add_special_tokens={mode})",
                f"Tokenizer.from_file IDs != AutoTokenizer IDs at pos {i}. "
                f"raw={raw[:80]!r} "
                f"tk.encode[:{i+5}]={a[:i+5]} at.encode[:{i+5}]={b[:i+5]}",
            )
            return
    log.info(f"[OK]  T5 loader equivalence on model_tokenizer/ "
             f"({len(probes)} probes, both add_special_tokens modes)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(dir_: Path, sample_size: int, seed: int) -> int:
    rng = random.Random(seed)
    failures: list = []

    langspec_paths = {
        _langspec_key(p): p for p in sorted(dir_.glob("langspec_*.json"))
    }
    if not langspec_paths:
        log.error(f"no langspec_*.json in {dir_}")
        return 2
    log.info(f"validating {dir_} ({len(langspec_paths)} langspec files)")

    # Build per-lang probe lists once.
    probe_by_lang: Dict[str, List[str]] = {}
    for lang_key in langspec_paths.keys():
        probes, src = _probe_source(dir_, lang_key, sample_size, rng)
        probe_by_lang[lang_key] = probes
        if not probes:
            _fail(failures, f"probe {lang_key}",
                  f"no FLORES+ or corpora/ text available for {lang_key}")
        else:
            log.info(f"  probes[{lang_key}]: {len(probes)} lines from {src}")

    # Pool a mixed probe set for T1 / T3 / T5 (the non-per-lang tests).
    mixed_probes: List[str] = []
    for lang_key, ps in probe_by_lang.items():
        mixed_probes.extend(ps[: max(1, sample_size // max(len(probe_by_lang), 1))])
    if not mixed_probes:
        _fail(failures, "mixed probes", "no probe text available across any language")
        return 3

    # T1 on base + langspec files (not pathpiece — it has a custom uniform-
    # score model that can legitimately fail this invariant; TODO confirm).
    base_path = dir_ / "base_tokenizer.json"
    if base_path.exists():
        t1_pretok_form_invariance(base_path, mixed_probes, failures)
    for lang_key, p in langspec_paths.items():
        t1_pretok_form_invariance(p, probe_by_lang.get(lang_key, mixed_probes), failures)

    # T3 — structural JSON equality across every *.json in the dir.
    t3_structural_equality(dir_, failures)

    # T4a — score-key consistency per langspec file (post-hoc routing guard).
    for lang_key, p in langspec_paths.items():
        t4a_score_key_consistency(p, probe_by_lang.get(lang_key, mixed_probes), failures)

    # T4b — routing decision sanity.
    t4b_routing_picks_target_lang(dir_, langspec_paths, probe_by_lang, failures)

    # T5 — two-loader stack agreement.
    t5_loader_equivalence(dir_, mixed_probes, failures)

    if failures:
        log.error(f"\n=== {len(failures)} FAILURES in {dir_} ===")
        for where, detail in failures:
            log.error(f"  {where}: {detail}")
        return 1
    log.info(f"\n=== {dir_} PASSED all checks ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="Path to saved tokenizer dir "
                    "(contains base_tokenizer.json + langspec_*.json)")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE,
                    help=f"Probes per language (default {DEFAULT_SAMPLE})")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.dir.exists():
        log.error(f"not a directory: {args.dir}")
        return 2
    return run(args.dir, args.sample_size, args.seed)


if __name__ == "__main__":
    sys.exit(main())
