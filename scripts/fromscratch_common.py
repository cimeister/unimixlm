#!/usr/bin/env python3
"""Shared constants + proportions + corpus prep for the BPE-vs-LangMAP
from-scratch experiment.

Two TRACKS:
  - morph: eng + {fin, hun, tur, swh, kor, arb} (FineWeb-v1 eng + FineWeb2 morph).
  - code:  eng + python (engcode lineage: FineWeb-v1 eng + StarCoder python).

Two SETS per track (= 4 model-config × 4-tokenizer = 16 runs total):
  - A: 125M params, 50k vocab, Chinchilla(20×)≈2.5B  tokens.
  - B: 500M params, 100k vocab, Chinchilla(20×)≈10B  tokens.

Per-track distribution = NATURAL: English fixed at 0.5; the remaining 0.5
split among non-English langs proportional to per-language corpus token
counts. This module is the SINGLE source of truth for those proportions; the
tokenizer-fit corpus AND the LM manifest both consume them via
`lang_proportions(track)` so they cannot diverge.
"""
import logging
import os
from pathlib import Path

# ── Special tokens (shared across all tracks/sets) ───────────────────────────
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


def _require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Point it at {purpose}.")
    return value


# ── English sources (shared by both tracks) ──────────────────────────────────
ENG_TOK_TRAINING_DIR = _require_env(
    "ENG_TOK_TRAINING_DIR",
    "the FineWeb tokenizer-training sample directory "
    "(tokenizer_training/downstream_lm_training_data/fineweb_sample)",
)
# Point at the `output/` (kept docs) subtree, NOT
# the parent `data/` — `data/` also contains a `removed/` subtree (robots-
# filtered / low-quality docs). select_files_for_target uses rglob("*.parquet")
# over this dir; at current SET-B eng targets the lexicographic prefix stays in
# output/, but pointing at the parent is a latent data-integrity landmine (a
# larger target would silently pull in filtered-OUT docs). Scope it explicitly.
ENG_LM_FULL_DIR = _require_env(
    "ENG_LM_FULL_DIR",
    "the FineWeb full LM-training data/output directory "
    "(fineweb-1_3_0-quality_10-filterrobots/data/output)",
)

# ── Morph (FineWeb2 quality_33 filterrobots, FULL — sample dir too small) ────
FW2_MORPH_ROOT = _require_env(
    "FW2_MORPH_ROOT",
    "the FineWeb2 full morph-languages data/output directory "
    "(fineweb-2_0_1-quality_33-filterrobots/data/output)",
)
MORPH_SCRIPT = {
    "fin": "_Latn", "hun": "_Latn", "tur": "_Latn", "swh": "_Latn",
    "kor": "_Hang", "arb": "_Arab",
}
# Crude token-count estimates from `du -sb` on the FULL FW2 dirs. Bytes/token ≈ 4 for Latin, ≈ 2 for Hangul/Arabic — purely for
# computing PROPORTIONS; never used as an absolute budget (the manifest
# in Part B measures real token counts during build).
MORPH_TOK_EST = {  # millions of tokens, approximate
    "fin": 4_405, "hun": 26_586, "tur": 38_305, "swh": 104,
    "kor": 14_854, "arb": 64_167,
}

# ── Code (StarCoder per-lang — engcode lineage extended to py/js/java) ───────
CODE_TOK_SAMPLE_ROOT = _require_env(
    "CODE_TOK_SAMPLE_ROOT",
    "the StarCoder tokenizer-training sample directory "
    "(tokenizer_training/downstream_lm_training_data/starcoder_sample)",
)
CODE_LM_FULL_ROOT = _require_env(
    "CODE_LM_FULL_ROOT",
    "the StarCoder full LM-training directory (starcoderdata-threshold-0)",
)
CODE_LANGS = ["python", "javascript", "java"]
# Token estimates from on-disk sizes ÷ ~3.5 B/token (millions of tokens):
#   python 22G→6.3B, javascript 24G→6.9B, java 29G→8.3B.
CODE_TOK_EST = {"python": 6_286, "javascript": 6_857, "java": 8_286}

# ── Track configuration ──────────────────────────────────────────────────────
TRACKS = {
    "morph": {
        "non_eng": ["fin", "hun", "tur", "swh", "kor", "arb"],
        "tok_est": MORPH_TOK_EST,
        "tok_fit_dir": {"eng": ENG_TOK_TRAINING_DIR,
                          **{l: f"{FW2_MORPH_ROOT}/{l}{s}"
                             for l, s in MORPH_SCRIPT.items()}},
        "lm_dir_A": {"eng": ENG_TOK_TRAINING_DIR,  # SET-A eng = 21 GB sample
                      **{l: f"{FW2_MORPH_ROOT}/{l}{s}"
                         for l, s in MORPH_SCRIPT.items()}},
        "lm_dir_B": {"eng": ENG_LM_FULL_DIR,         # SET-B eng = full FW1
                      **{l: f"{FW2_MORPH_ROOT}/{l}{s}"
                         for l, s in MORPH_SCRIPT.items()}},
        "text_col": {l: "text" for l in ["eng", "fin", "hun", "tur",
                                            "swh", "kor", "arb"]},
    },
    "code": {
        "non_eng": CODE_LANGS,                       # python, javascript, java
        "tok_est": CODE_TOK_EST,
        "eng_fraction": 0.8,                         # 80/20 eng/code (realistic)
        # python = 50% of the non-eng (code) share — applies to BOTH the
        # tokenizer-fit natural corpus AND the LM-training token budgets
        # (since both consume the same `lang_proportions(track)`). Remaining
        # 50% of the code share is split among unspecified code langs (js,
        # java) proportional to their corpus size (tok_est).
        "non_eng_shares": {"python": 0.5},
        "tok_fit_dir": {"eng": ENG_TOK_TRAINING_DIR,
                          **{l: f"{CODE_TOK_SAMPLE_ROOT}/{l}" for l in CODE_LANGS}},
        "lm_dir_A": {"eng": ENG_TOK_TRAINING_DIR,
                      **{l: f"{CODE_LM_FULL_ROOT}/{l}" for l in CODE_LANGS}},
        # A5B: SET-A scale (125M/50k) but 5B-token budget — a
        # continuation/scaling variant to test whether the code BPE-vs-LangMAP
        # val-BpB gap closes with more UNIQUE data. eng=80%×5B=4B exceeds the
        # 21GB fineweb_sample, so A5B eng uses FULL FW1 (output/), like SET-B.
        "lm_dir_A5B": {"eng": ENG_LM_FULL_DIR,
                        **{l: f"{CODE_LM_FULL_ROOT}/{l}" for l in CODE_LANGS}},
        "lm_dir_B": {"eng": ENG_LM_FULL_DIR,
                      **{l: f"{CODE_LM_FULL_ROOT}/{l}" for l in CODE_LANGS}},
        "text_col": {"eng": "text",
                      **{l: "content" for l in CODE_LANGS}},
    },
}

# ── Default Chinchilla budgets per set (20× params) ──────────────────────────
# A5B = SET-A scale (125M/50k) at a 5B-token budget (2× A) — code-only scaling
# variant (see lm_dir_A5B); uses full-FW1 eng.
SET_TOTAL_TOKENS = {"A": 2_500_000_000, "A5B": 5_000_000_000, "B": 10_000_000_000}
SET_VOCAB = {"A": 50_000, "A5B": 50_000, "B": 100_000}
# Default English share used when a track doesn't override `eng_fraction`.
# morph: 0.5. code: 0.8 (more
# realistic for code LMs).
ENG_FRACTION_DEFAULT = 0.5


def track_langs(track: str) -> list[str]:
    return ["eng"] + TRACKS[track]["non_eng"]


def lm_dir(track: str, set_name: str, lang: str) -> str:
    key = f"lm_dir_{set_name}"
    d = TRACKS[track][key]
    if lang not in d:
        raise KeyError(f"{lang!r} not in track {track!r} (have {list(d)})")
    return d[lang]


def tok_fit_dir(track: str, lang: str) -> str:
    return TRACKS[track]["tok_fit_dir"][lang]


def text_col(track: str, lang: str) -> str:
    return TRACKS[track]["text_col"][lang]


def eng_fraction(track: str) -> float:
    """Per-track English share. morph=0.5, code=0.8."""
    return TRACKS[track].get("eng_fraction", ENG_FRACTION_DEFAULT)


def non_eng_shares(track: str) -> dict[str, float]:
    """Per-non-English-language fraction of the (1 - eng_fraction) non-eng
    portion. Sums to 1.0. Used by `lang_proportions` for BOTH tokenizer-fit
    AND LM-training (they share the same proportions).

    Sourced from track config:
      - If `non_eng_shares` is in the config: those entries are FIXED.
        Remaining non-eng langs (not listed) split the residual proportional
        to `tok_est` (corpus-size estimate).
      - If absent: ALL non-eng langs split 100% proportional to `tok_est`
        (natural distribution — current morph behavior).
    """
    cfg = TRACKS[track]
    non_eng = cfg["non_eng"]
    fixed: dict[str, float] = dict(cfg.get("non_eng_shares", {}))
    fixed_sum = sum(fixed.values())
    if fixed_sum > 1.0 + 1e-9:
        raise AssertionError(
            f"track {track}: fixed non-eng shares sum {fixed_sum} > 1"
        )
    remaining = [l for l in non_eng if l not in fixed]
    if remaining:
        rem_est = sum(cfg["tok_est"][l] for l in remaining)
        if rem_est <= 0:
            raise AssertionError(f"track {track}: remaining tok_est sum 0")
        residual = 1.0 - fixed_sum
        out = dict(fixed)
        for l in remaining:
            out[l] = residual * (cfg["tok_est"][l] / rem_est)
    else:
        out = dict(fixed)
    s = sum(out.values())
    if abs(s - 1.0) > 1e-6:
        raise AssertionError(f"non_eng_shares sum {s} != 1 for track {track}")
    return out


def lang_proportions(track: str) -> dict[str, float]:
    """Per-language fraction of the TOTAL training tokens for this track.
    eng = eng_fraction(track); non-eng splits the residual per `non_eng_shares`.

    **One source of truth for BOTH tokenizer-fit corpus AND LM-training token
    budgets.** Tokenizer fit (BPE + base-Unigram natural corpus) consumes
    `lines_per_lang` derived from these; LM training consumes
    `lang_token_budget` (this × SET_TOTAL_TOKENS). LangMAP per-lang EM is the
    only exception — it uses a separate uniform-per-lang corpus, NOT these
    proportions.
    """
    ef = eng_fraction(track)
    props = {"eng": ef}
    for l, s in non_eng_shares(track).items():
        props[l] = (1.0 - ef) * s
    total = sum(props.values())
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"proportions sum {total} != 1 for {track}")
    return props


def lang_token_budget(track: str, set_name: str) -> dict[str, int]:
    """Per-language LM-training token budget = SET_TOTAL_TOKENS · proportion."""
    total = SET_TOTAL_TOKENS[set_name]
    return {l: int(round(p * total)) for l, p in lang_proportions(track).items()}


def lines_per_lang(track: str, total_lines: int,
                    min_lines_per_lang: int = 1_000) -> dict[str, int]:
    """Per-language tokenizer-fit line counts at the SAME proportions, with an
    explicit minimum floor per language (default 1,000 lines).

    The floor exists because tiny natural shares (e.g. swh, ~0.04% of morph
    ⇒ ~140 lines @ total=350k) starve the tokenizer-fit for that language;
    even though the BPE/base vocab is shared, an effectively-empty per-lang
    sample produces ragged byte coverage. This is NOT a silent rebalancing —
    when the floor activates we (a) log a warning naming the affected
    languages, and (b) accept the resulting overshoot vs `total_lines` rather
    than silently shrink other languages' shares. Set
    `min_lines_per_lang=0` to disable.
    """
    out: dict[str, int] = {}
    floored: list[tuple[str, int, int]] = []  # (lang, proportional, floored)
    for l, p in lang_proportions(track).items():
        raw = int(round(p * total_lines))
        if raw < min_lines_per_lang:
            floored.append((l, raw, min_lines_per_lang))
            out[l] = min_lines_per_lang
        else:
            out[l] = raw
    if floored:
        for l, raw, n in floored:
            logging.warning(
                "  Floor activated for %r: proportional=%d < min=%d → using %d "
                "(overshoot vs target total accepted, not silently rebalanced)",
                l, raw, min_lines_per_lang, n,
            )
        overshoot = sum(n - raw for _, raw, n in floored)
        logging.warning(
            "  Total overshoot from floor(s): +%d lines (target %d, actual %d)",
            overshoot, total_lines, sum(out.values()),
        )
    return out


# Per-track tokenizer-fit defaults. Two CORPUS REGIMES — different consumers:
#   - "natural": per-lang lines ∝ lang_proportions(track). Used by BPE and
#     base-Unigram. The shared vocab they learn reflects the realistic
#     distribution the model will see at training time.
#   - "uniform": LANGSPEC_LINES_PER_LANG lines/lang (default 100k). Used by
#     the per-language LangMAP EM re-estimation — each per-lang scoring table
#     needs adequate signal regardless of natural share (else tiny shares
#     like swh would be starved and the LangMAP comparison would be unfair).
NATURAL_TOTAL_LINES = 350_000
LANGSPEC_LINES_PER_LANG = 100_000


def _write_corpus(corpus_dir: Path, track: str, lang: str, n_lines: int,
                  filename: str) -> str:
    """Stream the first `n_lines` non-empty rows of `track`/`lang` to
    `corpus_dir/filename`. Idempotent (skips existing non-empty)."""
    from datasets import load_dataset  # local import: heavy

    out_file = corpus_dir / filename
    if out_file.exists() and out_file.stat().st_size > 0:
        logging.info(f"  Corpus exists: {out_file}")
        return str(out_file)
    data_dir = tok_fit_dir(track, lang)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Tokenizer-fit corpus dir missing for {track}/{lang}: {data_dir}"
        )
    col = text_col(track, lang)
    ds = load_dataset("parquet", data_dir=data_dir, split="train",
                      streaming=True)
    lines: list[str] = []
    for ex in ds:
        t = (ex.get(col) or "").strip()
        if t:
            lines.append(t)
        if len(lines) >= n_lines:
            break
    if not lines:
        raise RuntimeError(
            f"0 non-empty rows read for {track}/{lang} from {data_dir} "
            f"(column {col!r}) — refusing to fit on empty corpus."
        )
    with open(out_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    logging.info(f"  Wrote {len(lines)} lines for {lang} -> {out_file}")
    return str(out_file)


def prepare_corpus_natural(track: str, corpus_dir,
                            total_lines: int = NATURAL_TOTAL_LINES,
                            min_lines_per_lang: int = 1_000) -> dict:
    """Per-lang lines ∝ lang_proportions(track) for BPE + base-Unigram fit,
    with a per-language floor (default 1,000 lines; see `lines_per_lang`).
    Returns {lang: path}.
    """
    corpus_dir = Path(corpus_dir); corpus_dir.mkdir(parents=True, exist_ok=True)
    targets = lines_per_lang(track, total_lines, min_lines_per_lang)
    logging.info(
        f"prepare_corpus_natural track={track} total={total_lines} "
        f"min_per_lang={min_lines_per_lang}: "
        + ", ".join(f"{l}={targets[l]}" for l in targets)
    )
    return {l: _write_corpus(corpus_dir, track, l, n, f"{l}.txt")
            for l, n in targets.items()}


def prepare_corpus_uniform(track: str, corpus_dir,
                            lines_per_lang: int = LANGSPEC_LINES_PER_LANG) -> dict:
    """`lines_per_lang` lines per language, uniform (NOT proportional). Used
    by the LangMAP per-language EM re-estimation. Returns {lang: path}.
    """
    corpus_dir = Path(corpus_dir); corpus_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        f"prepare_corpus_uniform track={track} lines_per_lang={lines_per_lang}"
    )
    return {l: _write_corpus(corpus_dir, track, l, lines_per_lang, f"{l}.txt")
            for l in track_langs(track)}


# ── Manifest sizing (Part B): no-repeat per-lang doc budgets ─────────────────
# Bytes-per-token estimates by script. We size the manifest with a safety
# margin (`OVERSAMPLE_FACTOR`) so even the MOST-compressing tokenizer (BPE on
# Latin web text typically the tightest) has enough docs to hit its per-lang
# token target without repeating. Each per-tokenizer training run stops at
# its own target; less-compressing tokenizers simply hit it sooner.
BYTES_PER_TOKEN_BY_LANG = {
    # Latin-script web text: ~4 bytes/token typical
    "eng": 4.0, "fin": 4.0, "hun": 4.0, "tur": 4.0, "swh": 4.0,
    # Non-Latin: ~2 bytes/token (multi-byte chars; tokenizers commonly split)
    "kor": 2.0, "arb": 2.0,
    # Code: slightly less compressing than English text on average
    "python": 3.5, "javascript": 3.5, "java": 3.5,
}
# Margin on the per-lang token target when selecting manifest docs. 2.0. Sizing by
# whole-parquet-file bytes (all columns) vs a text-byte/token estimate
# lets fat non-text columns (FW2 url/scores/...) cause far too few docs to be picked
# (hun: 2 files / 294k docs → only 112M of 224M target tokens). The selector is
# TOKEN-AWARE (samples the text column for avg tokens/doc, then selects by
# row count); 2.0× margin absorbs sampling variance across files. Over-selection
# is harmless — the cache build stops at target_tokens (no-repeat preserved).
OVERSAMPLE_FACTOR = 2.0


def target_bytes_per_lang(track: str, set_name: str,
                           oversample: float = OVERSAMPLE_FACTOR) -> dict[str, int]:
    """Bytes of parquet text each language's manifest slice must cover.
    target_bytes = target_tokens × bytes_per_token × oversample."""
    budgets = lang_token_budget(track, set_name)
    out = {}
    for l, tgt_tok in budgets.items():
        bpt = BYTES_PER_TOKEN_BY_LANG.get(l)
        if bpt is None:
            raise KeyError(f"No bytes/token estimate for {l!r}")
        out[l] = int(round(tgt_tok * bpt * oversample))
    return out


# max_seq_len used by the from-scratch trainer (train_fromscratch_v2 default).
# The memmap cache TRUNCATES each doc to this and counts only post-truncation
# tokens toward the per-lang target, so manifest sizing MUST use the same cap.
# Keep in sync with train_fromscratch_v2.py --max_seq_len default.
MANIFEST_MAX_SEQ_LEN = 512


def _measure_tokens_per_doc(first_file: str, text_col: str, tokenizer_path: str,
                            max_seq_len: int = MANIFEST_MAX_SEQ_LEN,
                            sample_docs: int = 20000) -> float:
    """Mean POST-TRUNCATION tokens/doc — the quantity the cache actually accrues.

    Tokenize a representative text sample with the ACTUAL BPE tokenizer and cap
    each doc at `max_seq_len` (the cache does `ids[:max_seq_len]` and counts
    only those). Two bugs this avoids: (1) bytes/token is not a usable constant
    (hun BPE ≈ 15 B/tok, not 4); (2) RAW tokens/doc wildly overcounts for
    long-doc corpora (hun raw ≈ 1661, but capped ≈ 380) → the cache reaches the
    target far slower than raw would predict → under-provisioning.
    BPE is the most-compressing tokenizer (fewest tokens/doc ⇒ most docs needed),
    so sizing for it covers base/pathpiece/langspec. Reads up to `sample_docs`
    rows for a stable mean (the first few-thousand can be atypically long)."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    pf = pq.ParquetFile(str(first_file))
    texts: list[str] = []
    for batch in pf.iter_batches(batch_size=4000, columns=[text_col]):
        for v in batch.column(0).to_pylist():
            if v:
                texts.append(v)
        if len(texts) >= sample_docs:
            break
    if not texts:
        raise RuntimeError(
            f"0 non-empty {text_col!r} rows sampled from {first_file}; cannot "
            f"size the manifest for this language."
        )
    encs = tok.encode_batch(texts, add_special_tokens=False)
    capped = sum(min(len(e.ids), max_seq_len) for e in encs)
    return capped / len(texts)


def select_files_for_token_target(data_dir: str, target_tokens: int,
                                   text_col: str, tokenizer_path: str,
                                   oversample: float = OVERSAMPLE_FACTOR) -> tuple[list[str], dict]:
    """TOKEN-AWARE file selection (replaces the old file-byte sizing, which
    sized by whole-parquet-file bytes vs a wrong text-byte/token constant and
    under-provisioned hun).

    Measure REAL tokens/doc by tokenizing a text sample with the BPE tokenizer
    (`tokenizer_path`), then select the smallest lexicographic prefix of parquet
    files whose cumulative ROW COUNT (metadata only) covers
    `target_tokens × oversample` docs. Over-selecting docs is harmless — the
    cache build consumes manifest rows in shuffled order and stops at
    `target_tokens` (no-repeat preserved). UNDER-selecting causes a hard
    `_build_memmap_cache` exhaustion. Fail loud if even ALL files can't supply
    enough docs (corpus genuinely too small for the no-repeat budget).

    Returns (chosen_files, sizing_info).
    """
    import math
    import pyarrow.parquet as pq
    files = sorted(Path(data_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    def _num_rows_or_none(p: str):
        """parquet row count, or None if the file is unreadable/corrupt.
        Real corpora contain truncated/corrupt shards (3 corrupt files each in
        StarCoder javascript/java). Skipping an
        UNREADABLE file is NOT a silent data substitution — it excludes garbage
        we cannot read; with 2.0× oversample + many valid shards, coverage and
        the token budget are unaffected. We log every skip loudly + record the
        file in `sizing['skipped_corrupt']`, and the file is excluded from the
        manifest so the cache build never reads it either."""
        try:
            return pq.ParquetFile(p).metadata.num_rows
        except Exception as e:  # pyarrow.lib.ArrowInvalid etc.
            logging.warning("  SKIPPING corrupt/unreadable parquet (%s): %s",
                            type(e).__name__, p)
            return None

    # Sample tokens/doc from the FIRST READABLE file (files[0] may be corrupt).
    skipped: list[str] = []
    sample_file = None
    for f in files:
        if _num_rows_or_none(str(f)) is not None:
            sample_file = str(f); break
        skipped.append(str(f))
    if sample_file is None:
        raise RuntimeError(f"All parquet files under {data_dir} are unreadable.")
    tokens_per_doc = max(1.0, _measure_tokens_per_doc(sample_file, text_col,
                                                      tokenizer_path))
    need_docs = int(math.ceil(target_tokens * oversample / tokens_per_doc))

    chosen: list[str] = []
    cum_rows = 0
    for f in files:
        fs = str(f)
        nr = _num_rows_or_none(fs)
        if nr is None:
            if fs not in skipped:
                skipped.append(fs)
            continue
        chosen.append(fs)
        cum_rows += nr
        if cum_rows >= need_docs:
            sizing = {
                "tokens_per_doc_measured": round(tokens_per_doc, 1),
                "tokenizer_used": str(tokenizer_path),
                "need_docs": need_docs, "selected_rows": cum_rows,
                "n_files": len(chosen), "n_files_available": len(files),
                "n_skipped_corrupt": len(skipped),
                "skipped_corrupt": skipped,
                "oversample": oversample,
            }
            if skipped:
                logging.warning("  %d corrupt file(s) skipped for %s: %s",
                                len(skipped), data_dir, skipped)
            return chosen, sizing
    raise RuntimeError(
        f"Corpus {data_dir} has only {cum_rows:,} READABLE docs across "
        f"{len(chosen)} valid files ({len(skipped)} corrupt skipped), < needed "
        f"{need_docs:,} (target {target_tokens:,} tokens × {oversample}× / "
        f"{tokens_per_doc:.0f} measured tok/doc). Corpus too small for the "
        f"no-repeat budget; reduce the target or add corpus."
    )
