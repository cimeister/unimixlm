#!/usr/bin/env python3
"""Fit all 4 tokenizers (BPE, base-Unigram, PathPiece, LangMAP) for the
from-scratch experiment at a single vocab size, for either track
(morph or code). Two CORPUS REGIMES are used:
  - BPE + base-Unigram see the NATURAL distribution corpus (per-lang lines ∝
    lang_proportions(track)). Shared vocab reflects realistic LM data.
  - LangMAP per-language EM sees a UNIFORM 100k-lines/lang corpus (else
    tiny natural shares like swh starve the per-lang estimation; LangMAP
    quality depends on adequate per-lang signal).
  - PathPiece is derived from base (no training data).

Run FOUR times total to cover all track/vocab-size combinations:
  --track morph --vocab_size 50000     # SET-A morph
  --track morph --vocab_size 100000    # SET-B morph
  --track code  --vocab_size 50000     # SET-A code
  --track code  --vocab_size 100000    # SET-B code

Idempotent (skips/repairs existing artifacts). Single-process; no
torch.distributed (avoids corpus-cache + file races seen with engcode).
"""
import argparse
import logging
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders

from las_experiment.tokenizer_training.training import (
    train_bpe_tokenizer,
    _generate_pathpiece_tokenizer,
    build_tekken_pretokenizer,
)
from scripts.fromscratch_common import (
    SPECIAL_TOKENS,
    TRACKS,
    track_langs,
    prepare_corpus_natural,
    prepare_corpus_uniform,
    NATURAL_TOTAL_LINES,
    LANGSPEC_LINES_PER_LANG,
)
from scripts.train_fromscratch_engcode import _assert_tokenizers_consistent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def _valid(path: Path, expected_vocab: int | None) -> bool:
    if not path.exists():
        return False
    try:
        tok = Tokenizer.from_file(str(path))
    except Exception as e:
        logging.warning(f"  Corrupt tokenizer {path} ({e}); deleting")
        path.unlink()
        return False
    if expected_vocab is not None and tok.get_vocab_size() != expected_vocab:
        logging.warning(
            f"  Vocab mismatch {path} "
            f"({tok.get_vocab_size()} != {expected_vocab}); deleting"
        )
        path.unlink()
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=list(TRACKS), required=True)
    p.add_argument("--vocab_size", type=int, required=True,
                   help="50000 (SET-A) or 100000 (SET-B)")
    p.add_argument("--natural_total_lines", type=int,
                   default=NATURAL_TOTAL_LINES,
                   help="Total lines for the BPE/base natural-distribution "
                        "corpus (proportionally split across langs).")
    p.add_argument("--langspec_lines_per_lang", type=int,
                   default=LANGSPEC_LINES_PER_LANG,
                   help="Per-lang lines for the LangMAP EM corpus (uniform).")
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="defaults to $SCRATCH_DIR/unimix_fromscratch_morph if unset",
    )
    args = p.parse_args()
    if args.output_dir is None:
        scratch_dir = os.environ.get("SCRATCH_DIR")
        if not scratch_dir:
            raise RuntimeError(
                "Neither --output_dir nor SCRATCH_DIR is set; pass "
                "--output_dir or set SCRATCH_DIR to your cluster scratch root."
            )
        args.output_dir = str(Path(scratch_dir) / "unimix_fromscratch_morph")
    V = args.vocab_size

    out = Path(args.output_dir)
    tok_dir = out / f"{args.track}_tokenizers_{V}"
    nat_dir = tok_dir / "corpus_natural"   # BPE + base
    uni_dir = tok_dir / "corpus_uniform"   # LangMAP per-lang EM
    tok_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Corpora (idempotent) ──────────────────────────────────────────────
    logging.info(f"=== track={args.track} vocab={V}: preparing corpora ===")
    nat_info = prepare_corpus_natural(args.track, nat_dir,
                                       total_lines=args.natural_total_lines)
    uni_info = prepare_corpus_uniform(args.track, uni_dir,
                                       lines_per_lang=args.langspec_lines_per_lang)
    nat_files = list(nat_info.values())

    # ── 1. BPE — trained on the NATURAL corpus ──────────────────────────────
    bpe_path = tok_dir / "bpe_tokenizer.json"
    if _valid(bpe_path, V):
        logging.info(f"BPE exists: {bpe_path}")
    else:
        logging.info(f"Training BPE (vocab={V}, corpus=natural)...")
        train_bpe_tokenizer(nat_files, str(bpe_path), vocab_size=V,
                             special_tokens=SPECIAL_TOKENS)

    # ── 2. Base Unigram — trained on the NATURAL corpus ─────────────────────
    base_path = tok_dir / "base_tokenizer.json"
    if _valid(base_path, V):
        logging.info(f"Base Unigram exists: {base_path}")
    else:
        logging.info(f"Training base Unigram (vocab={V}, corpus=natural)...")
        from code.standard_unigramlm import StandardUnigramLMTokenizer
        base_tok = StandardUnigramLMTokenizer(
            vocab_size=V, em_mode="soft", byte_level=True,
            unk_token="<unk>", pretokenizer=build_tekken_pretokenizer(),
            decoder_override=decoders.ByteLevel(),
        )
        base_tok.train(nat_files, str(base_path), corpus_info=nat_info)

    # ── 3. PathPiece — derived from base ────────────────────────────────────
    pp_path = tok_dir / "pathpiece_tokenizer.json"
    if _valid(pp_path, V):
        logging.info(f"PathPiece exists: {pp_path}")
    else:
        logging.info("Generating PathPiece from base (scores=-1)...")
        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

    # ── 4. LangMAP per-lang EM — UNIFORM 100k lines/lang ────────────────────
    langs = track_langs(args.track)
    langspec_paths = [tok_dir / f"langspec_{l}.json" for l in langs]
    if all(_valid(p, V) for p in langspec_paths):
        logging.info("LangMAP per-lang tokenizers exist.")
    else:
        logging.info(f"LangMAP per-lang EM (corpus=uniform, {args.langspec_lines_per_lang}/lang)...")
        from code.language_specific_unigramlm import (
            LanguageSpecificUnigramLMTokenizer,
        )
        las_tok = LanguageSpecificUnigramLMTokenizer(
            reestimation_em_mode="soft", unk_token="<unk>",
        )
        las_tok.train(
            corpus_info=uni_info,                  # uniform per-lang, NOT natural
            base_tokenizer_path=str(base_path),
            tokenizer_path_format=str(tok_dir / "langspec_{lang_code}.json"),
            load_base_if_exists=True,
            load_tokenizers_if_exists=True,
        )

    # Consistency gate (vocab + special-token ids: base vs every langspec).
    _assert_tokenizers_consistent(
        str(base_path), [str(p) for p in langspec_paths],
        special_tokens=tuple(SPECIAL_TOKENS),
    )
    logging.info(f"=== Done. 4 tokenizers @ track={args.track} vocab={V} in {tok_dir} ===")


if __name__ == "__main__":
    main()
