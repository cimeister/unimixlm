#!/usr/bin/env python3
"""Verify that every LAS tokenizer set has matching vocab_size and special-token
ids between the base tokenizer and all per-language tokenizers.

Checks the tokenizer sets named on the command line and calls
`_assert_tokenizers_consistent` on each. Fails loudly on the first mismatch.
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

from tokenizers import Tokenizer


def _assert_tokenizers_consistent(base_path: str, per_lang_paths, special_tokens):
    base = Tokenizer.from_file(base_path)
    base_vocab = base.get_vocab_size()
    base_specials = {s: base.token_to_id(s) for s in special_tokens}
    for p in per_lang_paths:
        tok = Tokenizer.from_file(p)
        v = tok.get_vocab_size()
        if v != base_vocab:
            raise AssertionError(
                f"Vocab-size mismatch: base={base_vocab} vs {p}={v}"
            )
        for s, bid in base_specials.items():
            tid = tok.token_to_id(s)
            if tid != bid:
                raise AssertionError(
                    f"Special-token id mismatch for {s!r}: base={bid} vs {p}={tid}"
                )
    logging.info(
        f"  vocab={base_vocab}, specials={base_specials}, {len(per_lang_paths)} per-lang files"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        required=True,
        metavar="LABEL:BASE_TOKENIZER_JSON:PER_LANG_DIR:GLOB_PATTERN",
        help=(
            "One tokenizer set to check, repeatable. LABEL is a free-text "
            "name for the report, BASE_TOKENIZER_JSON is the path to the "
            "base tokenizer's tokenizer.json, PER_LANG_DIR is the directory "
            "to glob for per-language tokenizers, and GLOB_PATTERN is the "
            "glob used within it (e.g. 'langspec_*.json'). No default sets "
            "are built in; pass at least one --set."
        ),
    )
    return p


def parse_sets(raw_sets):
    parsed = []
    for raw in raw_sets:
        parts = raw.split(":", 3)
        if len(parts) != 4:
            raise RuntimeError(
                f"--set value {raw!r} must have exactly 4 ':'-separated "
                f"fields: LABEL:BASE_TOKENIZER_JSON:PER_LANG_DIR:GLOB_PATTERN"
            )
        label, base, dir_, pattern = parts
        parsed.append((label, Path(base), Path(dir_), pattern))
    return parsed

# Each base uses a different special-token set. We derive them from the base
# tokenizer itself: look up the ids of its standard specials, and require
# per-lang tokenizers to match.
CANDIDATE_SPECIALS = (
    # HF-style
    "<unk>", "<s>", "</s>", "<pad>",
    # Qwen / generic
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    # Gemma
    "<bos>", "<eos>",
    # OLMo
    "<|pad|>",
)


def check_set(label: str, base_path: Path, dir_: Path, pattern: str) -> bool:
    if not base_path.exists():
        logging.warning(f"[SKIP] {label}: base not found at {base_path}")
        return True
    per_lang = sorted(dir_.glob(pattern))
    if not per_lang:
        logging.warning(f"[SKIP] {label}: no per-lang files matching {pattern}")
        return True

    # Detect which specials the base actually has
    from tokenizers import Tokenizer
    base = Tokenizer.from_file(str(base_path))
    present = tuple(s for s in CANDIDATE_SPECIALS if base.token_to_id(s) is not None)

    logging.info(f"--- {label} ---")
    logging.info(f"  base:      {base_path}")
    logging.info(f"  per-lang:  {[p.name for p in per_lang]}")
    logging.info(f"  specials:  {present}")

    try:
        _assert_tokenizers_consistent(
            str(base_path), [str(p) for p in per_lang],
            special_tokens=present,
        )
        logging.info(f"[OK] {label}")
        return True
    except AssertionError as e:
        logging.error(f"[FAIL] {label}: {e}")
        return False


def main():
    args = build_arg_parser().parse_args()
    tokenizer_sets = parse_sets(args.sets)

    failed = []
    for label, base, dir_, pattern in tokenizer_sets:
        if not check_set(label, base, dir_, pattern):
            failed.append(label)

    print()
    if failed:
        print(f"FAILED: {len(failed)} tokenizer set(s) have vocab/special-token mismatches:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("All checked tokenizer sets are consistent.")


if __name__ == "__main__":
    main()
