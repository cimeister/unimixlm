#!/usr/bin/env python3
"""
check_hf_sp_alignment.py

Usage:
  python check_hf_sp_alignment.py \
      --hf_tokenizer path/to/tokenizer.json \
      --sp_model path/to/sp.model \
      --texts "Hello world!" "  multiple  spaces" "newlines\nok" \
      --max_errors 5

Or provide a file with one test string per line:
  python check_hf_sp_alignment.py \
      --hf_tokenizer path/to/tokenizer.json \
      --sp_model path/to/sp.model \
      --textfile path/to/samples.txt
"""

import argparse
import json
from typing import List, Tuple

# pip install tokenizers sentencepiece
from tokenizers import Tokenizer
import sentencepiece as spm


def load_hf_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def load_sp(path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor(model_file=path)
    return sp


def hf_encode_tokens(hf_tok: Tokenizer, text: str) -> List[str]:
    """
    Returns the HF token *strings* (the 'pieces' as HF sees them), without special tokens.
    These strings already contain the ByteLevel surface markers (e.g., 'Ġ' for spaces).
    """
    enc = hf_tok.encode(text, add_special_tokens=False)
    return enc.tokens


def hf_roundtrip_stable(hf_tok: Tokenizer, text: str) -> bool:
    """
    Check HF round-trip stability: encode -> ids -> decode -> encode yields identical tokens.
    """
    enc1 = hf_tok.encode(text, add_special_tokens=False)
    decoded = hf_tok.decode(enc1.ids, skip_special_tokens=True)
    enc2 = hf_tok.encode(decoded, add_special_tokens=False)
    return enc1.tokens == enc2.tokens


def check_surface_equivalence(
    hf_tok: Tokenizer,
    sp: spm.SentencePieceProcessor,
    text: str
) -> Tuple[bool, List[Tuple[str, str, int]]]:
    """
    For a single raw input string:
      1) HF tokens on raw text: hf_tokens
      2) Build the HF surface string by concatenating those tokens: surf = ''.join(hf_tokens)
      3) Tokenize surf with SP to get sp_pieces (strings)
      4) Compare hf_tokens == sp_pieces

    Returns:
      (is_equal, mismatches)
        - is_equal: overall equality
        - mismatches: list of (hf_token, sp_piece, index) for any differing positions
    """
    hf_tokens = hf_encode_tokens(hf_tok, text)
    hf_surface = "".join(hf_tokens)
    print("HF tokens:", hf_tokens)
    # Let SP segment the HF surface string
    sp_pieces = sp.encode(hf_surface, out_type=str)
    print("SP tokens:", sp_pieces)
    # Compare sequences
    is_equal = (hf_tokens == sp_pieces)
    mismatches = []
    if not is_equal:
        # Align by index; capture mismatches
        L = max(len(hf_tokens), len(sp_pieces))
        for i in range(L):
            t = hf_tokens[i] if i < len(hf_tokens) else "<HF:⊘>"
            p = sp_pieces[i] if i < len(sp_pieces) else "<SP:⊘>"
            if t != p:
                mismatches.append((t, p, i))
    return is_equal, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_tokenizer", required=True, help="Path to HF tokenizer JSON used at runtime")
    ap.add_argument("--sp_model", required=True, help="Path to SentencePiece .model trained on HF-preprocessed text")
    ap.add_argument("--texts", nargs="*", default=[], help="Inline test strings")
    ap.add_argument("--textfile", default=None, help="File with one test string per line (added to --texts)")
    ap.add_argument("--max_errors", type=int, default=10, help="Stop after this many reported mismatches across all inputs")
    args = ap.parse_args()

    hf_tok = load_hf_tokenizer(args.hf_tokenizer)
    sp = load_sp(args.sp_model)

    corpus = list(args.texts)
    if args.textfile:
        with open(args.textfile, "r", encoding="utf-8") as f:
            corpus.extend([line.rstrip("\n") for line in f])

    if not corpus:
        corpus = [
            "Hello world!",
            "  multiple   spaces",
            "newlines\nare fine",
            "Tabs\tand bytes:\x00\x01",
            "Punctuation—em dash, ellipsis… and unicode: naïve café",
            "Leading space",
            "Trailing space ",
            "Emoji 👍🏽 mixed in.",
        ]

    total = len(corpus)
    n_ok_surface = 0
    n_ok_roundtrip = 0
    error_budget = args.max_errors

    for idx, s in enumerate(corpus):
        # Round-trip stability on raw text (HF only)
        rt_ok = hf_roundtrip_stable(hf_tok, s)
        if rt_ok:
            n_ok_roundtrip += 1
        else:
            print(f"[HF-RT] MISMATCH on sample {idx}: {repr(s)}")

        # Surface-form equivalence HF vs SP
        eq, mism = check_surface_equivalence(hf_tok, sp, s)
        if eq:
            n_ok_surface += 1
        else:
            print(f"[HF↔SP] MISMATCH on sample {idx}: {repr(s)}")
            # Show a compact diff
            for (hf_tok_str, sp_piece_str, i) in mism[:5]:
                print(f"  idx={i}: HF={repr(hf_tok_str)}  |  SP={repr(sp_piece_str)}")
            if len(mism) > 5:
                print(f"  ... {len(mism)-5} more diffs")

            error_budget -= 1
            if error_budget <= 0:
                print(f"\nStopping early after reaching --max_errors={args.max_errors}.")
                break

    print("\n=== SUMMARY ===")
    print(f"Samples checked: {total}")
    print(f"HF round-trip stable: {n_ok_roundtrip}/{total}")
    print(f"HF tokens == SP pieces on HF-surface: {n_ok_surface}/{total}")

    # Helpful exit code for CI
    if n_ok_surface == total and n_ok_roundtrip == total:
        print("OK: Alignment holds for all samples.")
        raise SystemExit(0)
    else:
        print("WARN: Some samples failed. See logs above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

