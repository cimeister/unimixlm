#!/usr/bin/env python3
"""
make_placebo.py — build a placebo tokenizer for the causal control arm.

A LangMAP overlay is a HF `tokenizers` Unigram model whose `model.vocab`
is a list of ``[piece, score]`` pairs (list index == token id). The
PLACEBO keeps every piece and every token id **exactly** as in the
langspec file and only **permutes the score column** under a fixed seed.

Two modes:

  --mode global   (the original code-tokenizer-branch design, PR #21): one permutation
      over all non-protected pieces. CONFOUNDED: measured on the real
      100M training corpora, a global permutation IMPROVES compression
      over langspec (e.g. tur 3.34 vs 3.28 bytes/token), because random
      long pieces receive high scores and Viterbi then picks fewer,
      longer pieces. A placebo difference could then be a compression
      effect, not an information effect. Kept only for provenance.

  --mode bucketed (default): sort pieces by langspec score, partition the
      rank order into consecutive buckets of --bucket-size, and permute
      scores only WITHIN each bucket. This preserves the score-magnitude
      profile (compression stays within ~1% of langspec — verify with
      scripts/prep output) while destroying which specific piece carries
      which score, i.e. the language-specific information. Pick
      --bucket-size via the calibration in the prep pipeline.

Protected pieces keep their original scores so the tokenizer stays
well-formed: added_tokens, the unk piece, SP-style byte-fallback pieces
(<0xNN>), and the 256 GPT-2 ByteLevel byte-proxy single characters (the
byte-coverage floor for ByteLevel vocabs like Qwen/OLMo).

Usage:
    python scripts/make_placebo.py \
        --langspec las/langspec_fin_Latn.json \
        --out      las/placebo_fin_Latn.json \
        --mode bucketed --bucket-size 1000 --seed 1234
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

_BYTE_FALLBACK = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _bytelevel_alphabet() -> set:
    from tokenizers.pre_tokenizers import ByteLevel
    return set(ByteLevel.alphabet())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langspec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mode", choices=["global", "bucketed"], default="bucketed")
    ap.add_argument("--bucket-size", type=int, default=1000,
                    help="bucketed mode: permute scores within consecutive "
                         "score-rank buckets of this size")
    args = ap.parse_args()

    tok = json.loads(args.langspec.read_text())
    model = tok.get("model", {})
    if model.get("type") != "Unigram":
        sys.exit(
            f"Expected a Unigram model, got {model.get('type')!r}. "
            f"make_placebo only applies to LangMAP/Unigram overlays."
        )
    vocab = model["vocab"]  # list of [piece, score]

    # Pieces whose score we must NOT scramble (keep tokenizer well-formed).
    protected = set()
    for at in tok.get("added_tokens", []):
        protected.add(at.get("content"))
    unk_id = model.get("unk_id")
    if isinstance(unk_id, int) and 0 <= unk_id < len(vocab):
        protected.add(vocab[unk_id][0])
    byte_proxies = _bytelevel_alphabet()

    permutable_idx = [
        i
        for i, (piece, _) in enumerate(vocab)
        if piece not in protected
        and not _BYTE_FALLBACK.match(piece)
        and piece not in byte_proxies
    ]

    rng = random.Random(args.seed)
    if args.mode == "global":
        scores = [vocab[i][1] for i in permutable_idx]
        shuffled = scores[:]
        rng.shuffle(shuffled)
        for slot, new_score in zip(permutable_idx, shuffled):
            vocab[slot][1] = new_score
    else:
        # Rank pieces by langspec score (desc), bucket the rank order,
        # shuffle scores within each bucket.
        by_score = sorted(permutable_idx, key=lambda i: vocab[i][1], reverse=True)
        bs = args.bucket_size
        if bs < 2:
            sys.exit("--bucket-size must be >= 2")
        for start in range(0, len(by_score), bs):
            bucket = by_score[start:start + bs]
            bucket_scores = [vocab[i][1] for i in bucket]
            rng.shuffle(bucket_scores)
            for i, s in zip(bucket, bucket_scores):
                vocab[i][1] = s

    # Hard invariants: pieces/ids unchanged; score multiset unchanged.
    orig = json.loads(args.langspec.read_text())["model"]["vocab"]
    assert [p for p, _ in orig] == [p for p, _ in vocab], \
        "PLACEBO CHANGED THE VOCAB — aborting."
    assert sorted(s for _, s in orig) == sorted(s for _, s in vocab), \
        "PLACEBO CHANGED THE SCORE MULTISET — aborting."

    args.out.write_text(json.dumps(tok, ensure_ascii=False))
    print(
        f"placebo written: {args.out}  "
        f"(|V|={len(vocab)}, mode={args.mode}"
        f"{f' bucket={args.bucket_size}' if args.mode == 'bucketed' else ''}, "
        f"scrambled={len(permutable_idx)}, "
        f"protected={len(vocab) - len(permutable_idx)}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
