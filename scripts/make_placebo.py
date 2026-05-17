#!/usr/bin/env python3
"""
make_placebo.py — build a placebo tokenizer for the causal control arm.

A LangMAP overlay is a HF `tokenizers` Unigram model whose `model.vocab`
is a list of ``[piece, score]`` pairs (list index == token id). The
PLACEBO keeps every piece and every token id **exactly** as in the
langspec file and only **permutes the score column** under a fixed seed.

Why this is the right control: the vocabulary and id->piece mapping are
byte-for-byte identical to langspec, so the model embedding table stays
perfectly aligned (no re-init, no confound). The ONLY thing destroyed is
the language-specific *information* in the scores. A placebo run must
therefore show no LangMAP gain; if it does, the "gain" is an artifact of
the pipeline / extra training, not the overlay.

Special / unk / byte-fallback pieces keep their original scores so the
tokenizer stays well-formed (otherwise everything can collapse to <unk>);
only the genuine sub-word pieces are scrambled among themselves.

Usage:
    python scripts/make_placebo.py \
        --langspec las_gemma2_2b/langspec_fin_Latn.json \
        --out      las_gemma2_2b/placebo_fin_Latn.json \
        --seed 1234
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

_BYTE_FALLBACK = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langspec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
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

    permutable_idx = [
        i
        for i, (piece, _) in enumerate(vocab)
        if piece not in protected and not _BYTE_FALLBACK.match(piece)
    ]
    scores = [vocab[i][1] for i in permutable_idx]

    rng = random.Random(args.seed)
    shuffled = scores[:]
    rng.shuffle(shuffled)

    for slot, new_score in zip(permutable_idx, shuffled):
        vocab[slot][1] = new_score

    # Hard invariant: pieces and ids are unchanged (embedding alignment).
    orig_pieces = [p for p, _ in json.loads(args.langspec.read_text())["model"]["vocab"]]
    new_pieces = [p for p, _ in vocab]
    assert orig_pieces == new_pieces, "PLACEBO CHANGED THE VOCAB — aborting."

    args.out.write_text(json.dumps(tok, ensure_ascii=False))
    print(
        f"placebo written: {args.out}  "
        f"(|V|={len(vocab)}, scrambled={len(permutable_idx)}, "
        f"protected={len(vocab) - len(permutable_idx)}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
