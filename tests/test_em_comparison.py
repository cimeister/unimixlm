"""
Tests comparing SentencePiece EM vs Python EM implementations.

Both EM paths receive the same preprocessed corpus and the same vocabulary.
The vocabulary only contains tokens that are plausible under the pretokenizer
(i.e., substrings of individual pretokenized pieces — no cross-boundary tokens).

Run with: pytest tests/test_em_comparison.py -v
"""

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

try:
    import sentencepiece as spm

    _HAS_SPM = (
        shutil.which("spm_train") is not None
        and hasattr(spm, "SentencePieceProcessor")
    )
except ImportError:
    spm = None
    _HAS_SPM = False

from tokenizers import Tokenizer, Regex
from tokenizers.models import Unigram as RawUnigram
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok, Split, Sequence
from tokenizers.trainers import UnigramTrainer

from code.tokenizer_formatting_utils import (
    _apply_pretok_mapping_hf,
    _write_sp_seed_vocab_file,
    write_hf_bytelevel_corpus_optimized,
)
from code.em_utils import _accumulate_usage
from code.constants import SP_DEFAULT_ARGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# OLMo-3 / GPT-4 style pretokenizer (realistic, with word-boundary splitting)
_OLMO_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _build_pretokenizer():
    """Build realistic pretokenizer (OLMo-3 style with word-boundary splitting)."""
    return Sequence([
        Split(pattern=Regex(_OLMO_REGEX), behavior="removed", invert=True),
        ByteLevelPreTok(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


TOY_CORPUS = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "the bird flew over the tree",
    "she sat on a big red chair",
    "the man walked to the store",
    "a small cat chased the dog",
    "the sun set over the hill",
    "he ran fast in the rain",
    "the fish swam in the pond",
    "a tall tree stood in the park",
    "the red car drove on the road",
    "she read a book on the bed",
    "the wind blew through the door",
    "a bird sat on the wall",
    "the kid ran to the gate",
] * 30


def _build_vocab_from_corpus(pretok, corpus, vocab_size=500):
    """Build vocabulary using HF UnigramTrainer with the pretokenizer.

    The resulting vocabulary only contains tokens that are substrings of
    pretokenized pieces — no cross-boundary tokens.
    """
    tok = Tokenizer(RawUnigram())
    tok.pre_tokenizer = pretok

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=ByteLevelPreTok.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    return sorted(tok.get_vocab().keys())


def _write_corpus(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _run_sp_em(processed_path, vocab_tokens, unk_token, tmp_dir):
    """Run SentencePiece EM (1 iteration) and return log-probabilities."""
    vocab_path = os.path.join(tmp_dir, "vocab.tsv")
    out_prefix = os.path.join(tmp_dir, "model")

    uniform_logp = math.log(1.0 / len(vocab_tokens))
    vocab_with_scores = [(tok, uniform_logp) for tok in vocab_tokens]

    token_count = _write_sp_seed_vocab_file(vocab_with_scores, vocab_path, unk_token)
    expected_vocab_size = token_count + 3  # <unk>, <s>, </s>

    command = (
        ["spm_train"]
        + SP_DEFAULT_ARGS
        + [
            f"--input={processed_path}",
            f"--model_prefix={out_prefix}",
            f"--seed_sentencepieces_file={vocab_path}",
            f"--vocab_size={expected_vocab_size}",
            "--num_sub_iterations=1",
            f"--unk_piece={unk_token}",
            "--num_threads=4",
        ]
    )

    subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")

    sp = spm.SentencePieceProcessor()
    sp.load(out_prefix + ".model")

    logps = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        logps[piece] = float(sp.get_score(i))

    return logps


def _run_python_em(raw_path, vocab_tokens, unk_token, pretok, mode="soft"):
    """Run Python EM (1 iteration) on RAW corpus and return log-probabilities.

    Uses raw (not preprocessed) corpus because _accumulate_usage applies the
    pretokenizer internally. Passing a preprocessed corpus would double-encode.
    """
    # Initialize with uniform probabilities (matching SP seed)
    if mode == "soft":
        # Forward-backward expects log-probabilities
        uniform_logp = math.log(1.0 / len(vocab_tokens))
        init_dist = {tok: uniform_logp for tok in vocab_tokens}
    else:
        # Viterbi also works with log-probabilities
        uniform_logp = math.log(1.0 / len(vocab_tokens))
        init_dist = {tok: uniform_logp for tok in vocab_tokens}

    c_dict, total = _accumulate_usage(
        str(raw_path),
        init_dist,
        unk_token,
        mode=mode,
        pretokenizer=pretok,
    )

    # Convert counts to log-probabilities
    logps = {}
    for tok, count in c_dict.items():
        if count > 0 and total > 0:
            logps[tok] = math.log(count / total)

    return logps


def _rank_correlation(dict_a, dict_b, keys):
    """Compute Spearman rank correlation."""
    ranked_a = sorted(keys, key=lambda t: dict_a.get(t, float("-inf")), reverse=True)
    ranked_b = sorted(keys, key=lambda t: dict_b.get(t, float("-inf")), reverse=True)

    rank_a = {tok: i for i, tok in enumerate(ranked_a)}
    rank_b = {tok: i for i, tok in enumerate(ranked_b)}

    n = len(keys)
    if n < 2:
        return 0.0

    d_sq_sum = sum((rank_a[t] - rank_b[t]) ** 2 for t in keys)
    return 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SPM, reason="Requires spm_train binary")
class TestSPvsPythonEM:

    def test_vocab_has_no_cross_boundary_tokens(self):
        """The vocabulary built by HF UnigramTrainer with a pretokenizer
        must not contain tokens that cross pretokenizer boundaries."""
        pretok = _build_pretokenizer()
        vocab = _build_vocab_from_corpus(pretok, TOY_CORPUS, vocab_size=500)

        cross_boundary = [t for t in vocab if 'Ġ' in t[1:] and len(t) > 2]
        assert len(cross_boundary) == 0, (
            f"Vocabulary contains {len(cross_boundary)} cross-boundary tokens: "
            f"{cross_boundary[:10]}"
        )

    def test_sp_em_vs_python_soft_em(self, tmp_path):
        """SP EM and Python soft EM should produce correlated probability
        distributions when given the same valid vocabulary."""
        pretok = _build_pretokenizer()
        vocab = _build_vocab_from_corpus(pretok, TOY_CORPUS, vocab_size=500)

        raw_path = str(tmp_path / "corpus.txt")
        processed_path = str(tmp_path / "processed.txt")
        _write_corpus(raw_path, TOY_CORPUS)
        write_hf_bytelevel_corpus_optimized(raw_path, processed_path, pretok)

        sp_logps = _run_sp_em(processed_path, vocab, "<unk>", str(tmp_path))
        py_logps = _run_python_em(raw_path, vocab, "<unk>", pretok, mode="soft")

        # Compare on tokens present in both with non-trivial probability
        common = set(sp_logps.keys()) & set(py_logps.keys())
        common = {t for t in common
                  if sp_logps[t] > -15 and py_logps[t] > -15
                  and t not in {"<unk>", "<s>", "</s>", "<pad>"}}

        assert len(common) > 20, f"Too few common tokens: {len(common)}"

        rho = _rank_correlation(sp_logps, py_logps, common)

        # With a valid vocabulary (no cross-boundary tokens), the two
        # implementations should broadly agree on which tokens are frequent
        assert rho > 0.5, (
            f"Rank correlation too low: {rho:.3f}. "
            f"SP and Python EM should agree when vocabulary is valid."
        )

    def test_sp_em_vs_python_hard_em(self, tmp_path):
        """SP EM and Python hard (Viterbi) EM with valid vocabulary."""
        pretok = _build_pretokenizer()
        vocab = _build_vocab_from_corpus(pretok, TOY_CORPUS, vocab_size=500)

        raw_path = str(tmp_path / "corpus.txt")
        processed_path = str(tmp_path / "processed.txt")
        _write_corpus(raw_path, TOY_CORPUS)
        write_hf_bytelevel_corpus_optimized(raw_path, processed_path, pretok)

        sp_logps = _run_sp_em(processed_path, vocab, "<unk>", str(tmp_path))
        py_logps = _run_python_em(raw_path, vocab, "<unk>", pretok, mode="hard")

        common = set(sp_logps.keys()) & set(py_logps.keys())
        common = {t for t in common
                  if sp_logps[t] > -15 and py_logps[t] > -15
                  and t not in {"<unk>", "<s>", "</s>", "<pad>"}}

        assert len(common) > 20, f"Too few common tokens: {len(common)}"

        rho = _rank_correlation(sp_logps, py_logps, common)
        # Hard mode (Viterbi) makes discrete segmentation choices, so it
        # correlates less with SP's soft EM than our forward-backward does
        assert rho > 0.2, (
            f"Rank correlation (hard mode) too low: {rho:.3f}."
        )

    def test_top_k_overlap(self, tmp_path):
        """The most probable tokens should largely overlap between SP and Python EM."""
        pretok = _build_pretokenizer()
        vocab = _build_vocab_from_corpus(pretok, TOY_CORPUS, vocab_size=500)

        raw_path = str(tmp_path / "corpus.txt")
        processed_path = str(tmp_path / "processed.txt")
        _write_corpus(raw_path, TOY_CORPUS)
        write_hf_bytelevel_corpus_optimized(raw_path, processed_path, pretok)

        sp_logps = _run_sp_em(processed_path, vocab, "<unk>", str(tmp_path))
        py_logps = _run_python_em(raw_path, vocab, "<unk>", pretok, mode="soft")

        # Filter out special tokens
        specials = {"<unk>", "<s>", "</s>", "<pad>"}
        sp_filtered = {t: s for t, s in sp_logps.items() if t not in specials}
        py_filtered = {t: s for t, s in py_logps.items() if t not in specials}

        sp_top = set(sorted(sp_filtered, key=sp_filtered.get, reverse=True)[:20])
        py_top = set(sorted(py_filtered, key=py_filtered.get, reverse=True)[:20])

        overlap = len(sp_top & py_top)
        assert overlap >= 8, (
            f"Top-20 overlap too low: {overlap}/20.\n"
            f"SP top-10: {sorted(sp_filtered, key=sp_filtered.get, reverse=True)[:10]}\n"
            f"PY top-10: {sorted(py_filtered, key=py_filtered.get, reverse=True)[:10]}"
        )

    def test_both_paths_use_same_preprocessed_corpus(self, tmp_path):
        """Both EM paths must operate on identically pretokenized text."""
        pretok = _build_pretokenizer()

        raw_path = str(tmp_path / "corpus.txt")
        processed_path = str(tmp_path / "processed.txt")
        _write_corpus(raw_path, ["the cat sat on the mat"])
        write_hf_bytelevel_corpus_optimized(raw_path, processed_path, pretok)

        with open(processed_path, "r", encoding="utf-8") as f:
            processed_text = f.read().strip()

        # The written file should be the join of pretokenized pieces
        direct_pieces = _apply_pretok_mapping_hf("the cat sat on the mat", pretok)
        direct_joined = "".join(direct_pieces)

        assert processed_text == direct_joined

    def test_python_soft_vs_hard_consistency(self, tmp_path):
        """Python soft and hard EM should rank tokens similarly."""
        pretok = _build_pretokenizer()
        vocab = _build_vocab_from_corpus(pretok, TOY_CORPUS, vocab_size=500)

        raw_path = str(tmp_path / "corpus.txt")
        processed_path = str(tmp_path / "processed.txt")
        _write_corpus(raw_path, TOY_CORPUS)
        write_hf_bytelevel_corpus_optimized(raw_path, processed_path, pretok)

        soft_logps = _run_python_em(raw_path, vocab, "<unk>", pretok, mode="soft")
        hard_logps = _run_python_em(raw_path, vocab, "<unk>", pretok, mode="hard")

        common = set(soft_logps.keys()) & set(hard_logps.keys())
        common = {t for t in common if soft_logps[t] > -15 and hard_logps[t] > -15}

        assert len(common) > 10

        rho = _rank_correlation(soft_logps, hard_logps, common)
        assert rho > 0.5, f"Soft vs hard Python EM correlation: {rho:.3f}"
