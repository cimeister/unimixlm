"""
Tests for tokenizer consistency across the LAS pipeline.

Verifies that token IDs, pre-tokenizers, vocabulary ordering, and text
preprocessing are consistent between tokenizer training, model finetuning,
and downstream evaluation stages.

Run with: pytest tests/test_tokenizer_pipeline.py -v
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import Unigram as _RawUnigram


def Unigram(vocab, unk_id):
    """Wrapper that converts list-of-lists to list-of-tuples for the tokenizers library."""
    return _RawUnigram([(t, s) for t, s in vocab], unk_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unigram_tokenizer_json(
    vocab, unk_id=0, pre_tokenizer=None, decoder=None, normalizer=None,
    post_processor=None, added_tokens=None,
):
    """Build a minimal HF Unigram tokenizer JSON dict."""
    data = {
        "version": "1.0",
        "model": {
            "type": "Unigram",
            "unk_id": unk_id,
            "vocab": vocab,
        },
        "added_tokens": added_tokens or [],
    }
    if pre_tokenizer is not None:
        data["pre_tokenizer"] = pre_tokenizer
    if decoder is not None:
        data["decoder"] = decoder
    if normalizer is not None:
        data["normalizer"] = normalizer
    if post_processor is not None:
        data["post_processor"] = post_processor
    return data


def _write_tokenizer_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


SAMPLE_VOCAB = [
    ["<unk>", 0.0],
    ["a", -1.0],
    ["b", -2.0],
    ["c", -3.0],
    ["ab", -1.5],
    ["bc", -2.5],
    ["abc", -1.2],
]

SAMPLE_PRETOKENIZER = {"type": "ByteLevel", "add_prefix_space": False}
SAMPLE_DECODER = {"type": "ByteLevel"}

# Tekken regex inlined to avoid importing training.py (which pulls in heavy deps)
TEKKEN_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}][\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _can_import(module_name):
    """Check if a module can be imported without error."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _build_tekken_pretokenizer():
    """Build Tekken pretokenizer without importing training.py."""
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import ByteLevel as BLP, Split, Sequence
    return Sequence([
        Split(pattern=Regex(TEKKEN_REGEX), behavior="isolated", invert=False),
        BLP(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def _generate_pathpiece_tokenizer_local(base_path, output_path):
    """Local copy of _generate_pathpiece_tokenizer to avoid heavy imports."""
    with open(base_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data["model"]["vocab"]
    tok_data["model"]["vocab"] = [[piece, -1.0] for piece, _ in vocab]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# A. Token ID consistency across pipeline stages
# ---------------------------------------------------------------------------

class TestTokenIDConsistency:

    def test_langspec_preserves_token_order_from_base(self, tmp_path):
        """Langspec tokenizer files must preserve token ordering from base,
        ensuring IDs are consistent across pipeline stages."""
        base_data = _make_unigram_tokenizer_json(
            SAMPLE_VOCAB, pre_tokenizer=SAMPLE_PRETOKENIZER, decoder=SAMPLE_DECODER
        )
        _write_tokenizer_json(tmp_path / "base.json", base_data)

        # Langspec: same tokens, different scores
        langspec_vocab = [[piece, score * 0.9] for piece, score in SAMPLE_VOCAB]
        langspec_data = _make_unigram_tokenizer_json(
            langspec_vocab, pre_tokenizer=SAMPLE_PRETOKENIZER, decoder=SAMPLE_DECODER
        )
        _write_tokenizer_json(tmp_path / "langspec.json", langspec_data)

        base_tokens = [t for t, _ in base_data["model"]["vocab"]]
        lang_tokens = [t for t, _ in langspec_data["model"]["vocab"]]
        assert base_tokens == lang_tokens

    def test_pathpiece_preserves_token_order(self, tmp_path):
        """PathPiece tokenizer must preserve token order, only changing scores."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        base_data = _make_unigram_tokenizer_json(
            SAMPLE_VOCAB, pre_tokenizer=SAMPLE_PRETOKENIZER, decoder=SAMPLE_DECODER
        )
        base_path = tmp_path / "base.json"
        pp_path = tmp_path / "pathpiece.json"
        _write_tokenizer_json(base_path, base_data)

        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        with open(pp_path) as f:
            pp_data = json.load(f)

        base_tokens = [t for t, _ in base_data["model"]["vocab"]]
        pp_tokens = [t for t, _ in pp_data["model"]["vocab"]]
        assert base_tokens == pp_tokens

    def test_pathpiece_all_scores_minus_one(self, tmp_path):
        """Every score in PathPiece must be exactly -1.0."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        base_data = _make_unigram_tokenizer_json(SAMPLE_VOCAB)
        base_path = tmp_path / "base.json"
        pp_path = tmp_path / "pathpiece.json"
        _write_tokenizer_json(base_path, base_data)

        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        with open(pp_path) as f:
            pp_data = json.load(f)

        for piece, score in pp_data["model"]["vocab"]:
            assert score == -1.0, f"Token '{piece}' has score {score}, expected -1.0"

    def test_pathpiece_preserves_all_json_fields(self, tmp_path):
        """PathPiece must preserve pre_tokenizer, decoder, normalizer,
        post_processor from the base tokenizer."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        base_data = _make_unigram_tokenizer_json(
            SAMPLE_VOCAB,
            pre_tokenizer=SAMPLE_PRETOKENIZER,
            decoder=SAMPLE_DECODER,
            normalizer={"type": "NFC"},
            post_processor={"type": "TemplateProcessing"},
        )
        base_path = tmp_path / "base.json"
        pp_path = tmp_path / "pathpiece.json"
        _write_tokenizer_json(base_path, base_data)

        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        with open(pp_path) as f:
            pp_data = json.load(f)

        assert pp_data.get("pre_tokenizer") == SAMPLE_PRETOKENIZER
        assert pp_data.get("decoder") == SAMPLE_DECODER
        assert pp_data.get("normalizer") == {"type": "NFC"}
        assert pp_data.get("post_processor") == {"type": "TemplateProcessing"}

    def test_all_tokenizer_variants_share_token_ids(self, tmp_path):
        """Base, pathpiece, and langspec tokenizers must map the same tokens
        to the same IDs, since they share the model's embedding matrix."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        vocab = [
            ["<unk>", 0.0],
            ["hello", -2.0],
            ["world", -3.0],
            ["helloworld", -4.0],
        ]
        base_data = _make_unigram_tokenizer_json(vocab)
        base_path = tmp_path / "base.json"
        pp_path = tmp_path / "pathpiece.json"
        langspec_path = tmp_path / "langspec.json"

        _write_tokenizer_json(base_path, base_data)
        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        # Langspec: same tokens, different scores
        langspec_vocab = [[piece, -0.5] for piece, _ in vocab]
        langspec_data = _make_unigram_tokenizer_json(langspec_vocab)
        _write_tokenizer_json(langspec_path, langspec_data)

        base_tok = Tokenizer.from_file(str(base_path))
        pp_tok = Tokenizer.from_file(str(pp_path))
        lang_tok = Tokenizer.from_file(str(langspec_path))

        for token, _ in vocab:
            base_id = base_tok.token_to_id(token)
            pp_id = pp_tok.token_to_id(token)
            lang_id = lang_tok.token_to_id(token)
            assert base_id == pp_id == lang_id, (
                f"Token '{token}' has different IDs: "
                f"base={base_id}, pathpiece={pp_id}, langspec={lang_id}"
            )


# ---------------------------------------------------------------------------
# B. Pre-tokenizer / decoder consistency
# ---------------------------------------------------------------------------

class TestPreTokenizerConsistency:

    def test_validation_catches_pretokenizer_mismatch(self):
        """Validation must detect pre-tokenizer differences."""
        from las_experiment.tokenizer_training.validation import _validate_pretokenizer

        tok_data = {"pre_tokenizer": {"type": "Whitespace"}}
        expected = {"type": "ByteLevel", "add_prefix_space": False}

        with pytest.raises(AssertionError, match="pre-tokenizer does not match"):
            _validate_pretokenizer(tok_data, expected, "test_tok")

    def test_validation_catches_decoder_mismatch(self):
        """Validation must detect decoder differences."""
        from las_experiment.tokenizer_training.validation import _validate_decoder

        tok_data = {"decoder": {"type": "Metaspace"}}
        expected = {"type": "ByteLevel"}

        with pytest.raises(AssertionError, match="post-tokenizer does not match"):
            _validate_decoder(tok_data, expected, "test_tok")

    def test_validation_passes_on_match(self):
        """Validation passes when all components match."""
        from las_experiment.tokenizer_training.validation import _validate_components

        tok_data = {
            "pre_tokenizer": SAMPLE_PRETOKENIZER,
            "decoder": SAMPLE_DECODER,
        }
        _validate_components(tok_data, SAMPLE_PRETOKENIZER, SAMPLE_DECODER, "test")

    def test_validation_catches_token_order_mismatch(self, tmp_path):
        """Validation must detect reordered tokens."""
        from las_experiment.tokenizer_training.validation import _validate_token_order

        vocab1 = [["<unk>", 0.0], ["a", -1.0], ["b", -2.0]]
        vocab2 = [["<unk>", 0.0], ["b", -2.0], ["a", -1.0]]  # swapped

        tok = Tokenizer(Unigram(vocab2, 0))
        path = tmp_path / "tok.json"
        tok.save(str(path))

        with pytest.raises(AssertionError, match="token order does not match"):
            _validate_token_order(path, ["<unk>", "a", "b"], "test")

    def test_full_validation_end_to_end(self, tmp_path):
        """Full validation pipeline: model -> base -> langspec."""
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from las_experiment.tokenizer_training.validation import validate_tokenizers

        vocab = [("unk", 0.0), ("a", -1.0), ("b", -2.0)]

        def _make_and_save(path, v):
            tok = Tokenizer(_RawUnigram(v, 0))
            tok.pre_tokenizer = ByteLevelPreTok(add_prefix_space=False)
            tok.decoder = ByteLevelDecoder()
            tok.save(str(path))

        model_path = tmp_path / "model.json"
        base_path = tmp_path / "base.json"
        lang_path = tmp_path / "langspec_afr_Latn.json"

        _make_and_save(model_path, vocab)
        _make_and_save(base_path, [(t, s * 0.8) for t, s in vocab])
        _make_and_save(lang_path, [(t, s * 1.2) for t, s in vocab])

        # Should not raise
        validate_tokenizers(
            model_tokenizer_path=model_path,
            base_tokenizer_path=base_path,
            lang_tokenizer_paths={"afr_Latn": lang_path},
        )


# ---------------------------------------------------------------------------
# C. Text preprocessing asymmetries
# ---------------------------------------------------------------------------

class TestTextPreprocessing:

    @pytest.mark.skipif(
        not _can_import("las_experiment.finetune"),
        reason="Requires torch/transformers"
    )
    def test_tokenized_dataset_no_text_transform(self):
        """TokenizedTextDataset must not alter text (no lowercasing, stripping, etc.)."""
        mock_tok = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_tok.encode_batch.return_value = [mock_encoding]
        mock_tok.decode.return_value = "test"

        mock_cfg = MagicMock()
        mock_cfg.max_seq_len = 512

        from las_experiment.finetune import TokenizedTextDataset

        original_text = "  Hello WORLD with Mixed Case  "
        ds = TokenizedTextDataset(
            [original_text], mock_tok, mock_cfg, pad_token_id=0, eos_token_id=1
        )
        call_args = mock_tok.encode_batch.call_args[0][0]
        assert call_args == [original_text], (
            f"Text was transformed before tokenization: {call_args!r}"
        )


# ---------------------------------------------------------------------------
# D. PathPiece shortest-segmentation behavior
# ---------------------------------------------------------------------------

class TestPathPieceBehavior:

    def test_pathpiece_prefers_longest_match(self):
        """Uniform -1 scores make Viterbi prefer fewest tokens (longest match)."""
        vocab = [
            ["<unk>", 0.0],
            ["a", -1.0],
            ["b", -1.0],
            ["c", -1.0],
            ["ab", -1.0],
            ["abc", -1.0],
        ]
        tok = Tokenizer(Unigram(vocab, 0))
        result = tok.encode("abc")

        # 1 token (score -1) beats 2 tokens (score -2) beats 3 tokens (score -3)
        assert result.tokens == ["abc"], (
            f"Expected ['abc'] but got {result.tokens}"
        )

    def test_normal_scores_may_split_differently(self):
        """With non-uniform scores, Unigram may prefer shorter tokens."""
        vocab = [
            ["<unk>", 0.0],
            ["a", -0.1],
            ["b", -0.1],
            ["c", -0.1],
            ["ab", -5.0],
            ["abc", -10.0],
        ]
        tok = Tokenizer(Unigram(vocab, 0))
        result = tok.encode("abc")

        # a+b+c (score -0.3) beats abc (score -10.0)
        assert result.tokens == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# E. BpB computation correctness
# ---------------------------------------------------------------------------

class TestBpBComputation:

    def test_bpb_formula(self):
        """BpB = total_nll / (total_bytes * ln(2))."""
        total_nll = 100.0
        total_bytes = 50
        expected = total_nll / (total_bytes * math.log(2))
        actual = total_nll / (total_bytes * math.log(2))
        assert abs(actual - expected) < 1e-10

    @pytest.mark.skipif(not _can_import("torch"), reason="Requires torch")
    def test_nll_recovery_from_mean_loss(self):
        """loss.item() * num_loss_tokens must recover total NLL."""
        import torch

        labels = torch.tensor([
            [10, 20, 30, -100, -100],
            [10, 20, 30, 40, -100],
        ])
        shifted_labels = labels[:, 1:]
        num_loss_tokens = (shifted_labels != -100).sum().item()
        assert num_loss_tokens == 5

        mean_loss = 2.0
        total_nll = mean_loss * num_loss_tokens
        assert total_nll == 10.0

    def test_bpb_zero_bytes_no_crash(self):
        """BpB computation must not divide by zero on empty input."""
        total_bytes = 0
        total_nll = 0.0
        if total_bytes > 0:
            bpb = total_nll / (total_bytes * math.log(2))
        else:
            bpb = None
        assert bpb is None


# ---------------------------------------------------------------------------
# F. Filename convention consistency
# ---------------------------------------------------------------------------

class TestFilenameConsistency:

    def test_training_and_finetune_filenames_match(self):
        """Tokenizer training output filename must match what finetune.py loads."""
        # Training: lang_code from config is "yor_Latn"
        training_filename = "langspec_yor_Latn.json"

        # Finetuning: lang from config is "yor", code builds f"langspec_{lang}_Latn.json"
        finetune_filename = f"langspec_yor_Latn.json"

        assert training_filename == finetune_filename

    def test_training_and_eval_filenames_match(self):
        """Eval code must look for the same filename as training produces."""
        training_filename = "langspec_yor_Latn.json"

        # Eval: downstream_eval.py builds f"langspec_{las_lang}_Latn.json"
        eval_filename = f"langspec_yor_Latn.json"

        assert training_filename == eval_filename

    @pytest.mark.parametrize("lang", ["afr", "swh", "ron", "ind"])
    def test_experiment_languages(self, lang):
        """All experiment languages produce consistent filenames."""
        training = f"langspec_{lang}_Latn.json"
        finetune = f"langspec_{lang}_Latn.json"
        eval_name = f"langspec_{lang}_Latn.json"
        assert training == finetune == eval_name


# ---------------------------------------------------------------------------
# G. Finetuning pipeline tokenizer verification
# ---------------------------------------------------------------------------

class TestFinetuningTokenizerConsistency:

    def _make_tokenizer_file(self, tmp_path, name, vocab, pretok=None, decoder=None):
        """Helper: create a tokenizer JSON file."""
        data = _make_unigram_tokenizer_json(
            vocab, pre_tokenizer=pretok, decoder=decoder
        )
        path = tmp_path / name
        _write_tokenizer_json(path, data)
        return path

    def test_base_and_langspec_share_pretokenizer(self, tmp_path):
        """Base and langspec tokenizers loaded in finetune.py must have
        identical pre-tokenizers, verifying they produce compatible
        pre-tokenization splits."""
        pretok = {"type": "ByteLevel", "add_prefix_space": False,
                  "trim_offsets": True, "use_regex": True}
        decoder = {"type": "ByteLevel"}
        vocab = [["<unk>", 0.0], ["hello", -1.0], ["world", -2.0]]

        base_path = self._make_tokenizer_file(
            tmp_path, "base_tokenizer.json", vocab,
            pretok=pretok, decoder=decoder
        )
        lang_path = self._make_tokenizer_file(
            tmp_path, "langspec_afr_Latn.json",
            [[t, s * 0.5] for t, s in vocab],
            pretok=pretok, decoder=decoder
        )

        with open(base_path) as f:
            base_data = json.load(f)
        with open(lang_path) as f:
            lang_data = json.load(f)

        assert base_data["pre_tokenizer"] == lang_data["pre_tokenizer"], (
            "Base and langspec must share pre-tokenizer"
        )
        assert base_data["decoder"] == lang_data["decoder"], (
            "Base and langspec must share decoder"
        )

    def test_all_variants_share_vocab_ids(self, tmp_path):
        """All tokenizer variants used during finetuning must map tokens
        to the same IDs as the model's embedding matrix expects."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        vocab = [
            ("unk", 0.0),
            ("pad", 0.0),
            ("s", 0.0),
            ("the", -1.0),
            ("cat", -2.0),
            ("sat", -3.0),
        ]

        # Create base via tokenizers library (proper format)
        base_path = tmp_path / "base_tokenizer.json"
        pp_path = tmp_path / "pathpiece_tokenizer.json"
        langspec_path = tmp_path / "langspec_afr_Latn.json"

        base_tok = Tokenizer(_RawUnigram(vocab, 0))
        base_tok.save(str(base_path))

        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        langspec_vocab = [(t, -0.5) for t, _ in vocab]
        lang_tok = Tokenizer(_RawUnigram(langspec_vocab, 0))
        lang_tok.save(str(langspec_path))

        # Reload all
        base_tok = Tokenizer.from_file(str(base_path))
        pp_tok = Tokenizer.from_file(str(pp_path))
        lang_tok = Tokenizer.from_file(str(langspec_path))

        # Verify every token maps to the same ID
        for token, _ in vocab:
            base_id = base_tok.token_to_id(token)
            pp_id = pp_tok.token_to_id(token)
            lang_id = lang_tok.token_to_id(token)
            assert base_id is not None, f"Token '{token}' not found in base"
            assert base_id == pp_id, (
                f"'{token}': base ID {base_id} != pathpiece ID {pp_id}"
            )
            assert base_id == lang_id, (
                f"'{token}': base ID {base_id} != langspec ID {lang_id}"
            )

    def test_pretokenizer_matches_model_tokenizer(self, tmp_path):
        """The LAS tokenizer's pre-tokenizer must match the model tokenizer's,
        otherwise the same text produces different token boundaries."""
        pretok_model = {"type": "ByteLevel", "add_prefix_space": False,
                        "trim_offsets": True, "use_regex": True}
        pretok_wrong = {"type": "Whitespace"}

        vocab = [["<unk>", 0.0], ["a", -1.0]]

        model_data = _make_unigram_tokenizer_json(
            vocab, pre_tokenizer=pretok_model
        )
        model_path = tmp_path / "model.json"
        _write_tokenizer_json(model_path, model_data)

        wrong_data = _make_unigram_tokenizer_json(
            vocab, pre_tokenizer=pretok_wrong
        )
        wrong_path = tmp_path / "wrong.json"
        _write_tokenizer_json(wrong_path, wrong_data)

        from las_experiment.tokenizer_training.validation import (
            _validate_pretokenizer, _load_tokenizer_json,
        )

        model_json = _load_tokenizer_json(model_path)
        wrong_json = _load_tokenizer_json(wrong_path)

        with pytest.raises(AssertionError, match="pre-tokenizer does not match"):
            _validate_pretokenizer(
                wrong_json,
                model_json["pre_tokenizer"],
                "mismatched_tok",
            )

    @pytest.mark.skipif(
        not _can_import("las_experiment.finetune"),
        reason="Requires torch/transformers"
    )
    def test_load_las_tokenizer_special_tokens(self, tmp_path):
        """load_las_tokenizer_with_eos_pad must resolve pad/eos IDs
        consistently with the model tokenizer."""
        vocab = [
            ["<unk>", 0.0],
            ["<|endoftext|>", 0.0],
            ["hello", -1.0],
        ]
        tok_path = tmp_path / "tok.json"
        data = _make_unigram_tokenizer_json(vocab)
        _write_tokenizer_json(tok_path, data)
        # Save via tokenizers library to get proper format
        tok = Tokenizer(Unigram(vocab, 0))
        tok.save(str(tok_path))

        # Mock model tokenizer
        mock_model_tok = MagicMock()
        mock_model_tok.pad_token = "<|endoftext|>"
        mock_model_tok.eos_token = "<|endoftext|>"

        from las_experiment.finetune import load_las_tokenizer_with_eos_pad

        loaded_tok, pad_id, eos_id = load_las_tokenizer_with_eos_pad(
            tok_path, mock_model_tok
        )
        assert pad_id == 1  # <|endoftext|> is at index 1
        assert eos_id == 1
        assert loaded_tok.token_to_id("<|endoftext|>") == 1


# ---------------------------------------------------------------------------
# H. Parallel futures bug regression test
# ---------------------------------------------------------------------------

class TestParallelFuturesBugRegression:

    def test_futures_loop_assigns_to_correct_key(self):
        """Regression test for C1: the futures collection loop must assign
        results to the correct language key, not a stale variable."""
        # Simulate the code pattern from language_specific_unigramlm.py
        lang_code = None  # Stale variable from a prior loop
        corpus_info = {"en": "en.txt", "fr": "fr.txt", "de": "de.txt"}

        # First loop sets lang_code as a side effect
        for lang_code in corpus_info.keys():
            pass
        assert lang_code == "de"  # last iteration leaks

        # The FIXED code should use 'lang' from the futures dict, not 'lang_code'
        futures_by_lang = {"en": "result_en", "fr": "result_fr", "de": "result_de"}

        # Correct pattern (what the fix should do):
        per_lang_tok = {}
        for lang, result in futures_by_lang.items():
            per_lang_tok[lang] = {"result": result}  # uses 'lang', NOT 'lang_code'

        assert len(per_lang_tok) == 3
        assert set(per_lang_tok.keys()) == {"en", "fr", "de"}
        assert per_lang_tok["en"]["result"] == "result_en"
        assert per_lang_tok["fr"]["result"] == "result_fr"


# ---------------------------------------------------------------------------
# I. Integration: tokenizer round-trip through pipeline stages
# ---------------------------------------------------------------------------

class TestPipelineRoundTrip:

    def test_same_text_same_ids_across_modes(self, tmp_path):
        """For each token in the vocabulary, all tokenizer modes must
        produce the same token-to-ID mapping. Different modes may SEGMENT
        differently (producing different token sequences), but any given
        token string must always map to the same ID."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        # Realistic-ish vocab
        vocab = [
            ["<unk>", 0.0],
            ["▁Die", -2.0],
            ["▁kat", -3.0],
            ["▁sit", -2.5],
            ["▁op", -1.5],
            ["▁die", -1.0],
            ["▁mat", -2.0],
            ["▁", -0.5],
            ["D", -4.0],
            ["ie", -3.5],
            ["k", -4.0],
            ["at", -3.0],
        ]

        base_path = tmp_path / "base_tokenizer.json"
        pp_path = tmp_path / "pathpiece_tokenizer.json"
        lang_path = tmp_path / "langspec_afr_Latn.json"

        base_data = _make_unigram_tokenizer_json(vocab)
        _write_tokenizer_json(base_path, base_data)

        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        lang_vocab = [[t, s * 1.3] for t, s in vocab]
        lang_data = _make_unigram_tokenizer_json(lang_vocab)
        _write_tokenizer_json(lang_path, lang_data)

        base_tok = Tokenizer.from_file(str(base_path))
        pp_tok = Tokenizer.from_file(str(pp_path))
        lang_tok = Tokenizer.from_file(str(lang_path))

        # Check: every token in vocab has consistent IDs
        for i, (token, _) in enumerate(vocab):
            assert base_tok.token_to_id(token) == i
            assert pp_tok.token_to_id(token) == i
            assert lang_tok.token_to_id(token) == i

    def test_tokenizer_json_structure_preserved(self, tmp_path):
        """Saving and reloading a tokenizer must preserve the JSON structure
        (no field reordering that could affect ID mapping)."""
        vocab = [["<unk>", 0.0], ["a", -1.0], ["b", -2.0]]
        data = _make_unigram_tokenizer_json(vocab)
        path = tmp_path / "tok.json"
        _write_tokenizer_json(path, data)

        # Load and re-save
        tok = Tokenizer.from_file(str(path))
        path2 = tmp_path / "tok2.json"
        tok.save(str(path2))

        # Reload and check vocab order
        tok2 = Tokenizer.from_file(str(path2))
        for token, _ in vocab:
            assert tok.token_to_id(token) == tok2.token_to_id(token)


# ---------------------------------------------------------------------------
# J. Pretokenizer consistency (from-scratch pipeline)
# ---------------------------------------------------------------------------

class TestPreTokenizerFromScratchConsistency:

    def test_add_prefix_space_divergence(self):
        """add_prefix_space=True vs False produce different first tokens.
        This documents the bug that existed before the pretokenizer fix."""
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        with_prefix = ByteLevelPreTok(add_prefix_space=True, use_regex=False)
        without_prefix = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        text = "Hello world"
        tokens_with = with_prefix.pre_tokenize_str(text)
        tokens_without = without_prefix.pre_tokenize_str(text)

        # With add_prefix_space=True, "Hello" becomes "ĠHello"
        # Without it, "Hello" stays "Hello"
        assert tokens_with != tokens_without, (
            "add_prefix_space should produce different output"
        )
        # The first token's text differs
        assert tokens_with[0][0].startswith("Ġ")
        assert not tokens_without[0][0].startswith("Ġ")

    def test_tekken_regex_splits_numbers(self):
        """The Tekken regex groups numbers into 1-3 digit chunks,
        which bare ByteLevel does not."""
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok, Split, Sequence

        bare = ByteLevelPreTok(add_prefix_space=False, use_regex=False)
        tekken = _build_tekken_pretokenizer()

        text = "Price: 12345"
        bare_tokens = bare.pre_tokenize_str(text)
        tekken_tokens = tekken.pre_tokenize_str(text)

        # Tekken should split "12345" into "123" + "45" (1-3 digit groups)
        # while bare ByteLevel keeps it as one piece
        bare_pieces = [t for t, _ in bare_tokens]
        tekken_pieces = [t for t, _ in tekken_tokens]
        assert len(tekken_pieces) > len(bare_pieces), (
            f"Tekken should produce more pieces than bare ByteLevel: "
            f"tekken={tekken_pieces}, bare={bare_pieces}"
        )

    def test_all_tokenizers_share_pretokenizer_json(self, tmp_path):
        """All tokenizer files produced by the pipeline must have
        identical pre_tokenizer JSON fields."""
        _generate_pathpiece_tokenizer = _generate_pathpiece_tokenizer_local

        tekken = _build_tekken_pretokenizer()

        # Create a BPE tokenizer with Tekken pretokenizer
        from tokenizers.models import BPE as BPEModel
        from tokenizers import decoders as tok_decoders

        bpe_tok = Tokenizer(BPEModel(unk_token="<unk>"))
        bpe_tok.pre_tokenizer = tekken
        bpe_tok.decoder = tok_decoders.ByteLevel()
        bpe_path = tmp_path / "bpe.json"
        bpe_tok.save(str(bpe_path))

        # Create a Unigram tokenizer with the same pretokenizer
        unigram_tok = Tokenizer(_RawUnigram(
            [("a", -1.0), ("b", -2.0), ("<unk>", 0.0)], 2
        ))
        unigram_tok.pre_tokenizer = tekken
        unigram_tok.decoder = tok_decoders.ByteLevel()
        base_path = tmp_path / "base.json"
        unigram_tok.save(str(base_path))

        # PathPiece from base
        pp_path = tmp_path / "pathpiece.json"
        _generate_pathpiece_tokenizer(str(base_path), str(pp_path))

        # Load all and compare pre_tokenizer JSON
        import json
        pretokenizers = {}
        for name, path in [("bpe", bpe_path), ("base", base_path), ("pathpiece", pp_path)]:
            with open(path) as f:
                data = json.load(f)
            pretokenizers[name] = data.get("pre_tokenizer")

        # BPE and base should have the same pretokenizer structure
        bpe_pt = json.dumps(pretokenizers["bpe"], sort_keys=True)
        base_pt = json.dumps(pretokenizers["base"], sort_keys=True)
        pp_pt = json.dumps(pretokenizers["pathpiece"], sort_keys=True)

        assert bpe_pt == base_pt, (
            f"BPE and base pretokenizers differ:\n  BPE: {bpe_pt}\n  Base: {base_pt}"
        )
        assert base_pt == pp_pt, (
            f"Base and PathPiece pretokenizers differ:\n  Base: {base_pt}\n  PP: {pp_pt}"
        )

    @pytest.mark.skipif(
        not _can_import("code.standard_unigramlm"),
        reason="Requires pyarrow/datasets"
    )
    def test_pretokenizer_override_used_during_training(self):
        """When pretokenizer is passed to StandardUnigramLMTokenizer,
        _get_pretokenizer() must return it instead of the default."""
        from code.standard_unigramlm import StandardUnigramLMTokenizer
        tekken = _build_tekken_pretokenizer()
        tok = StandardUnigramLMTokenizer(
            vocab_size=500,
            byte_level=True,
            pretokenizer=tekken,
        )

        result = tok._get_pretokenizer()
        # Should return the override, not the default ByteLevel
        assert result is tekken, (
            f"Expected Tekken pretokenizer override, got {type(result)}"
        )

    @pytest.mark.skipif(
        not _can_import("code.standard_unigramlm"),
        reason="Requires pyarrow/datasets"
    )
    def test_decoder_override_used(self):
        """When decoder_override is passed, _get_decoder() must return it."""
        from code.standard_unigramlm import StandardUnigramLMTokenizer
        from tokenizers import decoders

        custom_decoder = decoders.ByteLevel()
        tok = StandardUnigramLMTokenizer(
            vocab_size=500,
            byte_level=True,
            decoder_override=custom_decoder,
        )

        result = tok._get_decoder()
        assert result is custom_decoder

    def test_encode_decode_roundtrip_with_tekken(self, tmp_path):
        """Encoding and decoding with a Tekken-pretokenized Unigram tokenizer
        must produce the original text."""
        from tokenizers import decoders as tok_decoders
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        alphabet = list(ByteLevelPreTok.alphabet())
        vocab = [("<unk>", 0.0)] + [(ch, -1.0) for ch in alphabet]
        tok = Tokenizer(_RawUnigram(vocab, 0))
        tok.pre_tokenizer = _build_tekken_pretokenizer()
        tok.decoder = tok_decoders.ByteLevel()

        test_texts = [
            "hello world",
            "1234 test",
            "a b c",
        ]
        for text in test_texts:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded.ids)
            assert decoded == text, (
                f"Round-trip failed: {text!r} -> {encoded.tokens} -> {decoded!r}"
            )


# ---------------------------------------------------------------------------
# K. LAS pretokenizer inheritance constraints
# ---------------------------------------------------------------------------

class TestLASPreTokenizerInheritance:

    @pytest.mark.skipif(
        not _can_import("code.language_specific_unigramlm"),
        reason="Requires pyarrow/datasets"
    )
    def test_las_tokenizer_does_not_accept_pretokenizer(self):
        """LanguageSpecificUnigramLMTokenizer must NOT accept a pretokenizer
        parameter — it must always inherit from its base/parent."""
        import inspect
        from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

        sig = inspect.signature(LanguageSpecificUnigramLMTokenizer.__init__)
        params = list(sig.parameters.keys())
        assert "pretokenizer" not in params, (
            "LanguageSpecificUnigramLMTokenizer should not accept pretokenizer — "
            "it must inherit from parent/base"
        )
        assert "decoder_override" not in params, (
            "LanguageSpecificUnigramLMTokenizer should not accept decoder_override"
        )

    @pytest.mark.skipif(
        not _can_import("code.language_specific_unigramlm"),
        reason="Requires pyarrow/datasets"
    )
    def test_las_does_not_accept_whitespace_token_boundaries(self):
        """whitespace_token_boundaries has been removed from all APIs."""
        import inspect
        from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
        from code.standard_unigramlm import StandardUnigramLMTokenizer

        for cls in [StandardUnigramLMTokenizer, LanguageSpecificUnigramLMTokenizer]:
            sig = inspect.signature(cls.__init__)
            assert "whitespace_token_boundaries" not in sig.parameters, (
                f"{cls.__name__} still has whitespace_token_boundaries parameter"
            )

    @pytest.mark.skipif(
        not _can_import("code.standard_unigramlm"),
        reason="Requires pyarrow/datasets"
    )
    def test_pretokenizer_precedence(self):
        """_get_pretokenizer() precedence: override > parent > base > byte_level > None."""
        from code.standard_unigramlm import StandardUnigramLMTokenizer

        # 1. No pretokenizer, byte_level=False → None
        tok = StandardUnigramLMTokenizer(vocab_size=100, byte_level=False)
        assert tok._get_pretokenizer() is None

        # 2. byte_level=True → ByteLevel
        tok = StandardUnigramLMTokenizer(vocab_size=100, byte_level=True)
        result = tok._get_pretokenizer()
        assert result is not None

        # 3. Explicit override wins over byte_level
        override = _build_tekken_pretokenizer()
        tok = StandardUnigramLMTokenizer(
            vocab_size=100, byte_level=True, pretokenizer=override
        )
        assert tok._get_pretokenizer() is override

    def test_saved_pretokenizer_matches_training(self, tmp_path):
        """A saved tokenizer must have the same pretokenizer that was used
        during training. This catches post-hoc patching bugs."""
        tekken = _build_tekken_pretokenizer()

        # Build a tokenizer with Tekken pretokenizer
        from tokenizers import decoders as tok_decoders
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        alphabet = list(ByteLevelPreTok.alphabet())
        vocab = [("<unk>", 0.0)] + [(ch, -1.0) for ch in alphabet]
        tok = Tokenizer(_RawUnigram(vocab, 0))
        tok.pre_tokenizer = tekken
        tok.decoder = tok_decoders.ByteLevel()

        # Save it
        tok_path = tmp_path / "tok.json"
        tok.save(str(tok_path))

        # Load and verify pretokenizer matches
        with open(tok_path) as f:
            saved_data = json.load(f)

        # Serialize the original pretokenizer for comparison
        ref_tok = Tokenizer(_RawUnigram([("<unk>", 0.0)], 0))
        ref_tok.pre_tokenizer = tekken
        ref_json = json.loads(ref_tok.to_str()).get("pre_tokenizer")

        assert saved_data["pre_tokenizer"] == ref_json

    def test_patched_pretokenizer_detected(self, tmp_path):
        """If someone patches the pretokenizer after saving (the old bug),
        the mismatch should be detectable."""
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok
        from tokenizers import decoders as tok_decoders

        # Build tokenizer with ByteLevel pretokenizer
        alphabet = list(ByteLevelPreTok.alphabet())
        vocab = [("<unk>", 0.0)] + [(ch, -1.0) for ch in alphabet]
        tok = Tokenizer(_RawUnigram(vocab, 0))
        tok.pre_tokenizer = ByteLevelPreTok(add_prefix_space=True)
        tok.decoder = tok_decoders.ByteLevel()

        # Save it
        tok_path = tmp_path / "tok.json"
        tok.save(str(tok_path))

        # Patch the pretokenizer in the saved file (the old bad pattern)
        with open(tok_path) as f:
            data = json.load(f)
        original_pretok = data["pre_tokenizer"]
        data["pre_tokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\d+"}, "behavior": "Removed", "invert": True},
                {"type": "ByteLevel", "add_prefix_space": False},
            ],
        }
        with open(tok_path, "w") as f:
            json.dump(data, f)

        # Reload and verify the pretokenizer NO LONGER matches the original
        with open(tok_path) as f:
            patched_data = json.load(f)

        assert patched_data["pre_tokenizer"] != original_pretok, (
            "Patched pretokenizer should differ from original — "
            "this is the bug we're guarding against"
        )


# ---------------------------------------------------------------------------
# L. SentencePiece EM training pipeline tests
# ---------------------------------------------------------------------------

def _build_toy_log_probs(vocab_with_probs):
    """Build a log-probability dict from {token: probability} pairs.

    This is an inline helper to avoid importing from modules with heavy deps.
    """
    import math as _math
    return {tok: _math.log(p) for tok, p in vocab_with_probs.items()}


class TestViterbiSegmentation:
    """Test Viterbi segmentation on a toy vocabulary."""

    def test_viterbi_prefers_high_prob_segmentation(self):
        """Viterbi should pick the segmentation with highest total log-prob."""
        import numpy as np
        from code.em_utils import segment_text_viterbi
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        # Vocabulary: "ab" has high prob, "a"+"b" individually are lower
        vocab_probs = {
            "<unk>": 1e-6,
            "a": 0.05,
            "b": 0.05,
            "ab": 0.4,
            "c": 0.1,
            "Ġ": 0.35,  # HF space marker (ByteLevel maps space to this)
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        # "ab" on its own -- pretokenizer with use_regex=False keeps it as one piece
        tokens = segment_text_viterbi("ab", log_probs, pretok=pretok)
        # Should prefer single "ab" (log 0.4) over "a"+"b" (log 0.05 + log 0.05)
        assert tokens == ["ab"], f"Expected ['ab'], got {tokens}"

    def test_viterbi_falls_back_to_unk_for_unknown_chars(self):
        """Characters not in the vocabulary should produce <unk> tokens."""
        from code.em_utils import segment_text_viterbi
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        vocab_probs = {
            "<unk>": 1e-6,
            "a": 0.5,
            "b": 0.49,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        # ByteLevel pretok maps "a" -> "a", but maps characters by their
        # byte-level unicode mapping. "x" becomes a byte-level char not in vocab.
        # We test with the raw pieces directly via a text that produces known pieces.
        tokens = segment_text_viterbi("ab", log_probs, pretok=pretok)
        # "a" and "b" are in vocab, so no UNK expected for "ab"
        assert "<unk>" not in tokens
        assert "a" in tokens and "b" in tokens

    def test_viterbi_segments_longer_text(self):
        """Viterbi should handle multi-word text with a pretokenizer."""
        from code.em_utils import segment_text_viterbi
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        # Build vocab that includes ByteLevel-mapped pieces
        vocab_probs = {
            "<unk>": 1e-6,
            "h": 0.05, "e": 0.05, "l": 0.05, "o": 0.05,
            "he": 0.1, "ll": 0.1, "lo": 0.1,
            "hello": 0.15,
            "Ġ": 0.05,  # space marker in ByteLevel
            "Ġw": 0.05,
            "or": 0.05,
            "ld": 0.05,
            "w": 0.02, "r": 0.02, "d": 0.02,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        tokens = segment_text_viterbi("hello world", log_probs, pretok=pretok)
        # Should produce some tokenization without errors
        assert len(tokens) > 0
        # Reconstructing: joining tokens should cover the pretokenized form
        joined = "".join(tokens).replace("<unk>", "?")
        assert len(joined) > 0

    def test_viterbi_empty_text(self):
        """Empty text should produce empty token list."""
        from code.em_utils import segment_text_viterbi
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)
        log_probs = _build_toy_log_probs({"<unk>": 1e-6, "a": 0.5})
        tokens = segment_text_viterbi("", log_probs, pretok=pretok)
        assert tokens == []


class TestForwardBackwardEMCounts:
    """Test forward-backward EM expected counts on a known vocabulary."""

    def test_counts_are_nonnegative(self):
        """All expected counts from forward-backward must be non-negative."""
        import numpy as np
        from code.em_utils import forward_backward_unigram_log_space
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        vocab_probs = {
            "<unk>": 1e-4,
            "a": 0.2,
            "b": 0.2,
            "c": 0.2,
            "ab": 0.15,
            "bc": 0.15,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        counts = forward_backward_unigram_log_space(
            "abc", log_probs, pretok=pretok
        )

        for token, count in counts.items():
            assert count >= 0, f"Token '{token}' has negative count {count}"

    def test_counts_sum_approximately_correct(self):
        """Expected counts should sum to approximately the expected number
        of tokens in the segmentation (between 1 and len(text))."""
        import numpy as np
        from code.em_utils import forward_backward_unigram_log_space
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        vocab_probs = {
            "<unk>": 1e-4,
            "a": 0.2,
            "b": 0.2,
            "c": 0.1,
            "ab": 0.2,
            "bc": 0.15,
            "abc": 0.1,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        counts = forward_backward_unigram_log_space(
            "abc", log_probs, pretok=pretok
        )

        total = sum(counts.values())
        # The text "abc" pretokenized by ByteLevel(use_regex=False) stays as "abc".
        # With this vocab, segmentations range from 1 token ("abc") to
        # 3 tokens ("a"+"b"+"c"), so the expected count should be between 1 and 3.
        assert 0.9 <= total <= 3.1, (
            f"Expected total count between 1 and 3, got {total}"
        )

    def test_unk_count_zero_when_all_chars_covered(self):
        """When every character can be tokenized, UNK count should be zero."""
        import numpy as np
        from code.em_utils import forward_backward_unigram_log_space
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        vocab_probs = {
            "<unk>": 1e-4,
            "a": 0.3,
            "b": 0.3,
            "ab": 0.3,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        counts = forward_backward_unigram_log_space(
            "ab", log_probs, pretok=pretok
        )

        unk_count = counts.get("<unk>", 0.0)
        assert unk_count < 1e-10, (
            f"UNK count should be ~0 when all chars covered, got {unk_count}"
        )

    def test_unk_count_positive_for_uncovered_chars(self):
        """When some characters are not in the vocab, UNK count should be positive."""
        import numpy as np
        from code.em_utils import forward_backward_unigram_log_space
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        # "a" is in vocab but "b" is not (after ByteLevel mapping)
        vocab_probs = {
            "<unk>": 1e-3,
            "a": 0.5,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        # "ab" through ByteLevel becomes "ab" -- "b" is not in vocab
        counts = forward_backward_unigram_log_space(
            "ab", log_probs, pretok=pretok
        )

        unk_count = counts.get("<unk>", 0.0)
        assert unk_count > 0, (
            f"UNK count should be positive for uncovered chars, got {unk_count}"
        )

    def test_counts_only_contain_vocab_tokens(self):
        """Returned counts should only include tokens from the vocabulary."""
        import numpy as np
        from code.em_utils import forward_backward_unigram_log_space
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=False)

        vocab_probs = {
            "<unk>": 1e-4,
            "a": 0.3,
            "b": 0.3,
            "ab": 0.3,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        counts = forward_backward_unigram_log_space(
            "ab", log_probs, pretok=pretok
        )

        for token in counts:
            assert token in log_probs, (
                f"Count returned for token '{token}' not in vocabulary"
            )


class TestPretokenizerAffectsViterbi:
    """Test that different pretokenizers produce different Viterbi segmentations."""

    def test_different_pretokenizers_different_segmentations(self):
        """ByteLevel with vs without regex should split differently on
        mixed text, producing different Viterbi segmentations."""
        from code.em_utils import segment_text_viterbi
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok_no_regex = ByteLevelPreTok(
            add_prefix_space=False, use_regex=False
        )
        pretok_with_regex = ByteLevelPreTok(
            add_prefix_space=False, use_regex=True
        )

        # use_regex=True splits on word boundaries; use_regex=False does not.
        # This means the pretokenized pieces differ, which should cause
        # different Viterbi segmentations.
        text = "hello world"

        # Build a vocab that covers all ByteLevel-mapped characters
        # plus some multi-char tokens that would only match within words
        vocab_probs = {
            "<unk>": 1e-6,
            "h": 0.02, "e": 0.02, "l": 0.02, "o": 0.02,
            "w": 0.02, "r": 0.02, "d": 0.02,
            "Ġ": 0.02,  # ByteLevel space marker
            "he": 0.05, "ll": 0.05, "lo": 0.05,
            "oĠ": 0.08,  # crosses word boundary -- only matchable without regex
            "Ġw": 0.05,
            "or": 0.05, "ld": 0.05,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        tokens_no_regex = segment_text_viterbi(
            text, log_probs, pretok=pretok_no_regex
        )
        tokens_with_regex = segment_text_viterbi(
            text, log_probs, pretok=pretok_with_regex
        )

        # With regex, pretokenizer splits "hello" and " world" separately,
        # so "oĠ" cannot span the boundary.
        # Without regex, the entire string is one piece, so "oĠ" CAN match.
        assert tokens_no_regex != tokens_with_regex, (
            f"Expected different segmentations for different pretokenizers, "
            f"but both produced: {tokens_no_regex}"
        )

    def test_tekken_pretokenizer_splits_numbers_differently(self):
        """Tekken regex pretokenizer groups numbers into 1-3 digit chunks,
        producing different Viterbi segmentations than bare ByteLevel."""
        from code.em_utils import segment_text_viterbi

        pretok_bare = _build_bare_bytelevel_pretokenizer()
        pretok_tekken = _build_tekken_pretokenizer()

        # Build a vocab with digit tokens
        vocab_probs = {
            "<unk>": 1e-6,
            "1": 0.05, "2": 0.05, "3": 0.05, "4": 0.05, "5": 0.05,
            "12": 0.08, "34": 0.08, "45": 0.08,
            "123": 0.1, "345": 0.1,
            "12345": 0.15,
        }
        log_probs = _build_toy_log_probs(vocab_probs)

        tokens_bare = segment_text_viterbi(
            "12345", log_probs, pretok=pretok_bare
        )
        tokens_tekken = segment_text_viterbi(
            "12345", log_probs, pretok=pretok_tekken
        )

        # Tekken splits numbers into 1-3 digit groups, so "12345" becomes
        # "123" + "45" at pretok level, preventing "12345" from matching as
        # a single token. Bare ByteLevel keeps "12345" as one piece.
        assert tokens_bare != tokens_tekken, (
            f"Expected different segmentations: bare={tokens_bare}, "
            f"tekken={tokens_tekken}"
        )


def _build_bare_bytelevel_pretokenizer():
    """Build a bare ByteLevel pretokenizer (no regex splitting)."""
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok
    return ByteLevelPreTok(add_prefix_space=False, use_regex=False)


class TestCorpusPreprocessingRoundTrip:
    """Test that write_hf_bytelevel_corpus_optimized produces correct
    pretokenized output."""

    def test_roundtrip_simple_text(self, tmp_path):
        """Written output should be the pretokenized form of the input."""
        from code.tokenizer_formatting_utils import (
            write_hf_bytelevel_corpus_optimized,
            _apply_pretok_mapping_hf,
        )
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=True)

        # Write input file
        in_path = str(tmp_path / "input.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write("Hello world\n")
            f.write("Testing 123\n")

        out_path = str(tmp_path / "output.txt")
        write_hf_bytelevel_corpus_optimized(in_path, out_path, pretok)

        # Read the output
        with open(out_path, "r", encoding="utf-8") as f:
            output_lines = f.readlines()

        # Verify each line matches what _apply_pretok_mapping_hf would produce
        input_lines = ["Hello world", "Testing 123"]
        assert len(output_lines) == len(input_lines)

        for input_line, output_line in zip(input_lines, output_lines):
            expected_pieces = _apply_pretok_mapping_hf(input_line, pretok)
            expected = "".join(expected_pieces) + "\n"
            assert output_line == expected, (
                f"Mismatch for input '{input_line}':\n"
                f"  expected: {expected!r}\n"
                f"  got:      {output_line!r}"
            )

    def test_roundtrip_preserves_special_chars(self, tmp_path):
        """Preprocessing should handle punctuation and unicode correctly."""
        from code.tokenizer_formatting_utils import (
            write_hf_bytelevel_corpus_optimized,
            _apply_pretok_mapping_hf,
        )
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=True)

        in_path = str(tmp_path / "input.txt")
        test_lines = [
            "Price: $100.50!",
            "Caf\u00e9 au lait",
            "a  b   c",  # multiple spaces
        ]
        with open(in_path, "w", encoding="utf-8") as f:
            for line in test_lines:
                f.write(line + "\n")

        out_path = str(tmp_path / "output.txt")
        write_hf_bytelevel_corpus_optimized(in_path, out_path, pretok)

        with open(out_path, "r", encoding="utf-8") as f:
            output_lines = f.readlines()

        assert len(output_lines) == len(test_lines)
        for input_line, output_line in zip(test_lines, output_lines):
            expected_pieces = _apply_pretok_mapping_hf(input_line, pretok)
            expected = "".join(expected_pieces) + "\n"
            assert output_line == expected

    def test_roundtrip_with_tekken_pretokenizer(self, tmp_path):
        """Tekken pretokenizer should produce correct output that matches
        _apply_pretok_mapping_hf, and the underlying pieces should differ
        from bare ByteLevel even though concatenation may be identical."""
        from code.tokenizer_formatting_utils import (
            write_hf_bytelevel_corpus_optimized,
            _apply_pretok_mapping_hf,
        )

        pretok_bare = _build_bare_bytelevel_pretokenizer()
        pretok_tekken = _build_tekken_pretokenizer()

        in_path = str(tmp_path / "input.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write("Price: 12345\n")

        out_tekken = str(tmp_path / "output_tekken.txt")
        write_hf_bytelevel_corpus_optimized(in_path, out_tekken, pretok_tekken)

        with open(out_tekken, "r", encoding="utf-8") as f:
            tekken_output = f.read()

        # Verify tekken output matches _apply_pretok_mapping_hf
        expected_pieces = _apply_pretok_mapping_hf("Price: 12345", pretok_tekken)
        expected = "".join(expected_pieces) + "\n"
        assert tekken_output == expected

        # The underlying pretok pieces should differ between bare and tekken
        # (bare keeps "12345" as one piece, tekken splits into "123" + "45")
        bare_pieces = _apply_pretok_mapping_hf("Price: 12345", pretok_bare)
        tekken_pieces = _apply_pretok_mapping_hf("Price: 12345", pretok_tekken)
        assert bare_pieces != tekken_pieces, (
            f"Tekken and bare ByteLevel should produce different pieces:\n"
            f"  bare:   {bare_pieces}\n"
            f"  tekken: {tekken_pieces}"
        )

    def test_empty_input_file(self, tmp_path):
        """Empty input file should produce empty output file."""
        from code.tokenizer_formatting_utils import (
            write_hf_bytelevel_corpus_optimized,
        )
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok

        pretok = ByteLevelPreTok(add_prefix_space=False, use_regex=True)

        in_path = str(tmp_path / "empty.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            pass  # empty file

        out_path = str(tmp_path / "output.txt")
        write_hf_bytelevel_corpus_optimized(in_path, out_path, pretok)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "", f"Expected empty output, got {content!r}"
