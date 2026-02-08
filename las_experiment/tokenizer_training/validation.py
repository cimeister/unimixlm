"""
Validation utilities for LAS tokenizer training.

Ensures trained tokenizers maintain consistency with the base model tokenizer
in terms of pre/post-tokenization components and token ordering.
"""

import json
import logging
from pathlib import Path


def _load_tokenizer_json(tokenizer_path: Path) -> dict:
    """Load tokenizer JSON file as a dictionary."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_tokens_from_tokenizer_data(tokenizer_data: dict) -> list[str]:
    """
    Extract ordered list of tokens from tokenizer JSON.

    Handles both Unigram (list) and BPE (dict) vocab formats.
    For dict format, tokens are sorted by their ID values to preserve correct order.

    This function reuses the format detection and extraction logic from
    _load_vocab_from_hf_tokenizer in code.tokenizer_formatting_utils, but
    operates on already-loaded tokenizer data and adds ID-based sorting for
    BPE tokenizers (using the pattern from _get_hf_unigram_tokenizer_vocab).

    Args:
        tokenizer_data: Tokenizer JSON as dictionary

    Returns:
        List of tokens in order
    """
    # Format detection and extraction logic from _load_vocab_from_hf_tokenizer
    vocab = tokenizer_data.get("model", {}).get("vocab")
    if vocab is None:
        raise ValueError("Tokenizer data missing 'model.vocab' field")

    if isinstance(vocab, list):
        # Unigram format: [["token", score], ...]
        tokens = [token for token, _score in vocab]
    elif isinstance(vocab, dict):
        # BPE format: {"token": id, ...}
        # Sort by ID value to preserve token order
        # (sorting pattern from _get_hf_unigram_tokenizer_vocab)
        sorted_items = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [token for token, _id in sorted_items]
    else:
        raise ValueError(f"Unsupported vocab format: {type(vocab)}")

    return tokens


def _validate_pretokenizer(
    tokenizer_data: dict,
    expected_pretokenizer: dict,
    tokenizer_name: str,
) -> None:
    """
    Validate that a tokenizer's pre-tokenizer matches the expected one.

    Args:
        tokenizer_data: Tokenizer JSON as dictionary
        expected_pretokenizer: Expected pre_tokenizer field
        tokenizer_name: Name for logging (e.g., "base", "en")

    Raises:
        AssertionError if pre-tokenizers don't match
    """
    actual_pretokenizer = tokenizer_data.get("pre_tokenizer")
    assert actual_pretokenizer is not None, (
        f"{tokenizer_name} tokenizer missing 'pre_tokenizer' field"
    )

    assert actual_pretokenizer == expected_pretokenizer, (
        f"{tokenizer_name} tokenizer pre-tokenizer does not match model tokenizer.\n"
        f"Expected: {expected_pretokenizer}\n"
        f"Got: {actual_pretokenizer}"
    )


def _validate_decoder(
    tokenizer_data: dict,
    expected_decoder: dict,
    tokenizer_name: str,
) -> None:
    """
    Validate that a tokenizer's decoder (post-tokenizer) matches the expected one.

    Args:
        tokenizer_data: Tokenizer JSON as dictionary
        expected_decoder: Expected decoder field
        tokenizer_name: Name for logging (e.g., "base", "en")

    Raises:
        AssertionError if decoders don't match
    """
    actual_decoder = tokenizer_data.get("decoder")
    assert actual_decoder is not None, (
        f"{tokenizer_name} tokenizer missing 'decoder' field"
    )

    assert actual_decoder == expected_decoder, (
        f"{tokenizer_name} tokenizer post-tokenizer does not match model tokenizer.\n"
        f"Expected: {expected_decoder}\n"
        f"Got: {actual_decoder}"
    )


def _validate_token_order(
    tokenizer_data: dict,
    expected_tokens: list[str],
    tokenizer_name: str,
) -> None:
    """
    Validate that a tokenizer's token ordering matches the expected order.

    Args:
        tokenizer_data: Tokenizer JSON as dictionary
        expected_tokens: List of tokens in expected order
        tokenizer_name: Name for logging (e.g., "base", "en")

    Raises:
        AssertionError if token order doesn't match
    """
    actual_tokens = _get_tokens_from_tokenizer_data(tokenizer_data)

    assert actual_tokens == expected_tokens, (
        f"{tokenizer_name} tokenizer token order does not match model tokenizer. "
        f"Expected {len(expected_tokens)} tokens, got {len(actual_tokens)} tokens"
    )
    logging.info(f"✓ {tokenizer_name}: token order matches model tokenizer")


def _validate_components(
    tokenizer_data: dict,
    expected_pretokenizer: dict,
    expected_decoder: dict,
    tokenizer_name: str,
) -> None:
    """
    Validate both pre-tokenizer and decoder components of a tokenizer.

    Args:
        tokenizer_data: Tokenizer JSON as dictionary
        expected_pretokenizer: Expected pre_tokenizer field
        expected_decoder: Expected decoder field
        tokenizer_name: Name for logging (e.g., "base", "en")

    Raises:
        AssertionError if any component doesn't match
    """
    _validate_pretokenizer(tokenizer_data, expected_pretokenizer, tokenizer_name)
    _validate_decoder(tokenizer_data, expected_decoder, tokenizer_name)
    logging.info(f"✓ {tokenizer_name}: pre/post-tokenizer match model tokenizer")


def validate_tokenizers(
    model_tokenizer_path: Path,
    base_tokenizer_path: Path,
    lang_tokenizer_paths: dict[str, Path],
) -> None:
    """
    Validate that all trained tokenizers have:
    1. Identical pre-tokenizer and post-tokenizer components
    2. Token order matching the original model tokenizer

    Compares actual JSON file contents rather than object equality.

    Raises AssertionError if any check fails.
    """
    logging.info("Validating tokenizers...")

    # Load model tokenizer JSON
    model_data = _load_tokenizer_json(model_tokenizer_path)
    model_tokens = _get_tokens_from_tokenizer_data(model_data)
    expected_pretokenizer = model_data.get("pre_tokenizer")
    expected_decoder = model_data.get("decoder")

    # Load and validate base tokenizer
    base_data = _load_tokenizer_json(base_tokenizer_path)
    _validate_components(base_data, expected_pretokenizer, expected_decoder, "base")
    _validate_token_order(base_data, model_tokens, "base")

    # Validate language-specific tokenizers
    for lang, lang_path in lang_tokenizer_paths.items():
        assert lang_path.exists(), f"Language tokenizer not found: {lang_path}"

        lang_data = _load_tokenizer_json(lang_path)
        _validate_components(lang_data, expected_pretokenizer, expected_decoder, lang)
        _validate_token_order(lang_data, model_tokens, lang)

    logging.info("✓ All validation checks passed!")
