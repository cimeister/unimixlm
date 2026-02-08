"""
Validation utilities for LAS tokenizer training.

Ensures trained tokenizers maintain consistency with the base model tokenizer
in terms of pre/post-tokenization components and token ordering.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer
from code.tokenizer_formatting_utils import _get_hf_unigram_tokenizer_vocab


def _load_tokenizer_json(tokenizer_path: Path) -> dict:
    """Load tokenizer JSON file as a dictionary."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_token_list_from_tokenizer(tokenizer_path: Path) -> list[str]:
    """
    Extract ordered list of tokens from a tokenizer file.

    Uses _get_hf_unigram_tokenizer_vocab to handle both BPE (dict) and
    Unigram (list) vocabulary formats uniformly.

    Args:
        tokenizer_path: Path to tokenizer JSON file

    Returns:
        List of tokens in order
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_with_scores = _get_hf_unigram_tokenizer_vocab(tokenizer)
    assert len(vocab_with_scores) > 0, f"Tokenizer vocab is empty: {tokenizer_path}"
    return [token for token, _score in vocab_with_scores]


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
    tokenizer_path: Path,
    expected_tokens: list[str],
    tokenizer_name: str,
) -> None:
    """
    Validate that a tokenizer's token ordering matches the expected order.

    Args:
        tokenizer_path: Path to tokenizer JSON file
        expected_tokens: List of tokens in expected order
        tokenizer_name: Name for logging (e.g., "base", "en")

    Raises:
        AssertionError if token order doesn't match
    """
    actual_tokens = _get_token_list_from_tokenizer(tokenizer_path)

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
    lang_tokenizer_paths: Optional[dict[str, Path]] = {},
) -> None:
    """
    Validate that all trained tokenizers have:
    1. Identical pre-tokenizer and post-tokenizer components
    2. Token order matching the original model tokenizer

    Uses _get_hf_unigram_tokenizer_vocab to handle both BPE and Unigram formats.
    Compares actual JSON file contents rather than object equality.

    Raises AssertionError if any check fails.
    """
    logging.info("Validating tokenizers...")

    # Load model tokenizer tokens (handles both BPE and Unigram formats)
    model_tokens = _get_token_list_from_tokenizer(model_tokenizer_path)

    # Load model tokenizer JSON for pre/post-tokenizer comparison
    model_data = _load_tokenizer_json(model_tokenizer_path)
    expected_pretokenizer = model_data.get("pre_tokenizer")
    expected_decoder = model_data.get("decoder")

    # Load and validate base tokenizer
    base_data = _load_tokenizer_json(base_tokenizer_path)
    _validate_components(base_data, expected_pretokenizer, expected_decoder, "base")
    _validate_token_order(base_tokenizer_path, model_tokens, "base")

    # Validate language-specific tokenizers
    for lang, lang_path in lang_tokenizer_paths.items():
        assert lang_path.exists(), f"Language tokenizer not found: {lang_path}"

        lang_data = _load_tokenizer_json(lang_path)
        _validate_components(lang_data, expected_pretokenizer, expected_decoder, lang)
        _validate_token_order(lang_path, model_tokens, lang)

    logging.info("✓ All validation checks passed!")
