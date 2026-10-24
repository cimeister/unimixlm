"""Patched copy of the MorphScore harness (offsets-based boundaries).

Identical to ~/tokenizer-intrinsic-evals/morphscore/morphscore.py except for the
per-token boundary / token-string derivation, which is taken from
tokenizers.Encoding.offsets instead of from per-id decoding. See
morphscore_patched.py for the exact diff.
"""

from .morphscore_patched import MorphScore

__version__ = "0.1.0-offsets-patched"

__all__ = ["MorphScore"]
