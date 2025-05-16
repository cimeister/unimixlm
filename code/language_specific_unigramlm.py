"""
Per-language UnigramLM built on a *shared* vocabulary.

Changes w.r.t. the original version
-----------------------------------
* For **every language** we now create a fully-fledged **HF Unigram
  tokenizer** whose scores come from the EM step.  This guarantees that
  encode / decode behave identically to the StandardUnigramLMTokenizer.
* Saved artefacts are `langspec_<em_mode>_<lang>.tokenizer.json`
  (no more *.npy* probability dumps).
* The public API is still the same; code that calls `.train()` or
  `.custom_viterbi_best_language()` needs **no modification**.

Only two external edits are required elsewhere – see § 2 below.
"""


from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace, ByteLevel

from code.em_unigram_trainer import EMUnigramTrainer
from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.utils import _build_hf_tokenizer_from_probs
from code.utils import SPECIAL_TOKENS, MIN_TOKEN_PROB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_LANGTOKENIZER_NAME = (
    "langspec_{em_mode}_{lang_code}.tokenizer.json"  # <- replaces .npy file
)
DEFAULT_BASE_NAME = "langspec_base_tokenizer.json"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_hf_unigram_tokenizer_vocab(tokenizer):
    state = tokenizer.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    return attributes["vocab"]

def _extract_log_scores(tok: Tokenizer) -> Dict[str, float]:
    """
    Return dict {token -> log_score} from a HF Unigram tokenizer.
    """
    vocab_tuples = _get_hf_unigram_tokenizer_vocab(tok)
    return {tok: score for tok, score in vocab_tuples}


def _build_hf_tokenizer_with_new_probs(
    base_tokenizer,
    probs: Dict[str, float],
    unk_token: str
) -> Tokenizer:
    """
    Turn a plain probability dictionary into a HF Unigram tokenizer.
    """
    vocab_tuples = _get_hf_unigram_tokenizer_vocab(base_tokenizer)
    tuples = [
        (t, math.log(max(probs[t], MIN_TOKEN_PROB))) for t, __ in vocab_tuples
    ]

    unk_id = base_tokenizer.get_vocab()[unk_token]
    tok = _build_hf_tokenizer_from_probs(tuples, unk_id)
    if hasattr(base_tokenizer, 'pre_tokenizer'):
        tok.pre_tokenizer = base_tokenizer.pre_tokenizer
    else:
        tok.pre_tokenizer = Whitespace()

    if hasattr(base_tokenizer, 'post_processor'):
        tok.post_processor = base_tokenizer.post_processor
    if hasattr(base_tokenizer, 'decoder'):
        tok.decoder = base_tokenizer.decoder
    if hasattr(base_tokenizer.model, 'byte_fallback'):
        tok.model.byte_fallback = base_tokenizer.model.byte_fallback
    return tok

# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------


class LanguageSpecificUnigramLMTokenizer(StandardUnigramLMTokenizer):
    """
    Adds one *HF* tokenizer per language while inheriting everything else
    from StandardUnigramLMTokenizer.
    """

    # The constructor is unchanged – we just remember how many EM iterations
    # the caller asked for so we can forward the value later.
    def __init__(
        self,
        vocab_size: int = 8_000,
        unk_token: str = SPECIAL_TOKENS["unk_token"],
        special_tokens=None,
        reestimation_em_mode: str = "soft",
        num_iterations: int = 20,
        byte_level: bool = False
    ):
        super().__init__(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            em_mode=reestimation_em_mode,
            num_iterations=num_iterations,
            byte_level=byte_level
        )

        self.reestimation_em_mode = reestimation_em_mode
        self.num_reestimation_iterations = num_iterations
        # lang_code -> {"tokenizer": HF Tokenizer, "scores": {tok->log_p}, "path": str}
        self.language_info: Dict[str, dict] = {}

    def load(self, base_path, language_paths=None):
        """
        Load a previously saved base tokenizer plus any saved EM distributions for languages.

        Args:
            base_path: path to base tokenizer .json
            language_paths: dict { lang_code: path_to_em_probs.npy }
        Returns:
            { 'base': self.base_tokenizer, 'languages': [lang_codes] }
        """
        # Load the base tokenizer using the parent class method
        super().load(base_path)
        vocab_tuples = _get_hf_unigram_tokenizer_vocab(self.base_tokenizer)
        base_vocab_order = [i for i, j in vocab_tuples]
        
        # Initialize language info
        self.language_info = {}

        # Load language-specific probability distributions
        if language_paths:
            for lang_code, lang_tok_path in language_paths.items():
                if os.path.isfile(lang_tok_path):
                    tok = Tokenizer.from_file(lang_tok_path)
                    vocab_order_per_lang = _get_hf_unigram_tokenizer_vocab(tok)
                    assert [i for i, j in vocab_order_per_lang] == base_vocab_order
                    self.language_info[lang_code] = {
                        "tokenizer": tok,
                        "scores": _extract_log_scores(tok),
                        "path": lang_tok_path,
                    } 
                    logger.info(f"Loaded per-lang tokenizer for {lang_code} from {lang_tok_path}")
                else:
                    logger.warning(f"No per-lang tokenizer file found at {lang_tok_path} for {lang_code}")

    # ------------------------------------------------------------------ train

    def train(
        self,
        corpus_info: Dict[str, str],
        base_tokenizer_path: str,
        output_dir: str = "",
        tokenizer_path_format: str | None = None,
        load_base_if_exists: bool = True,
        load_tokenizers_if_exists: bool = True,
    ):
        """
        Build / load the *shared* base tokenizer via the parent class,
        then obtain a HF Unigram tokenizer for every language.

        `corpus_info`  …… {lang_code: path_to_train_file}
        """
        output_dir = Path(output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        if load_base_if_exists and os.path.isfile(base_tokenizer_path):
            logger.info(f"Loading specified base tokenizer from {base_tokenizer_path}")
            super().load(base_tokenizer_path)
            logger.warning(f"Will use byte-level settings from {base_tokenizer_path} regardless of user input: byte-level={self.byte_level}")
        
        training_new_base_tokenizer = not self.base_tokenizer
        
        # Gather train files for all languages
        all_train_files = []
        for lang_code, f in corpus_info.items():
            all_train_files.append(f)
        
        if not all_train_files:
            logger.error("No training files found for any language in corpus_info.")
            return None

        # Train base tokenizer if needed (using parent class method)
        if training_new_base_tokenizer:
            logger.info("Training base Unigram tokenizer for shared vocab.")
            # Use parent class's train method for the base tokenizer
            self.base_tokenizer = super().train(all_train_files, output_path=base_tokenizer_path, load_if_exists=False)
        else:
            logger.info("Using previously loaded base tokenizer.")

        # 2) one tokenizer per language
        for lang_code, train_file in corpus_info.items():
            # where to store / look for the tokenizer
            tk_path: Path = (
                Path(tokenizer_path_format.format(lang_code=lang_code))
                if tokenizer_path_format is not None
                else output_dir
                / DEFAULT_LANGTOKENIZER_NAME.format(
                    em_mode=self.reestimation_em_mode, lang_code=lang_code
                )
            )

            if load_tokenizers_if_exists and tk_path.is_file():
                tok = Tokenizer.from_file(str(tk_path))
                self.language_info[lang_code] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": str(tk_path),
                }
                logger.info(f"Loaded tokenizer for {lang_code} from {tk_path}")
                continue

            # ----------  EM training on the language corpus  ----------
            em_trainer = EMUnigramTrainer(
                vocab=self.base_tokenizer.get_vocab(),
                unk_token=self.unk_token,
                special_tokens=self.special_tokens,
                em_mode=self.reestimation_em_mode,
                max_iterations=self.num_reestimation_iterations,
                unk_prior=self.unk_prior,
                byte_level=self.byte_level
            )
            token_probs = em_trainer.train(train_file)  # token -> prob

            # ----------  HF tokenizer from these probabilities  ----------
            tok = _build_hf_tokenizer_with_new_probs(self.base_tokenizer, token_probs, self.unk_token)
            tok.save(str(tk_path))

            self.language_info[lang_code] = {
                "tokenizer": tok,
                "scores": _extract_log_scores(tok),
                "path": str(tk_path),
            }
            logger.info(f"Saved tokenizer for {lang_code} to {tk_path}")
        return {
            "base_path": base_tokenizer_path,
            "language_paths": {k: v['path'] for k, v in self.language_info.items()}
        }

    # ---------------------------------------------------------------- encode / decode helpers

    def encode_lang(self, text: str, lang_code: str):
        if lang_code not in self.language_info:
            raise ValueError(f"No tokenizer for language '{lang_code}'.")
        return self.language_info[lang_code]["tokenizer"].encode(text)

    def decode_lang(self, ids: List[int], lang_code: str):
        if lang_code not in self.language_info:
            raise ValueError(f"No tokenizer for language '{lang_code}'.")
        return self.language_info[lang_code]["tokenizer"].decode(ids)

    # ------------------------------------------------ best-language selection

    def best_language_encode(self, text: str):
        """
        Segment `text` with *each* language model and return the one that
        yields the highest log-probability.

        Returns
        -------
        (best_lang_code, tokens_as_list_of_str)
        """
        best_lang, best_tokens, best_logp = None, [], float("-inf")

        for lang_code, info in self.language_info.items():
            enc = info["tokenizer"].encode(text)
            toks = enc.tokens
            logp = sum(info["scores"].get(t, math.log(MIN_TOKEN_PROB)) for t in toks)

            if logp > best_logp:
                best_lang, best_tokens, best_logp = lang_code, toks, logp

        return best_lang, best_tokens

    def best_language_encode_batch(self, texts):
        """
        For each input text return
            best_lang, Encoding           # Encoding from HF tokenizers lib
        This avoids re-encoding after we have already picked the language.
        """
        if not self.language_info:
            raise ValueError("No per-language tokenizers loaded.")

        # First pass: encode each text with EVERY language and keep the log-prob
        best_langs  = [None] * len(texts)
        best_scores = [float("-inf")] * len(texts)
        best_encs   = [None] * len(texts)

        for lang_code, info in self.language_info.items():
            encs = info["tokenizer"].encode_batch(texts)
            for i, enc in enumerate(encs):
                score = sum(info["scores"].get(t, math.log(MIN_TOKEN_PROB)) for t in enc.tokens)
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_langs[i]  = lang_code
                    best_encs[i]   = enc

        return best_langs, best_encs


def train_language_specific_tokenizer(corpus_info, vocab_size=8000,
                                      reestimation_em_mode = 'soft',
                                      output_dir="",
                                      byte_level=False,
                                      base_tokenizer_path=None,
                                      tokenizer_path_format=None,
                                      load_base_if_exists=True,
                                      load_tokenizers_if_exists=False):
    """
    Convenience function that:
      1) Creates a LanguageSpecificUnigramLMTokenizer
      2) Calls .train(...) with the given corpus info
      3) Returns a dict with paths: { 'base': <base_path>, 'languages': {lang_code: <em_probs_path>} }
         so you can load them later if you want.

    You can skip re-training if 'langspec_base_tokenizer.json' is found (load_if_exists=True),
    and skip re-running EM if 'langspec_{lang_code}_em_probs.npy' is found (load_available_probs=True).
    """
    tok = LanguageSpecificUnigramLMTokenizer(vocab_size=vocab_size, reestimation_em_mode=reestimation_em_mode, byte_level=byte_level)
    if not base_tokenizer_path:
        base_tokenizer_path = os.path.join(output_dir, DEFAULT_BASE_NAME)
        
    if not tokenizer_path_format:
        tokenizer_path_format = os.path.join(output_dir, DEFAULT_LANGTOKENIZER_NAME)

    tokenizer_paths = tok.train(
        corpus_info,
        base_tokenizer_path=base_tokenizer_path,
        tokenizer_path_format=tokenizer_path_format,
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists
    )
    if not tokenizer_paths:
        logger.error("Failed to train or load language-specific tokenizer.")
        return None

    return tokenizer_paths