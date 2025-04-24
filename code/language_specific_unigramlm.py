import os
import logging
import math
import json
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer

from code.em_unigram_trainer import EMUnigramTrainer
from code.utils import segment_text_viterbi, SPECIAL_TOKENS
from code.standard_unigramlm import StandardUnigramLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_NAME = "langspec_base_tokenizer.json"
DEFAULT_PROB_FILES_NAME = "langspec_{em_mode}_{lang_code}_em_probs.npy"

from code.language_selection import LANGUAGES


class LanguageSpecificUnigramLMTokenizer(StandardUnigramLMTokenizer):
    """
    Extends StandardUnigramLMTokenizer to handle language-specific models.
    
    Maintains:
      1) A single base tokenizer for shared vocab (trained or loaded) - inherited from parent.
      2) Per-language EM distributions (token->prob).
      3) Methods to do "true" Viterbi segmentation under the EM distribution:
         - custom_viterbi_tokenize(text, lang_code)
         - custom_viterbi_best_language(text)

    The base HF tokenizer is used for:
      - Vocab ID mapping
      - Possibly for decode (ID->text)
      - Not for actual segmentation, since we do a separate Viterbi approach.
    """

    def __init__(self, vocab_size=8000, 
                 unk_token=SPECIAL_TOKENS['unk_token'], 
                 special_tokens=None,
                 reestimation_em_mode="hard",
                 num_iterations=20,
                 max_sub_len=15):
        
        # Initialize the parent class with the base tokenizer settings
        super().__init__(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            em_mode="soft",
            max_sub_len=max_sub_len
        )
        self.num_reestimation_iterations = num_iterations
        self.reestimation_em_mode = reestimation_em_mode
        # Add language-specific attributes
        self.language_info = {}

    def train(self, corpus_info, 
              base_tokenizer_path,
              probability_path_format,
              load_base_if_exists=True,
              load_probs_if_exists=True):
        """
        Train (or load) the base tokenizer on the combined corpus,
        then run (or load) per-language EM distributions.

        Args:
            corpus_info: dict: {lang_code: {"splits": {"train": ...}, ...}, ...}
            base_tokenizer_path: path to save/load the base tokenizer
            probability_path_format: format string for per-language probability files
            load_base_if_exists: if True, try to load the base tokenizer from disk
            load_probs_if_exists: if True, try to load per-language EM .npy if it exists
        Returns:
            { 'base': self.base_tokenizer, 'languages': [lang_codes used] }
        """
        # First, load existing base tokenizer if specified
        if load_base_if_exists and os.path.isfile(base_tokenizer_path):
            logger.info(f"Loading specified base tokenizer from {base_tokenizer_path}")
            self.base_tokenizer = Tokenizer.from_file(base_tokenizer_path)
        
        training_new_base_tokenizer = not self.base_tokenizer
        
        # Gather train files for all languages
        all_train_files = []
        for lang_code, info in corpus_info.items():
            if 'splits' in info and 'train' in info['splits']:
                all_train_files.append(info['splits']['train'])
        
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

        # Next, train per-language EM distributions
        base_vocab = self.base_tokenizer.get_vocab()  # token->id
        language_paths = {}
        
        for lang_code, info in corpus_info.items():
            if 'splits' not in info or 'train' not in info['splits']:
                logger.warning(f"No 'train' split for {lang_code}, skipping.")
                continue

            train_file = info['splits']['train']
            logger.info(f"Processing language {lang_code} with train file: {train_file}")

            # Try to load existing probability distributions
            token_probs = None
            np_path = probability_path_format.format(lang_code=lang_code, em_mode=self.reestimation_em_mode)
            if load_probs_if_exists:
                if os.path.isfile(np_path):
                    logger.info(f"Loading existing EM distribution from {np_path}")
                    arr = np.load(np_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        arr = arr.item()
                    token_probs = arr

            # If no saved distribution, train a new one
            if token_probs is None:
                logger.info(f"Running EM trainer for {lang_code} (no saved file found/used).")

                # Check resource level
                is_low_resource = False
                if lang_code in LANGUAGES and LANGUAGES[lang_code].get('resource_level') == 'low':
                    is_low_resource = True

                em_trainer = EMUnigramTrainer(
                    vocab=base_vocab,
                    unk_token=self.unk_token,
                    special_tokens=self.special_tokens,
                    em_mode=self.reestimation_em_mode,
                    max_iterations = self.num_reestimation_iterations
                )
                token_probs = em_trainer.train(train_file, is_low_resource=is_low_resource)
                np.save(np_path, token_probs)
                logger.info(f"Saved EM distribution for {lang_code} to {np_path}")
            else:
                logger.info(f"Already have EM distribution for {lang_code}, skipping training.")
            
            # Store language-specific information
            self.language_info[lang_code] = {
                "token_probs": token_probs
            }
            language_paths[lang_code] = np_path

        # Finalize the vocabulary indices
        self.finalize_vocab_indices()
        if training_new_base_tokenizer:
            self.base_tokenizer.save(base_tokenizer_path)
            logger.info(f"Replacing base tokenizer file with newly indexed tokenizer info at {base_tokenizer_path}")

        logger.info("Finished training (or loading) base + per-language distributions.")
        return {
            'base': base_tokenizer_path,
            'languages': language_paths
        }

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
        
        # Initialize language info
        self.language_info = {}

        # Load language-specific probability distributions
        if language_paths:
            for lang_code, prob_path in language_paths.items():
                if os.path.isfile(prob_path):
                    arr = np.load(prob_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        arr = arr.item()
                    self.language_info[lang_code] = {
                        "token_probs": arr
                    }
                    logger.info(f"Loaded EM distribution for {lang_code} from {prob_path}")
                else:
                    logger.warning(f"No EM prob file found at {prob_path} for {lang_code}")

        return {
            'base': self.base_tokenizer,
            'languages': list(self.language_info.keys())
        }

    def custom_viterbi_tokenize(self, text, lang_code):
        """
        Segment `text` via the language's per-token distribution (EM results).
        """
        if lang_code not in self.language_info:
            logger.warning(f"No language info for {lang_code}; fallback to base tokens.")
            return self.base_tokenizer.encode(text).tokens

        token_probs = self.language_info[lang_code]["token_probs"]
        tokens = segment_text_viterbi(text, token_probs, unk_token=self.unk_token)
        return tokens

    def custom_viterbi_best_language(self, text):
        """
        Try each language's distribution, do full Viterbi segmentation, sum log-probs,
        pick best. Returns (best_lang, tokens).
        """
        best_lang = None
        best_tokens = []
        best_logp = float('-inf')

        for lang_code, info in self.language_info.items():
            token_probs = info["token_probs"]
            candidate_tokens = segment_text_viterbi(text, token_probs, unk_token=self.unk_token)
            # sum log-probs
            logp = 0.0
            for t in candidate_tokens:
                p = token_probs.get(t, 1e-12)
                logp += math.log(p)

            if logp > best_logp:
                best_logp = logp
                best_tokens = candidate_tokens
                best_lang = lang_code

        return best_lang, best_tokens

    def get_token_probs(self, lang_code):
        """Return the dictionary of token->prob for a given language, if it exists."""
        if lang_code not in self.language_info:
            return None
        return self.language_info[lang_code]["token_probs"]


def train_language_specific_tokenizer(corpus_info, vocab_size=8000,
                                      reestimation_em_mode = 'soft',
                                      output_dir="",
                                      base_tokenizer_path=None,
                                      probability_path_format=None,
                                      load_base_if_exists=True,
                                      load_probs_if_exists=False):
    """
    Convenience function that:
      1) Creates a LanguageSpecificUnigramLMTokenizer
      2) Calls .train(...) with the given corpus info
      3) Returns a dict with paths: { 'base': <base_path>, 'languages': {lang_code: <em_probs_path>} }
         so you can load them later if you want.

    You can skip re-training if 'langspec_base_tokenizer.json' is found (load_if_exists=True),
    and skip re-running EM if 'langspec_{lang_code}_em_probs.npy' is found (load_available_probs=True).
    """
    tok = LanguageSpecificUnigramLMTokenizer(vocab_size=vocab_size, reestimation_em_mode=reestimation_em_mode)
    if not base_tokenizer_path:
        base_tokenizer_path = os.path.join(output_dir, DEFAULT_BASE_NAME)
    if not probability_path_format:
        probability_path_format = os.path.join(output_dir, DEFAULT_PROB_FILES_NAME)

    tokenizer_paths = tok.train(
        corpus_info,
        base_tokenizer_path=base_tokenizer_path,
        probability_path_format=probability_path_format,
        load_base_if_exists=load_base_if_exists,
        load_probs_if_exists=load_probs_if_exists
    )
    if not tokenizer_paths:
        logger.error("Failed to train or load language-specific tokenizer.")
        return None

    return tokenizer_paths
