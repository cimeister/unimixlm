import os
import logging
import json
import math
import multiprocessing
import numpy as np

from collections import defaultdict
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from code.utils import _accumulate_usage, get_baseline_characters
from code.utils import SPECIAL_TOKENS, INITIAL_VOCAB_MULT_FACTOR, NUM_TRIMMING_ITERATIONS, MIN_CHAR_OCCURENCE_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandardUnigramLMTokenizer:
    """
    A 'standard' UnigramLM tokenizer that can be trained in various modes:
      1) HF mode ("em_mode='hf'") => use the built-in HuggingFace UnigramTrainer
      2) Hard EM ("em_mode='hard'") => single-language custom approach with Viterbi E-step
      3) Soft EM ("em_mode='soft'") => single-language custom approach with forward-backward E-step
      4) Multigram ("multigram=True") => per-language EM with mixture combining
    """
    def __init__(
        self, 
        vocab_size=8000, 
        unk_token=SPECIAL_TOKENS['unk_token'], 
        special_tokens=None,
        em_mode=None,            # "hf", "hard", or "soft"
        num_iterations=10,       # If em_mode != "hf", how many EM iterations
        max_sub_len=15,          # for forward-backward, to speed up
        multigram=False,         # Whether to use per-language EM with mixture combining
        mixture_weights=None     # Used only if multigram=True
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        if special_tokens is None:
            special_tokens = list(SPECIAL_TOKENS.values())
        self.special_tokens = special_tokens

        self.base_tokenizer = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.max_sub_len = max_sub_len
        self.multigram = multigram
        self.mixture_weights = mixture_weights
        
        # Additional attributes for multigram approach
        self.token_probs = None      # final distribution (multigram or otherwise)
        self.per_lang_probs = {}     # per-language distributions (multigram only)
        self.lang_codes = []         # language codes (multigram only)
    
    def load(self, base_path):
        """
        Load a saved tokenizer and optionally, global probabilities.
        
        Args:
            base_path: Path to the tokenizer file
        """
        from tokenizers import Tokenizer
        self.base_tokenizer = Tokenizer.from_file(base_path)

    def train(self, corpus_files, output_path=None, load_if_exists=False, corpus_info=None):
        """
        Train a Unigram tokenizer using the specified approach.
        
        Args:
            corpus_files: List of corpus files to train on
            output_path: Path to save the tokenizer
            load_if_exists: Whether to load existing files if present
            corpus_info: Dict mapping language codes to corpus info (required for multigram)
            
        Returns:
            The trained tokenizer
        """
        if load_if_exists and output_path and os.path.isfile(output_path):
            logger.info(f"Loading existing tokenizer from {output_path}")
            self.base_tokenizer = Tokenizer.from_file(output_path)
                        
            return self.base_tokenizer

        # Different training approaches
        if self.em_mode == "hf" and not self.multigram:
            # ============= HF Approach =============
            return self._train_hf(corpus_files, output_path)
            
        elif self.multigram:
            # ============= Multigram Approach =============
            if not corpus_info:
                raise ValueError("corpus_info is required for multigram training")
            return self._train_multigram(corpus_info, output_path)
            
        elif self.em_mode in ["hard", "soft"]:
            # ============= Custom EM Approach (hard/soft) =============
            return self._train_custom_em(corpus_files, output_path)
        else:
            raise ValueError("Invalid settings for StandardUnigramLMTokenizer: no EM mode specified")

    def _train_hf(self, corpus_files, output_path=None):
        """
        Train using the HuggingFace UnigramTrainer.
        """
        logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = Whitespace()

        trainer = UnigramTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token
        )
        tok.train(corpus_files, trainer)
        self.base_tokenizer = tok

        if output_path:
            self.base_tokenizer.save(output_path)
            logger.info(f"Saved HF-based standard tokenizer to {output_path}")
        return self.base_tokenizer

    def _train_custom_em(self, corpus_files, output_path=None):
        """
        Train using custom EM algorithm (hard or soft).
        """

        logger.info("Getting baseline char set...")
        chars_to_keep, char_freq = get_baseline_characters(corpus_files, MIN_CHAR_OCCURENCE_THRESHOLD)
        logger.info(f"{len(char_freq)} unique characters found in dataset. Attempting to keep all with "
            f"at least {MIN_CHAR_OCCURENCE_THRESHOLD} occurrences, which is {len(chars_to_keep)} characters.")
        
        # 1) Build an initial big Unigram model
        logger.info(f"Custom EM approach ({self.em_mode}). Building an initial oversized vocab with HF tools...")
        init_tok = Tokenizer(Unigram())
        init_tok.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(
            vocab_size=self.vocab_size * INITIAL_VOCAB_MULT_FACTOR,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token
        )
        init_tok.train(corpus_files, trainer)
        vocab_dict = init_tok.get_vocab()
        all_tokens = set(vocab_dict.keys())
        all_tokens.update(chars_to_keep)
        all_tokens.update(self.special_tokens)

        # uniform distribution initially, making UNK token smaller
        p_token = {}
        for tk in all_tokens:
            p_token[tk] = 1.0 / len(all_tokens)
        
        # 2) EM loop
        num_tokens_to_trim = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR - self.vocab_size
        tokens_to_remove_per_loop = [math.floor(num_tokens_to_trim/NUM_TRIMMING_ITERATIONS)] * NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - NUM_TRIMMING_ITERATIONS)
        keys_to_keep = set(self.special_tokens + list(chars_to_keep))

        def _order_token_list_by_prob(_p_token):
            tokens_to_keep = []
            remaining_items = []
            for tk, pr in _p_token.items():
                if tk in keys_to_keep:
                    tokens_to_keep.append((tk, pr))
                else:
                    remaining_items.append((tk, pr))
            # Get the rest of the tuples that are not in items_to_move.
            return tokens_to_keep + sorted(remaining_items, key=lambda x: x[1], reverse=True)

        for it in range(self.num_iterations):
            logger.info(f"EM iteration {it+1}/{self.num_iterations} for single-language ({self.em_mode})")

            usage_counts = {tk: 0.0 for tk in all_tokens}
            total_subwords = 0.0

            # keep UNK prob small
            if self.unk_token in p_token:
                p_token[self.unk_token] = 1.0 / len(all_tokens) 
            logger.info(f"Resetting UNK prob to {p_token[self.unk_token]}")

            # For each corpus file
            for fpath in corpus_files:
                c_dict, tot = _accumulate_usage(fpath, p_token, self.unk_token, mode=self.em_mode)

                for tk, val in c_dict.items():
                    usage_counts[tk] += val
                total_subwords += tot

            # M-step
            # prune if needed
            if tokens_to_remove_per_loop[it] > 0:                    
                sorted_final = _order_token_list_by_prob(p_token)
                all_tokens = set([tk for tk, sc in sorted_final[:-tokens_to_remove_per_loop[it]]])
                logger.info(f"Current vocab size: {len(all_tokens)}")

            sum_counts = sum([usage_counts[t] for t in all_tokens])
            p_token = {}
            if sum_counts > 0:
                for tk in all_tokens:
                    p_token[tk] = usage_counts[tk] / sum_counts
            else:
                logger.warning("No usage => fallback uniform distribution.")
                for tk in all_tokens:
                    p_token[tk] = 1.0 / len(all_tokens)

            logger.info(f"Current UNK prob: {p_token[self.unk_token]}")
            
        # Finalize the token list after EM
        if len(p_token) > self.vocab_size:
            sorted_final = _order_token_list_by_prob(p_token)
            sorted_final = sorted_final[:self.vocab_size]
            remaining_prob = sum([pr for tk, pr in sorted_final])
            sorted_final = [(tk, pr/remaining_prob) for tk, pr in sorted_final]
        else:
            sorted_final = sorted(p_token.items(), key=lambda x: x[1], reverse=True)

        total_prob_coverage = sum([p for tk, p in sorted_final])
        logger.info(f"Token probabilities sum to {total_prob_coverage}")
        
        # build final Unigram model, keeping special tokens at beginning
        vocab_list = []
        special_tokens = []
        for tk, prob in sorted_final:
            score = math.log(max(prob, 1e-12))
            if tk in self.special_tokens:
                special_tokens.append((tk, score))
            else:
                vocab_list.append((tk, score))
        vocab_list = special_tokens + vocab_list

        logger.info(f"Custom EM done. final vocab size = {len(vocab_list)}")
        
        # figure out unk_id
        unk_id = 0
        for i, (tk, sc) in enumerate(vocab_list):
            if tk == self.unk_token:
                logger.info(f"UNK score: {sc}")
                unk_id = i
                break

        final_tok = Tokenizer(Unigram(tuple(vocab_list), unk_id))
        final_tok.pre_tokenizer = Whitespace()
        self.base_tokenizer = final_tok
        
        # Save the final token probabilities
        self.token_probs = {tk: math.exp(score) for tk, score in vocab_list}
        
        vocab = self.base_tokenizer.get_vocab()
        id_check = {k: vocab[k] for k in self.special_tokens}
        logger.info(f"Special tokens: {id_check}")

        if output_path:
            self.base_tokenizer.save(output_path)
            logger.info(f"Saved custom-EM standard tokenizer to {output_path}")

        return self.base_tokenizer

    def _train_multigram(self, corpus_info, output_path=None):
        """
        Train using the multigram approach (per-language distributions with mixture combining).
        This follows the same structure as _train_custom_em, but estimates per-language
        distributions in each iteration before combining them.
        
        Args:
            corpus_info: Dict mapping language codes to corpus info
            output_path: Path to save the tokenizer
        """        
        # Gather train files and language codes
        train_files = []
        self.lang_codes = []
        for lang_code, info in corpus_info.items():
            if 'splits' in info and 'train' in info['splits']:
                train_files.append(info['splits']['train'])
                self.lang_codes.append(lang_code)

        if not train_files:
            logger.error("No train files found for any language.")
            return None
        
        # Get baseline characters (identical to _train_custom_em)
        logger.info("Getting baseline char set...")
        chars_to_keep, char_freq = get_baseline_characters(train_files, MIN_CHAR_OCCURENCE_THRESHOLD)
        logger.info(f"{len(char_freq)} unique characters found in dataset. Attempting to keep all with "
            f"at least {MIN_CHAR_OCCURENCE_THRESHOLD} occurrences, which is {len(chars_to_keep)} characters.")

        # 1) Build an initial big Unigram model (identical to _train_custom_em)
        logger.info(f"Multigram approach with {self.em_mode} EM. Building an initial oversized vocab with HF tools...")
        init_tok = Tokenizer(Unigram())
        init_tok.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(
            vocab_size=self.vocab_size * INITIAL_VOCAB_MULT_FACTOR,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token
        )
        init_tok.train(train_files, trainer)
        vocab_dict = init_tok.get_vocab()
        all_tokens = set(vocab_dict.keys())
        all_tokens.update(chars_to_keep)
        all_tokens.update(self.special_tokens)

        # Initialize mixture weights if not provided (specific to multigram)
        if self.mixture_weights is None:
            # Default = uniform
            w = 1.0 / len(self.lang_codes)
            self.mixture_weights = {lc: w for lc in self.lang_codes}
        else:
            # Ensure mixture weights sum to 1
            s = sum(self.mixture_weights.values())
            if s <= 0:
                logger.warning("Sum of mixture weights is non-positive. Using uniform instead.")
                w = 1.0 / len(self.lang_codes)
                self.mixture_weights = {lc: w for lc in self.lang_codes}
            else:
                for lc in self.mixture_weights:
                    self.mixture_weights[lc] /= s
        
        # Initialize per-language distributions (specific to multigram)
        self.per_lang_probs = {}
        for lang_code in self.lang_codes:
            # uniform initialization
            p_init = 1.0 / len(all_tokens)
            self.per_lang_probs[lang_code] = {tk: p_init for tk in all_tokens}
        
        # Initialize global distribution (will be updated in each iteration)
        p_token = {tk: 1.0 / len(all_tokens) for tk in all_tokens}
        
        # Setup for pruning (identical to _train_custom_em)
        num_tokens_to_trim = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR - self.vocab_size
        tokens_to_remove_per_loop = [math.floor(num_tokens_to_trim/NUM_TRIMMING_ITERATIONS)] * NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - NUM_TRIMMING_ITERATIONS)
        keys_to_keep = set(self.special_tokens + list(chars_to_keep))

        def _order_token_list_by_prob(_p_token):
            tokens_to_keep = []
            remaining_items = []
            for tk, pr in _p_token.items():
                if tk in keys_to_keep:
                    tokens_to_keep.append((tk, pr))
                else:
                    remaining_items.append((tk, pr))
            # Get the rest of the tuples that are not in items_to_move.
            return tokens_to_keep + sorted(remaining_items, key=lambda x: x[1], reverse=True)
        
        # 2) EM loop
        for it in range(self.num_iterations):
            logger.info(f"[Multigram] EM iteration {it+1}/{self.num_iterations}")
            
            # First, initialize per-language usage counts
            lang_usage_counts = {
                lc: {tk: 0.0 for tk in all_tokens} for lc in self.lang_codes
            }
            lang_total_subwords = {lc: 0.0 for lc in self.lang_codes}
            
            # Keep UNK prob small in each language distribution
            for lang_code in self.lang_codes:
                if self.unk_token in self.per_lang_probs[lang_code]:
                    self.per_lang_probs[lang_code][self.unk_token] = 1.0 / len(all_tokens) 
                logger.info(f"Resetting UNK prob for {lang_code} to {self.per_lang_probs[lang_code][self.unk_token]}")
            
            # E-step: For each language, collect usage counts
            for lang_code in self.lang_codes:
                train_file = corpus_info[lang_code]['splits']['train']
                logger.info(f"Processing {lang_code} corpus...")
                
                # Use the language's distribution for segmentation
                c_dict, tot = _accumulate_usage(
                    train_file, 
                    self.per_lang_probs[lang_code], 
                    self.unk_token, 
                    mode=self.em_mode
                )
                
                # Update language-specific counts
                for tk, val in c_dict.items():
                    lang_usage_counts[lang_code][tk] += val
                lang_total_subwords[lang_code] += tot
            
            # M-step: Update per-language distributions
            for lang_code in self.lang_codes:
                total_subwords = lang_total_subwords[lang_code]
                usage_counts = lang_usage_counts[lang_code]
                
                if total_subwords > 0:
                    for tk in all_tokens:
                        self.per_lang_probs[lang_code][tk] = usage_counts[tk] / total_subwords
                else:
                    logger.warning(f"No usage for {lang_code} => fallback uniform distribution.")
                    for tk in all_tokens:
                        self.per_lang_probs[lang_code][tk] = 1.0 / len(all_tokens)
            
            # Combine per-language distributions into global distribution
            for tk in all_tokens:
                mix_sum = 0.0
                for lc in self.lang_codes:
                    alpha_l = self.mixture_weights[lc]
                    mix_sum += alpha_l * self.per_lang_probs[lc].get(tk, 0.0)
                p_token[tk] = mix_sum
            
            # Pruning step (similar to _train_custom_em but using global distribution)
            if tokens_to_remove_per_loop[it] > 0:                    
                sorted_final = _order_token_list_by_prob(p_token)
                all_tokens = set([tk for tk, sc in sorted_final[:-tokens_to_remove_per_loop[it]]])
                logger.info(f"Current vocab size: {len(all_tokens)}")
                
                # Also update per-language distributions to have the same token set
                for lang_code in self.lang_codes:
                    # Remove pruned tokens
                    for tk in list(self.per_lang_probs[lang_code].keys()):
                        if tk not in all_tokens:
                            del self.per_lang_probs[lang_code][tk]
                    
                    # Renormalize
                    sum_post = sum(self.per_lang_probs[lang_code].values())
                    if sum_post > 0:
                        for tk in self.per_lang_probs[lang_code]:
                            self.per_lang_probs[lang_code][tk] /= sum_post
            
            logger.info(f"Current UNK prob in global distribution: {p_token[self.unk_token]}")
        
        # Final pruning if needed (identical to _train_custom_em)
        if len(p_token) > self.vocab_size:
            sorted_final = _order_token_list_by_prob(p_token)
            sorted_final = sorted_final[:self.vocab_size]
            remaining_prob = sum([pr for tk, pr in sorted_final])
            sorted_final = [(tk, pr/remaining_prob) for tk, pr in sorted_final]
        else:
            sorted_final = sorted(p_token.items(), key=lambda x: x[1], reverse=True)

        total_prob_coverage = sum([p for tk, p in sorted_final])
        logger.info(f"Token probabilities sum to {total_prob_coverage}")
        
        # Build final Unigram model (identical to _train_custom_em)
        vocab_list = []
        special_tokens = []
        for tk, prob in sorted_final:
            score = math.log(max(prob, 1e-12))
            if tk in self.special_tokens:
                special_tokens.append((tk, score))
            else:
                vocab_list.append((tk, score))
        vocab_list = special_tokens + vocab_list

        logger.info(f"Multigram EM done. final vocab size = {len(vocab_list)}")
        
        # Figure out unk_id (identical to _train_custom_em)
        unk_id = 0
        for i, (tk, sc) in enumerate(vocab_list):
            if tk == self.unk_token:
                logger.info(f"UNK score: {sc}")
                unk_id = i
                break

        final_tok = Tokenizer(Unigram(tuple(vocab_list), unk_id))
        final_tok.pre_tokenizer = Whitespace()
        self.base_tokenizer = final_tok
        
        # Save the final token probabilities
        self.token_probs = {tk: math.exp(score) for tk, score in vocab_list}
        
        vocab = self.base_tokenizer.get_vocab()
        id_check = {k: vocab[k] for k in self.special_tokens}
        logger.info(f"Special tokens: {id_check}")

        if output_path:
            self.base_tokenizer.save(output_path)
            logger.info(f"Saved multigram tokenizer to {output_path}")

        return self.base_tokenizer

    def finalize_vocab_indices(self):
        """
        Re-map tokens in self.base_tokenizer => IDs [0..N-1] 
        as a standard Unigram. 
        """
        if not self.base_tokenizer:
            logger.warning("No base_tokenizer to finalize in StandardUnigramLMTokenizer.")
            return

        unigram_model = self.base_tokenizer.model
        state = unigram_model.__getstate__()
        attributes = json.loads(state.decode('utf-8'))
        old_vocab = attributes['vocab']
        unk_id = attributes['unk_id']

        from tokenizers import Tokenizer
        from tokenizers.models import Unigram

        new_tok = Tokenizer(Unigram(tuple((x,y) for x,y in old_vocab), unk_id))
        vocab = new_tok.get_vocab()
        id_check = {k: vocab.get(k, -1) for k in self.special_tokens}
        logger.info(f"Special tokens: {id_check}")
        new_tok.pre_tokenizer = self.base_tokenizer.pre_tokenizer
        self.base_tokenizer = new_tok

    def encode(self, text):
        """
        Encode text using the tokenizer.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded or trained.")
        return self.base_tokenizer.encode(text).ids

    def decode(self, ids):
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer loaded or trained.")
        return self.base_tokenizer.decode(ids)

    def get_vocab(self):
        """
        Get the vocabulary as a dict mapping tokens to IDs.
        
        Returns:
            Dict mapping tokens to IDs
        """
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer.")
        return self.base_tokenizer.get_vocab()

    def get_vocab_size(self):
        """
        Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        if not self.base_tokenizer:
            return 0
        return len(self.base_tokenizer.get_vocab())
        
    def get_token_probs(self):
        """
        Get the token probabilities learned during training.
        
        Returns:
            Dict mapping tokens to probabilities, or None if not trained
        """
        return self.token_probs

def train_standard_tokenizer(corpus_info, vocab_size=8000, em_mode='soft', multigram=False, output_path=None):
    """
    Train a standard UnigramLM tokenizer on all languages.
    
    Args:
        corpus_info: Dictionary with corpus information for all languages
        vocab_size: Size of the vocabulary
    
    Returns:
        Path to the trained tokenizer
    """
    # Collect all training files
    train_files = []
    for lang_code, info in corpus_info.items():
        if 'splits' in info and 'train' in info['splits']:
            train_files.append(info['splits']['train'])
    
    if not train_files:
        logger.error("No training files found")
        return None
    
    # Create and train the tokenizer
    if not output_path:
        output_path = f"standard_unigramlm_{em_mode}.json" if not multigram else f"multigramlm_{em_mode}.json"

    tokenizer = StandardUnigramLMTokenizer(vocab_size=vocab_size, em_mode=em_mode, multigram=multigram)
    tokenizer.train(train_files, output_path, corpus_info=corpus_info)
    
    return output_path
