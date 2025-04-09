import os
import logging
import math
import numpy as np
import json
from collections import Counter, defaultdict

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer

# If you already have some function for Viterbi segmentation under a Unigram distribution:
from code.utils import segment_text_viterbi, forward_backward_unigram_fast, _accumulate_usage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKENIZERS_DIR = os.path.join(DATA_DIR, "tokenizers")
os.makedirs(TOKENIZERS_DIR, exist_ok=True)
DEFAULT_BASE_NAME = "multi_base_tokenizer.json"
DEFAULT_BASE_NAME_SOFT = "multi_base_tokenizer_soft.json"
DEFAULT_PROBS_NAME = "multigram_global_probs.npy"
DEFAULT_PROBS_NAME_SOFT = "multigram_global_probs_soft.npy"


class MultigramLMTokenizer:
    """
    Implementation of a mixture-of-languages Unigram approach:
      1) Train or load a base tokenizer to get an initial large set of tokens.
      2) For each language ℓ, maintain a distribution p_l(token).
      3) Combine them via p_global(token) = sum_l alpha_l * p_l(token).
      4) Prune tokens with the smallest p_global, re-normalize, repeat.
      5) The final p_global is used for "true" Viterbi segmentation at inference.
    """

    def __init__(self,
                 vocab_size=8000,
                 unk_token="<unk>",
                 special_tokens=None,
                 num_iterations=5,
                 mixture_weights=None,
                 soft_em=False):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        if special_tokens is None:
            special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
        self.special_tokens = special_tokens
        self.num_iterations = num_iterations
        self.mixture_weights = mixture_weights
        self.base_tokenizer = None  # a tokenizers.Tokenizer
        self.token_probs = None     # final p_global
        self.per_lang_probs = {}
        self.lang_codes = []
        self.soft_em = soft_em

    def train(self, 
        corpus_info, 
        base_path,
        global_probs_path, 
        load_if_exists=True):
        """
        Train the mixture-of-languages Unigram model.

        Steps:
          1) Possibly load an existing "multi_base_tokenizer.json" if load_if_exists is True.
          2) Else train a standard Unigram tokenizer on all languages combined => initial vocab set V.
          3) Initialize per-language distributions p_l(token) = uniform (or partial).
          4) Run mixture EM + pruning for self.num_iterations, capping at vocab_size.
          5) Save final token_probs (p_global), possibly to .npy
        """
        self.base_tokenizer = None
        self.token_probs = None
        if load_if_exists:
            if os.path.isfile(base_path):
                logger.info(f"Loading existing base tokenizer from {base_path}")
                self.base_tokenizer = Tokenizer.from_file(base_path)
            else:
                logger.info("No existing base tokenizer found; will train anew.")
            if os.path.isfile(global_probs_path):
                logger.info(f"Loading existing multigram probabilities from {global_probs_path}")
                arr = np.load(global_probs_path, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    arr = arr.item()
                self.token_probs = arr
            else:
                logger.info("No existing base tokenizer found; will train anew.")

        if self.token_probs is not None and self.base_tokenizer is not None:
            
            return {
            'base': self.base_tokenizer,
            'token_probs': self.token_probs,
            'vocab_size': len(self.token_probs)
            }
        # Gather train files
        train_files = []
        self.lang_codes = []
        for lang_code, info in corpus_info.items():
            if 'splits' in info and 'train' in info['splits']:
                train_files.append(info['splits']['train'])
                self.lang_codes.append(lang_code)

        if not train_files:
            logger.error("No train files found for any language.")
            return None

        # Step 1: If needed, train a base tokenizer for an initial big vocabulary
        if not self.base_tokenizer:
            logger.info("Training a base Unigram tokenizer on the combined data for initial vocab.")
            tok = Tokenizer(Unigram())
            tok.pre_tokenizer = Whitespace()

            trainer = UnigramTrainer(
                vocab_size=self.vocab_size * 5,  # Start bigger to get initial pool
                special_tokens=self.special_tokens,
                unk_token=self.unk_token
            )
            tok.train(train_files, trainer)
            self.base_tokenizer = tok
        else:
            logger.info("Using pre-loaded base tokenizer as initial set of tokens.")

        # Step 2: get the initial big set of tokens
        vocab_dict = self.base_tokenizer.get_vocab()  # token->id
        # invert so we can sort by ID, etc.
        sorted_items = sorted(vocab_dict.items(), key=lambda x: x[1])
        all_tokens = [t for t,_ in sorted_items]
        # Guarantee special tokens are included
        for sp_tk in self.special_tokens:
            if sp_tk not in all_tokens:
                all_tokens.append(sp_tk)

        orig_vocab_size = len(vocab_dict.keys())
        mode = "soft" if self.soft_em else "hard"
        logger.info(f"Starting with tokenizer with vocab size {orig_vocab_size}. Using '{mode}' method for EM estimation")
        # Step 3: Initialize per_lang_probs
        self.per_lang_probs = {}
        for lang_code in self.lang_codes:
            # uniform
            p_init = 1.0 / len(all_tokens)
            self.per_lang_probs[lang_code] = {tk: p_init for tk in all_tokens}

        # mixture weights
        if self.mixture_weights is None:
            # default = uniform
            w = 1.0 / len(self.lang_codes)
            self.mixture_weights = {lc: w for lc in self.lang_codes}
        else:
            # ensure they sum to 1 (or rescale them)
            s = sum(self.mixture_weights.values())
            if s <= 0:
                logger.warning("Sum of mixture weights is non-positive. Using uniform instead.")
                w = 1.0 / len(self.lang_codes)
                self.mixture_weights = {lc: w for lc in self.lang_codes}
            else:
                for lc in self.mixture_weights:
                    self.mixture_weights[lc] /= s

        # Step 4: Mixture EM + pruning
        for iteration in range(self.num_iterations):
            logger.info(f"[Mixture] EM iteration {iteration+1}/{self.num_iterations}")

            # E-step: for each language, we do usage counting
            usage_counts = {
                lc: defaultdict(float) for lc in self.lang_codes
            }
            total_usage = {lc: 0.0 for lc in self.lang_codes}

            # For each language, accumulate usage c_l(token)
            for lang_code in self.lang_codes:
                train_file = corpus_info[lang_code]['splits']['train']
                logger.info(f"On {lang_code} corpus...")
                
                c, tot = _accumulate_usage(
                    train_file,
                    self.per_lang_probs[lang_code],
                    self.unk_token,
                    mode='soft' if self.soft_em else 'hard'
                )
                
                for tk, val in c.items():
                    usage_counts[lang_code][tk] += val
                total_usage[lang_code] += tot

            logger.info("M step...")
            # M-step: update p_l(token)
            for lang_code in self.lang_codes:
                tot = total_usage[lang_code]
                if tot <= 0:
                    # corner case
                    continue
                for tk in self.per_lang_probs[lang_code]:
                    self.per_lang_probs[lang_code][tk] = usage_counts[lang_code][tk] / tot

            # Combine into p_global
            self.token_probs = {}
            for tk in all_tokens:
                mix_sum = 0.0
                for lc in self.lang_codes:
                    alpha_l = self.mixture_weights[lc]
                    mix_sum += alpha_l * self.per_lang_probs[lc].get(tk, 0.0)
                self.token_probs[tk] = mix_sum

            # Prune tokens if above vocab_size
            if len(self.token_probs) > self.vocab_size:
                # sort tokens by p_global
                sorted_tks = sorted(self.token_probs.items(), key=lambda x: x[1], reverse=True)
                keep = set([tk for tk,_ in sorted_tks[:self.vocab_size]])
                # filter
                new_all_tokens = []
                for tk in all_tokens:
                    if tk in keep:
                        new_all_tokens.append(tk)
                all_tokens = new_all_tokens

                # remove pruned from each per_lang_probs
                for lang_code in self.lang_codes:
                    # re-normalize
                    sum_post = 0.0
                    for tk in list(self.per_lang_probs[lang_code].keys()):
                        if tk not in keep:
                            del self.per_lang_probs[lang_code][tk]
                        else:
                            sum_post += self.per_lang_probs[lang_code][tk]
                    # scale
                    if sum_post > 0:
                        for tk in self.per_lang_probs[lang_code]:
                            self.per_lang_probs[lang_code][tk] /= sum_post

                # also filter p_global
                new_token_probs = {}
                sum_glob = 0.0
                for tk in keep:
                    val = self.token_probs[tk]
                    new_token_probs[tk] = val
                    sum_glob += val
                # renormalize p_global as well
                if sum_glob > 0:
                    for tk in new_token_probs:
                        new_token_probs[tk] /= sum_glob

                self.token_probs = new_token_probs
            else:
                # also renormalize p_global
                s_g = sum(self.token_probs.values())
                if s_g > 0:
                    for tk in self.token_probs:
                        self.token_probs[tk] /= s_g

        logger.info("[Mixture] Done with EM training. Final vocabulary size = %d" % len(self.token_probs))
        self.finalize_vocab_indices()
        # the final p_global distribution is in self.token_probs

        np.save(global_probs_path, self.token_probs)
        logger.info(f"Saved final p_global distribution to {global_probs_path}")
        self.base_tokenizer.save(base_path)
        logger.info(f"Saved re-indexed base tokenizer to {base_path}")

        return {
            'base': self.base_tokenizer,
            'token_probs': self.token_probs,
            'vocab_size': len(self.token_probs)
        }


    def load(self, base_path, global_probs_path=None):
        from tokenizers import Tokenizer
        self.base_tokenizer = Tokenizer.from_file(base_path)
        if global_probs_path and os.path.isfile(global_probs_path):
            arr = np.load(global_probs_path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr.item()
            self.token_probs = arr

    def finalize_vocab_indices(self):
        """
        We rely on self.token_probs as the final set of subwords. 
        We re-map them to IDs 0..N-1, build a new Unigram model in self.base_tokenizer.
        """
        if not self.base_tokenizer:
            logger.warning("No base_tokenizer to finalize in MultigramLMTokenizer.")
            return
        if not self.token_probs:
            logger.warning("No final token_probs. Possibly not trained.")
            return

        # Build final subword list from self.token_probs
        # We'll sort them by probability descending or by token alphabetical, up to self.vocab_size
        sorted_toks = sorted(self.token_probs.items(), key=lambda x: x[1], reverse=True)
        # We might also need to limit to self.vocab_size if not done. 
        if len(sorted_toks)>self.vocab_size:
            sorted_toks = sorted_toks[:self.vocab_size]

        # Also ensure special tokens exist. If not in final, add them with small score
        for sp_tk in self.special_tokens:
            if not any(sp_tk==p[0] for p in sorted_toks):
                sorted_toks.append((sp_tk, -99999.0))

        original_unigram_model = self.base_tokenizer.model
        state = original_unigram_model.__getstate__()
        unk_id = json.loads(state.decode('utf-8'))['unk_id']

        new_tok = Tokenizer(Unigram(tuple((x,y) for x,y in sorted_toks), unk_id))
        # keep same pretokenizer
        new_tok.pre_tokenizer = self.base_tokenizer.pre_tokenizer
        self.base_tokenizer = new_tok

    def mixture_viterbi_tokenize(self, text):
        """
        Use self.token_probs as distribution. 
        If not present, fallback to base_tokenizer's segmentation.
        """
        if not self.token_probs:
            return self.base_tokenizer.encode(text).tokens
        return segment_text_viterbi(text, self.token_probs, unk_token=self.unk_token)

    def decode(self, token_ids):
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded.")
        return self.base_tokenizer.decode(token_ids)

    def get_vocab(self):
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded.")
        return self.base_tokenizer.get_vocab()

    def get_vocab_size(self):
        if self.token_probs:
            return len(self.token_probs)
        elif self.base_tokenizer:
            return len(self.base_tokenizer.get_vocab())
        return 0


def train_multigram_tokenizer(corpus_info,
                              vocab_size=8000,
                              output_dir=TOKENIZERS_DIR,
                              load_if_exists=True,
                              num_iterations=10,
                              mixture_weights=None,
                              soft_em=False):
    """
    Convenience function that:
      1) Creates a MultigramLMTokenizer
      2) Calls .train(...) with the given corpus info
      3) Returns paths to base tokenizer .json and final p_global .npy
    """
    tokenizer = MultigramLMTokenizer(
        vocab_size=vocab_size,
        num_iterations=num_iterations,
        mixture_weights=mixture_weights,
        soft_em=soft_em
    )
    base_path = os.path.join(output_dir, DEFAULT_BASE_NAME_SOFT if soft_em else DEFAULT_BASE_NAME)

    # The final p_global distribution is presumably saved to "multigram_global_probs.npy"
    global_probs_path = os.path.join(output_dir, DEFAULT_PROBS_NAME_SOFT if soft_em else DEFAULT_PROBS_NAME)

    result = tokenizer.train(corpus_info, base_path=base_path, global_probs_path=global_probs_path, load_if_exists=load_if_exists)
    if not result:
        logger.error("Failed to train or load multi-gram tokenizer.")
        return None

    # The base tokenizer is presumably saved to "multi_base_tokenizer.json"
   

    return {
        'base': base_path,
        'global_probs': global_probs_path
    }

