import os
import logging
import math
import numpy as np
from collections import Counter, defaultdict

from code.utils import _accumulate_usage, segment_text_viterbi, MIN_TOKEN_PROB

logger = logging.getLogger(__name__)

class EMUnigramTrainer:
    """
    EM-based trainer for a fixed vocabulary Unigram distribution.
    Uses the same EM approach as StandardUnigramLMTokenizer for consistency.
    """

    def __init__(self, vocab, unk_token="<unk>", special_tokens=None,
                 max_iterations=20, em_mode="hard", convergence_threshold=1e-7):
        """
        Args:
            vocab: dict(token -> ID), the shared vocabulary.
            unk_token: string for unknown token.
            special_tokens: list of special tokens (like <s>, </s>, <pad>, <unk>).
            max_iterations: how many EM steps max.
            em_mode: Either "hard" or "soft" to match StandardUnigramLMTokenizer
            convergence_threshold: stop if sum of parameter differences falls below this
        """
        self.vocab = vocab  # e.g. from your base HF tokenizer
        self.unk_token = unk_token
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>"]
        self.max_iterations = max_iterations
        self.em_mode = em_mode
        self.convergence_threshold = convergence_threshold
        
        # Ensure em_mode is valid
        if self.em_mode not in ["hard", "soft"]:
            logger.warning(f"Invalid em_mode '{em_mode}', defaulting to 'hard'")
            self.em_mode = "hard"

        # Create a set for quick membership testing
        self.vocab_set = set(self.vocab.keys())

        # Ensure special tokens exist in vocab
        for tok in self.special_tokens:
            if tok not in self.vocab_set:
                logger.info(f"Adding missing special token '{tok}' to vocab.")
                self.vocab[tok] = len(self.vocab)
                self.vocab_set.add(tok)

        self.token_probs = None  # final distribution

    def train(self, corpus_file, batch_size=1000, num_processes=None):
        """
        Run the EM algorithm over `corpus_file` using the same approach as
        StandardUnigramLMTokenizer.

        Args:
            corpus_file: path to text
            batch_size: how many lines to process in memory (passed to _accumulate_usage)
            num_processes: concurrency (passed to _accumulate_usage)

        Returns:
            dict(token -> float probability)
        """
        logger.info(f"Starting EM training on {corpus_file} with mode={self.em_mode}")

        # 1) Initialize with uniform distribution
        vocab_size = len(self.vocab)
        p_token = {t: 1.0 / vocab_size for t in self.vocab}

        # EM loop
        prev_p_token = None
        for iteration in range(self.max_iterations):
            logger.info(f"EM iteration {iteration+1}/{self.max_iterations}")
            
            # Reset usage counts for E-step
            usage_counts = {tk: 0.0 for tk in self.vocab}
            total_subwords = 0.0
            
            # Save current parameters for convergence check
            if iteration > 0:
                prev_p_token = p_token.copy()
            
            # E-step: directly use _accumulate_usage from utils
            # This function handles parallelization internally
            c_dict, tot = _accumulate_usage(
                corpus_file, 
                p_token,
                self.unk_token, 
                batch_size=batch_size,
                num_processes=num_processes,
                mode=self.em_mode
            )
            
            # Update counts
            for tk, val in c_dict.items():
                usage_counts[tk] += val
            total_subwords += tot
            
            # M-step: Update token probabilities based on counts
            if total_subwords > 0:
                # Calculate new probabilities based on observed counts
                for tk in self.vocab:
                    p_token[tk] = usage_counts[tk] / total_subwords
            else:
                logger.warning("No usage => fallback to uniform distribution.")
                for tk in self.vocab:
                    p_token[tk] = 1.0 / len(self.vocab)
            
            # Check for convergence
            if prev_p_token and iteration > 0:
                diffs = [abs(p_token[tk] - prev_p_token[tk]) for tk in self.vocab]
                total_diff = sum(diffs)
                logger.info(f"Iteration {iteration+1} parameter change: {total_diff:.8f}")
                
                if total_diff < self.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration+1}, total_diff={total_diff:.8g}")
                    break

        # Keep probabilities positive
        for tk in self.vocab:
            p_token[tk] = max(p_token[tk], MIN_TOKEN_PROB)
        
        del p_token[self.unk_token] # don't include UNK prob in counts   
        total_prob = sum(p_token.values())
        if total_prob > 0:
            p_token = {tk: prob/total_prob for tk, prob in p_token.items()}

        p_token[self.unk_token] = MIN_TOKEN_PROB
        logger.info(f"Resetting UNK prob to {p_token[self.unk_token]}")
        
        logger.info(f"EM training completed with {self.max_iterations} iterations")
        
        return p_token
    
