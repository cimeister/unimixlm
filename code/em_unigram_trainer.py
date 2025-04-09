import os
import logging
import math
import numpy as np
from collections import Counter
import multiprocessing
from functools import partial
from tqdm import tqdm

from code.utils import segment_text_viterbi

logger = logging.getLogger(__name__)

def _segment_batch_worker(args):
    """
    Top-level / module-level function so it is pickleable by multiprocessing.
    Replicates the logic of the old _segment_batch method.

    args is a tuple:
      (lines, token_probs, unk_token)
    """
    lines, token_probs, unk_token = args
    c = Counter()
    tok_count = 0
    for text in lines:
        tokens = segment_text_viterbi(text, token_probs, unk_token=unk_token)
        c.update(tokens)
        tok_count += len(tokens)
    return (c, tok_count)


class EMUnigramTrainer:
    """
    EM-based trainer for a fixed vocabulary Unigram distribution.
    We rely on segment_text_viterbi(...) from utils.py to ensure consistent
    segmentation with what's used at inference time.
    """

    def __init__(self, vocab, unk_token="<unk>", special_tokens=None,
                 max_iterations=20, convergence_threshold=1e-7):
        """
        Args:
            vocab: dict(token -> ID), the shared vocabulary.
            unk_token: string for unknown token.
            special_tokens: list of special tokens (like <s>, </s>, <pad>, <unk>).
            max_iterations: how many EM steps max.
            convergence_threshold: stop if sum of param diffs is below this.
        """
        self.vocab = vocab  # e.g. from your base HF tokenizer
        self.unk_token = unk_token
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>"]
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # create a set for quick membership testing
        self.vocab_set = set(self.vocab.keys())

        # ensure special tokens exist in vocab
        for tok in self.special_tokens:
            if tok not in self.vocab_set:
                logger.info(f"Adding missing special token '{tok}' to vocab.")
                self.vocab[tok] = len(self.vocab)
                self.vocab_set.add(tok)

        self.token_probs = None  # final distribution
        self.log_probs = None

    def train(self, corpus_file, batch_size=1000, num_processes=None, is_low_resource=False):
        """
        Run the EM algorithm over `corpus_file`.

        Args:
            corpus_file: path to text
            batch_size: how many lines to process in memory
            num_processes: concurrency
            is_low_resource: don't skip lines for large corpora if True

        Returns:
            dict(token -> float probability)
        """
        logger.info(f"Starting EM training on {corpus_file}")

        # 1) Initialize
        vocab_size = len(self.vocab)
        token_probs = {t: 1.0 / vocab_size for t in self.vocab}
        # keep <unk> smaller
        min_prob = min(token_probs.values())

        if self.unk_token in token_probs:
            token_probs[self.unk_token] = min_prob / 1000.0

        # EM loop
        for iteration in range(self.max_iterations):
            logger.info(f"EM iteration {iteration+1}/{self.max_iterations}")

            from collections import Counter
            token_counts = Counter()
            total_tokens = 0
            lines_processed = 0

            with open(corpus_file, 'r', encoding='utf-8') as fin:
                buffer_lines = []
                file_ended = False
                while not file_ended:
                    # read a batch of lines
                    while len(buffer_lines) < batch_size:
                        line = fin.readline()
                        if not line:
                            file_ended = True
                            break
                        line = line.strip()
                        if line:
                            buffer_lines.append(line)

                    # process this batch
                    if buffer_lines:
                        # E-step: segment these lines & collect counts
                        c, t = self._process_lines(buffer_lines, token_probs, num_processes)
                        token_counts.update(c)
                        total_tokens += t
                        lines_processed += len(buffer_lines)
                        buffer_lines = []

                        # For large corpora, skip reading the rest 
                        if (not is_low_resource) and lines_processed > 50000 and total_tokens > 1000000:
                            logger.info(f"Sufficient lines {lines_processed} / tokens {total_tokens}, skipping rest.")
                            break

            if total_tokens == 0:
                logger.error("No tokens processed! Possibly empty corpus?")
                break

            # M-step: update distribution
            new_probs = {}
            for tok in self.vocab:
                c = token_counts.get(tok, 0)
                new_probs[tok] = c / float(total_tokens) if total_tokens else 1e-12

            # keep them nonzero
            for tok in self.vocab:
                if new_probs[tok] <= 0.0:
                    new_probs[tok] = 1.0 / (total_tokens + vocab_size * 1000)

            # keep <unk> small
            if self.unk_token in new_probs:
                new_probs[self.unk_token] = min(new_probs[self.unk_token], min(new_probs.values()) / 1000.0)

            # check convergence
            diffs = [abs(new_probs[t] - token_probs[t]) for t in self.vocab]
            total_diff = sum(diffs)
            if iteration >= 2 and total_diff < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration+1}, total_diff={total_diff:.3g}")
                token_probs = new_probs
                break

            token_probs = new_probs

        self.token_probs = token_probs
        # we can store log_probs if we want
        self.log_probs = {t: math.log(max(p, 1e-12)) for t, p in token_probs.items()}
        return token_probs

    def _process_lines(self, lines, token_probs, num_processes=None):
        """
        Segment lines using segment_text_viterbi, count usage, return (Counter, total_tokens).
        Parallel if needed.
        """
        if num_processes is None:
            num_processes = max(1, min(20, (multiprocessing.cpu_count() - 1)))

        if len(lines) < 100:
            # just do it in the current process
            return _segment_batch_worker((lines, token_probs, self.unk_token))

        # chunk for parallel
        chunk_size = max(1, len(lines)//num_processes)
        chunks = []
        idx = 0
        while idx < len(lines):
            chunk = lines[idx:idx+chunk_size]
            idx += chunk_size
            chunks.append(chunk)

        # We'll build a list of arguments so each worker can do its own batch
        # pool.map requires a top-level function. We pass partial arguments.
        worker_args = [(chunk, token_probs, self.unk_token) for chunk in chunks]

        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_segment_batch_worker, worker_args)

        from collections import Counter
        final_counts = Counter()
        final_tokens = 0
        for c, t in results:
            final_counts.update(c)
            final_tokens += t

        return final_counts, final_tokens

    

    def tokenize(self, text):
        """
        If you want to do a single-line tokenize with the final distribution:
        """
        if not self.token_probs:
            raise ValueError("No token_probs learned yet!")
        return segment_text_viterbi(text, self.token_probs, unk_token=self.unk_token)
