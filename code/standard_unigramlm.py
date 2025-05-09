from __future__ import annotations
from pathlib import Path
import math
import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional

from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace

from code.utils import (
    get_baseline_characters,
    get_baseline_bytes,
    _accumulate_usage,
    _accumulate_usage_dummy,
    load_tokenizer_with_mode,
    _build_hf_tokenizer_from_probs,
    tokmix_merge,
    SPECIAL_TOKENS,
    INITIAL_VOCAB_MULT_FACTOR,
    NUM_TRIMMING_ITERATIONS,
    MIN_TOKEN_PROB,
    MIN_CHAR_OCCURENCE_THRESHOLD
)

logger = logging.getLogger(__name__)


class StandardUnigramLMTokenizer:
    """Unified implementation supporting HF, EM and Multigram."""

    # ---------------------------------------------------------------------
    # constructor + basic public API (unchanged except for kwargs defaults)
    # ---------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int = 8_000,
        unk_token: str = SPECIAL_TOKENS["unk_token"],
        special_tokens: Optional[List[str]] = None,
        em_mode: Optional[str] = None,  # "hf", "hard", "soft"
        num_iterations: int = 10,
        multigram: bool = False,
        mixture_weights: Optional[Dict[str, float]] = None,
        byte_level: bool = False,
        min_char_occurence_threshold: Optional[int] = None
    ) -> None:
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.unk_prior = 1.0 / (32 * vocab_size)  # SentencePiece default
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS.values())
        self.byte_level = byte_level
        self.min_char_occurence_threshold = min_char_occurence_threshold
        if not self.min_char_occurence_threshold:
            self.min_char_occurence_threshold = MIN_CHAR_OCCURENCE_THRESHOLD

        self.base_tokenizer: Optional[Tokenizer] = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.multigram = multigram
        self.mixture_weights = mixture_weights or {}

        # multigram‑specific
        self.lang_codes: List[str] = []
        self.per_lang_probs: Dict[str, Dict[str, float]] = {}
        self.global_probs: Dict[str, float] = {}

    # ------------------------- convenience helpers -----------------------
    def _baseline_tokens(self, files: List[str]) -> Set[str]:
        return (
            get_baseline_bytes()[0]
            if self.byte_level
            else set(get_baseline_characters(files, min_occurence_threshold=self.min_char_occurence_threshold)[0])
        )

    def _initial_vocab(self, train_files: List[str]) -> Set[str]:
        """Oversized HF trainer to bootstrap an initial token set."""
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = (
            ByteLevel(add_prefix_space=False)
            if self.byte_level
            else Whitespace()
        )
        seed_vocab_size = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR
        logger.info(f"Attempting to find seed vocab of size {seed_vocab_size}...")
        trainer = UnigramTrainer(
            vocab_size=seed_vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token,
            initial_alphabet=ByteLevel.alphabet() if self.byte_level else [],
            #n_sub_iterations=1
        )
        tok.train(train_files, trainer)
        init_vocab = tok.get_vocab()
        logger.info(f"Initial vocabulary size: {len(init_vocab)}")
        if self.byte_level:
            # Get actual bytes from the vocabulary
            byte_tokens = [k for k in init_vocab.keys() if any(ord(c) > 127 for c in k)]
            logger.info(f"Byte-level tokens in vocabulary: {len(byte_tokens)}")
        
        return set(init_vocab.keys())

    @staticmethod
    def _normalise(counts: Dict[str, float], unk_token: str, unk_prior: float) -> Dict[str, float]:
        """Convert raw counts to a prob dist with frozen unk_prior."""
        total = sum(counts.values()) - counts.get(unk_token, 0.0)
        if total <= 0:
            uniform = 1.0 / max(1, len(counts) - 1)
            return {
                t: (unk_prior if t == unk_token else uniform * (1 - unk_prior))
                for t in counts
            }
        scale = 1.0 - unk_prior
        return {
            t: (unk_prior if t == unk_token else counts[t] / total * scale)
            for t in counts
        }

    @staticmethod
    def _tokens_to_remove(token_loss: List[Tuple[str, float]], k_drop: int):
        if k_drop <= 0:
            return set()
        
        token_loss.sort(key=lambda x: x[1])
        to_remove = {tk for tk, _ in token_loss[:k_drop]}
        return to_remove

    def _build_final_vocab(
        self, p_token: Dict[str, float]
    ) -> Tuple[List[Tuple[str, float]], int]:
        del p_token[self.unk_token]
        sorted_items = sorted(p_token.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]
        remaining_prob = sum([pr for _, pr in sorted_items]) + self.unk_prior
        sorted_items = [
            (self.unk_token, self.unk_prior)
        ] + [
            (tk, pr / remaining_prob) for tk, pr in sorted_items
        ]
        vocab_list = []
        special = []
        for tk, pr in sorted_items:
            log_sc = math.log(max(pr, MIN_TOKEN_PROB))
            if tk in self.special_tokens:
                special.append((tk, log_sc))
            else:
                vocab_list.append((tk, log_sc))
        vocab_list = special + vocab_list
        unk_id = next(i for i, (tk, _) in enumerate(vocab_list) if tk == self.unk_token)
        return vocab_list, unk_id

    # ------------------------------------------------------------------
    # main EM routine (shared by mono & multigram)
    # ------------------------------------------------------------------
    def _run_em(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
    ) -> None:
        if per_lang and not corpus_info:
            raise ValueError("corpus_info required when multigram=True")

        # gather training files and language codes
        if per_lang:
            self.lang_codes = [lc for lc, info in corpus_info.items()]
            corpus_files = [corpus_info[lc] for lc in self.lang_codes]
        else:
            self.lang_codes = []

        # ------------------ shared preparation ------------------
        baseline = self._baseline_tokens(corpus_files)
        all_tokens = self._initial_vocab(corpus_files)
        all_tokens.update(baseline)
        all_tokens.update(self.special_tokens)

        pinned = set(self.special_tokens) | baseline | {self.unk_token}

        # init distributions ------------------------------------------------
        if per_lang:
            self.per_lang_probs = {lc: {tk: 1.0 / len(all_tokens) for tk in all_tokens} for lc in self.lang_codes}
            self.mixture_weights = self._sanitize_mixture_weights(self.lang_codes, self.mixture_weights)
            p_token: Dict[str, float] = {tk: 1.0 / len(all_tokens) for tk in all_tokens}
            prev_lang_probs = {lc: self.per_lang_probs[lc].copy() for lc in self.lang_codes}
        else:
            p_token = {tk: 1.0 / len(all_tokens) for tk in all_tokens}
        prev_p_token = p_token.copy()

        # pruning schedule ---------------------------------------------------
        k_total = len(all_tokens) - self.vocab_size
        k_step = [math.floor(k_total / NUM_TRIMMING_ITERATIONS)] * NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - NUM_TRIMMING_ITERATIONS)
        for it in range(self.num_iterations):
            logger.info("EM %d/%d – starting | vocab=%d | to trim=%d", it + 1, self.num_iterations, len(all_tokens), k_total)

            # ===================== E & M – identical maths ==================
            
            if per_lang:
                lang_counts = {lc: {tk: 0.0 for tk in all_tokens} for lc in self.lang_codes}
                unk_count = 0
                total_counts = 0
                for lc, file in zip(self.lang_codes, corpus_files):
                    c_dict, _ = _accumulate_usage(file, self.per_lang_probs[lc], self.unk_token, mode=self.em_mode, byte_level=self.byte_level)
                    for tk, v in c_dict.items():
                        lang_counts[lc][tk] += v
                    unk_count += lang_counts[lc][self.unk_token]
                    total_counts += sum(lang_counts[lc].values())
                observed_tokens = set([k for lang_probs in lang_counts.values() for k in lang_probs])
                for lc in self.lang_codes:
                    self.per_lang_probs[lc] = self._normalise(lang_counts[lc], self.unk_token, self.unk_prior)
                p_token = {tk: sum(self.mixture_weights[lc] * self.per_lang_probs[lc][tk] for lc in self.lang_codes) for tk in all_tokens}
            else:
                counts = {tk: 0.0 for tk in all_tokens}
                for file in corpus_files:
                    c_dict, _ = _accumulate_usage(file, p_token, self.unk_token, mode=self.em_mode, byte_level=self.byte_level)
                    for tk, v in c_dict.items():
                        counts[tk] += v
                unk_count = counts[self.unk_token]
                total_counts = sum(counts.values())
                observed_tokens = counts.keys()
                p_token = self._normalise(counts, self.unk_token, self.unk_prior)

            logger.info(f"Observed {len(observed_tokens)} unique tokens out of {len(all_tokens)} possible tokens across corpus...")
            logger.info(f"Observed {unk_count} UNK tokens out of {total_counts}...")

            # -------------- diagnostics BEFORE pruning ---------------------
            unk_prob = p_token.get(self.unk_token, 0.0)
            l1_delta = 0.5 * sum(abs(p_token.get(tk, 0) - prev_p_token.get(tk, 0)) for tk in all_tokens)
            logger.info(" UNK prob: %.8g | Δ‑dist: %.3e | min=%.3e max=%.3e", unk_prob, l1_delta, min(p_token.values()), max(p_token.values()))
            if per_lang:
                for lc in self.lang_codes:
                    up = self.per_lang_probs[lc][self.unk_token]
                    d  = 0.5 * sum(abs(self.per_lang_probs[lc].get(tk, 0) - prev_lang_probs[lc].get(tk, 0)) for tk in all_tokens)
                    logger.info("      [%s] UNK=%.8g | Δ‑dist=%.3e", lc, up, d)

            #import ipdb; ipdb.set_trace()
            # -------------------- pruning ----------------------------------
            k_drop = k_step[it]
            k_drop = min(k_drop, len(all_tokens) - self.vocab_size)
            if k_drop > 0:
                before = len(all_tokens)
                under_consideration = all_tokens - pinned
                if per_lang:
                    per_lang_loss = {}
                    for lc in self.lang_codes:
                        per_lang_loss[lc] = {
                            tk: -lang_counts[lc][tk] * math.log(max(self.per_lang_probs[lc][tk], 1e-30)) for tk in under_consideration
                        }  # noqa: E127
                    loss = [(tk, sum(self.mixture_weights[lc] * per_lang_loss[lc][tk] for lc in self.lang_codes)) for tk in under_consideration]
            
                else:
                    loss = [
                        (tk, -counts[tk] * math.log(max(p_token[tk], 1e-30))) for tk in under_consideration
                    ]
                tokens_to_remove = self._tokens_to_remove(loss, k_drop)
                all_tokens -= tokens_to_remove
                pruned = before - len(all_tokens)
                k_total -= pruned
                logger.info("   tokens_pruned=%d | new_vocab=%d", pruned, len(all_tokens))

                # restrict & renormalise ---------------------------------------
                if per_lang:
                    for lc in self.lang_codes:
                        self.per_lang_probs[lc] = {tk: self.per_lang_probs[lc].get(tk, 0.0) for tk in all_tokens}
                        self.per_lang_probs[lc] = self._normalise(self.per_lang_probs[lc], self.unk_token, self.unk_prior)
                        prev_lang_probs[lc] = self.per_lang_probs[lc].copy()
                        
                p_token = {tk: p_token[tk] for tk in all_tokens}
                p_token = self._normalise(p_token, self.unk_token, self.unk_prior)

            prev_p_token = p_token.copy()

            logger.info(f"Total probability mass: {sum(p_token.values())}")
            assert all(x in p_token for x in pinned)

        # ---------------- store results ----------------
        self.global_probs = p_token

    # ------------------------------------------------------------------
    # thin wrappers that call _run_em then build the HF tokenizer
    # ------------------------------------------------------------------
    def _train_custom_em(self, corpus_files: List[str], output_path=None):
        self._run_em(corpus_files, per_lang=False)
        return self._build_tokenizer(self.global_probs, output_path)

    def _train_multigram(self, corpus_info: Dict[str, str], output_path=None):
        self._run_em(None, per_lang=True, corpus_info=corpus_info)
        return self._build_tokenizer(self.global_probs, output_path)

    def _train_hf(self, corpus_files, output_path=None):
        """
        Train using the HuggingFace UnigramTrainer.
        """
        logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = Whitespace() if self.byte_level else ByteLevel(add_prefix_space=False)

        #TODO: Add byte_fallback=self.byte_level when feature becomes available
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

    def _train_bpe(self, corpus_files: List[str], output_path=None):
        """Plain HF BPE trainer; shares most hyper‑params with EM modes."""
        logger.info("Using HuggingFace BPETrainer.")
        tok = Tokenizer(BPE(unk_token=self.unk_token))
        tok.pre_tokenizer = (
            ByteLevel(add_prefix_space=False) if self.byte_level else Whitespace()
        )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=ByteLevel.alphabet() if self.byte_level else [],
        )
        tok.train(corpus_files, trainer)
        self.base_tokenizer = tok
        if output_path:
            tok.save(output_path)
            logger.info("Saved BPE tokenizer to %s", output_path)
        return tok

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

        token_level = "byte" if self.byte_level else "character"
        logger.info(f"Training new tokenizer with EM mode: {self.em_mode} at {token_level}-level")
        # Different training approaches
        if self.em_mode == "bpe" and not self.multigram:
            return self._train_bpe(corpus_files, output_path)

        elif self.em_mode == "hf" and not self.multigram:
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

    # ------------------------------------------------------------------
    # helper to create the final HF tokenizer & save it
    # ------------------------------------------------------------------
    def _build_tokenizer(self, probs: Dict[str, float], output_path):
        vocab_list, unk_id = self._build_final_vocab(probs)
        tok = _build_hf_tokenizer_from_probs(
            vocab_list,
            unk_id,
            byte_level=self.byte_level,
        )
        self.base_tokenizer = tok
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(output_path)
            logger.info("Tokenizer saved to %s", output_path)
        return tok

    # --------- utilities ---------
    @staticmethod
    def _sanitize_mixture_weights(lang_codes: List[str], user_weights: Dict[str, float]) -> Dict[str, float]:
        if not lang_codes:
            return {}
        if not user_weights:
            uniform = 1.0 / len(lang_codes)
            return {lc: uniform for lc in lang_codes}
        weights = {lc: user_weights.get(lc, 0.0) for lc in lang_codes}
        s = sum(weights.values())
        if s <= 0:
            uniform = 1.0 / len(lang_codes)
            return {lc: uniform for lc in lang_codes}
        return {lc: w / s for lc, w in weights.items()}

    def load(self, base_path):
        """
        Load a saved tokenizer
        
        Args:
            base_path: Path to the tokenizer file
        """
        self.base_tokenizer, self.byte_level = load_tokenizer_with_mode(base_path)
        self.vocab_size = len(self.base_tokenizer.get_vocab())
        with open(base_path, encoding="utf-8") as fh:
            data = json.load(fh)
    
        # 3) recover unigram scores --------------------------------------
        if data.get("model", {}).get("type") == "Unigram":
            vocab_pairs = data["model"]["vocab"]      # list [piece, score]
            # scores are log‑probs (SentencePiece convention)
            tokens, log_scores = zip(*vocab_pairs)
            probs = [math.exp(s) for s in log_scores]
            Z = sum(probs)
            self.global_probs = {t: p / Z for t, p in zip(tokens, probs)}
        else:
            logger.warning(f"No global probs for tokenizer at {base_path} found")

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

def train_lang_tokenizers(corpus_info: Dict[str, Dict[str, str]],
                          out_dir: str,
                          vocab_size=8000, 
                          byte_level=False,
                          use_per_lang_if_exists=True) -> Dict[str, str]:
    """
    Args
    ----
    config       : experiment config (for hyper‑params)
    corpus_info  : { lang : { 'train': <file>, 'val': <file>, … } }
    out_dir      : directory where <lang>_tokenizer.json are written

    Returns
    -------
    paths : { lang : tokenizer_path }
    """
    paths = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 2) shared tokenizer hyper‑parameters
    for lg, sample_path in corpus_info.items():
        outfile = os.path.join(out_dir, f"tokmix_{lg}_tokenizer.json")
        if not os.path.exists(outfile) or not use_per_lang_if_exists:

            logger.info("Training per‑language tokenizer for %s …", lg)
            tok = StandardUnigramLMTokenizer(
                vocab_size   = vocab_size,
                em_mode      = "soft",      # full forward‑backward EM
                byte_level   = byte_level,
                multigram    = False
            )
        
            tok.train([sample_path], output_path=outfile)
        paths[lg] = outfile

    # save manifest for downstream stages
    return paths

def train_tokmix(per_lang_paths: Dict[str, str],
                 k_lang: int,
                 k_final: int,
                 byte_level: bool,
                 output_path: str):
    """
    Combine language‑specific tokenizers trained earlier.
    `per_lang_paths` maps lang → path/to/json
    """
    tkzs = {lg: StandardUnigramLMTokenizer()for lg in per_lang_paths}
    for lg, path in per_lang_paths.items():
        tkzs[lg].load(path)

    merged = tokmix_merge({lg: tkz.global_probs for lg, tkz in  tkzs.items()}, 
                          k_lang, k_final,
                          byte_level,
                          SPECIAL_TOKENS["unk_token"],
                          list(SPECIAL_TOKENS.values()))
    merged.save(output_path)
    logger.info("Saved TokMix tokenizer to %s", output_path)
    return output_path

def train_standard_tokenizer(corpus_info, 
                             vocab_size=8000, 
                             em_mode='soft', 
                             multigram=False, 
                             byte_level=False, 
                             output_path=None,
                             min_char_occurence_threshold=None
                             ):
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
    for lang_code, train_file in corpus_info.items():
        train_files.append(train_file)
    
    if not train_files:
        logger.error("No training files found")
        return None
    
    # Create and train the tokenizer
    if not output_path:
        output_path = f"standard_unigramlm_{em_mode}.json" if not multigram else f"multigramlm_{em_mode}.json"

    tokenizer = StandardUnigramLMTokenizer(vocab_size=vocab_size, em_mode=em_mode, multigram=multigram, byte_level=byte_level)
    tokenizer.train(train_files, output_path, corpus_info=corpus_info)
    
    return output_path
