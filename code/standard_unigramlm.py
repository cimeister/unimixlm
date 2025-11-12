from __future__ import annotations
from pathlib import Path
import math
import os
import re
import json
import copy
import torch
import logging
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
import random
import tempfile
import subprocess
import shutil
import atexit
from abc import ABC, abstractmethod


try:
    import sentencepiece as spm
except Exception:
    spm = None
import numpy as np
import pickle
import hashlib
import time
from datetime import datetime
import platform
import sys
import multiprocessing

from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer

from code.utils import (
    get_baseline_characters,
    get_baseline_bytes,
    tokmix_merge,
    compute_intersection_pruning_tokens,
    proportional_corpus_iterator,
    _save_tokenizer_metadata,
    _get_metadata_logging_path,
    _create_base_metadata
)

from code.tokenizer_formatting_utils import (
    load_tokenizer_with_mode,
    _build_unigramlm_hf_tokenizer_from_lprobs,
    _load_custom_vocab_from_file,
     _write_sp_seed_vocab_file,
    _process_file_worker
)

from code.em_utils import (
    _accumulate_usage,
    _accumulate_usage_dummy,
    _compute_loss_with_alternative_sequences,
    _compute_alternatives_direct
)

from code import constants

logger = logging.getLogger(__name__)


@dataclass
class EMState:
    """Encapsulates state needed during EM training iterations."""
    corpus_files: List[str]
    per_lang: bool
    lang_codes: List[str]
    all_tokens: Set[str]  # Mutable - tokens removed during pruning
    pinned: Set[str]
    global_lprobs: Dict[str, float]
    per_lang_lprobs: Optional[Dict[str, Dict[str, float]]]
    k_total: int  # Mutable - decremented during pruning
    k_step: List[int]


# ---------------------------------------------------------------------
# Refactor 3: Pruning Strategy Pattern
# ---------------------------------------------------------------------

class PruningStrategy(ABC):
    """Abstract base class for a pruning loss computation strategy."""
    @abstractmethod
    def compute_loss(
        self,
        token: str,
        counts_dict: Dict[str, float],
        lprob_dict: Dict[str, float],
        alternatives_dict: Optional[Dict],
        unk_token: str
    ) -> float:
        pass

class ExactLossPruning(PruningStrategy):
    """Computes exact loss using alternative sequences."""
    def compute_loss(self, token, counts_dict, lprob_dict, alternatives_dict, unk_token):
        return _compute_loss_with_alternative_sequences(
            token, counts_dict, alternatives_dict, lprob_dict, unk_token
        )

class ApproxLossPruning(PruningStrategy):
    """Computes approximate loss (count * lprob)."""
    def compute_loss(self, token, counts_dict, lprob_dict, alternatives_dict, unk_token):
        return counts_dict[token] * lprob_dict[token]

class ProbabilityPruning(PruningStrategy):
    """Uses the token's probability directly as its 'loss' (score)."""
    def compute_loss(self, token, counts_dict, lprob_dict, alternatives_dict, unk_token):
        return np.exp(lprob_dict[token])


class StandardUnigramLMTokenizer:
    """Unified implementation supporting HF, EM and Multigram."""

    # ---------------------------------------------------------------------
    # 1. constructor + public API (HF compatibility)
    # ---------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int = 8_000,
        unk_token: str = constants.SPECIAL_TOKENS["unk_token"],
        special_tokens: Optional[List[str]] = None,
        em_mode: Optional[str] = None,  # "hf", "hard", "soft"
        num_iterations: int = 10,
        multigram: bool = False,
        mixture_weights: Optional[Dict[str, float]] = None,
        byte_level: bool = True,
        whitespace_token_boundaries: bool = False,
        min_char_occurence_threshold: Optional[int] = None,
        dataset_size_normalization: bool = True,
        use_intersection_pruning: bool = False,
        initial_vocab_tokens: Optional[str] = None,  # Path to file containing tokens
        #TODO: add check that this is a legal value 
        pruning_criterion: str = "exact_loss",  # "exact_loss", "approx_loss", "probability"
        use_sp_seed_vocab: bool = True,
        use_sp_em: bool = True,
    ) -> None:
        """
        Initialize StandardUnigramLMTokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            unk_token: Unknown token string
            special_tokens: List of special tokens to include
            em_mode: EM algorithm mode - "hf" (HuggingFace), "hard" (Viterbi), "soft" (forward-backward)
            num_iterations: Number of EM iterations
            multigram: Whether to use multigram mode (per-language distributions)
            mixture_weights: Language mixture weights for multigram mode
            byte_level: Whether to use byte-level tokenization
            whitespace_token_boundaries: Whether to use whitespace as token boundaries
            min_char_occurence_threshold: Minimum character occurrence threshold
            dataset_size_normalization: Whether to normalize loss by dataset size in multigram mode.
                When True, prevents bias toward larger datasets during pruning. Recommended for
                multilingual scenarios with imbalanced dataset sizes.
            use_intersection_pruning: Whether to use intersection-based pruning in multigram mode.
                When True, only prunes tokens that are consistently unimportant (low loss/prob) across ALL languages.
                When False, uses traditional average-loss pruning.
            initial_vocab_tokens: Path to file containing custom initial vocabulary tokens.
                Supported formats: plain text (one token per line), JSON list, or HF tokenizer JSON.
                If provided, replaces the HF-based vocabulary initialization.
            pruning_criterion: Loss function used for token pruning during EM training.
                "exact_loss": Compute exact loss using alternative sequences (most accurate, slowest)
                "approx_loss": Use approximate loss computation (good balance)
                "probability": Use token probabilities directly (fastest, least sophisticated)
        """

        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.special_tokens = special_tokens or list(constants.SPECIAL_TOKENS.values())
        self.byte_level = byte_level
        self.whitespace_token_boundaries = whitespace_token_boundaries
        self.min_char_occurence_threshold = min_char_occurence_threshold
        if not self.min_char_occurence_threshold:
            self.min_char_occurence_threshold = constants.MIN_CHAR_OCCURENCE_THRESHOLD

        self.base_tokenizer: Optional[Tokenizer] = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.multigram = multigram
        self.mixture_weights = mixture_weights or {}
        self.dataset_size_normalization = dataset_size_normalization
        self.use_intersection_pruning = use_intersection_pruning
        self.initial_vocab_tokens = initial_vocab_tokens
        self.pruning_criterion = pruning_criterion
        self.use_sp_seed_vocab = use_sp_seed_vocab
        self.use_sp_em = use_sp_em

        # Validate parameters
        if self.initial_vocab_tokens and not os.path.exists(self.initial_vocab_tokens):
            raise FileNotFoundError(f"Initial vocabulary file not found: {self.initial_vocab_tokens}")
        
        if pruning_criterion not in ["exact_loss", "approx_loss", "probability"]:
            raise ValueError(f"Invalid pruning_criterion: {pruning_criterion}. Must be one of: exact_loss, approx_loss, probability")

        # Refactor 3: Instantiate the correct pruning strategy
        if self.pruning_criterion == "exact_loss":
            self._pruning_strategy = ExactLossPruning()
        elif self.pruning_criterion == "approx_loss":
            self._pruning_strategy = ApproxLossPruning()
        elif self.pruning_criterion == "probability":
            self._pruning_strategy = ProbabilityPruning()
        else:
            # This case is already handled by the validation above, but good for defense.
            raise ValueError(f"Invalid pruning_criterion: {pruning_criterion}")

        # multigram‑specific
        self.lang_codes: List[str] = []
        self.global_lprobs: Dict[str, float] = {}
        
        # metadata collection
        self._training_metadata = {
            "start_time": None,
            "training_metrics": {}
        }

    # ------------------------- convenience helpers -----------------------
    def __getattr__(self, name):
        """
        Delegate every unknown attribute to the underlying HF `Tokenizer`
        once it is loaded / trained.  That gives us `id_to_token`,
        `decode_batch`, `.model`, etc. “for free”.
        """
        if self.base_tokenizer and hasattr(self.base_tokenizer, name):
            return getattr(self.base_tokenizer, name)
        raise AttributeError(name)

    def __call__(self, texts, *, add_special_tokens: bool = False, **kwargs):
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not yet loaded or trained.")

        if isinstance(texts, list):
            return self.encode_batch(texts, add_special_tokens=add_special_tokens)
        return self.base_tokenizer.encode(texts, add_special_tokens=add_special_tokens)

    _padding_side_default = "left"    # decoder-only models

    @property
    def padding_side(self):
        """Return 'left' or 'right'. Default is set at class level."""
        return getattr(self, "_padding_side", self._padding_side_default)

    @padding_side.setter
    def padding_side(self, value: str):
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value

    def save_pretrained(self, output_dir: str, **kwargs):
        """
        HuggingFace-style saver so the Trainer can checkpoint.
        It writes:
            • tokenizer.json            (the HF fast tokenizer)
            • special_tokens_map.json   (so `AutoTokenizer` can reload)
        For simplicity we store only the *base* tokenizer; language-specific
        probability tables are not needed for decoding and would bloat
        checkpoints.
        """
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not initialised / trained.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1) write the tokenizer.json
        tok_json_path = out / "tokenizer.json"
        tok_json_path.write_text(self.base_tokenizer.to_str(), encoding="utf-8")

        # 2) write special_tokens_map.json  (HF expects it)
        special_map = {
            "unk_token": self.unk_token,
            "pad_token": getattr(self, "pad_token", None),
            "bos_token": getattr(self, "bos_token", None),
            "eos_token": getattr(self, "eos_token", None),
        }
        # drop Nones
        special_map = {k: v for k, v in special_map.items() if v is not None}
        (out / "special_tokens_map.json").write_text(
            json.dumps(special_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(out)

    @classmethod
    def from_pretrained(cls, pretrained_dir: str, **kwargs):
        """
        Lightweight loader that mirrors save_pretrained.
        It initialises the class and loads the *base* tokenizer; language-
        specific add-ons are not restored here.
        """
        tok_file = Path(pretrained_dir) / "tokenizer.json"
        if not tok_file.is_file():
            raise FileNotFoundError(tok_file)
        obj = cls(**kwargs)
        obj.base_tokenizer = Tokenizer.from_file(str(tok_file))
        # attempt to restore special tokens
        sp_map = Path(pretrained_dir) / "special_tokens_map.json"
        if sp_map.is_file():
            specials = json.loads(sp_map.read_text(encoding="utf-8"))
            for attr, val in specials.items():
                setattr(obj, attr, val)
        return obj
    
    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        """
        Minimal subset of PreTrainedTokenizerBase.pad expected by
        DataCollatorWithPadding.
        • `encoded_inputs` is a list of dicts coming from your Dataset.
        • We pad input_ids (and, if present, attention_mask / token_type_ids)
          to the longest length in the batch (or `max_length` if provided).
        """
        if not isinstance(encoded_inputs, (list, tuple)):
            encoded_inputs = [encoded_inputs]

        # ---------------- figure out target length -------------------------
        pad_id = getattr(self.base_tokenizer, "pad_token_id", self.get_vocab()[constants.SPECIAL_TOKENS["pad_token"]])
        longest = max(len(f["input_ids"]) for f in encoded_inputs)
        if max_length is not None:
            longest = max_length
        if pad_to_multiple_of:
            longest = (longest + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        # ---------------- pad each field we know about --------------------
        batch = {}
        for field in encoded_inputs[0].keys():
            values = [f[field] for f in encoded_inputs]
            if all(v is None for v in values):
                continue
            # Handle already tensor values
            if len(values) > 0 and isinstance(values[0], torch.Tensor):
                # Stack tensors
                batch[field] = torch.stack(values)
                continue
            
            # Check if this is a sequence field that needs padding
            if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
                # This is a sequence field (input_ids, attention_mask, etc.)
                if field == "input_ids":
                    padded = [v[:longest] + [pad_id] * (longest - len(v[:longest])) for v in values]
                elif field == "attention_mask":
                    padded = [v[:longest] + [0] * (longest - len(v[:longest])) for v in values]
                elif field == "token_type_ids":
                    # For token_type_ids, pad with the last value (usually 1 for second segment)
                    padded = []
                    for v in values:
                        v_truncated = v[:longest]
                        last_value = v_truncated[-1] if v_truncated else 0
                        padded.append(v_truncated + [last_value] * (longest - len(v_truncated)))
                elif field == "labels":
                    # For labels, pad with -100 (ignored in loss)
                    padded = [v[:longest] + [-100] * (longest - len(v[:longest])) for v in values]
                else:
                    # For other sequence fields, pad with 0
                    logger.info(f"Unknown field type {field}; padding with 0s")
                    padded = [v[:longest] + [0] * (longest - len(v[:longest])) for v in values]
                
                batch[field] = torch.tensor(padded, dtype=torch.long) if return_tensors == "pt" else padded
            
            else:
                # This is a scalar field (label, language tag, etc.)
                # Only convert to tensor if it's numeric
                if len(values) > 0:
                    first_val = values[0]
                    
                    # Check if values are numeric
                    if isinstance(first_val, (int, float)):
                        try:
                            batch[field] = torch.tensor(values) if return_tensors == "pt" else values
                        except:
                            # If conversion fails, keep as list
                            batch[field] = values
                    else:
                        # Non-numeric field (like language tags), keep as is
                        batch[field] = values
    
        # If attention_mask wasn’t in the features, create it now
        if "attention_mask" not in batch:
            batch["attention_mask"] = (
                torch.ones_like(batch["input_ids"]) if return_tensors == "pt"
                else [[1]*len(f["input_ids"]) + [0]*(longest-len(f["input_ids"]))
                      for f in encoded_inputs]
            )

        return batch

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
            self.global_lprobs = self._log_normalize({tk: lp for tk, lp in vocab_pairs})
        else:
            logger.warning(f"No global probs for tokenizer at {base_path} found")

    def encode(self, text, add_special_tokens=False):
        """
        Encode text using the tokenizer.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded or trained.")
        return self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts, *, add_special_tokens: bool = False):
        """
        Vectorised encoding that returns *Encoding* objects, identical to the
        HuggingFace `tokenizers.Tokenizer.encode_batch`.
        """
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not yet loaded or trained.")
        return self.base_tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

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

    # ------------------------------------------------------------------
    # 2. top-level training dispatch
    # ------------------------------------------------------------------

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

        # Collect initial metadata
        self._training_metadata.update(
            self._collect_training_metadata(corpus_files, corpus_info)
        )
        self._training_metadata["start_time"] = time.time()
        
        # Track training metrics
        self._training_metadata["training_metrics"] = {
            "initial_vocab_size": None,
            "pruned_per_iteration": [],
            "convergence": {"l1_deltas": [], "unk_probs": []},
            "training_time": None,
            "final_loss": None,
            "pruning_criterion": self.pruning_criterion
        }

        token_level = "byte" if self.byte_level else "character"
        logger.info(f"Training new tokenizer with EM mode: {self.em_mode} at {token_level}-level")
        logger.info(f"Training on {len(corpus_files)} files. First file: '{corpus_files[0]}'")
        
        # Different training approaches
        if self.em_mode == "bpe":
            result = self._train_bpe(corpus_files, output_path)

        elif self.em_mode == "hf" and not self.multigram:
            # ============= HF Approach =============
            result = self._train_hf(corpus_files, output_path)
            
        elif self.multigram:
            # ============= Multigram Approach =============
            if not corpus_info:
                raise ValueError("corpus_info with per-language data is required for multigram training")
            result = self._train_multigram(corpus_info, output_path)
            
        elif self.em_mode in ["hard", "soft"]:
            # ============= Custom EM Approach (hard/soft) =============
            result = self._train_custom_em(corpus_files, output_path)
        else:
            raise ValueError("Invalid settings for StandardUnigramLMTokenizer: no EM mode specified")
        
        # Finalize metadata
        if self._training_metadata["start_time"]:
            self._training_metadata["training_metrics"]["training_time"] = time.time() - self._training_metadata["start_time"]
        
        self._update_training_results(self._training_metadata["training_metrics"])
        
        return result

    def _train_custom_em(self, corpus_files: List[str], output_path=None):
        logging_path = _get_metadata_logging_path(output_path)
        self.global_lprobs = self._run_em_sp(corpus_files, per_lang=False, logging_path=logging_path)
        return self._build_tokenizer(self.global_lprobs, output_path)

    def _train_multigram(self, corpus_info: Dict[str, str], output_path=None):
        logging_path = _get_metadata_logging_path(output_path)
        self.global_lprobs = self._run_em_sp(None, per_lang=True, corpus_info=corpus_info, logging_path=logging_path)
        return self._build_tokenizer(self.global_lprobs, output_path)

    def _train_hf(self, corpus_files, output_path=None):
        """
        Train using the HuggingFace UnigramTrainer.
        """
        logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = self._get_pretokenizer()
        tok.decoder = self._get_decoder()

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
            
            # Add metadata to the saved JSON
            self._save_tokenizer_metadata_file(output_path)
            
            logger.info(f"Saved HF-based standard tokenizer with metadata to {output_path}")
        return self.base_tokenizer

    def _train_bpe(self, corpus_files: List[str], output_path=None):
        """Plain HF BPE trainer; shares most hyper‑params with EM modes."""
        logger.info("Using HuggingFace BPETrainer.")
        tok = Tokenizer(BPE(unk_token=self.unk_token))
        tok.pre_tokenizer = self._get_pretokenizer()
        tok.decoder = self._get_decoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.byte_level else [],
        )
        corpus_iter = proportional_corpus_iterator(corpus_files, total_bytes=1_500_000_000)
        tok.train_from_iterator(corpus_iter, trainer)
        self.base_tokenizer = tok
        if output_path:
            tok.save(output_path)
            
            # Add metadata to the saved JSON
            self._save_tokenizer_metadata_file(output_path)
            
            logger.info("Saved BPE tokenizer with metadata to %s", output_path)
        return tok

    # ------------------------------------------------------------------
    # 3. EM training loop & helpers
    # ------------------------------------------------------------------

    def _run_em_sp(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        logging_path: str = None
    ) -> None:
        #Pre-process files for SPM if needed; initialize EM state
        if self.use_sp_em or self.use_sp_seed_vocab:
            if per_lang:
                self.dataset_size_normalization = True  # normalized already by nature of using probabilities rather than counts
            self._temp_dir = tempfile.mkdtemp()
            processed_file_paths = None
            # Register a cleanup function to remove the dirpectory when the program exits.
            atexit.register(shutil.rmtree, self._temp_dir, ignore_errors=True)
            text_files = corpus_files if not per_lang else list(corpus_info.values())

            num_workers = min(min(os.cpu_count() or 1, 20), len(text_files))
            logger.info(f"Using {num_workers} worker processes.")

            # Prepare arguments for each worker
            worker_args = [(path, self._temp_dir, self._get_pretokenizer(), i) for i, path in enumerate(text_files)]

            with multiprocessing.Pool(processes=num_workers) as pool:
                # preserves order of input
                processed_file_paths = pool.map(_process_file_worker, worker_args)
                processed_mapping = {orig: processed for orig, processed in zip(text_files, processed_file_paths)}

            state = self._initialize_em_state(corpus_files, per_lang, corpus_info, processed_mapping)
        else:
            state = self._initialize_em_state(corpus_files, per_lang, corpus_info)

        # Store initial vocab size in metadata
        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["initial_vocab_size"] = len(state.all_tokens)

        target_vocab_size = self.vocab_size  # The final vocabulary size we are aiming for
        current_vocab_size = len(state.all_tokens) # Keep track of the current vocab size
        
        # This loop continues until the vocabulary is pruned down to the target size.
        pruning_iteration = 0
        
        counts = {}
        lang_counts = {}
        
        while current_vocab_size > target_vocab_size:
            pruning_iteration += 1 # Increment pruning iteration
            
            # --- Inner EM Loop (SP's `num_sub_iterations`) ---
            # Run E-M steps multiple times *before* the next pruning step.
            logger.info("SP-style Pruning Iteration %d – Vocab=%d | Target=%d", 
                       pruning_iteration, current_vocab_size, target_vocab_size) 
            
            prev_lprobs, prev_lang_lprobs = None, None # Init before inner loop
            
            for sub_it in range(constants.NUM_SUB_ITERATIONS):
                logger.info("  EM sub-iteration %d/%d", sub_it + 1, constants.NUM_SUB_ITERATIONS)
                prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)
                
                # Refactor 1: Extracted E+M step logic
                (counts, 
                 lang_counts, 
                 unk_count, 
                 total_counts, 
                 observed_tokens) = self._run_em_iteration(state)
                
                global_step = (pruning_iteration - 1) * constants.NUM_SUB_ITERATIONS + sub_it 
                self._log_iteration_diagnostics(
                    global_step, 
                    state, 
                    prev_lprobs, 
                    prev_lang_lprobs,
                    observed_tokens, 
                    unk_count, 
                    total_counts
                )
            # --- (End of Inner EM Loop) ---
            
            # 1. Determine new target size for this iteration
            num_candidates = current_vocab_size - len(state.pinned) # NEW: Get # of non-pinned tokens
            # Keep `shrinking_factor` % of candidates, plus all pinned tokens
            num_to_keep = int(num_candidates * constants.SHRINKING_FACTOR) + len(state.pinned) 
            # The new size is the max of the final target or the shrunk size
            new_target_size = max(target_vocab_size, num_to_keep) 
            
            # 2. Calculate how many tokens to drop
            k_drop = current_vocab_size - new_target_size 
            
            if k_drop <= 0: # Stop if no more tokens to drop
                logger.info("Pruning finished, vocab size %d reached target %d",
                           current_vocab_size, target_vocab_size)
                break

            logger.info("SP Pruning Iteration %d – Pruning %d tokens (from %d to %d)",
                       pruning_iteration, k_drop, current_vocab_size, new_target_size) # NEW: Logging

            # --- Pruning Logic: Calculate Loss and Remove Tokens) ---
            logging_dict = {}
            if state.per_lang:
                counts_for_logging = {tk: sum(lang_counts[lc].get(tk, 0.0) for lc in state.lang_codes) for tk in state.all_tokens}
            else:
                counts_for_logging = counts
                
            for tk in state.all_tokens:
                logging_dict[tk] = {
                    "pre-pruning-lprob": state.global_lprobs[tk],
                    "status": "pinned" if tk in state.pinned else "kept",  # default, changed later
                    "loss": "NA",
                    "post-pruning-prob": "NA",
                    "expected-counts": counts_for_logging.get(tk, 0.0) # NEW: Use counts from last EM
                }

            before = len(state.all_tokens)
            
            # Refactor 1: Extracted pruning loss calculation
            tokens_to_remove = self._get_tokens_to_remove_by_pruning(
                state, counts, lang_counts, k_drop, logging_dict
            )

            # Common token removal logic for both multigram and standard
            for tk in tokens_to_remove:
                logging_dict[tk]["status"] = "pruned"
            state.all_tokens -= tokens_to_remove
            pruned = before - len(state.all_tokens)
            logger.info("   tokens_pruned=%d | new_vocab=%d", pruned, len(state.all_tokens))
            
            # Store pruning metrics
            if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
                self._training_metadata["training_metrics"]["pruned_per_iteration"].append(pruned)

            if state.per_lang:
                for lc in state.lang_codes:
                    state.per_lang_lprobs[lc] = {tk: state.per_lang_lprobs[lc].get(tk, -np.inf) for tk in state.all_tokens}
                    state.per_lang_lprobs[lc] = self._log_normalize(state.per_lang_lprobs[lc])
                    
            state.global_lprobs = {tk: state.global_lprobs[tk] for tk in state.all_tokens}
            state.global_lprobs = self._log_normalize(state.global_lprobs)
            
            for tk, new_lprob in state.global_lprobs.items():
                if tk in logging_dict: 
                    logging_dict[tk]["post-pruning-prob"] = np.exp(new_lprob)
            
            if logging_path is not None:
                logging_file = f"{logging_path}_prune_{pruning_iteration}.pkl" 
                with open(logging_file, 'wb') as handle:
                    pickle.dump(logging_dict, handle)

            current_vocab_size = len(state.all_tokens) # Update loop variable
            
            assert all(x in state.global_lprobs for x in state.pinned)
        # --- (End of Outer Pruning Loop) ---
        
        # --- Final EM run ---
        # SP runs an E-M steps one last time *after* the final vocabulary
        # has been selected, to get the final probabilities.
        logger.info("Running final EM pass on the final vocabulary...")
        prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)

        # Refactor 1: Use extracted E+M step
        (counts, 
         lang_counts, 
         unk_count, 
         total_counts, 
         observed_tokens) = self._run_em_iteration(state)
        
        # Final M-Step (Normalization)
        if state.per_lang:
             for lc in state.lang_codes:
                 state.per_lang_lprobs[lc] = self._log_normalize(state.per_lang_lprobs[lc])
        state.global_lprobs = self._log_normalize(state.global_lprobs)
        
        logger.info(f"Final EM pass complete. Final vocab size: {len(state.all_tokens)}")
        
        return state.global_lprobs

    # Refactor 1: Extracted E+M iteration logic
    def _run_em_iteration(
        self, 
        state: EMState
    ) -> Tuple[Dict, Dict, float, float, Set]:
        """
        Runs one E-step and M-step of the EM algorithm by dispatching
        to the appropriate implementation (SentencePiece or custom).

        Mutates state.global_lprobs and state.per_lang_lprobs.
        Returns counts and diagnostics.
        """
        if self.use_sp_em:
            return self._run_em_iteration_sp(state)
        else:
            return self._run_em_iteration_custom(state)

    def _run_em_iteration_sp(
        self,
        state: EMState
    ) -> Tuple[Dict, Dict, float, float, Set]:
        """Runs one E-M iteration using the SentencePiece backend."""
        logger.info("Using SentencePiece for EM computation...")
        counts = {}
        lang_counts = {}

        if state.per_lang:
            # per-language re-estimation, then mix
            for lc, f in zip(state.lang_codes, state.corpus_files):
                state.per_lang_lprobs[lc] = self._sp_em(state.per_lang_lprobs[lc], f)
            
            state.global_lprobs = {
                tk: np.logaddexp.reduce([self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf)
                        for lc in state.lang_codes], axis=-1)
                for tk in state.all_tokens
            }
            
            lang_counts = {lc: {tk: np.exp(lp) for tk, lp in state.per_lang_lprobs[lc].items()} for lc in state.lang_codes}
            unk_count = sum(lang_counts[lc].get(self.unk_token, 0.0) for lc in state.lang_codes)
        else:
            # single-corpus re-estimation
            state.global_lprobs = self._sp_em(state.global_lprobs, ','.join(state.corpus_files))
            counts = {tk: np.exp(lp) for tk, lp in state.global_lprobs.items()}
            unk_count = counts.get(self.unk_token, 0.0)

        observed_tokens = set(state.all_tokens)
        total_counts = 1.0  # probabilities-as-counts            
        return counts, lang_counts, unk_count, total_counts, observed_tokens

    def _run_em_iteration_custom(
        self,
        state: EMState
    ) -> Tuple[Dict, Dict, float, float, Set]:
        """Runs one E-M iteration using the custom Python implementation."""
        logger.info("Using custom implementation for EM computation...")
        counts = {}
        lang_counts = {}

        if state.per_lang:
            lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
            unk_count = 0
            total_counts = 0
            for lc, file in zip(state.lang_codes, state.corpus_files):
                c_dict, _ = _accumulate_usage(file, 
                                              state.per_lang_lprobs[lc], 
                                              self.unk_token, 
                                              mode=self.em_mode, 
                                              whitespace_boundaries=self.whitespace_token_boundaries,
                                              pretokenizer=self._get_pretokenizer())
                for tk, v in c_dict.items():
                    lang_counts[lc][tk] += v
                unk_count += lang_counts[lc].get(self.unk_token, 0.0) # Ensure unk is handled
                total_counts += sum(lang_counts[lc].values())
            
            observed_tokens = set([k for counts in lang_counts.values() for k in counts])
            
            for lc in state.lang_codes:
                state.per_lang_lprobs[lc] = self._counts_to_log_probs(lang_counts[lc])
            
            state.global_lprobs = {tk: np.logaddexp.reduce(
                [self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf) for lc in state.lang_codes]
                ) for tk in state.all_tokens}
        else:
            counts = {tk: 0.0 for tk in state.all_tokens}
            for file in state.corpus_files:
                c_dict, _ = _accumulate_usage(file, state.global_lprobs, self.unk_token, 
                                              mode=self.em_mode, 
                                              whitespace_boundaries=self.whitespace_token_boundaries, 
                                              pretokenizer=self._get_pretokenizer())
                for tk, v in c_dict.items():
                    counts[tk] += v
            
            unk_count = counts.get(self.unk_token, 0.0) # Ensure unk is handled
            total_counts = sum(counts.values())
            observed_tokens = counts.keys()
            state.global_lprobs = self._counts_to_log_probs(counts)
            
        return counts, lang_counts, unk_count, total_counts, observed_tokens

    # Refactor 1: Extracted pruning loss calculation logic
    def _get_tokens_to_remove_by_pruning(
        self,
        state: EMState,
        counts: Dict[str, float],
        lang_counts: Dict[str, Dict[str, float]],
        k_drop: int,
        logging_dict: Optional[Dict] = None
    ) -> Set[str]:
        """
        Calculates pruning losses and determines which tokens to remove.
        If logging_dict is provided, it mutates it with loss information.
        Returns a set of tokens to remove.
        """
        if state.per_lang:
            # For multilingual, compute alternatives for each language
            lang_alternative_sequences = {}
            if self.pruning_criterion == "exact_loss":
                for lc, file in zip(state.lang_codes, state.corpus_files):
                    lang_alternative_sequences[lc] = _compute_alternatives_direct(state.per_lang_lprobs[lc], self.unk_token, byte_level=self.byte_level)
        else:
            alternative_sequences = _compute_alternatives_direct(state.global_lprobs, self.unk_token, byte_level=self.byte_level) if self.pruning_criterion != "probability" else None

        under_consideration = state.all_tokens - state.pinned
        if state.per_lang:
            # For multilingual case
            # Compute per-language losses first
            per_lang_losses = self._compute_per_language_losses(
                state.per_lang_lprobs, lang_counts, under_consideration, lang_alternative_sequences
            )
            if self.use_intersection_pruning:
                tokens_to_remove = compute_intersection_pruning_tokens(
                    per_lang_losses, k_drop, self.lang_codes, under_consideration
                )

            else:
                # Average-loss approach (with dataset size normalization)
                # Compute dataset sizes for normalization if enabled
                dataset_sizes = {}
                if self.dataset_size_normalization and not self.use_sp_em:
                    dataset_sizes = {lc: sum(lang_counts[lc].values()) for lc in state.lang_codes}
                    logger.info("Dataset sizes for normalization: %s", 
                               {lc: f"{size:.0f}" for lc, size in dataset_sizes.items()})
                # Then aggregate them
                loss = self._compute_multigram_pruning_scores(
                    state, per_lang_losses, dataset_sizes
                )
                
                if logging_dict:
                    for tk, total_loss in loss:
                        logging_dict[tk]["loss"] = total_loss
                
                tokens_to_remove = self._tokens_to_remove(loss, k_drop)
        else:
            # Use centralized standard pruning score computation
            loss = self._compute_standard_pruning_scores(
                state, counts, alternative_sequences
            )
            
            # Update logging
            if logging_dict:
                for tk, tk_loss in loss:
                    logging_dict[tk]["loss"] = tk_loss

            tokens_to_remove = self._tokens_to_remove(loss, k_drop)
        
        return tokens_to_remove

    def _log_iteration_diagnostics(
        self,
        iteration: int,
        state: EMState,
        prev_global_lprobs: Dict[str, float],
        prev_per_lang_lprobs: Optional[Dict[str, Dict[str, float]]],
        observed_tokens: Set[str],
        unk_count: float,
        total_counts: float
    ) -> None:
        """Log diagnostic information for the current EM iteration."""
        logger.info(f"Observed {len(observed_tokens)} unique tokens out of {len(state.all_tokens)} possible tokens across corpus...")
        logger.info(f"Observed {unk_count} UNK tokens out of {total_counts}...")

        # Diagnostics before pruning
        unk_prob = np.exp(state.global_lprobs.get(self.unk_token, -np.inf))
        l1_delta = 0.5 * sum(abs(np.exp(state.global_lprobs.get(tk, -np.inf)) - np.exp(prev_global_lprobs.get(tk, -np.inf))) for tk in state.all_tokens)
        logger.info(" UNK prob: %.8g | Δ‑dist: %.3e | min=%.3e max=%.3e", 
                   unk_prob, l1_delta, np.exp(min(state.global_lprobs.values())), np.exp(max(state.global_lprobs.values())))
        
        # Store convergence metrics
        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["convergence"]["unk_probs"].append(float(unk_prob))
            self._training_metadata["training_metrics"]["convergence"]["l1_deltas"].append(float(l1_delta))
        
        if state.per_lang:
            for lc in state.lang_codes:
                up = np.exp(state.per_lang_lprobs[lc][self.unk_token])
                d = 0.5 * sum(abs(np.exp(state.per_lang_lprobs[lc].get(tk, -np.inf)) - np.exp(prev_per_lang_lprobs[lc].get(tk, -np.inf))) 
                             for tk in state.all_tokens)
                logger.info("      [%s] UNK=%.8g | Δ‑dist=%.3e", lc, up, d)

    # ------------------------------------------------------------------
    # 4. pruning & loss helpers
    # ------------------------------------------------------------------    

    def _compute_standard_pruning_scores(
        self,
        state: EMState,
        counts: Dict[str, float],
        alternative_sequences: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute pruning scores for standard (non-multigram) case.
        Returns list of (token, score) tuples ready for _tokens_to_remove.
        """
        under_consideration = state.all_tokens - state.pinned
        return [
            (tk, self._pruning_strategy.compute_loss(
            tk, counts, state.global_lprobs, alternative_sequences, self.unk_token))
            for tk in under_consideration
        ]

    def _compute_multigram_pruning_scores(
        self,
        state: EMState,
        per_lang_losses: Dict[str, List[Tuple[str, float]]],
        dataset_sizes: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute aggregated pruning scores from pre-computed per-language losses.
        
        Args:
            state: EM state containing language codes and mixture weights
            per_lang_losses: Dict[lang_code, List[Tuple[token, loss]]] from _compute_per_language_losses
            dataset_sizes: Optional dataset sizes for normalization
        
        Returns:
            List[Tuple[token, aggregated_loss]] ready for _tokens_to_remove
        """
        # Convert per-language losses to a more efficient lookup structure
        per_lang_loss_dict = {
            lc: {token: loss for token, loss in losses} 
            for lc, losses in per_lang_losses.items()
        }
        
        # Get tokens to consider (assuming all languages have same token set)
        under_consideration = set(per_lang_loss_dict[state.lang_codes[0]].keys())
        
        # Aggregate losses across languages
        aggregated_losses = []
        for tk in under_consideration:
            total_loss = 0.0
            for lc in state.lang_codes:
                lang_loss = per_lang_loss_dict[lc][tk]
                
                # Apply dataset size normalization if enabled
                if self.dataset_size_normalization and dataset_sizes and dataset_sizes.get(lc, 0) > 0:
                    normalized_lang_loss = lang_loss / dataset_sizes[lc]
                else:
                    normalized_lang_loss = lang_loss
                    
                total_loss += self.mixture_weights[lc] * normalized_lang_loss
            
            aggregated_losses.append((tk, total_loss))
        
        return aggregated_losses

    def _compute_per_language_losses(
        self,
        per_lang_lprobs: Dict[str, Dict[str, float]],
        per_lang_counts: Dict[str, Dict[str, float]],
        under_consideration: Set[str],
        lang_alternative_sequences: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute raw per-language losses without aggregation for intersection pruning.
        Returns Dict[lang_code, List[Tuple[token, loss]]]
        """
        return {
            lc: [
                (tk, self._pruning_strategy.compute_loss(
                    tk, per_lang_counts[lc], per_lang_lprobs[lc], 
                    lang_alternative_sequences[lc] if lang_alternative_sequences else None, 
                    self.unk_token))
                for tk in under_consideration
            ]
            for lc in self.lang_codes
        }

    # ------------------------------------------------------------------
    # 5. vocabulary & initialization
    # ------------------------------------------------------------------

    def _initialize_em_state(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        processed_mapping: Optional[Dict[str, str]] = None
    ) -> EMState:
        """Initialize EM training state and data structures."""
        # Validation
        if per_lang and not corpus_info:
            raise ValueError("corpus_info required when multigram=True")

        # Gather training files and language codes
        lang_codes = []
        if per_lang:
            lang_codes = list(corpus_info.keys())
            corpus_files = [corpus_info[lc] for lc in lang_codes]
            self.lang_codes = lang_codes  # Keep instance variable for compatibility

        if processed_mapping:
            processed_corpus_files = [processed_mapping[f] for f in corpus_files]

        # Shared preparation
        files_for_init_vocab = processed_corpus_files if self.use_sp_seed_vocab else corpus_files
        all_tokens = self._initial_vocab(files_for_init_vocab)
        
        # TODO: make adding baseline tokens optional if using custom vocabulary
        # TODO: check for char-level whether this should be done pre or post processing
        baseline = self.baseline_tokens(corpus_files, self.byte_level, self.min_char_occurence_threshold)
        pinned = set(self.special_tokens) | baseline | {self.unk_token}
        if len(pinned) > self.vocab_size:
            raise ValueError(f"Number of pinned tokens ({len(pinned)}) exceeds vocabulary size ({self.vocab_size}). "
                "Change vocab size, special token set size, or baseline character set.")
        
        # double check all pinned tokens are in vocab
        all_tokens.update(pinned)

        # Initialize probability distributions
        uniform_lprob = np.log(1.0 / len(all_tokens))
        if per_lang:
            per_lang_lprobs = {lc: {tk: uniform_lprob for tk in all_tokens} for lc in lang_codes}
            self.mixture_weights = self._sanitize_mixture_weights(lang_codes, self.mixture_weights)
            self.log_mixture_weights = {lc: np.log(w) for lc, w in self.mixture_weights.items()}

        else:
            per_lang_lprobs = {}

        global_lprobs = {tk: uniform_lprob for tk in all_tokens}
        
        # Pruning schedule
        k_total = len(all_tokens) - self.vocab_size
        k_step = [math.floor(k_total / constants.NUM_TRIMMING_ITERATIONS)] * constants.NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - constants.NUM_TRIMMING_ITERATIONS)

        return EMState(
            corpus_files=corpus_files if not self.use_sp_em else processed_corpus_files,
            per_lang=per_lang,
            lang_codes=lang_codes,
            all_tokens=all_tokens,
            pinned=pinned,
            global_lprobs=global_lprobs,
            per_lang_lprobs=per_lang_lprobs,
            k_total=k_total,
            k_step=k_step
        )

    def _initial_vocab(self, train_files: List[str]) -> Set[str]:
        """Bootstrap an initial token set - either from custom vocab file or HF trainer."""
        if self.initial_vocab_tokens:
            # Use custom vocabulary from file
            logger.info(f"Loading custom initial vocabulary from {self.initial_vocab_tokens}")
            custom_tokens = _load_custom_vocab_from_file(self.initial_vocab_tokens)
            
            vocab = set(custom_tokens)
            vocab.update(self.special_tokens)
            logger.info(f"Custom vocabulary prepared: {len(vocab)} tokens")

        elif self.use_sp_seed_vocab:
            logger.info("Using SentencePiece suffix-array seeding for initial vocabulary...")
            vocab = self._sp_seed_vocab(train_files)
            # ensure specials are present
            vocab.update(self.special_tokens)
            logger.info(f"SP-seeded initial vocabulary size: {len(vocab)}")
        else:
            # Use existing HF-based approach  
            vocab = self._initial_vocab_hf(train_files)
            logger.info(f"HF-seeded initial vocabulary size: {len(vocab)}")

        ws_count, ws_in_word_count = 0, 0
        for tk in vocab:
            if constants._HF_SPACE in tk:
                ws_count += 1
                if constants._HF_SPACE in tk[1:-1]:
                    ws_in_word_count += 1
        logger.info(f"{ws_count} tokens with whitespace characters ({ws_in_word_count} where whitespace is within token)")
        logger.info(f"Seed vocab sample: {random.sample(list(vocab), 50)}")
        
        return vocab

    def _sp_seed_vocab(self, train_files: List[str]) -> Set[str]:
        """
        Use patched SP (num_sub_iterations=0, no seed file) to extract a seed vocabulary
        via suffix-array seeding. Returns a set of token strings (HF-style).
        """
        
        if spm is None:
            raise RuntimeError("sentencepiece is not installed but use_sp_seed_vocab=True")

        seed_vocab_size = self.vocab_size * constants.INITIAL_VOCAB_MULT_FACTOR

        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "model")

            # 2. Construct the spm_train command
            command = ["spm_train"] + constants.SP_DEFAULT_ARGS + [
                f"--input={','.join(train_files)}",
                f"--model_prefix={out_prefix}",

                f"--seed_sentencepiece_size={seed_vocab_size}",
                f"--max_sentencepiece_length={constants.MAX_BYTE_TOKEN_LEN if self.byte_level else constants.MAX_CHAR_TOKEN_LEN}",
                f"--split_by_whitespace={'true' if self.whitespace_token_boundaries else 'false'}",  
                # -----------------------

                "--num_sub_iterations=0",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count()-1)}",
                f"--vocab_size={seed_vocab_size}"
            ] 

            # 3. Execute the command
            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            try:
                output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            except subprocess.CalledProcessError as e:
                match = re.search(r'<=\s*(\d+)', e.stderr)
                if match:
                    max_vocab = int(match.group(1))
                    logger.warning(f"Failed to create seed vocabulary with size {seed_vocab_size}. Maximum allowed vocabulary size: {max_vocab}. Trying again...")
                    
                    command[-1] = f"--vocab_size={max_vocab}"
                    output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                else:
                    raise e
 
            logger.info("SentencePiece seed creation successful.")
            logger.info(f"SentencePiece STDOUT: {output.stdout}")
            logger.info(f"SentencePiece STDERR (logs + errors): : {output.stderr}")

            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            vocab = set()
            for i in range(sp.get_piece_size()):
                p = sp.id_to_piece(i)
                hf_p = p#sp_piece_to_hf(p)
                exceeds_len = (self.byte_level and len(hf_p.encode("utf-8")) > constants.MAX_BYTE_TOKEN_LEN) or (not self.byte_level and len(hf_p) > constants.MAX_CHAR_TOKEN_LEN)
                if not exceeds_len: 
                    vocab.add(hf_p if p not in self.special_tokens else p)
                
            return vocab
    
    def _initial_vocab_hf(self, train_files: List[str]) -> Set[str]:
        """Original HF trainer to bootstrap an initial token set."""
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = self._get_pretokenizer()
        logger.info("--- Pre-tokenizer Sample Output ---: " + str(tok.pre_tokenizer.pre_tokenize_str("This is a test")))
        seed_vocab_size = self.vocab_size * constants.INITIAL_VOCAB_MULT_FACTOR
        logger.info(f"Attempting to find seed vocab of size {seed_vocab_size} with HuggingFace library...")
        trainer = UnigramTrainer(
            vocab_size=seed_vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token,
            max_piece_length=constants.MAX_BYTE_TOKEN_LEN if self.byte_level else constants.MAX_CHAR_TOKEN_LEN,
            # not strictly necessary since these will be added back later
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.byte_level else [],
            n_sub_iterations=0
        )

        # Need to use train_from_iterator for pretokenization to be applied
        # corpus_iterator = file_iterator(train_files)
        # tok.train(train_files, trainer)
        corpus_iter = proportional_corpus_iterator(train_files, total_bytes=1_500_000_000)
        tok.train_from_iterator(corpus_iter, trainer)
        vocab = tok.get_vocab()        
        
        return set(vocab.keys())

    def _sp_em(
            self,
            vocab_dict: Dict[str, int],
            text_paths: str
        ) -> Dict[str, float]:       
            # Check if spm_train is available
            if spm is None:
                raise RuntimeError("spm_train executable not found. Make sure sentencepiece is installed and in your PATH.")

            tmpdir = self._temp_dir
            out_prefix = os.path.join(tmpdir, "model")
            vocab_path = os.path.join(tmpdir, "vocab.tsv")
            # 1. Prepare the vocabulary file
            token_count = _write_sp_seed_vocab_file(list(vocab_dict.items()), vocab_path, self.unk_token)

            # The exact vocab size must account for meta pieces
            # <unk>, <s> and </s> 
            expected_vocab_size = token_count + 3
            
            # 2. Construct the spm_train command
            command = ["spm_train"] + constants.SP_DEFAULT_ARGS + [
                f"--input={text_paths}",
                f"--model_prefix={out_prefix}",
                
                f"--seed_sentencepieces_file={vocab_path}",
                f"--vocab_size={expected_vocab_size}",
                # -----------------------

                f"--num_sub_iterations=1",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count()-1)}"
            ]

            # 3. Execute the command
            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info("SentencePiece training successful.")
            logger.info(f"SentencePiece STDOUT: {output.stdout}")
            logger.info(f"SentencePiece STDERR (logs + errors): : {output.stderr}")

            # 4. Load the resulting model and extract probabilities
            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            # 1) Collect raw log-scores (SentencePiece scores are already log-scale).
            token_logps = {}
            additional_tokens = set()
            for i in range(sp.get_piece_size()):
                p = sp.id_to_piece(i)
                p_hf = p  # sp_piece_to_hf(p)
                if p_hf not in vocab_dict:
                    additional_tokens.add(p_hf)
                if p_hf in set(constants.SPECIAL_TOKENS.values()) | {self.unk_token}:
                    logp = float(vocab_dict[p_hf])
                else:
                    logp = float(sp.get_score(i))  # SP log-score
                token_logps[p_hf] = logp
            if additional_tokens:
                logger.warning(f"unknown tokens: {additional_tokens}")
            # 2) Ensure every base token is present; default to base log-prob if missing.
            missing_tokens = set()
            for t in vocab_dict.keys():

                if t not in token_logps:
                    missing_tokens.add(t)
                    token_logps[t] = float(vocab_dict[t])
            if missing_tokens:
                logger.warning(f"Tokens {missing_tokens} not in sentencepiece model; defaulting to base "
                               f"tokenizer log-prob {vocab_dict[t]:.5f}")
            # 3) Normalize in log-space: log p_i := log p_i - logsumexp(log p)
            token_logps = self._log_normalize(token_logps)
            return token_logps

    def _build_final_vocab(
        self, lprobs: Dict[str, float]
    ) -> Tuple[List[Tuple[str, float]], int]:
        normalized_lprobs = self._log_normalize(lprobs, use_floor=True)
        sorted_items = sorted(normalized_lprobs.items(), key=lambda x: x[1], reverse=True)
        vocab_list = []
        special = []
        for tk, lp in sorted_items:
            if tk in self.special_tokens:
                special.append((tk, lp))
            else:
                vocab_list.append((tk, lp))
        vocab_list = special + vocab_list
        unk_id = next(i for i, (tk, _) in enumerate(vocab_list) if tk == self.unk_token)
        return vocab_list, unk_id

    # ------------------------------------------------------------------
    # 6. tokenizer building & pre-processing
    # ------------------------------------------------------------------

    def _build_tokenizer(self, lprobs: Dict[str, float], output_path):
        vocab_list, unk_id = self._build_final_vocab(lprobs)
        tok = _build_unigramlm_hf_tokenizer_from_lprobs(
            vocab_list,
            unk_id,
            pretokenizer=self._get_pretokenizer(),
            decoder=self._get_decoder()
        )
        self.base_tokenizer = tok
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(output_path)
            
            # Add metadata to the saved JSON
            self._save_tokenizer_metadata_file(output_path)
            
            logger.info("Tokenizer with metadata saved to %s", output_path)
        return tok

    def _get_tokenizer_type(self):
        """Determine tokenizer type string"""
        if self.em_mode == "bpe":
            return "bpe"
        elif self.multigram:
            return "multigram" 
        else:
            return "standard"

    def _get_pretokenizer(self):
        if self.base_tokenizer:
            logger.info("Using pretokenizer from base tokenizer")
            return self.base_tokenizer.pre_tokenizer
        if self.byte_level:
            logger.info("Using byte-level pretokenizer")
            return pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=self.whitespace_token_boundaries)
        elif self.whitespace_token_boundaries:
            logger.info("Using whitespace pretokenizer")
            return pre_tokenizers.Whitespace()
        else:
            logger.info("No pretokenizer found")
            return None

    def _get_decoder(self):
        if self.base_tokenizer:
            return self.base_tokenizer.decoder
        if self.byte_level:
            return decoders.ByteLevel()
        elif self.whitespace_token_boundaries: 
            raise NotImplementedError
        else:
            return None

    # ------------------------------------------------------------------
    # 7. metadata collection
    # ------------------------------------------------------------------
    
    def _save_tokenizer_metadata_file(self, tokenizer_path):
        """Save metadata to a separate .metadata.json file alongside the tokenizer"""

        try:
            logging_path = _get_metadata_logging_path(tokenizer_path)
            metadata_path = _save_tokenizer_metadata(logging_path, tokenizer_path, self._training_metadata)
            if metadata_path:
                logger.debug(f"Saved metadata to {metadata_path}")
            else:
                logger.warning(f"Failed to save metadata for {tokenizer_path}")

        except Exception as e:
            logger.warning(f"Failed to save metadata for {tokenizer_path}: {e}")

    # --------- metadata collection utilities ---------
    def _collect_training_metadata(self, corpus_files, corpus_info=None):
        """Collect metadata about training configuration and corpus"""
        metadata = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": self.__class__.__name__,
            "tokenizer_type": self._get_tokenizer_type(),
        }
        
        # Training configuration
        training_config = {
            "vocab_size": self.vocab_size,
            "em_mode": self.em_mode,
            "num_iterations": self.num_iterations,
            "multigram": self.multigram,
            "byte_level": self.byte_level,
            "dataset_size_normalization": self.dataset_size_normalization,
            "use_intersection_pruning": self.use_intersection_pruning,
            "pruning_criterion": self.pruning_criterion,
            "special_tokens": list(self.special_tokens)
        }
        
        # Add custom vocabulary info if used
        if self.initial_vocab_tokens:
            training_config["initial_vocab_source"] = "custom_file"
            training_config["initial_vocab_file"] = self.initial_vocab_tokens
        elif self.use_sp_seed_vocab:
            training_config["initial_vocab_source"] = "sp_generated"
        else:
            training_config["initial_vocab_source"] = "hf_generated"
        
        if self.multigram:
            training_config["mixture_weights"] = dict(self.mixture_weights)
            
        metadata["training_config"] = training_config
        
        # Corpus information
        corpus_metadata = self._collect_corpus_metadata(corpus_files, corpus_info)
        metadata["corpus_info"] = corpus_metadata
        
        # Environment info
        metadata["environment_info"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "num_processes": multiprocessing.cpu_count()
        }
        
        return metadata
    
    def _collect_corpus_metadata(self, corpus_files, corpus_info=None):
        """Collect metadata about training corpus"""
        corpus_metadata = {
            "training_files": corpus_files or [],
            "total_files": len(corpus_files) if corpus_files else 0,
        }
        
        if corpus_info:  # multigram case
            corpus_metadata["language_codes"] = list(corpus_info.keys())
            corpus_metadata["per_language_files"] = dict(corpus_info)
        
        # Compute file checksums and sizes
        corpus_sizes = {}
        file_checksums = {}
        total_lines = 0
        
        files_to_check = corpus_files or []
        if corpus_info:
            files_to_check = list(corpus_info.values())
            
        for file_path in files_to_check:
            if os.path.exists(file_path):
                # File checksum
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    file_checksums[file_path] = file_hash
                
                # Line count
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                    corpus_sizes[file_path] = lines
                    total_lines += lines
        
        corpus_metadata["corpus_sizes"] = corpus_sizes
        corpus_metadata["file_checksums"] = file_checksums
        corpus_metadata["total_lines"] = total_lines
        
        return corpus_metadata

    
    def _update_training_results(self, training_metrics):
        """Update training results in metadata"""
        self._training_metadata["training_results"] = {
            "initial_vocab_size": training_metrics.get("initial_vocab_size"),
            "final_vocab_size": len(self.global_lprobs) if hasattr(self, 'global_lprobs') else None,
            "tokens_pruned_per_iteration": training_metrics.get("pruned_per_iteration", []),
            "convergence_metrics": training_metrics.get("convergence", {}),
            "training_time_seconds": training_metrics.get("training_time"),
            "final_loss": training_metrics.get("final_loss")
        }

    # ------------------------------------------------------------------
    # 8. static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def baseline_tokens(files: List[str], 
                    byte_level: bool, 
                    min_char_occurence_threshold: int = constants.MIN_CHAR_OCCURENCE_THRESHOLD
                    ) -> Set[str]:
        return (
            get_baseline_bytes()[0]
            if byte_level
            else set(get_baseline_characters(files, min_occurence_threshold=min_char_occurence_threshold)[0])
        )

    @staticmethod
    def _sanitize_mixture_weights(lang_codes: List[str], user_weights: Dict[str, float]) -> Dict[str, float]:
        if not lang_codes:
            return {}
        if not user_weights:
            uniform = 1.0 / len(lang_codes)
            return {lc: uniform for lc in lang_codes}
        else: 
            weights = {lc: user_weights.get(lc, 0.0) for lc in lang_codes}
            s = sum(weights.values()) 
            if s <= 0:
                raise ValueError("Language mixture weights cannot sum to <= 0")
            return {lc: w / s for lc, w in weights.items()}


    @staticmethod
    def _normalize(values: Dict[str, float]) -> Dict[str, float]:
        """Normalize to a valid prob dist."""
        total = sum(values.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(values))
            return {t: uniform for t in values}
        return {t: values[t] / total for t in values}


    @staticmethod
    def _counts_to_log_probs(counts: Dict[str, float]) -> Dict[str, float]:
        total = sum(counts.values())
        log_total = math.log(total)
        return {k: math.log(v) - log_total if v > 0 else constants.MIN_TOKEN_LOG_PROB for k, v in counts.items()}

    @staticmethod
    def _log_normalize(logvals: Dict[str, float], use_floor=False) -> Dict[str, float]:
        """Normalize logged values to a valid (log) prob dist"""

        if use_floor: 
            #TODO: make this smarter so that normalization includes the floor
            logvals = {tk: max(lp, constants.MIN_TOKEN_LOG_PROB) for tk, lp in logvals.items()}
        m = max(logvals.values())
        # guard against pathological all -inf (unlikely); fallback to uniform
        if not np.isfinite(m):
            logger.warning("Token log-probabilities sum to non-finite value; defaulting to uniform distribution")
            uniform = -np.log(len(logvals))
            token_logps = {k: uniform for k in logvals.keys()}
        else:
            # compute log-sum-exp in a numerically stable way
            logZ = m + np.log(sum(np.exp(v - m) for v in logvals.values()))
            token_logps = {k: (v - logZ) for k, v in logvals.items()}

        return token_logps

    @staticmethod
    def _tokens_to_remove(token_loss: List[Tuple[str, float]], k_drop: int):
        if k_drop <= 0:
            return set()
        
        token_loss.sort(key=lambda x: x[1])
        to_remove = {tk for tk, _ in token_loss[:k_drop]}
        return to_remove


# ---------------------------------------------------------------------
# external training functions
# ---------------------------------------------------------------------

def train_lang_tokenizers(corpus_info: Dict[str, str],
                          out_dir: str,
                          vocab_size=8000, 
                          byte_level=False,
                          use_per_lang_if_exists=True,
                          shared_seed_vocab=False
                          ) -> Dict[str, str]:
    """
    Args
    ----
    config       : experiment config (for hyper‑params)
    corpus_info  : { lang :  <file>, … }
    out_dir      : directory where <lang>_tokenizer.json are written

    Returns
    -------
    paths : { lang : tokenizer_path }
    """
    paths = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if shared_seed_vocab:
        seed_outfile = os.path.join(out_dir, "tokmix_seed_tokenizer.json") 
        if not os.path.exists(seed_outfile):
            logger.info("Training seed tokenizer.")
            tok = StandardUnigramLMTokenizer(
                vocab_size   = vocab_size*constants.INITIAL_VOCAB_MULT_FACTOR,
                em_mode      = "soft",      # full forward‑backward EM
                byte_level   = byte_level,
                multigram    = False
            )
            tok.train(corpus_info.values(), output_path=seed_outfile)
    else:
        seed_outfile = None

    # 2) shared tokenizer hyper‑parameters
    for lg, sample_path in corpus_info.items():
        outfile = os.path.join(out_dir, f"tokmix_{lg}_tokenizer{'_shared_seed' if shared_seed_vocab else ''}.json")
        if not os.path.exists(outfile) or not use_per_lang_if_exists:

            logger.info("Training per‑language tokenizer for %s …", lg)
            tok = StandardUnigramLMTokenizer(
                vocab_size   = vocab_size,
                em_mode      = "soft",      # full forward‑backward EM
                byte_level   = byte_level,
                multigram    = False,
                initial_vocab_tokens = seed_outfile
            )
        
            tok.train([sample_path], output_path=outfile)
        paths[lg] = outfile

    # save manifest for downstream stages
    return paths

def train_tokmix(per_lang_paths: Dict[str, str],
                 k_final: int,
                 output_path: str,
                 byte_level: bool = False,
                 pruning_criterion: str = "probability",
                 use_intersection_pruning: bool = False,
                 baseline_tokens: set = None):
    """
    Combine language‑specific tokenizers trained earlier using TokMix approach.
    `per_lang_paths` maps lang → path/to/json
    
    Args:
        per_lang_paths: Dictionary mapping language codes to tokenizer paths
        k_final: Final vocabulary size
        output_path: Path to save the merged tokenizer
        byte_level: Whether tokenizers use byte-level encoding
        pruning_criterion: Ranking criterion - "probability", "approx_loss", or "exact_loss" 
        use_intersection_pruning: If True, use intersection pruning; if False, use average-based pruning
    """    
    # Collect source tokenizer metadata
    source_metadata = {}
    for lang, path in per_lang_paths.items():
        try:
            with open(path, 'r') as f:
                tok_data = json.load(f)
                if "tokenizer_metadata" in tok_data:
                    source_metadata[lang] = {
                        "path": path,
                        "em_mode": tok_data["tokenizer_metadata"]["training_config"].get("em_mode"),
                        "vocab_size": tok_data["tokenizer_metadata"]["training_config"].get("vocab_size")
                    }
                else:
                    source_metadata[lang] = {"path": path, "em_mode": "unknown", "vocab_size": "unknown"}
        except:
            source_metadata[lang] = {"path": path, "em_mode": "unknown", "vocab_size": "unknown"}
    
    tkzs = {lg: StandardUnigramLMTokenizer(
        ) for lg in per_lang_paths}
    #TODO: figure out nicer way to set pretokenizer/decoder from a base tokenizer
    pretokenizer, decoder = None, None
    for lg, path in per_lang_paths.items():
        tkzs[lg].load(path)
        pretokenizer = tkzs[lg]._get_pretokenizer()
        decoder = tkzs[lg]._get_decoder()

    # Use unified tokmix_merge function with new parameters
    strategy_desc = f"intersection-{pruning_criterion}" if use_intersection_pruning else f"average-{pruning_criterion}"
    logger.info(f"Using TokMix strategy: {strategy_desc}")
    
    tokens_to_include = baseline_tokens if baseline_tokens else set()
    tokens_to_include.update(constants.SPECIAL_TOKENS.values())
    merged = tokmix_merge(
        {lg: tkz.global_lprobs for lg, tkz in tkzs.items()}, 
        k_final,
        constants.SPECIAL_TOKENS["unk_token"],
        tokens_to_include,
        pretokenizer,
        decoder,
        pruning_criterion=pruning_criterion,
        use_intersection_pruning=use_intersection_pruning,
        byte_level=byte_level
    )
    
    merged.save(output_path)
    
    # Create TokMix metadata
    tokmix_metadata = _create_base_metadata("TokMixTokenizer", "tokmix")
    tokmix_metadata["training_config"] = {
        "vocab_size": k_final,
        "source_tokenizer_configs": source_metadata,
        "pruning_criterion": pruning_criterion,
        "use_intersection_pruning": use_intersection_pruning,
        "merging_strategy": strategy_desc,
        "byte_level": byte_level
    }
    
    # Save metadata to separate file
    logging_path = _get_metadata_logging_path(output_path)
    metadata_path = _save_tokenizer_metadata(logging_path, output_path, tokmix_metadata)

    if metadata_path:
        logger.info("Saved TokMix tokenizer to %s with metadata to %s", output_path, metadata_path)
    else:
        logger.info("Saved TokMix tokenizer to %s", output_path)
    return output_path

def train_standard_tokenizer(corpus_info, 
                             vocab_size=8000, 
                             em_mode='soft', 
                             multigram=False, 
                             byte_level=False, 
                             output_path=None,
                             dataset_size_normalization=True,
                             use_intersection_pruning=False,
                             pruning_criterion="exact_loss",
                             initial_vocab_tokens=None, 
                             use_sp_seed_vocab=True
                             ):
    """
    Train a standard UnigramLM tokenizer on all languages.
    
    Args:
        corpus_info: Dictionary with corpus information for all languages
        vocab_size: Size of the vocabulary
        em_mode: EM algorithm mode ("hf", "hard", "soft")
        multigram: Whether to use multigram mode with per-language distributions
        byte_level: Whether to use byte-level tokenization
        output_path: Path to save the trained tokenizer
        dataset_size_normalization: Whether to normalize loss by dataset size in multigram mode.
            Prevents bias toward larger datasets during pruning.
        use_intersection_pruning: Whether to use intersection-based pruning in multigram mode.
        pruning_criterion: Loss function for token pruning ("exact_loss", "approx_loss", "probability")
        initial_vocab_tokens: Path to file containing custom initial vocabulary tokens
    
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

    tokenizer = StandardUnigramLMTokenizer(
        vocab_size=vocab_size, 
        em_mode=em_mode, 
        multigram=multigram, 
        byte_level=byte_level,
        dataset_size_normalization=dataset_size_normalization,
        use_intersection_pruning=use_intersection_pruning,
        pruning_criterion=pruning_criterion,
        initial_vocab_tokens=initial_vocab_tokens,
        use_sp_seed_vocab=use_sp_seed_vocab
    )
    tokenizer.train(train_files, output_path, corpus_info=corpus_info)
    
    return output_path