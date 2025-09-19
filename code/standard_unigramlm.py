from __future__ import annotations
from pathlib import Path
import math
import os
import json
import torch
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import numpy as np
import pickle
import hashlib
import time
from datetime import datetime
import platform
import sys
import multiprocessing

from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, Sequence

from code.utils import (
    get_baseline_characters,
    get_baseline_bytes,
    tokmix_merge
)

from code.tokenizer_formatting_utils import (
    load_tokenizer_with_mode,
    _build_hf_tokenizer_from_probs,
    _load_custom_vocab_from_file
)

from code.em_utils import (
    _accumulate_usage,
    _accumulate_usage_dummy,
    _compute_loss_with_alternative_sequences,
    _compute_alternatives_direct
)

from code.constants import (
    SPECIAL_TOKENS,
    INITIAL_VOCAB_MULT_FACTOR,
    NUM_TRIMMING_ITERATIONS,
    MIN_TOKEN_PROB,
    MIN_CHAR_OCCURENCE_THRESHOLD,
    MAX_BYTE_TOKEN_LEN,
    MAX_CHAR_TOKEN_LEN
)

logger = logging.getLogger(__name__)


@dataclass
class EMState:
    """Encapsulates state needed during EM training iterations."""
    corpus_files: List[str]
    per_lang: bool
    lang_codes: List[str]
    all_tokens: Set[str]  # Mutable - tokens removed during pruning
    pinned: Set[str]
    p_token: Dict[str, float]
    prev_p_token: Dict[str, float]
    prev_lang_probs: Optional[Dict[str, Dict[str, float]]]
    k_total: int  # Mutable - decremented during pruning
    k_step: List[int]


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
        byte_level: bool = True,
        whitespace_token_boundaries: bool = False,
        min_char_occurence_threshold: Optional[int] = None,
        dataset_size_normalization: bool = True,
        use_intersection_pruning: bool = True,
        initial_vocab_tokens: Optional[str] = None,  # Path to file containing tokens
        pruning_criterion: str = "exact_loss"  # "exact_loss", "approx_loss", "probability"
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
                When True, only prunes tokens that are consistently low-loss across ALL languages.
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
        self.unk_prior = 1.0 / (32 * vocab_size)  # SentencePiece default
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS.values())
        self.byte_level = byte_level
        self.whitespace_token_boundaries = whitespace_token_boundaries
        self.min_char_occurence_threshold = min_char_occurence_threshold
        if not self.min_char_occurence_threshold:
            self.min_char_occurence_threshold = MIN_CHAR_OCCURENCE_THRESHOLD

        self.base_tokenizer: Optional[Tokenizer] = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.multigram = multigram
        self.mixture_weights = mixture_weights or {}
        self.dataset_size_normalization = dataset_size_normalization
        self.use_intersection_pruning = use_intersection_pruning
        self.initial_vocab_tokens = initial_vocab_tokens
        self.pruning_criterion = pruning_criterion

        # Validate parameters
        if self.initial_vocab_tokens and not os.path.exists(self.initial_vocab_tokens):
            raise FileNotFoundError(f"Initial vocabulary file not found: {self.initial_vocab_tokens}")
        
        if pruning_criterion not in ["exact_loss", "approx_loss", "probability"]:
            raise ValueError(f"Invalid pruning_criterion: {pruning_criterion}. Must be one of: exact_loss, approx_loss, probability")

        # multigram‑specific
        self.lang_codes: List[str] = []
        self.per_lang_probs: Dict[str, Dict[str, float]] = {}
        self.global_probs: Dict[str, float] = {}
        
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
        pad_id = getattr(self.base_tokenizer, "pad_token_id", self.get_vocab()[SPECIAL_TOKENS["pad_token"]])
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

    def _baseline_tokens(self, files: List[str]) -> Set[str]:
        return (
            get_baseline_bytes()[0]
            if self.byte_level
            else set(get_baseline_characters(files, min_occurence_threshold=self.min_char_occurence_threshold)[0])
        )

    def _get_pretokenizer(self):
        if self.base_tokenizer:
            return self.base_tokenizer.pre_tokenizer
        components = []
        if self.whitespace_token_boundaries:
            components.append(Whitespace())
        if self.byte_level:
            components.append(ByteLevel(add_prefix_space=True))

        if components:
            return components[0] if len(components) == 1 else Sequence(components)
    
    def _prepare_custom_vocab(self, custom_tokens: List[str], target_size: int) -> Set[str]:
        """Prepare custom vocabulary with automatic size handling."""
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in custom_tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        # Ensure special tokens are included
        special_set = set(self.special_tokens)
        non_special = [t for t in unique_tokens if t not in special_set]
        
        # Add all special tokens first
        final_tokens = list(self.special_tokens)
        
        if len(unique_tokens) <= target_size:
            # Use all tokens - no trimming needed
            final_tokens.extend(non_special)
            logger.info(f"Using all {len(unique_tokens)} custom tokens (target: {target_size})")
        else:
            # Trim to fit target size, keeping special tokens
            keep_count = target_size - len(self.special_tokens)
            if keep_count < 0:
                logger.warning(f"Target size {target_size} is smaller than number of special tokens {len(self.special_tokens)}")
                keep_count = 0
            
            final_tokens.extend(non_special[:keep_count])
            logger.info(f"Trimmed custom vocabulary from {len(unique_tokens)} to {len(final_tokens)} tokens")
        
        return set(final_tokens)
    
    def _initial_vocab(self, train_files: List[str]) -> Set[str]:
        """Bootstrap an initial token set - either from custom vocab file or HF trainer."""
        if self.initial_vocab_tokens:
            # Use custom vocabulary from file
            logger.info(f"Loading custom initial vocabulary from {self.initial_vocab_tokens}")
            custom_tokens = _load_custom_vocab_from_file(self.initial_vocab_tokens)
            
            # Prepare vocabulary with automatic size adjustment
            seed_vocab_size = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR
            vocab = self._prepare_custom_vocab(custom_tokens, seed_vocab_size)
            
            logger.info(f"Custom vocabulary prepared: {len(vocab)} tokens (target: {seed_vocab_size})")
            return vocab
        else:
            # Use existing HF-based approach  
            return self._initial_vocab_hf(train_files)
    
    def _initial_vocab_hf(self, train_files: List[str]) -> Set[str]:
        """Original HF trainer to bootstrap an initial token set."""
        tok = Tokenizer(Unigram())
        pretokenizer = self._get_pretokenizer()
        if pretokenizer:
            tok.pre_tokenizer = pretokenizer
        seed_vocab_size = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR
        logger.info(f"Attempting to find seed vocab of size {seed_vocab_size}...")
        trainer = UnigramTrainer(
            vocab_size=seed_vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token,
            max_piece_length=MAX_BYTE_TOKEN_LEN if self.byte_level else MAX_CHAR_TOKEN_LEN,
            # not strictly necessary since these will be added back later
            initial_alphabet=ByteLevel.alphabet() if self.byte_level else [],
        )
        tok.train(train_files, trainer)
        init_vocab = tok.get_vocab()
        logger.info(f"Initial vocabulary size: {len(init_vocab)}")
        
        return set(init_vocab.keys())

    def _compute_token_loss(
        self,
        token: str,
        counts_dict: Dict[str, float],
        prob_dict: Dict[str, float], 
        alternatives_dict: Optional[Dict] = None
    ) -> float:
        """
        Compute loss for a single token using the configured pruning criterion.
        
        Args:
            token: Token to compute loss for
            counts_dict: Count dictionary for this token's context
            prob_dict: Probability dictionary for this token's context  
            alternatives_dict: Alternative sequences dictionary (for exact loss only)
        
        Returns:
            Loss value for the token
        """
        if self.pruning_criterion == "probability":
            return prob_dict[token]
        elif self.pruning_criterion == "approx_loss":
            return counts_dict[token] * math.log(max(prob_dict[token], 1e-30))
        else:  # exact_loss
            return _compute_loss_with_alternative_sequences(
                token, counts_dict, alternatives_dict, prob_dict, self.unk_token
            )

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
            (tk, self._compute_token_loss(tk, counts, state.p_token, alternative_sequences))
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
        under_consideration: Set[str],
        lang_counts: Dict[str, Dict[str, float]],
        lang_alternative_sequences: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute raw per-language losses without aggregation for intersection pruning.
        Returns Dict[lang_code, List[Tuple[token, loss]]]
        """
        return {
            lc: [
                (tk, self._compute_token_loss(
                    tk, lang_counts[lc], self.per_lang_probs[lc],
                    lang_alternative_sequences[lc] if lang_alternative_sequences else None
                ))
                for tk in under_consideration
            ]
            for lc in self.lang_codes
        }

    @staticmethod
    def _normalize(counts: Dict[str, float], unk_token: str, unk_prior: float) -> Dict[str, float]:
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

    def _compute_intersection_pruning_tokens(self, under_consideration, per_lang_losses, k_drop):
        """
        Find tokens to prune using rank-based intersection approach.
        Uses shared logic from utils.compute_intersection_pruning_tokens.
        """
        from code.utils import compute_intersection_pruning_tokens
        return compute_intersection_pruning_tokens(
            per_lang_losses, k_drop, self.lang_codes, under_consideration
        )

    # ------------------------------------------------------------------
    # main EM routine (shared by standard & multigram)
    # ------------------------------------------------------------------
    def _run_em(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        logging_path: str = None
    ) -> None:
        # Initialize EM training state
        state = self._initialize_em_state(corpus_files, per_lang, corpus_info)
        
        # Store initial vocab size in metadata
        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["initial_vocab_size"] = len(state.all_tokens)
        
        for it in range(self.num_iterations):
            logger.info("EM %d/%d – starting | vocab=%d | to trim=%d | Pruning criterion: %s", 
                       it + 1, self.num_iterations, len(state.all_tokens), state.k_total, self.pruning_criterion)            
            if state.per_lang:
                lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
                unk_count = 0
                total_counts = 0
                for lc, file in zip(state.lang_codes, state.corpus_files):
                    c_dict, _ = _accumulate_usage(file, 
                                                  self.per_lang_probs[lc], 
                                                  self.unk_token, 
                                                  mode=self.em_mode, 
                                                  whitespace_boundaries=self.whitespace_token_boundaries,
                                                  pretokenizer=self._get_pretokenizer())
                    for tk, v in c_dict.items():
                        lang_counts[lc][tk] += v
                    unk_count += lang_counts[lc][self.unk_token]
                    total_counts += sum(lang_counts[lc].values())
                observed_tokens = set([k for lang_probs in lang_counts.values() for k in lang_probs])
                for lc in state.lang_codes:
                    self.per_lang_probs[lc] = self._normalize(lang_counts[lc], self.unk_token, self.unk_prior)
                state.p_token = {tk: sum(self.mixture_weights[lc] * self.per_lang_probs[lc][tk] for lc in state.lang_codes) for tk in state.all_tokens}
            else:
                counts = {tk: 0.0 for tk in state.all_tokens}
                for file in state.corpus_files:
                    c_dict, _ = _accumulate_usage(file, state.p_token, self.unk_token, 
                                                  mode=self.em_mode, 
                                                  whitespace_boundaries=self.whitespace_token_boundaries, 
                                                  pretokenizer=self._get_pretokenizer())
                    for tk, v in c_dict.items():
                        counts[tk] += v
                unk_count = counts[self.unk_token]
                total_counts = sum(counts.values())
                observed_tokens = counts.keys()
                state.p_token = self._normalize(counts, self.unk_token, self.unk_prior)

            # Use extracted diagnostics method
            self._log_iteration_diagnostics(it, state, observed_tokens, unk_count, total_counts)

            # -------------------- pruning ----------------------------------
            k_drop = state.k_step[it]
            k_drop = min(k_drop, len(state.all_tokens) - self.vocab_size)
            if k_drop > 0:
                logging_dict = {}
                for tk in state.all_tokens:
                    logging_dict[tk] = {
                        "pre-pruning-prob": state.p_token[tk],
                        "status": "pinned" if tk in state.pinned else "kept",  # default, changed later
                        "loss": "NA",
                        "post-pruning-prob": "NA",
                        "expected-counts": sum([lang_counts[lc][tk] for lc in state.lang_codes]) if state.per_lang else counts[tk]
                    }

                if state.per_lang:
                    # For multilingual, compute alternatives for each language
                    lang_alternative_sequences = {}
                    if self.pruning_criterion != "probability":
                        for lc, file in zip(state.lang_codes, state.corpus_files):
                            lang_alternative_sequences[lc] = _compute_alternatives_direct(self.per_lang_probs[lc], self.unk_token, byte_level=self.byte_level)
                else:
                    alternative_sequences = _compute_alternatives_direct(state.p_token, self.unk_token, byte_level=self.byte_level) if self.pruning_criterion != "probability" else None

                before = len(state.all_tokens)
                under_consideration = state.all_tokens - state.pinned
                if state.per_lang:
                # For multilingual case
                    if self.use_intersection_pruning:
                        # Compute per-language losses for intersection pruning
                        per_lang_losses = self._compute_per_language_losses(
                            under_consideration, lang_counts, lang_alternative_sequences
                        )
                        
                        # New intersection-based approach
                        tokens_to_remove = self._compute_intersection_pruning_tokens(
                            under_consideration, per_lang_losses, k_drop
                        )
                    else:
                        # Existing average-loss approach (with dataset size normalization)
                        # Compute dataset sizes for normalization if enabled
                        dataset_sizes = {}
                        if self.dataset_size_normalization:
                            dataset_sizes = {lc: sum(lang_counts[lc].values()) for lc in state.lang_codes}
                            logger.info("Dataset sizes for normalization: %s", 
                                       {lc: f"{size:.0f}" for lc, size in dataset_sizes.items()})
                        
                        # Compute per-language losses first
                        per_lang_losses = self._compute_per_language_losses(
                            under_consideration, lang_counts, lang_alternative_sequences
                        )
                        
                        # Then aggregate them
                        loss = self._compute_multigram_pruning_scores(
                            state, per_lang_losses, dataset_sizes
                        )
                        
                        # Update logging
                        for tk, total_loss in loss:
                            logging_dict[tk]["loss"] = total_loss
                        
                        tokens_to_remove = self._tokens_to_remove(loss, k_drop)
                    
                    # Update logging for removed tokens
                    for tk in tokens_to_remove:
                        logging_dict[tk]["status"] = "pruned"
                else:
                    # Use centralized standard pruning score computation
                    loss = self._compute_standard_pruning_scores(
                        state, counts, alternative_sequences
                    )
                    
                    # Update logging
                    for tk, tk_loss in loss:
                        logging_dict[tk]["loss"] = tk_loss

                    tokens_to_remove = self._tokens_to_remove(loss, k_drop)
                    for tk in tokens_to_remove:
                        logging_dict[tk]["status"] = "pruned"
                
                # Common token removal logic for both multigram and standard
                state.all_tokens -= tokens_to_remove
                pruned = before - len(state.all_tokens)
                state.k_total -= pruned
                logger.info("   tokens_pruned=%d | new_vocab=%d", pruned, len(state.all_tokens))
                
                # Store pruning metrics
                if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
                    self._training_metadata["training_metrics"]["pruned_per_iteration"].append(pruned)

                if state.per_lang:
                    for lc in state.lang_codes:
                        self.per_lang_probs[lc] = {tk: self.per_lang_probs[lc].get(tk, 0.0) for tk in state.all_tokens}
                        self.per_lang_probs[lc] = self._normalize(self.per_lang_probs[lc], self.unk_token, self.unk_prior)
                        state.prev_lang_probs[lc] = self.per_lang_probs[lc].copy()
                        
                state.p_token = {tk: state.p_token[tk] for tk in state.all_tokens}
                state.p_token = self._normalize(state.p_token, self.unk_token, self.unk_prior)
                for tk, new_prob in state.p_token.items():
                    logging_dict[tk]["post-pruning-prob"] = new_prob
                if logging_path is not None:
                    logging_file = f"{logging_path}_em_{it}.pkl"
                    with open(logging_file, 'wb') as handle:
                        pickle.dump(logging_dict, handle)

            state.prev_p_token = state.p_token.copy()

            logger.info(f"Total probability mass: {sum(state.p_token.values())}")
            assert all(x in state.p_token for x in state.pinned)

        # ---------------- store results ----------------
        self.global_probs = state.p_token

    def _initialize_em_state(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None
    ) -> EMState:
        """Initialize EM training state and data structures."""
        # Validation
        if per_lang and not corpus_info:
            raise ValueError("corpus_info required when multigram=True")

        # Gather training files and language codes
        lang_codes = []
        if per_lang:
            lang_codes = [lc for lc, info in corpus_info.items()]
            corpus_files = [corpus_info[lc] for lc in lang_codes]
            self.lang_codes = lang_codes  # Keep instance variable for compatibility

        # Shared preparation
        all_tokens = self._initial_vocab(corpus_files)
        
        # Only add baseline tokens if not using custom vocabulary
        if not self.initial_vocab_tokens:
            baseline = self._baseline_tokens(corpus_files)
            all_tokens.update(baseline)
            pinned = set(self.special_tokens) | baseline | {self.unk_token}
        else:
            baseline = set()  # Empty baseline for custom vocabulary
            pinned = set(self.special_tokens) | {self.unk_token}
        
        all_tokens.update(self.special_tokens)

        # Initialize probability distributions
        if per_lang:
            self.per_lang_probs = {lc: {tk: 1.0 / len(all_tokens) for tk in all_tokens} for lc in lang_codes}
            self.mixture_weights = self._sanitize_mixture_weights(lang_codes, self.mixture_weights)
            p_token: Dict[str, float] = {tk: 1.0 / len(all_tokens) for tk in all_tokens}
            prev_lang_probs = {lc: self.per_lang_probs[lc].copy() for lc in lang_codes}
        else:
            p_token = {tk: 1.0 / len(all_tokens) for tk in all_tokens}
            prev_lang_probs = None
        
        prev_p_token = p_token.copy()

        # Pruning schedule
        k_total = len(all_tokens) - self.vocab_size
        k_step = [math.floor(k_total / NUM_TRIMMING_ITERATIONS)] * NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - NUM_TRIMMING_ITERATIONS)

        return EMState(
            corpus_files=corpus_files,
            per_lang=per_lang,
            lang_codes=lang_codes,
            all_tokens=all_tokens,
            pinned=pinned,
            p_token=p_token,
            prev_p_token=prev_p_token,
            prev_lang_probs=prev_lang_probs,
            k_total=k_total,
            k_step=k_step
        )

    def _log_iteration_diagnostics(
        self,
        iteration: int,
        state: EMState,
        observed_tokens: Set[str],
        unk_count: float,
        total_counts: float
    ) -> None:
        """Log diagnostic information for the current EM iteration."""
        logger.info(f"Observed {len(observed_tokens)} unique tokens out of {len(state.all_tokens)} possible tokens across corpus...")
        logger.info(f"Observed {unk_count} UNK tokens out of {total_counts}...")

        # Diagnostics before pruning
        unk_prob = state.p_token.get(self.unk_token, 0.0)
        l1_delta = 0.5 * sum(abs(state.p_token.get(tk, 0) - state.prev_p_token.get(tk, 0)) for tk in state.all_tokens)
        logger.info(" UNK prob: %.8g | Δ‑dist: %.3e | min=%.3e max=%.3e", 
                   unk_prob, l1_delta, min(state.p_token.values()), max(state.p_token.values()))
        
        # Store convergence metrics
        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["convergence"]["unk_probs"].append(float(unk_prob))
            self._training_metadata["training_metrics"]["convergence"]["l1_deltas"].append(float(l1_delta))
        
        if state.per_lang:
            for lc in state.lang_codes:
                up = self.per_lang_probs[lc][self.unk_token]
                d = 0.5 * sum(abs(self.per_lang_probs[lc].get(tk, 0) - state.prev_lang_probs[lc].get(tk, 0)) 
                             for tk in state.all_tokens)
                logger.info("      [%s] UNK=%.8g | Δ‑dist=%.3e", lc, up, d)

    # ------------------------------------------------------------------
    # thin wrappers that call _run_em then build the HF tokenizer
    # ------------------------------------------------------------------
    def _train_custom_em(self, corpus_files: List[str], output_path=None):
        logging_path = output_path.split('.')[0] if output_path else None
        self._run_em(corpus_files, per_lang=False, logging_path=logging_path)
        return self._build_tokenizer(self.global_probs, output_path)

    def _train_multigram(self, corpus_info: Dict[str, str], output_path=None):
        logging_path = output_path.split('.')[0] if output_path else None
        self._run_em(None, per_lang=True, corpus_info=corpus_info, logging_path=logging_path)
        return self._build_tokenizer(self.global_probs, output_path)

    def _train_hf(self, corpus_files, output_path=None):
        """
        Train using the HuggingFace UnigramTrainer.
        """
        logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
        tok = Tokenizer(Unigram())
        pretokenizer = self._get_pretokenizer()
        if pretokenizer:
            tok.pre_tokenizer = pretokenizer

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
            self._add_metadata_to_saved_tokenizer(output_path)
            
            logger.info(f"Saved HF-based standard tokenizer with metadata to {output_path}")
        return self.base_tokenizer

    def _train_bpe(self, corpus_files: List[str], output_path=None):
        """Plain HF BPE trainer; shares most hyper‑params with EM modes."""
        logger.info("Using HuggingFace BPETrainer.")
        tok = Tokenizer(BPE(unk_token=self.unk_token))
        pretokenizer = self._get_pretokenizer()
        if pretokenizer:
            tok.pre_tokenizer = pretokenizer

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
            
            # Add metadata to the saved JSON
            self._add_metadata_to_saved_tokenizer(output_path)
            
            logger.info("Saved BPE tokenizer with metadata to %s", output_path)
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
        
        # Different training approaches
        if self.em_mode == "bpe":
            result = self._train_bpe(corpus_files, output_path)

        elif self.em_mode == "hf" and not self.multigram:
            # ============= HF Approach =============
            result = self._train_hf(corpus_files, output_path)
            
        elif self.multigram:
            # ============= Multigram Approach =============
            if not corpus_info:
                raise ValueError("corpus_info is required for multigram training")
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

    # ------------------------------------------------------------------
    # helper to create the final HF tokenizer & save it
    # ------------------------------------------------------------------
    def _build_tokenizer(self, probs: Dict[str, float], output_path):
        vocab_list, unk_id = self._build_final_vocab(probs)
        tok = _build_hf_tokenizer_from_probs(
            vocab_list,
            unk_id,
            pretokenizer=self._get_pretokenizer()
        )
        self.base_tokenizer = tok
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(output_path)
            
            # Add metadata to the saved JSON
            self._add_metadata_to_saved_tokenizer(output_path)
            
            logger.info("Tokenizer with metadata saved to %s", output_path)
        return tok
    
    def _add_metadata_to_saved_tokenizer(self, output_path):
        """Save metadata to a separate .metadata.json file alongside the tokenizer"""
        from code.utils import _save_tokenizer_metadata

        try:
            metadata_path = _save_tokenizer_metadata(output_path, self._training_metadata)
            if metadata_path:
                logger.debug(f"Saved metadata to {metadata_path}")
            else:
                logger.warning(f"Failed to save metadata for {output_path}")

        except Exception as e:
            logger.warning(f"Failed to save metadata for {output_path}: {e}")

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
            "special_tokens": list(self.special_tokens),
            "unk_prior": self.unk_prior,
        }
        
        # Add custom vocabulary info if used
        if self.initial_vocab_tokens:
            training_config["initial_vocab_source"] = "custom_file"
            training_config["initial_vocab_file"] = self.initial_vocab_tokens
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
    
    def _get_tokenizer_type(self):
        """Determine tokenizer type string"""
        if self.em_mode == "bpe":
            return "bpe"
        elif self.multigram:
            return "multigram" 
        else:
            return "standard"
    
    def _update_training_results(self, training_metrics):
        """Update training results in metadata"""
        self._training_metadata["training_results"] = {
            "initial_vocab_size": training_metrics.get("initial_vocab_size"),
            "final_vocab_size": len(self.global_probs) if hasattr(self, 'global_probs') else None,
            "tokens_pruned_per_iteration": training_metrics.get("pruned_per_iteration", []),
            "convergence_metrics": training_metrics.get("convergence", {}),
            "training_time_seconds": training_metrics.get("training_time"),
            "final_loss": training_metrics.get("final_loss")
        }

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

def train_lang_tokenizers(corpus_info: Dict[str, Dict[str, str]],
                          out_dir: str,
                          vocab_size=8000, 
                          byte_level=False,
                          use_per_lang_if_exists=True,
                          ) -> Dict[str, str]:
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
                whitespace_token_boundaries=not byte_level,
                multigram    = False
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
                 use_intersection_pruning: bool = False):
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
    from code.utils import _save_tokenizer_metadata, _create_base_metadata
    
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
    
    tkzs = {lg: StandardUnigramLMTokenizer() for lg in per_lang_paths}
    pretokenizer = None
    for lg, path in per_lang_paths.items():
        tkzs[lg].load(path)
        pretokenizer = tkzs[lg]._get_pretokenizer()

    # Use unified tokmix_merge function with new parameters
    strategy_desc = f"intersection-{pruning_criterion}" if use_intersection_pruning else f"average-{pruning_criterion}"
    logger.info(f"Using TokMix strategy: {strategy_desc}")
    
    merged = tokmix_merge(
        {lg: tkz.global_probs for lg, tkz in tkzs.items()}, 
        k_final,
        SPECIAL_TOKENS["unk_token"],
        list(SPECIAL_TOKENS.values()),
        pretokenizer,
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
    metadata_path = _save_tokenizer_metadata(output_path, tokmix_metadata)

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
                             initial_vocab_tokens=None
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
        whitespace_token_boundaries=not byte_level,
        dataset_size_normalization=dataset_size_normalization,
        use_intersection_pruning=use_intersection_pruning,
        pruning_criterion=pruning_criterion,
        initial_vocab_tokens=initial_vocab_tokens
    )
    tokenizer.train(train_files, output_path, corpus_info=corpus_info)
    
    return output_path
