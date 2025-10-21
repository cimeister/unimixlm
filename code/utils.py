import math
from collections import defaultdict, Counter
import multiprocessing
import json
from typing import Dict, List, Tuple, Set, Any, Optional, Union

from pathlib import Path
import hashlib
from datetime import datetime
import platform
import sys
import code.constants as constants
from code.tokenizer_formatting_utils import _build_unigramlm_hf_tokenizer_from_lprobs

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_baseline_bytes() -> Tuple[Set[str], Dict[str,int]]:
    alphabet = ByteLevel.alphabet()
    return set(alphabet), {c:1 for c in alphabet}


def get_baseline_characters(training_data_files, character_coverage=0.9995, min_occurence_threshold=constants.MIN_CHAR_OCCURENCE_THRESHOLD):
    """
    Get baseline characters from training data files, matching SentencePiece's approach.
    
    Args:
        training_data_files: List of file paths to process
        character_coverage: Target character coverage ratio (between 0.0 and 1.0)
                           Default is 0.9995 which is SentencePiece's default
        min_occurence_threshold: Minimum number of occurrences for a character
                                to be included regardless of coverage
        
    Returns:
        Tuple of (chars_to_keep, char_freqs) where chars_to_keep includes:
        1. Characters that are part of the character_coverage set
        2. Characters that appear at least min_occurence_threshold times
    """
    # Use multiprocessing for faster file processing
    num_processes = min(multiprocessing.cpu_count() or 1, len(training_data_files), 20)
    
    # Process files in parallel
    chars_by_file = {}
    total_char_counts = Counter()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_count_worker, training_data_files)
        
        # Aggregate results
        for file_path, counts in results:
            chars_by_file[file_path] = counts
            total_char_counts.update(counts)
    
    # Calculate total character occurrences
    total_chars = sum(total_char_counts.values())
    
    # Get relative frequencies
    char_freqs = {char: count/total_chars for char, count in total_char_counts.items()}
    
    # Sort characters by frequency (descending)
    sorted_chars = sorted(total_char_counts.items(), key=lambda x: x[1], reverse=True)
    
    # First set: Characters that meet the coverage threshold
    coverage_chars = set()
    cumulative_coverage = 0.0
    target_coverage = total_chars * character_coverage
    
    for char, count in sorted_chars:
        coverage_chars.add(char)
        cumulative_coverage += count
        
        # Stop once we reach the target coverage
        if cumulative_coverage >= target_coverage:
            break
    
    # Second set: Characters that appear at least min_occurence_threshold times
    frequency_chars = {char for char, count in total_char_counts.items() 
                      if count >= min_occurence_threshold}
    
    # Union of the two sets
    chars_to_keep = list(coverage_chars.union(frequency_chars))
    
    # Calculate actual character coverage percentage
    covered_count = sum(total_char_counts[char] for char in coverage_chars)
    actual_coverage = covered_count / total_chars * 100.0
    
    # Count chars included only due to frequency threshold
    freq_only_chars = frequency_chars - coverage_chars
    freq_only_count = len(freq_only_chars)
    
    # Log coverage statistics
    logger.info(f"Character coverage: {len(coverage_chars)}/{len(total_char_counts)} "
                f"characters ({actual_coverage:.4f}%)")
    logger.info(f"Additional characters from frequency threshold: {freq_only_count}")
    logger.info(f"Total characters kept: {len(chars_to_keep)}")
    logger.info(f"Characters not included: {len(total_char_counts) - len(chars_to_keep)}")
    
    return chars_to_keep, char_freqs


def _count_worker(file_path):
    """
    Worker function to count character/byte frequencies in a file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Tuple of (file_path, token_counts)
    """
    try:
        # Initialize token counter
        token_counts = Counter()
        
        # Process the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                token_counts.update(line)
        
        return file_path, token_counts
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return file_path, Counter()


def compute_intersection_pruning_tokens(
    per_lang_losses: Dict[str, List[Tuple[str, float]]],
    k_drop: int,
    lang_codes: List[str],
    under_consideration: Optional[Set[str]] = None
) -> Set[str]:
    """
    Find tokens to prune using rank-based intersection approach.
    Suitable for both standard multilingual and TokMix tokenizers with different vocabularies.

    Args:
        per_lang_losses: Dict[lang_code, List[Tuple[token, loss]]] - pre-computed losses per language
        k_drop: Number of tokens to remove
        lang_codes: List of language codes
        under_consideration: Optional set of tokens to consider (if None, considers all tokens)

    Returns:
        Set of tokens to remove

    Note: For each token, computes its rank within each language (1-indexed, lower loss = lower rank).
    Tokens missing from a language's vocabulary get rank 0 (highest removal priority).
    Each token's score is its maximum rank across all languages.
    Returns the k_drop tokens with the smallest max-ranks (best worst-case performance).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if k_drop <= 0:
        return set()
    
    if not lang_codes or len(lang_codes) < 2:
        logger.warning("Intersection pruning requires at least 2 languages")
        return set()
    
    # Determine tokens under consideration
    if under_consideration is None:
        under_consideration = set()
        for lang_losses in per_lang_losses.values():
            under_consideration.update(token for token, _ in lang_losses)
    
    logger.info(f"Computing intersection pruning for {len(under_consideration)} tokens across {len(lang_codes)} languages")

    # Rank tokens within each language based on pre-computed losses
    per_lang_ranks = {}  # lang -> {token: rank}

    for lc in lang_codes:
        # Get losses for this language and sort by loss (ascending = lowest loss first)
        lang_loss_tuples = per_lang_losses[lc].copy()
        lang_loss_tuples.sort(key=lambda x: x[1])  # Sort by loss value

        # Assign ranks (1-indexed)
        token_to_rank = {}
        for rank, (tk, loss) in enumerate(lang_loss_tuples):
            if tk in under_consideration:  # Only rank tokens under consideration
                token_to_rank[tk] = rank + 1

        per_lang_ranks[lc] = token_to_rank
        total_under_consideration_in_lang = len(token_to_rank)
        missing_tokens = len(under_consideration) - total_under_consideration_in_lang
        logger.info(f"Language {lc}: ranked {total_under_consideration_in_lang} tokens, {missing_tokens} missing (assigned rank 0)")
    
    # For each token, compute its maximum rank across all languages
    # Missing tokens get rank 0 (highest removal priority for loss-based systems)
    token_max_ranks = []  # (token, max_rank)

    for tk in under_consideration:
        # Get this token's rank in each language
        # Missing tokens get rank 0 (highest priority for removal)
        token_ranks = []
        for lc in lang_codes:
            if tk in per_lang_ranks[lc]:
                token_ranks.append(per_lang_ranks[lc][tk])
            else:
                # Token not in this language's vocabulary - assign rank 0 (highest removal priority)
                token_ranks.append(0)

        # For intersection pruning, take the MAXIMUM (worst) rank across all languages
        # Tokens missing from any language will have a rank of 0 contributing to their max
        max_rank = max(token_ranks)
        token_max_ranks.append((tk, max_rank))
    
    # Sort by max rank (ascending = tokens with best worst-case rank)
    token_max_ranks.sort(key=lambda x: x[1])
    
    # Select k_drop tokens with smallest max-ranks
    if len(token_max_ranks) < k_drop:
        logger.warning(f"Only {len(token_max_ranks)} tokens available, but {k_drop} requested")
        selected_tokens = set([tk for tk, _ in token_max_ranks])
    else:
        selected_tokens = set([tk for tk, _ in token_max_ranks[:k_drop]])
    
    if selected_tokens:
        selected_ranks = [max_rank for tk, max_rank in token_max_ranks[:len(selected_tokens)]]
        logger.info(f"Intersection pruning: selected {len(selected_tokens)} tokens with max-ranks from {min(selected_ranks)} to {max(selected_ranks)}")
    
    return selected_tokens


def _as_batch(txt):
    """Return (list version, was_single_flag)."""
    if isinstance(txt, (str, tuple)):
        return [txt], True
    if isinstance(txt, list):
        return txt, False
    raise TypeError(type(txt))

def _apply_trunc_pad(
    ids_batch, toks_batch, offs_batch,
    *,                      # keyword-only args below
    truncation: bool, max_length: int | None,
    padding: str | None, pad_id: int, pad_tok: str,
    return_attention_mask: bool,
):
    """
    • ids_batch / toks_batch / offs_batch are lists of equal length
      (offs_batch can be None)
    • Returns: ids_padded, toks_padded, masks_or_None, offs_padded_or_None
    """
    out_ids, out_toks, out_offs = [], [], []
    masks = [] if return_attention_mask else None

    for ids, toks, offs in zip(ids_batch, toks_batch,
                               offs_batch or [None] * len(ids_batch)):
        # ---- 1) truncation ------------------------------------------
        if truncation and max_length:
            ids, toks = ids[:max_length], toks[:max_length]
            if offs is not None:
                offs = offs[:max_length]

        # ---- 2) padding --------------------------------------------
        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids  += [pad_id] * pad_len
            toks += [pad_tok] * pad_len
            if offs is not None:
                offs += [(0, 0)] * pad_len
        else:
            pad_len = 0  # used for attention_mask

        # ---- 3) attention mask -------------------------------------
        if masks is not None:
            real_len = len(ids) - pad_len
            masks.append([1] * real_len + [0] * pad_len)

        out_ids.append(ids)
        out_toks.append(toks)
        if offs is not None:
            out_offs.append(offs)

    return (
        out_ids,
        out_toks,
        masks,
        out_offs if offs_batch is not None else None,
    )


def mk_id(raw: str) -> str:
    # 8-byte / 16-char hex digest, utf-8, replace errors
    digest = hashlib.blake2s(
        raw.encode("utf-8", "replace"),
        digest_size=8                      # 64-bit
    ).hexdigest()
    return digest


def encode_text(
    tokenizer: Any,
    text: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]],
    *,
    add_special_tokens: bool = False,
    padding: Optional[str] = None,
    truncation: Union[bool, str] = False,
    max_length: Optional[int] = None,
    stride: Optional[int] = None,                     # only honoured by HF path
    return_offsets_mapping: bool=False,
    return_attention_mask: bool=False,
    return_special_tokens_mask: bool=False,
    return_token_type_ids: bool=False,
    return_tensors: Optional[str]=None,
    return_word_ids: bool=False,
    language: Optional[str]=None,
    use_best_language: bool=True,
) -> Dict[str, Any]:

    batch_txt, was_single = _as_batch(text)
    pad_token = tokenizer.pad_token if hasattr(tokenizer, "pad_token") else constants.SPECIAL_TOKENS["pad_token"]
    pad_id = tokenizer.get_vocab()[pad_token]

    # ------------------------------------------------------------------#
    # ➊  Language-specific UnigramLM path                                #
    # ------------------------------------------------------------------#
    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
    if isinstance(tokenizer, LanguageSpecificUnigramLMTokenizer):
        if use_best_language:
            best_langs, enc_objects = tokenizer.best_language_encode_batch(batch_txt)
            ids_batch  = [enc.ids    for enc in enc_objects]
            toks_batch = [enc.tokens for enc in enc_objects]
            offsets_b  = [enc.offsets for enc in enc_objects]
        else:
            if language is None:
                raise ValueError("`language` must be provided when use_best_language=False")
            enc_objects = tokenizer.language_info[language]["tokenizer"].encode_batch(batch_txt)
            best_langs  = None
            ids_batch   = [enc.ids    for enc in enc_objects]
            toks_batch  = [enc.tokens for enc in enc_objects]
            offsets_b   = [enc.offsets for enc in enc_objects]

        # ----- trunc / pad (helper unchanged) ---------------------------------
        ids_batch, toks_batch, masks, offsets_b = _apply_trunc_pad(
            ids_batch, toks_batch, offsets_b,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=pad_id,
            pad_tok=pad_token,
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if best_langs is not None:
            result["best_lang"] = best_langs
        if return_attention_mask:
            result["attention_mask"] = masks
        if return_offsets_mapping:
            result["offset_mapping"] = offsets_b
        # unwrap single example
        if was_single:
            for k, v in result.items():
                if isinstance(v, list) and len(v) == 1:
                    result[k] = v[0]
        return result

    # ------------------------------------------------------------------#
    # ➋  Generic fast tokenizer path                                     #
    # ------------------------------------------------------------------#
    if hasattr(tokenizer, "encode_batch"):
        encodings = tokenizer.encode_batch(batch_txt, add_special_tokens=add_special_tokens)

        ids_batch   = [enc.ids    for enc in encodings]
        toks_batch  = [enc.tokens for enc in encodings]
        word_ids_b  = [enc.word_ids() for enc in encodings] if return_word_ids else None
        offsets_b  = [enc.offsets for enc in encodings] if return_offsets_mapping else None

        ids_batch, toks_batch, masks, offsets_b = _apply_trunc_pad(
            ids_batch, toks_batch, offsets_b,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=pad_id,
            pad_tok=pad_token,
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if masks is not None:
            result["attention_mask"] = masks
        if word_ids_b is not None:
            result["word_ids"] = word_ids_b
        if return_offsets_mapping:
            result["offset_mapping"] = offsets_b

    else:
        # ------------------------------------------------------------------
        # ➌  Fallback to high-level HF __call__ (handles stride/overflow,…)
        # ------------------------------------------------------------------
        hf_kwargs = dict(
            add_special_tokens   = add_special_tokens,
            padding              = padding,
            truncation           = truncation,
            max_length           = max_length,
            stride               = stride,
            return_offsets_mapping    = return_offsets_mapping,
            return_attention_mask     = return_attention_mask,
            return_special_tokens_mask= return_special_tokens_mask,
            return_token_type_ids     = return_token_type_ids,
            return_tensors            = return_tensors,
        )
        enc = tokenizer(batch_txt, **hf_kwargs)
        result = enc.to_dict() if hasattr(enc, "to_dict") else dict(enc)
        if "tokens" not in result and hasattr(tokenizer, "convert_ids_to_tokens"):
            result["tokens"] = [tokenizer.convert_ids_to_tokens(ids) for ids in result["input_ids"]]

    # unwrap single example
    if was_single:
        for k, v in result.items():
            if isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
    return result


def proportional_sample(train_files_by_lang: Dict[str, str],
                        out_root: str,
                        total_lines: int) -> Dict[str, str]:
    """
    For each language copy ≤ n lines into <out_root>/<lang>/sample.txt,
    where n is proportional to the byte count recorded in meta.json.
    Returns   lang -> sampled_file_path
    """
    byte_tot = {lg: json.load(open(Path(path).parent/"meta.json"))["bytes_written"]
                for lg, path in train_files_by_lang.items()}
    total = sum(byte_tot.values())

    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    sampled_files = {}
    for lg, path in train_files_by_lang.items():
        # ensure at least MIN_TOKENIZER_LANG_TRAINING_LINES lines
        k = max(constants.MIN_TOKENIZER_LANG_TRAINING_LINES, int(total_lines * byte_tot[lg] / total))
        dest_dir = out_path / lg
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "sample.txt"
        sampled_files[lg] = str(dest)

        if dest.exists():               # reuse cache
            continue

        with open(path, encoding="utf-8", errors="ignore") as src, \
             open(dest, "w", encoding="utf-8") as tgt:
            lines_written = 0
            for line in src:
                if lines_written == k:
                    break
                if constants._RNG.random() > 0.2:   # light intra‑lang subsample
                    continue
                tgt.write(line)
                lines_written += 1
    return sampled_files


def _sample_lines_from_file(file_path: str, sample_proportion: float, seed: int) -> List[str]:
    """
    Efficiently sample lines from a file using reservoir sampling.
    """
    import random
    random.seed(seed)
    
    # First, try to estimate file size
    sample_size = None
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Read first 1000 lines to estimate
        first_lines = []
        i = 0
        for i, line in enumerate(f):
            if i >= 1000:
                break
            if line.strip():
                first_lines.append(line.strip())
        
        # If file is small, just return all lines
        if i < 1000:
            return first_lines
        
        # Estimate total lines (rough approximation)
        f.seek(0, 2)  # End of file
        file_size = f.tell()
        f.seek(0)
        avg_line_size = sum(len(line) for line in first_lines) / len(first_lines)
        estimated_lines = int(file_size / avg_line_size)
        sample_size = max(1000, int(estimated_lines * sample_proportion))
    
    # Now do reservoir sampling
    reservoir = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
    
    return reservoir


def np_logaddexp(x, y):
    """Numerically stable log(exp(x) + exp(y))"""
    if x == float("-inf"):
        return y
    if y == float("-inf"):
        return x
    return max(x, y) + math.log(1 + math.exp(-abs(x - y)))


class SafeFormattingDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def load_tokenizer_from_config(config):
    tokenizer_class = config.get('class', 'standard')
    byte_level = config.get('byte_level', False)
    
    if tokenizer_class == 'standard':
        from code.standard_unigramlm import StandardUnigramLMTokenizer
        tokenizer = StandardUnigramLMTokenizer(
            byte_level=byte_level,
            multigram=False
        )
        tokenizer.load(config['path'])
        
    elif tokenizer_class == 'langspec':
        from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
        tokenizer = LanguageSpecificUnigramLMTokenizer(byte_level=byte_level)
        tokenizer.load(config['base_path'], config.get('language_paths'))
        
    else:
        # Try loading as a HuggingFace tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['path'])
        except Exception as e:
            logger.info(f"Tried to load tokenizer via default method and could not {e}")
            raise
    return tokenizer


def tokmix_merge(
    per_lang_tokenizers: Dict[str, Dict[str, float]],
    k_final: int,
    unk_token: str,
    special: List[str],
    pretokenizer: Any,
    decoder: Any = None,
    pruning_criterion: str = "probability",      # "probability", "approx_loss", "exact_loss"
    use_intersection_pruning: bool = False,      # True = intersection, False = average
    byte_level: bool = False
) -> Tokenizer:
    """
    Unified TokMix merge function supporting all combinations of pruning criteria and strategies.
    
    Args:
        per_lang_tokenizers: Dict[lang, Dict[token, prob]] - per-language tokenizer probabilities
        k_final: Final vocabulary size
        unk_token: UNK token string
        special: List of special tokens to pin
        pretokenizer: Tokenizer pretokenizer to use
        pruning_criterion: What values to use for ranking ("probability", "approx_loss", "exact_loss")
        use_intersection_pruning: Selection strategy (intersection vs average-based)
        byte_level: Whether to use byte-level processing for exact loss computation
    
    Returns:
        HuggingFace Tokenizer object
    """
    from code.em_utils import _compute_alternatives_direct, _compute_loss_with_alternative_sequences
    import math
    
    lang_codes = list(per_lang_tokenizers.keys())
    n_langs = len(lang_codes)
    pinned = set(special)
    
    # Get all unique tokens across languages
    all_tokens = set()
    for token_probs in per_lang_tokenizers.values():
        all_tokens.update(token_probs.keys())
    candidate_tokens = all_tokens - pinned
    
    logger.info(f"TokMix merge: {len(all_tokens)} total tokens, {len(candidate_tokens)} candidates, "
               f"criterion={pruning_criterion}, intersection={use_intersection_pruning}")
    
    # Step 1: Compute per-language values based on pruning criterion
    per_lang_values = {}  # lang -> List[Tuple[token, value]]
    
    if pruning_criterion == "probability":
        # Use probabilities directly
        for lang, token_probs in per_lang_tokenizers.items():
            per_lang_values[lang] = [(token, prob) for token, prob in token_probs.items() 
                                   if token in candidate_tokens]
    
    elif pruning_criterion in ["approx_loss", "exact_loss"]:
        # Compute losses
        if pruning_criterion == "exact_loss":
            # Pre-compute alternative sequences for exact loss computation
            logger.info("Pre-computing alternative sequences...")
            per_lang_alternatives = {}
            for lang, token_probs in per_lang_tokenizers.items():
                alternatives = _compute_alternatives_direct(token_probs, unk_token, byte_level=byte_level)
                per_lang_alternatives[lang] = alternatives
        
        logger.info(f"Computing {pruning_criterion} values...")
        for lang, token_probs in per_lang_tokenizers.items():
            lang_values = []
            for token in candidate_tokens:
                if token not in token_probs:
                    continue  # Skip tokens not in this language
                    
                if pruning_criterion == "approx_loss":
                    # Use probability as implicit count: prob * log(prob) 
                    # Positive values to match exact_loss semantics (higher = more important)
                    loss = token_probs[token] * math.log(max(token_probs[token], 1e-30))
                else:  # exact_loss
                    # Use exact loss with alternatives
                    loss = _compute_loss_with_alternative_sequences(
                        token, token_probs, per_lang_alternatives[lang], 
                        token_probs, unk_token
                    )
                lang_values.append((token, loss))
            per_lang_values[lang] = lang_values
    
    # Step 2: Token selection based on strategy
    if use_intersection_pruning:
        # Use intersection-based selection
        tokens_to_keep_count = k_final - len(pinned)
        tokens_to_drop = len(candidate_tokens) - tokens_to_keep_count
        
        if tokens_to_drop > 0:
            selected_to_remove = compute_intersection_pruning_tokens(
                per_lang_values, tokens_to_drop, lang_codes, candidate_tokens
            )
            kept_candidates = candidate_tokens - selected_to_remove
        else:
            kept_candidates = candidate_tokens
            
        kept_tokens = list(special) + list(kept_candidates)
        
    else:
        # Use average-based selection
        if pruning_criterion == "probability":
            # Average probabilities (original tokmix_merge logic)
            avg_values = defaultdict(float)
            for lang_values in per_lang_values.values():
                for token, prob in lang_values:
                    avg_values[token] += prob
            for token in avg_values:
                avg_values[token] /= n_langs
            
            # Sort by averaged probability (descending - higher prob = keep)
            sorted_tokens = sorted(
                avg_values.items(),
                key=lambda x: x[1], 
                reverse=True
            )
        else:
            # Average losses (original tokmix_merge_loss_based logic)  
            avg_values = defaultdict(float)
            for lang_values in per_lang_values.values():
                for token, loss in lang_values:
                    avg_values[token] += loss
            for token in avg_values:
                avg_values[token] /= n_langs
            
            # Sort by averaged loss (ascending - lower loss = less important = remove first)
            # But we want to KEEP tokens with highest removal loss, so reverse=True
            sorted_tokens = sorted(
                avg_values.items(),
                key=lambda x: x[1], 
                reverse=True
            )
        
        # Keep top tokens
        num_regular_tokens = max(0, k_final - len(special))
        kept_regular = [token for token, _ in sorted_tokens[:num_regular_tokens]]
        kept_tokens = list(special) + kept_regular
    
    logger.info(f"Selected {len(kept_tokens)} tokens for final vocabulary")
    
    # Step 3: Build final tokenizer with probability averaging
    # Always use probability averaging for final tokenizer construction
    avg_prob = defaultdict(float)
    
    for token in kept_tokens:
        for lang_probs in per_lang_tokenizers.values():
            avg_prob[token] += lang_probs.get(token, 0.0)
        avg_prob[token] /= n_langs
    
    # Normalize probabilities
    Z = sum(avg_prob[t] for t in kept_tokens if t not in pinned)
    lprobs_tuples = []
    for token in kept_tokens:
        if token in pinned:
            # Special tokens keep their averaged probability  
            p = max(avg_prob[token], constants.MIN_TOKEN_PROB)
        else:
            # Regular tokens get renormalized
            p = avg_prob[token] / Z if Z > 0 else constants.MIN_TOKEN_PROB
        lprobs_tuples.append((token, math.log(max(p, constants.MIN_TOKEN_PROB))))
    
    # Build HF tokenizer
    unk_id = kept_tokens.index(unk_token)
    return _build_unigramlm_hf_tokenizer_from_lprobs(lprobs_tuples, unk_id, pretokenizer=pretokenizer, decoder=decoder)


def _get_metadata_path(tokenizer_path: str) -> str:
    """Convert tokenizer path to metadata path."""
    if tokenizer_path.endswith('.json'):
        return tokenizer_path[:-5] + '.metadata.json'  # Remove .json, add .metadata.json
    else:
        return tokenizer_path + '.metadata.json'  # Fallback


def _save_tokenizer_metadata(tokenizer_path: str, metadata: Dict[str, Any]) -> str:
    """
    Save metadata to a separate .metadata.json file alongside the tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer JSON file (e.g., "tokenizer.json")
        metadata: Metadata dictionary to save

    Returns:
        Path to the created metadata file (e.g., "tokenizer.metadata.json")
    """
    try:
        # Generate metadata file path
        metadata_path = _get_metadata_path(tokenizer_path)

        # Add reference to the original tokenizer file
        metadata_with_ref = metadata.copy()
        metadata_with_ref["tokenizer_file"] = Path(tokenizer_path).name

        # Save metadata to separate file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_ref, f, indent=2, ensure_ascii=False, separators=(',', ': '))

        logger.debug(f"Successfully saved metadata to {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.error(f"Failed to save metadata for {tokenizer_path}: {e}")
        # Return empty string to indicate failure, but don't crash
        return ""


def _create_base_metadata(tokenizer_class: str, tokenizer_type: str) -> Dict[str, Any]:
    """Create base metadata structure for any tokenizer"""
    try:
        return {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": str(tokenizer_class),
            "tokenizer_type": str(tokenizer_type),
            "environment_info": {
                "python_version": str(sys.version).replace('\n', ' ').strip(),
                "platform": str(platform.platform()),
                "num_processes": int(multiprocessing.cpu_count())
            }
        }
    except Exception as e:
        # Fallback to minimal metadata if environment info fails
        return {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": str(tokenizer_class),
            "tokenizer_type": str(tokenizer_type),
            "environment_info": {}
        }
