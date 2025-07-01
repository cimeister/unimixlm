import math
from collections import defaultdict, Counter
import multiprocessing
import json
from typing import Dict, List, Tuple, Set, Any, Optional, Union

from pathlib import Path
import hashlib
import code.constants as constants

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from transformers import AutoTokenizer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# build once, reuse
_B2U = {i: c for i, c in enumerate(ByteLevel.alphabet())}
BY_PRETOK = ByteLevel(add_prefix_space=False)

def _apply_bytelevel_mapping(raw: str) -> str:
    return "".join(_B2U[b] for b in raw.encode("utf-8"))

def _apply_bytelevel_mapping_hf(text: str) -> str:
    
    normalized = BY_PRETOK.pre_tokenize_str(text)
    return ''.join(piece[0] for piece in normalized)

def mk_id(raw: str) -> str:
    # 8-byte / 16-char hex digest, utf-8, replace errors
    digest = hashlib.blake2s(
        raw.encode("utf-8", "replace"),
        digest_size=8                      # 64-bit
    ).hexdigest()
    return digest


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
                        out_root: str) -> Dict[str, str]:
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
        # ensure at least 1000 lines
        k = max(1000, int(constants._TOTAL_SAMPLE_LINES * byte_tot[lg] / total))
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

def get_baseline_bytes() -> Tuple[Set[str], Dict[str,int]]:
    alphabet = ByteLevel.alphabet()
    return set(alphabet), {c:1 for c in alphabet}


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


def set_special_tokens(tokenizer):
    vocab = tokenizer.get_vocab()

    for token_name, token in constants.SPECIAL_TOKENS.items():
        if token in vocab:
                # This sets the pad_token to <pad> if the user wants it recognized as special
            setattr(tokenizer, token_name, token)
            setattr(tokenizer, token_name + '_id', vocab[token])

class SafeFormattingDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def load_tokenizer_with_mode(path: str):
    tok = Tokenizer.from_file(path)
    byte_level = isinstance(tok.pre_tokenizer, ByteLevel)
    try:
        # sanity-check the model as well
        byte_fallback = bool(tok.model.byte_fallback)
    except AttributeError:
        byte_fallback = False
    if byte_level and not byte_fallback:
        logger.warning(f"{path} looks byte-level but byte_fallback==False (currently known issue for HF Unigram Tokenizers).")
    return tok, byte_level


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


def _build_hf_tokenizer_from_probs(
    probs_tuples: List[Tuple[str, float]],
    unk_id: int,
    byte_level: bool = False
) -> Tokenizer:
    """
    Turn a plain probability dictionary into a HF Unigram tokenizer.
    """
    #TODO: Add byte_fallback=self.byte_level if/when feature becomes available
    model = Unigram(probs_tuples, unk_id)
    tok = Tokenizer(model)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False) if byte_level else Whitespace()

    return tok


def tokmix_merge(
    per_lang_tokenizers: Dict[str, Dict[str, float]],
    k_final: int,
    byte_level: bool,
    unk_token: str,
    special: List[str],
) -> Tokenizer:
    """
    Merge per-language UnigramLM vocabularies according to TokMix alg.
    Each `per_lang_tokenizers[lg]` is a dict {token -> prob} with
    probabilities summing to 1 inside that language.
    The merged vocabulary keeps the `k_final` highest averaged probs.
    """
    langs          = list(per_lang_tokenizers)
    n_langs        = len(langs)
    avg_prob: Dict[str, float] = defaultdict(float)

    # 1. accumulate probabilities
    for lg_probs in per_lang_tokenizers.values():
        for tok, p in lg_probs.items():
            avg_prob[tok] += p
    # 2. average (tokens missing in a language contribute 0)
    for tok in avg_prob:
        avg_prob[tok] /= n_langs

    # 3. sort by averaged probability, keep top-k (specials are pinned)
    pinned = set(special)
    sorted_tokens = sorted(
        (t for t in avg_prob if t not in pinned),
        key=lambda t: avg_prob[t],
        reverse=True,
    )
    kept = list(special) + sorted_tokens[: max(0, k_final - len(special))]

    # 4. renormalise probs over the kept set
    Z = sum(avg_prob[t] for t in kept if t not in pinned)
    probs_tuples = []
    for t in kept:
        p = constants.MIN_TOKEN_PROB if t not in avg_prob else (avg_prob[t] / Z)
        probs_tuples.append((t, math.log(p)))

    # 5. build HF tokenizer
    unk_id = kept.index(unk_token)
    return _build_hf_tokenizer_from_probs(probs_tuples, unk_id, byte_level)
