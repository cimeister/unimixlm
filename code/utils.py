import math
from collections import OrderedDict, defaultdict, Counter
import multiprocessing
import json
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional, Union

from pathlib import Path
import random

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

SPECIAL_TOKENS = OrderedDict([("bos_token", "<s>"), ("eos_token", "</s>"), ("pad_token","<pad>"), ("unk_token","<unk>")])
NUM_TRIMMING_ITERATIONS = 3
INITIAL_VOCAB_MULT_FACTOR = 5
MIN_CHAR_OCCURENCE_THRESHOLD = 2
MIN_TOKEN_PROB = 1e-10
_TOTAL_SAMPLE_LINES = 2_000_000
MAX_VAL_TEST_SIZE = 3000
_RNG = random.Random(42)

# build once, reuse
_B2U = {i: c for i, c in enumerate(ByteLevel.alphabet())}
BY_PRETOK = ByteLevel(add_prefix_space=False)

def _apply_bytelevel_mapping(raw: str) -> str:
    return "".join(_B2U[b] for b in raw.encode("utf-8"))

def _apply_bytelevel_mapping_hf(text: str) -> str:
    
    normalized = BY_PRETOK.pre_tokenize_str(text)
    return ''.join(piece[0] for piece in normalized)

def get_baseline_characters(training_data_files, character_coverage=0.9995, min_occurence_threshold=MIN_CHAR_OCCURENCE_THRESHOLD):
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
    ids_batch, toks_batch,
    *,
    truncation, max_length,
    padding, pad_id, pad_tok,
    return_attention_mask,
):
    masks = [] if return_attention_mask else None
    out_ids, out_toks = [], []
    for ids, toks in zip(ids_batch, toks_batch):
        # trunc
        if truncation and max_length:
            ids, toks = ids[:max_length], toks[:max_length]
        # pad
        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids += [pad_id] * pad_len
            toks += [pad_tok] * pad_len
        out_ids.append(ids)
        out_toks.append(toks)
        if masks is not None:
            masks.append([1 if i < len(ids) - (pad_len if padding == "max_length" and max_length else 0) else 0
                          for i in range(len(ids))])
    return out_ids, out_toks, masks

def encode_text(
    tokenizer: Any,
    text: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]],
    *,
    add_special_tokens: bool = True,
    padding: Optional[str] = None,
    truncation: Union[bool, str] = False,
    max_length: Optional[int] = None,
    stride: Optional[int] = None,                     # only honoured by HF path
    return_overflowing_tokens: bool=False,            # idem
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
        ids_batch, toks_batch, masks = _apply_trunc_pad(
            ids_batch, toks_batch,
            truncation=truncation, max_length=max_length,
            padding=padding,
            pad_id=getattr(tokenizer, "pad_token_id", 0),
            pad_tok="<pad>",
            return_attention_mask=return_attention_mask,
        )
        if truncation and max_length:
            offsets_b = [off[:max_length] for off in offsets_b]
        if padding == "max_length" and max_length:
            offsets_b = [
                off + [(0,0)]*(max_length - len(off)) for off in offsets_b
            ]

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

        ids_batch, toks_batch, masks = _apply_trunc_pad(
            ids_batch, toks_batch,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=getattr(tokenizer, "pad_token_id", 0), pad_tok="<pad>",
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if masks is not None:
            result["attention_mask"] = masks
        if word_ids_b is not None:
            result["word_ids"] = word_ids_b
        if return_offsets_mapping:
            result["offset_mapping"] = [enc.offsets for enc in encodings]

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
            return_overflowing_tokens = return_overflowing_tokens,
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
        k = max(1000, int(_TOTAL_SAMPLE_LINES * byte_tot[lg] / total))
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
                if _RNG.random() > 0.2:   # light intra‑lang subsample
                    continue
                tgt.write(line)
                lines_written += 1
    return sampled_files

def get_baseline_bytes() -> Tuple[Set[str], Dict[str,int]]:
    alphabet = ByteLevel.alphabet()
    return set(alphabet), {c:1 for c in alphabet}


def segment_text_viterbi(text, token_probs, unk_token="<unk>", max_sub_len=15, beam_width=5.0):
    """
    Segment `text` into tokens using a whitespace split + Viterbi subword approach,
    under the distribution given by `token_probs` (a dict mapping token->prob).
    Returns a list of tokens.

    Args:
        text: A string (possibly containing multiple words).
        token_probs: dict(token -> float probability).
        unk_token: which token to use when no match can be found.
        max_sub_len: maximum subword length to consider in Viterbi.
        beam_width: pruning parameter, paths with score < best_score - beam_width are pruned

    Returns:
        A list of tokens (subwords).
    """
    # The final list of subword tokens across all whitespace-delimited words
    # TODO: make work for byte level
    all_tokens = []

    # 1) Whitespace split
    words = text.strip().split()
    if not words:
        return all_tokens

    # Here we process the whole batch of words together
    for word in words:
        # 2) For each word, run Viterbi
        sub_tokens = _segment_word_viterbi(word, token_probs, unk_token, max_sub_len, beam_width)
        all_tokens.extend(sub_tokens)

    return all_tokens


def _segment_word_viterbi(
    word: str,
    token_probs: dict,
    unk_token: str = "<unk>",
    max_sub_len: int = 20,
    beam_width: float = 5.0,
):
    """
    Efficient Viterbi-style sub-word segmentation (SentencePiece UnigramLM).

    * First pass  – explore paths that use only **known** tokens.
    * Second pass – insert required <unk> tokens **and immediately keep
                    exploring from the position that follows each <unk>**,
                    so later characters/bytes are no longer forced to become <unk>.
    The algorithm still runs in O(n · max_sub_len) time with beam pruning.
    """
    n = len(word)
    if n == 0:
        return []

    # --- tables -------------------------------------------------------------
    best_score = np.full(n + 1, float("-inf"), dtype=np.float64)
    best_edge = np.full(n + 1, -1, dtype=np.int32)
    max_score_at_pos = np.full(n + 1, float("-inf"), dtype=np.float64)
    best_score[0] = max_score_at_pos[0] = 0.0

    unk_logp = math.log(max(token_probs.get(unk_token, 1e-12), 1e-12))
    requires_unk = np.zeros(n + 1, dtype=bool)

    # ------------------------------------------------------------------------
    # Pass 1 – only known tokens
    # ------------------------------------------------------------------------
    for start in range(n):
        cur_score = best_score[start]
        if cur_score == float("-inf"):
            continue
        if cur_score < max_score_at_pos[start] - beam_width:
            continue

        max_end = min(n, start + max_sub_len)
        found = False
        for end in range(start + 1, max_end + 1):
            sub = word[start:end]
            if sub in token_probs:
                score = cur_score + math.log(max(token_probs[sub], 1e-12))
                if score > best_score[end]:
                    best_score[end] = score
                    best_edge[end] = start
                    max_score_at_pos[end] = max(max_score_at_pos[end], score)
                found = True

        if not found and best_edge[start + 1] == -1:
            requires_unk[start + 1] = True

    # ------------------------------------------------------------------------
    # Pass 2 – insert <unk> where required *and* keep exploring forward
    # ------------------------------------------------------------------------
    for start in range(n):                       # <— no early-exit break
        cur_score = best_score[start]
        if cur_score == float("-inf"):
            continue

        # 2a. insert exactly one-char <unk> if we still need it here
        end = start + 1
        if end <= n and requires_unk[end]:
            score = cur_score + unk_logp
            if score > best_score[end]:
                best_score[end] = score
                best_edge[end] = start
                max_score_at_pos[end] = max(max_score_at_pos[end], score)
            requires_unk[end] = False    # handled – don’t revisit

        # 2b. now try *normal* tokens that begin right after start
        if cur_score < max_score_at_pos[start] - beam_width:
            continue                      # (reuse beam pruning)
        max_end = min(n, start + max_sub_len)
        for end in range(start + 1, max_end + 1):
            sub = word[start:end]
            if sub in token_probs:
                score = cur_score + math.log(max(token_probs[sub], 1e-12))
                if score > best_score[end]:
                    best_score[end] = score
                    best_edge[end] = start
                    max_score_at_pos[end] = max(max_score_at_pos[end], score)

    # ------------------------------------------------------------------------
    # Back-trace
    # ------------------------------------------------------------------------
    tokens = []
    pos = n
    while pos > 0:
        prev = int(best_edge[pos])
        if prev == -1:
            tokens.append(unk_token)
            pos -= 1
        else:
            sub = word[prev:pos]
            tokens.append(sub if sub in token_probs else unk_token)
            pos = prev

    return tokens[::-1]


def forward_backward_unigram(text, p_lang, unk_token="<unk>", max_sub_len=20, prob_threshold=1e-12, byte_level=False):
    """
    Forward-backward algorithm for unigram language model training that matches SentencePiece's
    UNK handling approach.
    
    Returns a dict { subword -> fractional_count }.
    """
    # Get probability for unknown token
    p_unk = p_lang.get(unk_token, prob_threshold)
    
    # Split on whitespace first, like in Viterbi
    if byte_level:
        text = _apply_bytelevel_mapping_hf(text)

    # --- 2. Use the *whole* line; leading spaces matter --------------
    words = [text.rstrip("\n")]
    
    # Initialize global counts
    total_counts = {}
    
    for word in words:
        n = len(word)
        if n == 0:
            continue
            
        # Initialize alpha and beta arrays
        alpha = [0.0] * (n+1)
        alpha_without_unk = [0.0] * (n+1)  # Tracks probability mass WITHOUT using UNK
        beta = [0.0] * (n+1)
        alpha[0] = 1.0
        alpha_without_unk[0] = 1.0
        
        # Track positions that need UNK (cannot be reached with regular tokens)
        requires_unk = [False] * (n+1)
        
        # Track furthest reachable position for early termination
        furthest_reachable = 0
        
        # Helper function for token probabilities with memoization
        token_prob_cache = {}
        def token_prob(token):
            if token not in token_prob_cache:
                token_prob_cache[token] = p_lang.get(token, 0.0)
            return token_prob_cache[token]
        
        # Forward pass - first compute alpha_without_unk
        for i in range(n):
            base_val = alpha_without_unk[i]
            if base_val == 0.0:
                continue
                
            # Early termination check - if we're beyond what we can reach with regular tokens
            if i > furthest_reachable:
                break
                
            # Try all valid subwords
            end_limit = min(n, i+max_sub_len)
            found_valid = False
            
            for j in range(i+1, end_limit+1):
                sub = word[i:j]
                p_sub = token_prob(sub)
                if p_sub > prob_threshold:
                    alpha_without_unk[j] += base_val * p_sub
                    found_valid = True
                    furthest_reachable = max(furthest_reachable, j)
            
            # Mark positions that need UNK
            if not found_valid and i+1 <= n:
                requires_unk[i+1] = True
        
        # Forward pass with UNK - compute full alpha values
        # Reset furthest reachable to account for UNK tokens
        furthest_reachable_with_unk = furthest_reachable
        
        for i in range(n):
            base_val = alpha[i]
            if base_val == 0.0:
                continue
                
            # Early termination check
            if i > furthest_reachable_with_unk:
                break
                
            # Only allow UNK for positions that require it
            if i+1 <= n and requires_unk[i+1]:
                alpha[i+1] += base_val * p_unk
                furthest_reachable_with_unk = max(furthest_reachable_with_unk, i+1)
            
            # Try all valid subwords
            end_limit = min(n, i+max_sub_len)
            for j in range(i+1, end_limit+1):
                sub = word[i:j]
                p_sub = token_prob(sub)
                if p_sub > prob_threshold:
                    alpha[j] += base_val * p_sub
                    furthest_reachable_with_unk = max(furthest_reachable_with_unk, j)
        
        # Calculate normalization factor
        Z = alpha[n]
        if Z <= 0:
            # If we can't segment this word, count it as one <unk>
            total_counts[unk_token] = total_counts.get(unk_token, 0.0) + 1.0
            continue
        
        # Backward pass
        beta[n] = 1.0
        for i in range(n-1, -1, -1):
            # Skip positions that can't contribute to any valid path
            if alpha[i] == 0.0:
                continue
                
            # Only allow UNK for positions that require it
            if i+1 <= n and requires_unk[i+1]:
                beta[i] += p_unk * beta[i+1]
                
            # Try all valid subwords
            end_limit = min(n, i+max_sub_len)
            for j in range(i+1, end_limit+1):
                sub = word[i:j]
                p_sub = token_prob(sub)
                if p_sub > prob_threshold:
                    beta[i] += p_sub * beta[j]
        
        # Accumulate expected counts
        for i in range(n):
            ai = alpha[i]
            if ai == 0.0:
                continue
                
            # Early termination check
            if i > furthest_reachable_with_unk:
                break
                
            # Count <unk> token usage only where required
            if i+1 <= n and requires_unk[i+1] and beta[i+1] > 0:
                frac_unk = (ai * p_unk * beta[i+1]) / Z
                if frac_unk > 0:
                    total_counts[unk_token] = total_counts.get(unk_token, 0.0) + frac_unk
            
            # Count subword token usage
            end_limit = min(n, i+max_sub_len)
            for j in range(i+1, end_limit+1):
                # Skip if this can't possibly contribute
                if beta[j] == 0.0:
                    continue
                    
                sub = word[i:j]
                p_sub = token_prob(sub)
                if p_sub > prob_threshold:
                    frac = (ai * p_sub * beta[j]) / Z
                    if frac > 0:
                        total_counts[sub] = total_counts.get(sub, 0.0) + frac
    
    return total_counts


def _accumulate_usage_worker_soft(args):
    """
    Worker for the 'soft' E-step, i.e. forward-backward to accumulate fractional usage.
    """
    chunk_lines, p_lang, unk_token, byte_level = args

    c_dict = defaultdict(float)
    total = 0.0

    for line in chunk_lines:
        usage_line = forward_backward_unigram(line, p_lang, unk_token=unk_token, byte_level=byte_level, max_sub_len=30 if byte_level else 20)
        line_count = sum(usage_line.values())  # total subwords in a segmentation => expected # of tokens
        total+= line_count
        for tk, frac in usage_line.items():
            c_dict[tk]+= frac

    return (dict(c_dict), total)

def _accumulate_usage_worker_hard(args):
    """
    Top-level function for multiprocessing usage, with optimized batch processing.
    Args:
        chunk_lines: list of text lines
        p_lang: dict(token -> prob) for the current language's distribution
        unk_token: string for unknown token
    Returns: (counts_dict, total_subwords)
    """
    chunk_lines, p_lang, unk_token, byte_level = args

    c_dict = defaultdict(float)
    total = 0
    
    # Process in batches for better CPU cache utilization
    batch_size = 50  # Adjust based on your typical line length
    
    for i in range(0, len(chunk_lines), batch_size):
        batch = chunk_lines[i:i+batch_size]
        
        for line in batch:
            # Apply beam search pruning in Viterbi (5.0 is a reasonable default beam width)
            tokens = segment_text_viterbi(
                line, 
                p_lang, 
                unk_token=unk_token, 
                max_sub_len=15,
                byte_level=byte_level
            )

            total += len(tokens)
            for tk in tokens:
                c_dict[tk] += 1

    return (dict(c_dict), total)


def _accumulate_usage(train_file,
                      p_lang,
                      unk_token,
                      batch_size=1000,
                      num_processes=None,
                      mode='soft',
                      byte_level=False):
        """
        For a single language's corpus:
          - segment each line with distribution p_lang
          - collect usage counts c(token), total usage
        We'll do it in parallel to speed up big corpora.
        """
        if num_processes is None:
            # at most 20
            num_processes = min(20, multiprocessing.cpu_count()-1)

        # 1) read entire file's lines into memory (if feasible)
        lines = []
        with open(train_file, 'r', encoding='utf-8', errors="replace") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    lines.append(line)

        from collections import defaultdict
        c_dict = defaultdict(float)
        total = 0

        if not lines:
            return (c_dict, total)

        # 2) chunk the lines for each process
        n = len(lines)
        chunk_size = max(1, n // num_processes)
        chunks = []
        idx = 0
        while idx < n:
            chunk = lines[idx:idx + chunk_size]
            idx += chunk_size
            chunks.append(chunk)

        # 3) Build worker args
        # we pass p_lang + unk_token for the segmentation
        worker_args = []
        for ch in chunks:
            worker_args.append((ch, p_lang, unk_token, byte_level))

        # 4) Launch pool
        worker_func = _accumulate_usage_worker_hard if mode == 'hard' else _accumulate_usage_worker_soft
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(worker_func, worker_args)

        # 5) combine partial results
        for partial_cdict, partial_total in results:
            total += partial_total
            for tk, cnt in partial_cdict.items():
                c_dict[tk] += cnt

        return (c_dict, total)


def _accumulate_usage_dummy(
    file_path: str,
    p_token: Dict[str, float],
    unk_token: str,
    mode: str = "soft",
    avg_tokens_per_line: int = 20,
) -> Tuple[Dict[str, float], float]:
    """
    Return synthetic token-usage counts without touching `file_path`.

    • Keeps the signature (and return types) identical to the real
      `_accumulate_usage`, so the rest of the pipeline works unmodified.
    • Draws counts from a Dirichlet distribution centred on the current
      probabilities `p_token`, then scales them to mimic `avg_tokens_per_line`
      over 1 000 virtual lines.
    """
    # ---------- synthetic corpus size ----------
    virtual_lines = 1_000
    tot_subwords  = avg_tokens_per_line * virtual_lines   # e.g. 20k

    # ---------- sample a Dirichlet centred on p_token ----------
    # α = p * β; choose β so the sample variance is reasonably small.
    beta = 10_000                                         # larger = lower var
    alpha = [max(p * beta, 1e-9) for p in p_token.values()]
    sample = random.gammavariate  # alias

    dirichlet = [sample(a, 1.0) for a in alpha]
    z = sum(dirichlet)
    probs = [d / z for d in dirichlet]

    counts = {tk: p * tot_subwords for tk, p in zip(p_token.keys(), probs)}

    # ---------- enforce fixed UNK prior ----------
    counts[unk_token] = p_token[unk_token] * tot_subwords

    return counts, float(tot_subwords)

   # Flatten/truncate
def fix_input_ids(example, max_length=1024):
    if len(example["input_ids"])>0 and isinstance(example["input_ids"][0], list):
        flat=[]
        for s in example["input_ids"]:
            flat.extend(s)
        example["input_ids"] = flat
    if len(example["input_ids"])>max_length:
        example["input_ids"] = example["input_ids"][:max_length]
    example["labels"] = list(example["input_ids"])
    return example

def nonempty(example):
    return (len(example["input_ids"])>0)


def set_special_tokens(fast_tokenizer):
    vocab = fast_tokenizer.get_vocab()

    for token_name, token in SPECIAL_TOKENS.items():
        if token in vocab:
                # This sets the pad_token to <pad> if the user wants it recognized as special
            setattr(fast_tokenizer, token_name, token)
            setattr(fast_tokenizer, token_name + '_id', vocab[token])

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
    #TODO: Add byte_fallback=self.byte_level when feature becomes available
    model = Unigram(probs_tuples, unk_id)
    tok = Tokenizer(model)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False) if byte_level else Whitespace()

    return tok

# ------------------------------------------------------------------
def tokmix_merge(per_lang_tokenizers: Dict[str, Dict],
                 k_lang: int, k_final: int,
                 byte_level: bool, unk_token: str, special: List[str]) -> Tokenizer:
    """
    Combine per‑language ULM tokenizers à la TokMix.
      k_lang   – #top tokens to take from each language model
      k_final  – desired merged vocab size
    """
    pool = []
    for lg, global_probs in per_lang_tokenizers.items():
        # sort by score (log prob) descending, skip specials
        top = [t for t, _ in sorted(global_probs.items(),
                                    key=lambda x: x[1], reverse=True)
               if t not in special][:k_lang]
        pool.extend(top)

    # majority voting favouring pieces shared across langs
    counts = Counter(pool)
    merged = [t for t, _ in counts.most_common()]
    merged = special + merged[:k_final-len(special)]

    # build HF tokenizer (probabilities initialised uniformly)
    probs = [(t, math.log(1/len(merged))) for t in merged]
    unk_id = merged.index(unk_token)
    return _build_hf_tokenizer_from_probs(probs, unk_id, byte_level)
