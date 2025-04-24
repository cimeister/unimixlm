import math
from collections import OrderedDict, defaultdict, Counter
import multiprocessing
from functools import partial
import os
import json
import numpy as np
import io 
from typing import Dict, List, Tuple, Set

SPECIAL_TOKENS = OrderedDict([("bos_token", "<s>"), ("eos_token", "</s>"), ("pad_token","<pad>"), ("unk_token","<unk>")])
NUM_TRIMMING_ITERATIONS = 3
INITIAL_VOCAB_MULT_FACTOR = 5
MIN_CHAR_OCCURENCE_THRESHOLD = 5


def count_characters_in_file(file_path: str, chunk_size: int = 1024 * 1024) -> Counter:
    """
    Count character frequencies in a single file by reading in chunks.
    
    Args:
        file_path: Path to the text file
        chunk_size: Size of chunks to read at once (default: 1MB)
        
    Returns:
        Counter object with character counts
    """
    char_counts = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read the file in chunks to avoid memory issues
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Update counts directly from the chunk
                char_counts.update(chunk)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
    
    return char_counts

def _count_worker(file_path):
    """Worker function for parallel processing"""
    return file_path, count_characters_in_file(file_path)

def get_baseline_characters(training_data_files, min_occurence_threshold=5):
    """
    Get baseline characters from training data files with efficient parallel processing.
    
    Args:
        training_data_files: List of file paths to process
        min_occurence_threshold: Minimum number of occurrences for a character to be kept
        
    Returns:
        Tuple of (chars_to_keep, char_freqs)
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
    
    total_chars = sum(total_char_counts.values())
    
    # Get relative frequencies
    char_freqs = {char: count/total_chars for char, count in total_char_counts.items()}
    
    # Filter by threshold
    chars_to_keep = [ch for ch, ct in total_char_counts.items() if ct >= min_occurence_threshold]
    
    return chars_to_keep, char_freqs


def segment_text_viterbi(text, token_probs, unk_token="<unk>", max_sub_len=20, beam_width=5.0):
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


def _segment_word_viterbi(word, token_probs, unk_token="<unk>", max_sub_len=20, beam_width=5.0):
    """
    Viterbi subword segmentation for a single 'word' (no internal spaces).
    Returns a list of sub tokens.
    
    Uses beam search pruning to speed up the algorithm significantly.
    Implements UNK handling in the style of SentencePiece's UnigramLM.
    Includes early termination for unreachable positions.
    """
    n = len(word)
    if n == 0:
        return []

    # Initialize arrays with appropriate values
    best_score = np.full(n+1, float('-inf'), dtype=np.float64)
    best_score[0] = 0.0
    best_edge = np.full(n+1, -1, dtype=np.int32)
    
    # Get the unknown token log probability
    unk_logp = math.log(max(token_probs.get(unk_token, 1e-12), 1e-12))
    
    # For each position, keep track of the best score seen so far (for pruning)
    max_score_at_pos = np.full(n+1, float('-inf'), dtype=np.float64)
    max_score_at_pos[0] = 0.0
    
    # Track positions where we need to use UNK
    requires_unk = np.zeros(n+1, dtype=bool)
    
    # Track the furthest position we've reached with valid tokens
    furthest_valid_pos = 0

    # First pass: Try to segment using only known tokens
    for start in range(n):
        current_score = best_score[start]
        
        # Skip positions we can't reach
        if current_score == float('-inf'):
            continue
            
        # Early termination: if we're beyond the furthest position we can reach,
        # we can stop the first pass
        if start > furthest_valid_pos:
            break
            
        # If this path is already too bad compared to the best path at this position, skip it
        if current_score < max_score_at_pos[start] - beam_width:
            continue

        # Try all substring lengths from start
        max_end = min(n, start + max_sub_len)
        found_valid = False

        for end in range(start+1, max_end+1):
            sub = word[start:end]
            if sub in token_probs:
                # Calculate log probability
                logp = math.log(max(token_probs[sub], 1e-12))
                score = current_score + logp
                
                # Update best score and path
                if score > best_score[end]:
                    best_score[end] = score
                    best_edge[end] = start
                    # Update the max score at this position for pruning
                    max_score_at_pos[end] = max(max_score_at_pos[end], score)
                    # Update furthest valid position
                    furthest_valid_pos = max(furthest_valid_pos, end)
                found_valid = True
        
        # Mark positions that cannot be reached with valid tokens
        if not found_valid and best_edge[start+1] == -1:
            requires_unk[start+1] = True

    # Second pass: Handle positions requiring UNK tokens
    for start in range(min(n, furthest_valid_pos + 1)):
        current_score = best_score[start]
        if current_score == float('-inf'):
            continue
            
        # Only consider UNK for positions that need it
        end = start + 1
        if end <= n and requires_unk[end]:
            score = current_score + unk_logp
            best_score[end] = score
            best_edge[end] = start
            max_score_at_pos[end] = max(max_score_at_pos[end], score)
            requires_unk[end] = False  # Mark as handled
            # Update furthest position in case UNK tokens help us reach further
            furthest_valid_pos = max(furthest_valid_pos, end)

    # Backtrack to build the token list
    tokens = []
    current = n
    
    while current > 0:
        prev = int(best_edge[current])
        if prev == -1:  # Should never happen with two-pass approach
            tokens.append(unk_token)
            current -= 1
        else:
            sub = word[prev:current]
            tokens.append(sub if sub in token_probs else unk_token)
            current = prev
    
    # Reverse the tokens list
    return tokens[::-1]


def forward_backward_unigram(text, p_lang, unk_token="<unk>", max_sub_len=15, prob_threshold=1e-12):
    """
    Forward-backward algorithm for unigram language model training that matches SentencePiece's
    UNK handling approach.
    
    Returns a dict { subword -> fractional_count }.
    """
    # Get probability for unknown token
    p_unk = p_lang.get(unk_token, prob_threshold)
    
    # Split on whitespace first, like in Viterbi
    words = text.strip().split()
    
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
    chunk_lines, p_lang, unk_token = args

    c_dict = defaultdict(float)
    total = 0.0

    for line in chunk_lines:
        usage_line = forward_backward_unigram(line, p_lang, unk_token=unk_token)
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
    chunk_lines, p_lang, unk_token = args

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
                max_sub_len=20
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
                      mode='hard'):
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
        with open(train_file, 'r', encoding='utf-8') as fin:
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
            worker_args.append((ch, p_lang, unk_token))

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

