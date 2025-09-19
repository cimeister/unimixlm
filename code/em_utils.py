import multiprocessing
import math
import random
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple

from code.utils import _sample_lines_from_file
from code.tokenizer_formatting_utils import _apply_pretok_mapping_hf
from code.constants import (
    MIN_TOKEN_LOG_PROB, 
    MIN_TOKEN_PROB, 
    SPECIAL_TOKENS,
    MAX_BYTE_TOKEN_LEN,
    MAX_CHAR_TOKEN_LEN
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def segment_text_viterbi(text, 
                         token_log_probs, 
                         unk_token="<unk>", 
                         max_sub_len=MAX_CHAR_TOKEN_LEN, 
                         beam_width=5.0, 
                         pretok=None, 
                         whitespace_boundaries=None):
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
    text = _apply_pretok_mapping_hf(text, pretok, whitespace_boundaries)
    if not isinstance(text, list):
        # For byte-level, treat the entire text as one sequence (no whitespace splitting)
        return _segment_word_viterbi_sp(text, token_log_probs, unk_token, max_sub_len, beam_width)
    
    all_tokens = []
    if not text:
        return all_tokens

    # Here we process the whole batch of words together
    for word in text:
        # 2) For each word, run Viterbi
        sub_tokens = _segment_word_viterbi(word, token_log_probs, unk_token, max_sub_len, beam_width)
        all_tokens.extend(sub_tokens)

    return all_tokens


def _segment_word_viterbi(
    word: str,
    token_log_probs: dict,
    unk_token: str = "<unk>",
    max_sub_len: int = MAX_CHAR_TOKEN_LEN,
    beam_width: float = 5.0,
):
    """
    Efficient Viterbi-style sub-word segmentation 

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

    unk_logp = token_log_probs.get(unk_token, MIN_TOKEN_LOG_PROB)
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
            if sub in token_log_probs:
                score = cur_score + max(token_log_probs[sub], MIN_TOKEN_LOG_PROB)
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
            if sub in token_log_probs:
                score = cur_score + max(token_log_probs[sub], MIN_TOKEN_LOG_PROB)
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
            tokens.append(sub if sub in token_log_probs else unk_token)
            pos = prev

    return tokens[::-1]


def _segment_word_viterbi_sp(
    word: str,
    token_log_probs: dict,
    unk_token: str = "<unk>",
    max_sub_len: int = MAX_CHAR_TOKEN_LEN,
    beam_width: float = 5.0,
):
    """
    Single-pass Viterbi segmentation closer to SentencePiece implementation.
    
    This implementation:
    - Considers all possible tokens (including UNK) in a single pass
    - Uses proper beam pruning based on score differences
    - Handles empty strings correctly
    - Matches SentencePiece's algorithm except for their UNK penalty, which requires nuance
    """
    n = len(word)
    if n == 0:
        return []
    
    # Initialize DP tables
    best_score = np.full(n + 1, float("-inf"), dtype=np.float64)
    best_edge = np.full(n + 1, -1, dtype=np.int32)
    
    # Track the best score seen so far for beam pruning
    best_score_so_far = 0.0
    best_score[0] = 0.0
    
    # Precompute log probabilities
    unk_logp = unk_logp = token_log_probs.get(unk_token, MIN_TOKEN_LOG_PROB)
    
    # Single pass: consider all possibilities
    for start in range(n):
        score = best_score[start]
        if score == float("-inf"):
            continue
            
        # Beam pruning: skip if too far from best
        if score < best_score_so_far - beam_width:
            continue
        
        # Try all possible token lengths
        found_any = False
        max_end = min(n, start + max_sub_len)
        
        for end in range(start + 1, max_end + 1):
            sub = word[start:end]
            
            # Check if this substring is a known token
            if sub in token_log_probs:
                new_score = score + max(token_log_probs[sub], MIN_TOKEN_LOG_PROB)
                found_any = True
            else:
                # Use UNK token for unknown substrings
                # In SentencePiece, UNK is typically single character/byte
                if end == start + 1:  # Single character UNK
                    new_score = score + unk_logp
                else:
                    continue  # Skip multi-character unknown sequences
            
            # Update if this path is better
            if new_score > best_score[end]:
                best_score[end] = new_score
                best_edge[end] = start
                best_score_so_far = max(best_score_so_far, new_score)
        
        # If no known tokens found, must use single-character UNK
        if not found_any and start + 1 <= n:
            new_score = score + unk_logp
            if new_score > best_score[start + 1]:
                best_score[start + 1] = new_score
                best_edge[start + 1] = start
                best_score_so_far = max(best_score_so_far, new_score)
    
    # Backtrace to get tokens
    tokens = []
    pos = n
    while pos > 0:
        prev = int(best_edge[pos])
        if prev == -1:
            # This shouldn't happen with proper implementation
            # Fall back to single character
            tokens.append(word[pos-1:pos])
            pos -= 1
        else:
            sub = word[prev:pos]
            # Check if it's a known token or UNK
            if sub in token_log_probs:
                tokens.append(sub)
            else:
                tokens.append(unk_token)
            pos = prev
    
    return tokens[::-1]


def forward_backward_unigram_log_space(
    text,
    log_p_lang,
    unk_token="<unk>",
    max_sub_len=MAX_CHAR_TOKEN_LEN,
    pretok=None,
    whitespace_boundaries=False
):
    """
    Forward–backward for UnigramLM in log-space, with SentencePiece-style UNK gating:
    UNK is allowed exactly at positions where *no real token* can start.

    Args:
        text: input string (the whole line is treated as one 'word'; leading spaces matter)
        log_p_lang: dict {token: log_prob} including <unk>
        unk_token: the unknown-token string
        max_sub_len: maximum subword length to consider
        byte_level: if True, apply your byte-level mapping before DP (expects _apply_bytelevel_mapping_hf)

    Returns:
        dict {token: expected_count} (fractional EM counts)
    """
    LOG_NEG_INF = -np.inf

    texts = _apply_pretok_mapping_hf(text, pretok, whitespace_boundaries)#.rstrip("\n")

    # Use membership (not thresholds) to decide which substrings are valid tokens.
    if unk_token not in log_p_lang or not np.isfinite(log_p_lang[unk_token]):
        raise ValueError(f"'{unk_token}' must be in log_p_lang with a finite log-prob.")
    log_p_unk = log_p_lang[unk_token]

    # "Real" vocab excludes UNK for the startability test and subword transitions.
    real_vocab = set(log_p_lang.keys()) - {unk_token}

    token_cache = {}
    def get_token_log_prob(token):
        if token not in token_cache:
            if token in real_vocab:
                token_cache[token] = log_p_lang[token]
            else:
                token_cache[token] = LOG_NEG_INF
        return token_cache[token]

    # --- Pre-pass: where can a real token start
    total_counts = {}
    # Helper to accumulate in linear space without instability
    def add_count(tok, log_count):
        if np.isfinite(log_count):
            total_counts[tok] = total_counts.get(tok, 0.0) + float(np.exp(log_count))

    for text in texts:
        text = text.rstrip("\n")
        n = len(text)
        if n == 0:
            continue
        can_start = [False] * n
        for i in range(n):
            end = min(n, i + max_sub_len)
            # early-exit once we find any matching token
            for j in range(i + 1, end + 1):
                if text[i:j] in real_vocab:
                    can_start[i] = True
                    break

        # UNK is *required* at i -> i+1 exactly when no real token can start at i
        requires_unk = [False] * (n + 1)
        for i in range(n):
            requires_unk[i + 1] = not can_start[i]

        # --- Forward pass (log-space)
        log_alpha = np.full(n + 1, LOG_NEG_INF)
        log_alpha[0] = 0.0  # log(1)

        for i in range(n):
            if log_alpha[i] == LOG_NEG_INF:
                continue

            # UNK transition (length-1) only where required
            if requires_unk[i + 1]:
                log_alpha[i + 1] = np.logaddexp(log_alpha[i + 1], log_alpha[i] + log_p_unk)

            # Real tokens
            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab:
                    log_alpha[j] = np.logaddexp(log_alpha[j], log_alpha[i] + log_p_lang[sub])

        log_Z = log_alpha[n]
        if not np.isfinite(log_Z):
            # This should not happen 
            raise RuntimeError(
                "Failed to reach the end of the word. Numerical stability issues."
            )

        # --- Backward pass (log-space)
        log_beta = np.full(n + 1, LOG_NEG_INF)
        log_beta[n] = 0.0  # log(1)

        for i in range(n - 1, -1, -1):
            # UNK transition (length-1) only where required
            if requires_unk[i + 1]:
                log_beta[i] = np.logaddexp(log_beta[i], log_p_unk + log_beta[i + 1])

            # Real tokens
            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab:
                    log_beta[i] = np.logaddexp(log_beta[i], log_p_lang[sub] + log_beta[j])

        # (Sanity) beta at 0 equals Z
        if not np.isclose(log_beta[0], log_Z, rtol=1e-8, atol=1e-8):
            logger.warning("Large floating point errors...")

        for i in range(n):
            if log_alpha[i] == LOG_NEG_INF:
                continue

            # UNK at i -> i+1 (only where required)
            if requires_unk[i + 1] and log_beta[i + 1] != LOG_NEG_INF:
                add_count(unk_token, log_alpha[i] + log_p_unk + log_beta[i + 1] - log_Z)

            # Real tokens
            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab and log_beta[j] != LOG_NEG_INF:
                    add_count(sub, log_alpha[i] + log_p_lang[sub] + log_beta[j] - log_Z)

    return total_counts


def forward_backward_unigram(text, 
                             p_lang, unk_token="<unk>", 
                             max_sub_len=MAX_CHAR_TOKEN_LEN, 
                             prob_threshold=MIN_TOKEN_PROB, 
                             pretok=None, 
                             whitespace_boundaries=False):
    """
    Forward-backward algorithm for unigram language model training that matches SentencePiece's
    UNK handling approach.
    
    Returns a dict { subword -> fractional_count }.
    """
    # Get probability for unknown token
    p_unk = p_lang.get(unk_token, prob_threshold)
    
    texts = _apply_pretok_mapping_hf(text, pretok, whitespace_boundaries)
    #Initialize global counts
    total_counts = {}

    token_prob_cache = {}
    def token_prob(token):
        if token not in token_prob_cache:
            token_prob_cache[token] = p_lang.get(token, 0.0)
        return token_prob_cache[token]

    # --- 2. Use the *whole* line; leading spaces matter --------------
    for word in texts:
        word = text.rstrip("\n")
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
            return {}
        
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
    chunk_lines, p_lang, unk_token, whitespace_boundaries, pretok, max_token_len = args

    c_dict = defaultdict(float)
    total = 0.0

    for line in chunk_lines:
        usage_line = forward_backward_unigram_log_space(line, 
                                                        p_lang, 
                                                        unk_token=unk_token, 
                                                        pretok=pretok,
                                                        max_sub_len=max_token_len,
                                                        whitespace_boundaries=whitespace_boundaries)
        line_count = sum(usage_line.values())  # total subwords in a segmentation => expected # of tokens
        total += line_count
        for tk, frac in usage_line.items():
            c_dict[tk] += frac

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
    chunk_lines, logp_lang, unk_token, whitespace_boundaries, pretok, max_token_len = args

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
                logp_lang, 
                unk_token=unk_token, 
                max_sub_len=max_token_len,
                pretok=pretok,
                whitespace_boundaries=whitespace_boundaries
            )

            total += len(tokens)
            for tk in tokens:
                c_dict[tk] += 1

    return (dict(c_dict), total)


def _accumulate_usage(train_file,
                      p_lang,
                      unk_token,
                      num_processes=None,
                      mode='soft',
                      whitespace_boundaries=False,
                      pretokenizer=None,
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
        logger.info(f"Reading lines from {train_file}...")
        with open(train_file, 'r', encoding='utf-8', errors="replace") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    lines.append(line)

        if not lines:
            return (defaultdict(float), 0)

        # 2) chunk the lines for each process
        n = len(lines)
        chunk_size = max(1, n // num_processes)
        chunks = []
        idx = 0
        while idx < n:
            chunk = lines[idx:idx + chunk_size]
            idx += chunk_size
            chunks.append(chunk)
        
        log_plang = {tk: math.log(p if p else MIN_TOKEN_PROB) for tk, p in p_lang.items()}
        max_token_len = MAX_BYTE_TOKEN_LEN if byte_level else MAX_CHAR_TOKEN_LEN
        # 3) Build worker args
        # we pass p_lang + unk_token for the segmentation
        worker_args = []
        for ch in chunks:
            worker_args.append(
                (ch, log_plang, unk_token, whitespace_boundaries, pretokenizer, max_token_len)
            )

        # 4) Launch pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            if mode == 'hard': 
                results = pool.map(_accumulate_usage_worker_hard, worker_args)
            else: 
                results = pool.map(_accumulate_usage_worker_soft, worker_args)

        # 5) combine partial results
        c_dict = defaultdict(float)
        total = 0
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


def _compute_loss_with_alternatives(token_to_remove: str, counts: Dict[str, float], 
                                    alternatives: Dict[str, List[Tuple[str, float]]], 
                                    p_token: Dict[str, float],
                                    unk_token: str) -> float:
    """
    Compute loss as the change in negative log-likelihood when removing a token.
    Loss = NLL(corpus | vocab_without_token) - NLL(corpus | vocab_with_token)
    
    Positive loss means removing the token hurts (increases NLL).
    Negative loss means removing the token helps (decreases NLL).
    """
    if token_to_remove not in counts or counts[token_to_remove] == 0:
        return 0.0
    
    if token_to_remove not in p_token:
        return 0.0
    
    # Total counts and current probability
    removed_count = counts[token_to_remove]
    
    # Current negative log-likelihood contribution from this token
    current_nll = -removed_count * math.log(p_token[token_to_remove])
    
    # Get weighted alternatives
    weighted_alternatives = alternatives.get(token_to_remove, [])
    
    if not weighted_alternatives:
        # All occurrences become UNK
        if unk_token in p_token and p_token[unk_token] > 0:
            new_nll = -removed_count * math.log(p_token[unk_token])
        else:
            new_nll = float('inf')
    else:
        # Compute new NLL contribution from alternatives
        new_nll = 0.0
        
        for alt_token, weight in weighted_alternatives:
            if alt_token in p_token and p_token[alt_token] > 0:
                # This alternative gets 'weight' fraction of the removed count
                alt_count = removed_count * weight
                
                # NLL contribution (using current probability of alternative)
                # Note: This is an approximation - exact computation would require
                # recomputing all probabilities after vocabulary reduction
                new_nll += -alt_count * math.log(p_token[alt_token])
    
    # Loss is the change in NLL
    # Positive loss = removing token hurts (increases NLL)
    # Negative loss = removing token helps (decreases NLL)
    loss = new_nll - current_nll
    
    return loss


def _compute_alternatives_direct(
    token_probs: dict, 
    unk_token: str, 
    byte_level: bool = False
) -> Dict[str, List[Tuple[List[str], float]]]:
    """
    Compute alternatives directly for each token without corpus iteration.
    """
    token_log_probs = {}
    for token, p in token_probs.items():
        token_log_probs[token] = math.log(max(p, MIN_TOKEN_PROB))  # Use MIN_TOKEN_PROB since that's what will be set at end
    alternatives = {}
    
    for token in token_probs:
        if token == unk_token or token in SPECIAL_TOKENS:
            continue
            
        # Get the string representation of the token
        token_string = token  # assuming token is already the string
        
        # Get alternative segmentation
        alt_sequence = _get_segmentation_excluding_token(
            token_string, 
            token_log_probs, 
            unk_token, 
            token, 
            max_token_len=MAX_BYTE_TOKEN_LEN if byte_level else MAX_CHAR_TOKEN_LEN
        )
        
        # Since we're using only second best seg, all weight is on the single alternative
        alternatives[token] = [(alt_sequence, 1.0)]
    
    return alternatives

def _get_segmentation_excluding_token(substring: str, token_log_probs: dict, unk_token: str, 
                                excluded_token: str, max_token_len: int) -> List[List[str]]:
    """
    Get alternative tokenization sequences when excluded_token is not available.
    """
    # Create modified probabilities without the excluded token
    exc_tok_lprob = token_log_probs.pop(excluded_token, None)
    
    # Get the best alternative segmentation
    alt_seg = _segment_word_viterbi_sp(substring, token_log_probs, unk_token, max_token_len)
    if exc_tok_lprob:
        token_log_probs[excluded_token] = exc_tok_lprob
    
    # Return the single best alternative
    # TODO: Could be extended to return top-k from beam search results
    return alt_seg


def _compute_loss_with_alternative_sequences(
    token_to_remove: str, 
    counts: Dict[str, float], 
    alternative_sequences: Dict[str, List[Tuple[List[str], float]]], 
    p_token: Dict[str, float],
    unk_token: str
) -> float:
    """
    Compute loss accounting for full sequence probabilities.
    Loss = NLL(corpus | vocab_without_token) - NLL(corpus | vocab_with_token)
    """
    if token_to_remove not in counts or counts[token_to_remove] == 0:
        return 0.0
    
    if token_to_remove not in p_token or p_token[token_to_remove] <= 0:
        return 0.0
    
    removed_count = counts[token_to_remove]
    current_nll = -removed_count * math.log(p_token[token_to_remove])

    unk_logp = math.log(p_token[unk_token])
    
    # Get alternative sequences
    weighted_sequences = alternative_sequences.get(token_to_remove, [])
    
    if not weighted_sequences:
        # Default: token becomes single UNK
        if unk_token in p_token and p_token[unk_token] > 0:
            new_nll = -removed_count * unk_logp
        else:
            new_nll = float('inf')
    else:
        if not np.isclose(sum([tup[1] for tup in weighted_sequences]), 1.0, atol=1e-5):
            logger.error("Alternative seq weights not adding to 1")
        # Compute NLL from alternative sequences
        new_nll = 0.0
        
        for alt_sequence, weight in weighted_sequences:
            if weight <= 0:
                continue
                
            # Compute log probability of the entire sequence
            seq_log_prob = 0.0
            valid_sequence = True
            
            for token in alt_sequence:
                if token in p_token and p_token[token] > 0:
                    seq_log_prob += math.log(p_token[token])
                else:
                    valid_sequence = False
                    break
            
            if valid_sequence:
                # This sequence occurs with frequency: removed_count * weight
                seq_count = removed_count * weight
                new_nll += -seq_count * seq_log_prob
            else:
                # Invalid sequence - use UNK fallback
                if unk_token in p_token and p_token[unk_token] > 0:
                    # Sequence length matters for UNK tokens
                    seq_count = removed_count * weight
                    new_nll += -seq_count * len(alt_sequence) * unk_logp
                else:
                    new_nll = float('inf')
                    break
    # Loss is the change in NLL
    loss = new_nll - current_nll
    
    return loss


##############################################
# Old approach for computing alternatives; keeping for posterity
##############################################
def _compute_alternatives(corpus_files: List[str], token_probs: dict, unk_token: str, 
                          byte_level: bool = False, 
                          sample_proportion: float = 0.2,
                          num_processes: int = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    Optimized alternative computation.
    """
    if num_processes is None:
        num_processes = min(20, multiprocessing.cpu_count() - 1)
    
    all_alternatives = defaultdict(lambda: defaultdict(float))
    processed_tokens = set()
    
    # Track total tokens to process
    tokens_to_process = set(token_probs.keys()) - {unk_token}
    token_log_probs = {}
    for token, p in token_probs.items():
        token_log_probs[token] = math.log(p) if p > 0 else MIN_TOKEN_LOG_PROB
    
    for file_idx, file_path in enumerate(corpus_files):
        # Stop early if we've found alternatives for all tokens
        remaining_tokens = tokens_to_process - processed_tokens
        if not remaining_tokens:
            logger.info(f"All tokens processed, stopping early at file {file_idx+1}/{len(corpus_files)}")
            break
            
        logger.info(f"Computing alternatives from {file_path} ({file_idx+1}/{len(corpus_files)}) "
                   f"- {len(processed_tokens)}/{len(tokens_to_process)} tokens processed")
        
        # Efficient line sampling
        lines = _sample_lines_from_file(file_path, sample_proportion, seed=42 + file_idx)
        
        if not lines:
            continue
        
        # Chunk lines for parallel processing
        chunk_size = max(100, len(lines) // num_processes)
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        # Only pass unprocessed tokens to workers
        worker_args = [(chunk, token_log_probs, unk_token, byte_level, processed_tokens.copy()) 
                       for chunk in chunks]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_accumulate_alternatives_worker_weighted, worker_args)
        
        # Merge results
        new_tokens_found = 0
        for partial_alternatives in results:
            for token, alt_weights in partial_alternatives.items():
                if token not in processed_tokens:
                    new_tokens_found += 1
                    processed_tokens.add(token)
                for alt, weight in alt_weights.items():
                    all_alternatives[token][alt] += weight
        
        logger.info(f"  Found alternatives for {new_tokens_found} new tokens")
    
    # Convert to final format
    final_alternatives = {}
    for token, alt_weights in all_alternatives.items():
        total_weight = sum(alt_weights.values())
        if total_weight > 0:
            weighted_alts = [(alt, weight / total_weight) 
                           for alt, weight in alt_weights.items()]
            weighted_alts.sort(key=lambda x: -x[1])
            final_alternatives[token] = weighted_alts
        else:
            final_alternatives[token] = []
    
    logger.info(f"Computed alternatives for {len(final_alternatives)}/{len(tokens_to_process)} tokens")
    
    return final_alternatives


def _accumulate_alternatives_worker_weighted(args):
    """
    Worker that processes tokens and returns weighted alternatives.
    Optimized to segment each line only once.
    """
    chunk_lines, logp_lang, unk_token, byte_level, processed_tokens, pretok = args

    alternatives = defaultdict(lambda: defaultdict(float))
    max_token_len = MAX_BYTE_TOKEN_LEN if byte_level else MAX_CHAR_TOKEN_LEN

    # Pre-process all lines to get tokens that need alternatives
    lines_and_tokens = []

    for line in chunk_lines:
        if not line.strip():
            continue

        # Get segmentation for this line
        mapped_line = _apply_pretok_mapping_hf(line, pretok)
        if isinstance(mapped_line, list):
            best_seg = []
            for word in mapped_line:
                best_seg.extend(_segment_word_viterbi(word, logp_lang, unk_token, max_token_len))
            text_for_alternatives = line
        else:
            best_seg = _segment_word_viterbi(mapped_line, logp_lang, unk_token, max_token_len)
            text_for_alternatives = line  # Pass original line
            

        # Find tokens we haven't processed yet
        unique_tokens = set(best_seg) & set(logp_lang.keys()) - processed_tokens - {unk_token}
        if unique_tokens:
            lines_and_tokens.append((text_for_alternatives, unique_tokens))

    # Batch process lines with tokens that need alternatives
    for line, tokens_to_process in lines_and_tokens:
        # NEED TO RE-IMPLEMENT _find_alternatives_for_tokens_batch
        line_alternatives = _find_alternatives_for_tokens_batch(
            line, tokens_to_process, logp_lang, unk_token, byte_level
        )

        # Accumulate results
        for token, alt_weights in line_alternatives.items():
            for alt, weight in alt_weights.items():
                alternatives[token][alt] += weight

    return dict(alternatives)
