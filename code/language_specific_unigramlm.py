from __future__ import annotations


import logging
import os
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from tokenizers import Tokenizer

from code.em_unigram_trainer import EMUnigramTrainer
from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.utils import get_baseline_bytes
from code.tokenizer_formatting_utils import (
    _build_unigramlm_hf_tokenizer_from_lprobs,
    _load_vocab_from_hf_tokenizer,
    _write_sp_seed_vocab_file,
    write_hf_bytelevel_corpus,
)
from code.constants import (
    SPECIAL_TOKENS,
    MIN_TOKEN_LOG_PROB,
    SP_DEFAULT_ARGS,
    NUM_FINETUNE_EM_ITERATIONS,
)

logger = logging.getLogger(__name__)


DEFAULT_LANGTOKENIZER_NAME = (
    "langspec_{em_mode}_{lang_code}.tokenizer.json"  # <- replaces .npy file
)
DEFAULT_BASE_NAME = "langspec_base_tokenizer.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _extract_log_scores(tok: Tokenizer) -> Dict[str, float]:
    """
    Return dict {token -> log_score} from a HF Unigram tokenizer.
    """
    state = tok.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    vocab = attributes["vocab"]
    if isinstance(vocab, dict):
        # This is the format the BPE vocab comes in
        return {tok: MIN_TOKEN_LOG_PROB for tok, _ in vocab.items()}
    return {tok: score for tok, score in vocab}


def _score_encoding(enc, scores: Dict[str, float], tokenizer: Tokenizer) -> float:
    """Score an Encoding by summing log-probs looked up via id_to_token.

    Walks `enc.ids` rather than `enc.tokens` because HF Unigram emits UNK
    with `enc.tokens[i]` = the source character (not the canonical UNK
    vocab entry), which would silently miss the score-dict lookup and push
    the routing toward MIN_TOKEN_LOG_PROB. Routing that string through
    `tokenizer.id_to_token(i)` canonicalizes every ID to its vocab entry —
    so the UNK token's real score is found, whatever string the tokenizer
    actually uses for UNK (Gemma: '<unk>', Qwen3: '<|endoftext|>', etc.).
    """
    total = 0.0
    for token_id in enc.ids:
        canonical = tokenizer.id_to_token(token_id)
        if canonical is None:
            total += MIN_TOKEN_LOG_PROB
        else:
            total += scores.get(canonical, MIN_TOKEN_LOG_PROB)
    return total


def _build_unigramlm_hf_tokenizer_with_new_lprobs(
    base_tokenizer, new_log_probs: Dict[str, float], unk_token: str
) -> Tokenizer:
    """
    Turn a plain probability dictionary into a HF Unigram tokenizer.
    """
    vocab = _load_vocab_from_hf_tokenizer(base_tokenizer)
    tuples = [(t, max(new_log_probs[t], MIN_TOKEN_LOG_PROB)) for t in vocab]

    # Find unk_id within the actual tuples list (not base_tokenizer.get_vocab(),
    # which includes added tokens that may not appear in the trainable vocab).
    unk_id = None
    for i, (t, _) in enumerate(tuples):
        if t == unk_token:
            unk_id = i
            break
    if unk_id is None:
        # unk_token is an added token not in the trainable vocab (e.g. Qwen3's
        # <|im_end|>).  Use index 0 as the Unigram-internal unk_id fallback.
        # The actual unk behaviour is handled by the added_tokens section,
        # which is preserved by _build_unigramlm_hf_tokenizer_from_lprobs
        # via the base_json rebuild.  We must NOT append the token here
        # because that would change the vocab size and break validation.
        logger.warning(
            f"unk_token {unk_token!r} not in trainable vocab; "
            f"using unk_id=0 as Unigram-internal fallback"
        )
        unk_id = 0

    tok = _build_unigramlm_hf_tokenizer_from_lprobs(
        tuples, unk_id, base_tokenizer=base_tokenizer
    )
    if not tok.pre_tokenizer:
        logger.warning("No pretokenizer set for new HF tokenizer")

    return tok


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------
class LanguageSpecificUnigramLMTokenizer(StandardUnigramLMTokenizer):
    def __init__(
        self,
        vocab_size: int = 8_000,
        unk_token: str = SPECIAL_TOKENS["unk_token"],
        special_tokens=None,
        reestimation_em_mode: str = "soft",
        num_iterations: int = NUM_FINETUNE_EM_ITERATIONS,
        byte_level: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            em_mode=reestimation_em_mode,
            num_iterations=num_iterations,
            byte_level=byte_level,
        )

        self.reestimation_em_mode = reestimation_em_mode
        self.num_reestimation_iterations = num_iterations
        # lang_code -> {"tokenizer": HF Tokenizer, "scores": {tok->log_p}, "path": str}
        self.per_lang_tok: Dict[str, dict] = {}

    def load(self, base_path, language_paths=None):
        """
        Load a previously saved base tokenizer plus any saved EM distributions for languages.

        Args:
            base_path: path to base tokenizer .json
            language_paths: dict { lang_code: path_to_em_probs.npy }
        Returns:
            { 'base': self.base_tokenizer, 'languages': [lang_codes] }
        """
        # Load the base tokenizer using the parent class method
        super().load(base_path)
        base_vocab_order = _load_vocab_from_hf_tokenizer(self.base_tokenizer)

        # Initialize language info
        self.per_lang_tok = {}

        # Load language-specific probability distributions
        if language_paths:
            for lang_code, lang_tok_path in language_paths.items():
                tok = Tokenizer.from_file(lang_tok_path)
                vocab_order_per_lang = _load_vocab_from_hf_tokenizer(tok)
                assert vocab_order_per_lang == base_vocab_order
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": lang_tok_path,
                }
                logger.info(
                    f"Loaded per-lang tokenizer for {lang_code} from {lang_tok_path}"
                )

    def verify_byte_level_vocabulary(self, vocab_dict: Dict[str, int]) -> bool:
        """
        Verify that a vocabulary contains all necessary byte-level tokens.
        """
        # Check if vocabulary contains byte tokens
        # Depending on your byte encoding scheme, this might be:
        # - Raw bytes: '\x00' through '\xff'
        # - GPT-2 style: 'Ġ' prefix for spaces, special byte encodings
        # - Custom byte representation

        # Example for GPT-2 style byte-level BPE
        tokens = set(vocab_dict.keys())
        byte_tokens = get_baseline_bytes()[0]
        not_present = [b for b in byte_tokens if b not in tokens]
        if len(not_present) > 0:
            logger.warning(
                f"Byte-level vocabulary may be incomplete: only {len(byte_tokens) - len(not_present)} byte tokens found (showing top 10): {not_present[:10]}"
            )
            return False

        return True

    def train_with_sentencepiece_direct(
        self, corpus_file: str, vocab_dict: Dict[str, int], num_iterations: int = 10
    ) -> Dict[str, float]:
        import tempfile
        import subprocess
        import shutil
        import sentencepiece as spm

        # Check if spm_train is available
        if not shutil.which("spm_train"):
            raise RuntimeError(
                "spm_train executable not found. Make sure sentencepiece is installed and in your PATH."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "model")
            vocab_path = os.path.join(tmpdir, "vocab.tsv")
            # 1. Prepare the vocabulary file
            base_vocab = _load_vocab_from_hf_tokenizer(
                self.base_tokenizer
            )
            excluded = set(self.special_tokens) | {self.unk_token}  # exclude <unk>, <s>, </s>, <pad>, etc.
            num_pieces = sum(1 for t in base_vocab if t not in excluded)
 
            initial_log_score = np.log(1.0 / num_pieces) if num_pieces > 0 else -10.0

            token_count = _write_sp_seed_vocab_file(
                [(t, initial_log_score) for t in base_vocab],
                vocab_path,
                self.unk_token,
            )

            hf_processed_text_path = os.path.join(tmpdir, "hf_text.txt")
            normalizer = self.base_tokenizer.normalizer if self.base_tokenizer else None
            write_hf_bytelevel_corpus(
                corpus_file, hf_processed_text_path, self._get_pretokenizer(),
                normalizer=normalizer,
            )

            # The exact vocab size must account for meta pieces
            # <unk>, <s> and </s>
            expected_vocab_size = token_count + 3

            # 2. Construct the spm_train command
            command = [
                "spm_train",
                f"--input={hf_processed_text_path}",
                f"--model_prefix={out_prefix}",
                f"--seed_sentencepieces_file={vocab_path}",
                f"--vocab_size={expected_vocab_size}",
                # -----------------------
                f"--num_sub_iterations={max(1, min(int(num_iterations), 20))}",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count() - 1)}",
            ] + SP_DEFAULT_ARGS

            # 3. Execute the command
            # check=False so we can surface stdout+stderr on failure (the default
            # CalledProcessError.__str__ only includes the command string, which
            # strips every actionable piece of information).
            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            output = subprocess.run(
                command, check=False, capture_output=True, text=True, encoding="utf-8"
            )
            if output.returncode != 0:
                raise RuntimeError(
                    f"spm_train failed with exit code {output.returncode}.\n"
                    f"Command: {' '.join(command)}\n"
                    f"--- stderr ---\n{output.stderr}\n"
                    f"--- stdout ---\n{output.stdout}"
                )
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
                if p_hf in set(SPECIAL_TOKENS.values()) | {self.unk_token}:
                    try:
                        logp = float(vocab_dict[p_hf])
                    except KeyError:
                        logging.warning(
                            f"Special token {p_hf} not found in vocab_dict; skipping"
                        )
                        continue
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
                logger.warning(
                    f"Tokens {missing_tokens} not in sentencepiece model; defaulting to base "
                    f"tokenizer log-prob {vocab_dict[t]:.5f}"
                )

            # -------- Interface-2 sanity guard (T2) --------
            # If too many SP pieces don't map into the HF base vocab, the corpus
            # written to disk was in a format SP didn't read as HF-byte-level,
            # and `p_hf = p` (line 259) silently fails for most tokens. Every
            # affected base token then falls back to `vocab_dict[t]` and the
            # resulting langspec is numerically ≈ base.
            sp_meta = set(SPECIAL_TOKENS.values()) | {self.unk_token}
            truly_unknown = additional_tokens - sp_meta
            n_truly_unknown = len(truly_unknown)
            n_sp_pieces = sp.get_piece_size()
            n_base_vocab = len(vocab_dict)
            n_fallback = len(missing_tokens)
            pct_unknown = n_truly_unknown / max(n_sp_pieces, 1)
            pct_fallback = n_fallback / max(n_base_vocab, 1)
            logger.info(
                f"SP->HF handoff: sp_pieces={n_sp_pieces} base_vocab={n_base_vocab} "
                f"matched={n_base_vocab - n_fallback} fallback={n_fallback} "
                f"unknown_from_sp={n_truly_unknown} (pct_unknown={pct_unknown:.4f}, pct_fallback={pct_fallback:.4f})"
            )
            if pct_unknown > 0.01:
                raise RuntimeError(
                    f"SP produced {pct_unknown:.1%} pieces outside HF base vocab "
                    f"({n_truly_unknown}/{n_sp_pieces}). Likely cause: corpus "
                    f"format mismatch between write_hf_bytelevel_corpus and what "
                    f"spm_train read. First 20 unknown: {list(truly_unknown)[:20]}"
                )
            if pct_fallback > 0.50:
                raise RuntimeError(
                    f"{pct_fallback:.1%} of base vocab fell back to base score "
                    f"({n_fallback}/{n_base_vocab}). SP training didn't see enough "
                    f"of the language, or the SP<->HF piece format diverged."
                )

            # 3) Normalize in log-space
            token_logps = self._log_normalize(token_logps)

            return token_logps  # normalized log-probabilities

    def train(
        self,
        corpus_info: Dict[str, str],
        base_tokenizer_path: str,
        output_dir: str = "",
        tokenizer_path_format: str | None = None,
        load_base_if_exists: bool = True,
        load_tokenizers_if_exists: bool = True,
        use_sentencepiece: bool = True,
        max_workers=None,
    ):
        """
        Build / load the *shared* base tokenizer via the parent class,
        then obtain a HF Unigram tokenizer for every language.

        `corpus_info`  …… {lang_code: path_to_train_file}
        """
        output_dir = Path(output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        if load_base_if_exists and os.path.isfile(base_tokenizer_path):
            logger.info(f"Loading specified base tokenizer from {base_tokenizer_path}")
            super().load(base_tokenizer_path)
            logger.warning(
                f"Will use byte-level settings from {base_tokenizer_path} regardless of user input: byte-level={self.byte_level}"
            )

        training_new_base_tokenizer = not self.base_tokenizer

        # Gather train files for all languages
        all_train_files = []
        for lang_code, f in corpus_info.items():
            all_train_files.append(f)

        if not all_train_files:
            logger.error("No training files found for any language in corpus_info.")
            return None

        # Train base tokenizer if needed (using parent class method)
        if training_new_base_tokenizer:
            logger.info("Training base Unigram tokenizer for shared vocab.")
            # Use parent class's train method for the base tokenizer
            self.base_tokenizer = super().train(
                all_train_files, output_path=base_tokenizer_path, load_if_exists=False
            )
        else:
            logger.info("Using previously loaded base tokenizer.")

        if self.byte_level:
            if not self.verify_byte_level_vocabulary(self.base_tokenizer.get_vocab()):
                logger.warning(
                    "Base vocabulary may not contain all necessary byte tokens for byte-level tokenization"
                )

        futures_by_lang = {}
        if max_workers is None:
            # at most 20
            max_workers = min(
                min(20, multiprocessing.cpu_count() - 1), len(corpus_info)
            )
        if use_sentencepiece and max_workers > 1:
            executor = ThreadPoolExecutor(max_workers=max_workers)
        # 2) one tokenizer per language
        for lang_code, train_file in corpus_info.items():
            # where to store / look for the tokenizer
            tk_path: Path = (
                Path(tokenizer_path_format.format(lang_code=lang_code))
                if tokenizer_path_format is not None
                else output_dir
                / DEFAULT_LANGTOKENIZER_NAME.format(
                    em_mode=self.reestimation_em_mode, lang_code=lang_code
                )
            )
            if load_tokenizers_if_exists and tk_path.is_file():
                tok = Tokenizer.from_file(str(tk_path))
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": str(tk_path),
                }
                logger.info(f"Loaded tokenizer for {lang_code} from {tk_path}")
                continue

            if use_sentencepiece and max_workers > 1:
                logging.info(f"Submitting job for {lang_code}")
                future = executor.submit(
                    self.per_lang_estimation,
                    lang_code=lang_code,
                    train_file=train_file,
                    tk_path=tk_path,
                    use_sentencepiece=use_sentencepiece,
                )

                futures_by_lang[lang_code] = future

            else:
                tok, tk_path = self.per_lang_estimation(
                    lang_code, train_file, tk_path, use_sentencepiece
                )

                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": str(tk_path),
                }
        if futures_by_lang:
            # Wait for all language-specific training to complete and collect failures
            failures = []
            for lang, future in futures_by_lang.items():
                try:
                    tok, tk_path = future.result()
                except Exception as e:
                    error_msg = f"Language-specific training failed for {lang}: {e}"
                    logging.error(error_msg)
                    failures.append((lang, e))
                    continue
                self.per_lang_tok[lang] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": str(tk_path),
                }

            executor.shutdown(wait=True)
            if failures:
                failure_summary = "\n".join(
                    [f"  - {lang}: {str(e)}" for lang, e in failures]
                )
                error_message = f"Failed to train {len(failures)} language-specific tokenizer(s):\n{failure_summary}"
                logging.error(error_message)
                raise RuntimeError(error_message)

        return {
            "base_path": base_tokenizer_path,
            "language_paths": {k: v["path"] for k, v in self.per_lang_tok.items()},
        }

    def per_lang_estimation(self, lang_code, train_file, tk_path, use_sentencepiece):
        vocab_score_dict = _extract_log_scores(self.base_tokenizer)

        if use_sentencepiece:
            # Use fast sentencepiece EM implementation
            token_log_probs = self.train_with_sentencepiece_direct(
                corpus_file=train_file,
                vocab_dict=vocab_score_dict,
                num_iterations=self.num_reestimation_iterations,
            )
        else:
            # ----------  EM training on the language corpus  ----------
            em_trainer = EMUnigramTrainer(
                vocab=vocab_score_dict,
                unk_token=self.unk_token,
                special_tokens=self.special_tokens,
                em_mode=self.reestimation_em_mode,
                max_iterations=self.num_reestimation_iterations,
                pretokenizer=self._get_pretokenizer(),
                byte_level=self.byte_level,
            )
            token_probs = em_trainer.train(train_file)  # token -> prob
            token_log_probs = {tk: np.log(p) for tk, p in token_probs.items()}

        # ---------------------------------------------------------------------------
        # DEBUGGERS
        # ---------------------------------------------------------------------------

        # common_keys = token_log_probs1.keys() & token_log_probs.keys()
        # diffs = {k: abs(token_log_probs1[k] - token_log_probs[k]) for k in common_keys}
        # top10 = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:20]
        # for k, v in top10:
        #     print(f"{k}: {token_log_probs1[k]} vs {token_log_probs[k]} (diff={v})")
        # x = 0
        # x += abs(np.exp(token_log_probs1[k]) - np.exp(token_log_probs[k]))
        # print("TVD: ", x)

        # ----------  HF tokenizer from these probabilities  ----------
        logger.info(f"Internal unk token is {self.unk_token}")
        tok = _build_unigramlm_hf_tokenizer_with_new_lprobs(
            self.base_tokenizer, token_log_probs, self.unk_token
        )
        tok.save(str(tk_path))

        # -------- T4c: in-memory ↔ reloaded equivalence --------
        # Serialize/deserialize must preserve encoding behavior AND the
        # _extract_log_scores output, because the non-standard inference path
        # (best_language_encode) looks up scores by token string against
        # `_extract_log_scores(tok)`. Any save/load drift silently breaks
        # routing. Probe with a handful of lines from the training corpus.
        _probe_lines = []
        with open(train_file, "r", encoding="utf-8", errors="strict") as _f:
            for _i, _l in enumerate(_f):
                if _i >= 10:
                    break
                _l = _l.rstrip("\n")
                if _l:
                    _probe_lines.append(_l)
        if not _probe_lines:
            raise RuntimeError(
                f"T4c: no probe lines read from {train_file} — cannot verify "
                f"save/load equivalence for {tk_path}. The training corpus "
                f"must not be empty."
            )
        _reloaded = Tokenizer.from_file(str(tk_path))
        for _raw in _probe_lines:
            _a = tok.encode(_raw).ids
            _b = _reloaded.encode(_raw).ids
            if _a != _b:
                raise RuntimeError(
                    f"T4c: save/load drift for {tk_path} on language "
                    f"{lang_code!r}. In-memory IDs differ from reloaded IDs. "
                    f"first 20 in-mem: {_a[:20]} / reloaded: {_b[:20]}"
                )
        _s_mem = _extract_log_scores(tok)
        _s_rel = _extract_log_scores(_reloaded)
        if _s_mem != _s_rel:
            _keys_mem = set(_s_mem.keys())
            _keys_rel = set(_s_rel.keys())
            raise RuntimeError(
                f"T4c: _extract_log_scores disagrees between in-memory and "
                f"reloaded {tk_path}. "
                f"in-mem-only keys: {list(_keys_mem - _keys_rel)[:5]} "
                f"reloaded-only: {list(_keys_rel - _keys_mem)[:5]}"
            )
        logger.info(f"T4c OK: {tk_path} survives save/load on {len(_probe_lines)} probes")
        logger.info(f"Saved tokenizer with metadata for {lang_code} to {tk_path}")
        return tok, tk_path

    # ---------------------------------------------------------------- encode / decode helpers
    def encode_lang(self, text: str, lang_code: str):
        if lang_code not in self.per_lang_tok:
            raise ValueError(f"No tokenizer for language '{lang_code}'.")
        return self.per_lang_tok[lang_code]["tokenizer"].encode(text)

    def decode_lang(self, ids: List[int], lang_code: str):
        if lang_code not in self.per_lang_tok:
            raise ValueError(f"No tokenizer for language '{lang_code}'.")
        return self.per_lang_tok[lang_code]["tokenizer"].decode(ids)

    def best_language_encode(self, text: str):
        """
        Segment `text` with *each* language model and return the one that
        yields the highest log-probability.

        Returns
        -------
        (best_lang_code, tokens_as_list_of_str)
        """
        best_lang, best_tokens, best_logp = None, [], float("-inf")

        for lang_code, info in self.per_lang_tok.items():
            # add_special_tokens=False to match training-time tokenization in
            # `las_experiment.finetune.TokenizedTextDataset` (which encodes with
            # add_special_tokens=False). Otherwise BOS/EOS from any post-processor
            # would inflate token counts and skew BpB / routing scores.
            enc = info["tokenizer"].encode(text, add_special_tokens=False)
            logp = _score_encoding(enc, info["scores"], info["tokenizer"])

            if logp > best_logp:
                best_lang, best_tokens, best_logp = lang_code, enc.tokens, logp

        return best_lang, best_tokens

    def best_language_encode_batch(self, texts):
        """
        For each input text return
            best_lang, Encoding           # Encoding from HF tokenizers lib
        This avoids re-encoding after we have already picked the language.
        """
        if not self.per_lang_tok:
            raise ValueError("No per-language tokenizers loaded.")

        # First pass: encode each text with EVERY language and keep the log-prob
        best_langs = [None] * len(texts)
        best_scores = [float("-inf")] * len(texts)
        best_encs = [None] * len(texts)

        for lang_code, info in self.per_lang_tok.items():
            # add_special_tokens=False to match training-time tokenization (see
            # `best_language_encode` above for rationale).
            encs = info["tokenizer"].encode_batch(texts, add_special_tokens=False)
            for i, enc in enumerate(encs):
                score = _score_encoding(enc, info["scores"], info["tokenizer"])
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_langs[i] = lang_code
                    best_encs[i] = enc

        return best_langs, best_encs


def train_language_specific_tokenizer(
    corpus_info,
    vocab_size=8000,
    reestimation_em_mode="soft",
    output_dir="",
    byte_level=True,
    base_tokenizer_path=None,
    tokenizer_path_format=None,
    load_base_if_exists=True,
    load_tokenizers_if_exists=False,
):

    tok = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        reestimation_em_mode=reestimation_em_mode,
        byte_level=byte_level,
    )
    if not base_tokenizer_path:
        base_tokenizer_path = os.path.join(output_dir, DEFAULT_BASE_NAME)

    if not tokenizer_path_format:
        tokenizer_path_format = os.path.join(output_dir, DEFAULT_LANGTOKENIZER_NAME)

    tokenizer_paths = tok.train(
        corpus_info,
        base_tokenizer_path=base_tokenizer_path,
        tokenizer_path_format=tokenizer_path_format,
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists,
    )
    if not tokenizer_paths:
        logger.error("Failed to train or load language-specific tokenizer.")
        return None

    return tokenizer_paths
