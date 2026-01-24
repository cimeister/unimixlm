from __future__ import annotations


import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import multiprocessing

from tokenizers import Tokenizer

from code.em_unigram_trainer import EMUnigramTrainer
from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.utils import _create_base_metadata, get_baseline_bytes
from code.tokenizer_formatting_utils import (
    _build_unigramlm_hf_tokenizer_from_lprobs, _get_hf_unigram_tokenizer_vocab, _write_sp_seed_vocab_file,
    write_hf_bytelevel_corpus
)
from code.constants import SPECIAL_TOKENS, MIN_TOKEN_LOG_PROB, SP_DEFAULT_ARGS

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
    vocab_tuples = _get_hf_unigram_tokenizer_vocab(tok)
    return {tok: score for tok, score in vocab_tuples }


def _build_unigramlm_hf_tokenizer_with_new_lprobs(
    base_tokenizer,
    new_log_probs: Dict[str, float],
    unk_token: str
) -> Tokenizer:
    """
    Turn a plain probability dictionary into a HF Unigram tokenizer.
    """
    vocab_tuples = _get_hf_unigram_tokenizer_vocab(base_tokenizer)
    tuples = [
        (t, max(new_log_probs[t], MIN_TOKEN_LOG_PROB)) for t, _ in vocab_tuples
    ]

    unk_id = base_tokenizer.get_vocab()[unk_token]
    tok = _build_unigramlm_hf_tokenizer_from_lprobs(tuples, unk_id, base_tokenizer=base_tokenizer)
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
        num_iterations: int = 10,
        byte_level: bool = False,
        whitespace_token_boundaries: bool = False
    ):
        super().__init__(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            em_mode=reestimation_em_mode,
            num_iterations=num_iterations,
            byte_level=byte_level,
            whitespace_token_boundaries=whitespace_token_boundaries
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
        vocab_tuples = _get_hf_unigram_tokenizer_vocab(self.base_tokenizer)
        base_vocab_order = [i for i, j in vocab_tuples]
        
        # Initialize language info
        self.per_lang_tok = {}

        # Load language-specific probability distributions
        if language_paths:
            for lang_code, lang_tok_path in language_paths.items():
                tok = Tokenizer.from_file(lang_tok_path)
                vocab_order_per_lang = _get_hf_unigram_tokenizer_vocab(tok)
                assert [i for i, j in vocab_order_per_lang] == base_vocab_order
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": _extract_log_scores(tok),
                    "path": lang_tok_path,
                } 
                logger.info(f"Loaded per-lang tokenizer for {lang_code} from {lang_tok_path}")
    
    
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
            logger.warning(f"Byte-level vocabulary may be incomplete: only {len(byte_tokens) - len(not_present)} byte tokens found (showing top 10): {not_present[:10]}")
            return False
        
        return True


    def train_with_sentencepiece_direct(
            self,
            corpus_file: str,
            vocab_dict: Dict[str, int],
            num_iterations: int = 10
        ) -> Dict[str, float]: 
            import tempfile
            import subprocess
            import shutil
            import sentencepiece as spm

            # Check if spm_train is available
            if not shutil.which("spm_train"):
                raise RuntimeError("spm_train executable not found. Make sure sentencepiece is installed and in your PATH.")

            with tempfile.TemporaryDirectory() as tmpdir:
                out_prefix = os.path.join(tmpdir, "model")
                vocab_path = os.path.join(tmpdir, "vocab.tsv")
                # 1. Prepare the vocabulary file
                base_vocab_with_scores = _get_hf_unigram_tokenizer_vocab(self.base_tokenizer)
                num_pieces = len([t for t, _ in base_vocab_with_scores if t not in set(self.special_tokens) | {self.unk_token}]) # exclude <unk>, <s>, </s>, <pad>, etc.
                initial_log_score = np.log(1.0 / num_pieces) if num_pieces > 0 else -10.0 

                token_count = _write_sp_seed_vocab_file([(t, initial_log_score) for t, _ in base_vocab_with_scores], vocab_path, self.unk_token)

                hf_processed_text_path = os.path.join(tmpdir, "hf_text.txt")
                write_hf_bytelevel_corpus(corpus_file, hf_processed_text_path, self._get_pretokenizer())

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
                    
                    f"--num_threads={min(20, multiprocessing.cpu_count()-1)}"
                ] + SP_DEFAULT_ARGS
                # if self.whitespace_token_boundaries:
                #     command.append("--split_by_whitespace=true")

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
                    logger.warning(f"Tokens {missing_tokens} not in sentencepiece model; defaulting to base "
                                   f"tokenizer log-prob {vocab_dict[t]:.5f}")
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
        use_sentencepiece: bool = True
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
            logger.warning(f"Will use byte-level settings from {base_tokenizer_path} regardless of user input: byte-level={self.byte_level}")
        
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
            self.base_tokenizer = super().train(all_train_files, output_path=base_tokenizer_path, load_if_exists=False)
        else:
            logger.info("Using previously loaded base tokenizer.")

        if self.byte_level:
            if not self.verify_byte_level_vocabulary(self.base_tokenizer.get_vocab()):
                logger.warning("Base vocabulary may not contain all necessary byte tokens for byte-level tokenization")

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
            if load_tokenizers_if_exists and tk_path.is_file() and False:
                tok = Tokenizer.from_file(str(tk_path))
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores":  _extract_log_scores(tok),
                    "path": str(tk_path),
                }
                logger.info(f"Loaded tokenizer for {lang_code} from {tk_path}")
                continue

            vocab_score_dict = _extract_log_scores(self.base_tokenizer)

            if use_sentencepiece:
                # Use fast sentencepiece EM implementation
                token_log_probs = self.train_with_sentencepiece_direct(
                        corpus_file=train_file,
                        vocab_dict=vocab_score_dict,
                        num_iterations=self.num_reestimation_iterations
                )
            else:
            # ----------  EM training on the language corpus  ----------
                em_trainer = EMUnigramTrainer(
                    vocab=vocab_score_dict,
                    unk_token=self.unk_token,
                    special_tokens=self.special_tokens,
                    em_mode=self.reestimation_em_mode,
                    max_iterations=self.num_reestimation_iterations,
                    whitespace_token_boundaries=self.whitespace_token_boundaries,
                    pretokenizer=self._get_pretokenizer(),
                    byte_level=self.byte_level
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
            tok = _build_unigramlm_hf_tokenizer_with_new_lprobs(self.base_tokenizer, token_log_probs, self.unk_token)
            tok.save(str(tk_path))
            
            # Add metadata to language-specific tokenizer
            lang_metadata = _create_base_metadata("LanguageSpecificUnigramLMTokenizer", "langspec")
            lang_metadata["training_config"] = {
                "vocab_size": self.vocab_size,
                "reestimation_em_mode": self.reestimation_em_mode,
                "byte_level": self.byte_level,
                "language_code": lang_code,
                "base_tokenizer_path": base_tokenizer_path,
                "num_reestimation_iterations": self.num_reestimation_iterations
            }
            lang_metadata["corpus_info"] = {
                "training_file": train_file,
                "language_code": lang_code
            }
            
            # Try to extract base tokenizer metadata
            try:
                with open(base_tokenizer_path, 'r') as f:
                    base_data = json.load(f)
                    if "tokenizer_metadata" in base_data:
                        lang_metadata["base_tokenizer_metadata"] = base_data["tokenizer_metadata"]["training_config"]
            except:
                lang_metadata["base_tokenizer_metadata"] = {"em_mode": "unknown", "vocab_size": "unknown"}
            
            self._save_tokenizer_metadata_file(str(tk_path))

            self.per_lang_tok[lang_code] = {
                "tokenizer": tok,
                "scores": _extract_log_scores(tok),
                "path": str(tk_path),
            }
            logger.info(f"Saved tokenizer with metadata for {lang_code} to {tk_path}")
        return {
            "base_path": base_tokenizer_path,
            "language_paths": {k: v['path'] for k, v in self.per_lang_tok.items()}
        }

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
            enc = info["tokenizer"].encode(text)
            toks = enc.tokens
            logp = sum(info["scores"].get(t, MIN_TOKEN_LOG_PROB) for t in toks)

            if logp > best_logp:
                best_lang, best_tokens, best_logp = lang_code, toks, logp

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
        best_langs  = [None] * len(texts)
        best_scores = [float("-inf")] * len(texts)
        best_encs   = [None] * len(texts)

        for lang_code, info in self.per_lang_tok.items():
            encs = info["tokenizer"].encode_batch(texts)
            for i, enc in enumerate(encs):
                score = sum(info["scores"].get(t, MIN_TOKEN_LOG_PROB) for t in enc.tokens)
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_langs[i]  = lang_code
                    best_encs[i]   = enc

        return best_langs, best_encs


def train_language_specific_tokenizer(corpus_info, vocab_size=8000,
                                      reestimation_em_mode = 'soft',
                                      output_dir="",
                                      byte_level=True,
                                      base_tokenizer_path=None,
                                      tokenizer_path_format=None,
                                      load_base_if_exists=True,
                                      load_tokenizers_if_exists=False):
   
    tok = LanguageSpecificUnigramLMTokenizer(vocab_size=vocab_size, reestimation_em_mode=reestimation_em_mode, byte_level=byte_level)
    if not base_tokenizer_path:
        base_tokenizer_path = os.path.join(output_dir, DEFAULT_BASE_NAME)
        
    if not tokenizer_path_format:
        tokenizer_path_format = os.path.join(output_dir, DEFAULT_LANGTOKENIZER_NAME)

    tokenizer_paths = tok.train(
        corpus_info,
        base_tokenizer_path=base_tokenizer_path,
        tokenizer_path_format=tokenizer_path_format,
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists
    )
    if not tokenizer_paths:
        logger.error("Failed to train or load language-specific tokenizer.")
        return None

    return tokenizer_paths
