#!/usr/bin/env python3
"""
training.py
End-to-end pipeline for training a family of language-specific LAS UnigramLM tokenizers.

Architecture:
1. Load a model tokenizer (e.g., BPE from OLMo-3) as initialization seed
2. Auto-detect byte_level setting from the model tokenizer's pretokenizer
3. Auto-detect unk_token from the model tokenizer (falls back to eos_token)
4. Train ONE shared base Unigram tokenizer on combined multilingual corpora
5. For each language, re-estimate token probabilities from the shared base (parallel)
6. Validate all tokenizers (pre/post-tokenizer consistency, token order preservation)
7. Push all tokenizers to a single HuggingFace Hub repository

The base tokenizer defines the shared vocabulary, and language-specific tokenizers
adapt the token probabilities to each language via EM. Settings are automatically
inherited from the model tokenizer. Training is parallelized via ThreadPoolExecutor.
All tokenizers are validated and pushed to a single flat-structure HF repository.
"""

import itertools
import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import List

from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE as BPEModel
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok, Split, Sequence
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.standard_unigramlm import StandardUnigramLMTokenizer
from las_experiment.tokenizer_training.validation import validate_tokenizers
from las_experiment.tokenizer_training.utils import (
    push_to_hf,
)

# Mistral Nemo / Tekken pretokenization regex.
# Case-aware, combining-mark-aware, handles contractions, numbers, whitespace.
TEKKEN_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}][\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_tekken_pretokenizer():
    """Build a Mistral Nemo / Tekken style pretokenizer.

    Byte-level encoding with a Unicode-aware regex that handles:
    - Case-sensitive word boundaries (camelCase splitting)
    - Combining marks / diacritics
    - English contractions
    - Number grouping (1-3 digits)
    - Whitespace/newline preservation
    """
    from tokenizers import Regex

    return Sequence([
        Split(pattern=Regex(TEKKEN_REGEX), behavior="isolated", invert=False),
        ByteLevelPreTok(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def train_bpe_tokenizer(
    corpus_files: List[str],
    output_path: str,
    vocab_size: int = 100000,
    special_tokens: List[str] = None,
) -> None:
    """Train a BPE tokenizer with Tekken-style pretokenization.

    This is the BPE baseline for the from-scratch experiment. Uses the same
    pretokenizer as the Unigram tokenizers so the comparison is purely about
    the subword algorithm.
    """
    if special_tokens is None:
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]

    tok = Tokenizer(BPEModel(unk_token="<unk>"))
    tok.pre_tokenizer = build_tekken_pretokenizer()
    tok.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevelPreTok.alphabet(),
    )

    tok.train(corpus_files, trainer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)
    logging.info(f"BPE tokenizer saved to {output_path} (vocab_size={tok.get_vocab_size()})")


def _generate_pathpiece_tokenizer(base_path: str, output_path: str) -> None:
    """Create a PathPiece tokenizer by setting all Unigram scores to -1.

    This makes the Viterbi decoder prefer the fewest possible tokens
    (longest match / shortest segmentation), since each token contributes
    -1 to the total score regardless of identity.
    """
    with open(base_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data["model"]["vocab"]
    tok_data["model"]["vocab"] = [[piece, -1.0] for piece, _ in vocab]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False)

    logging.info(
        f"PathPiece tokenizer: {len(vocab)} pieces, all scores set to -1.0"
    )


def train_base_tokenizer(
    *,
    corpus_info: dict,
    model_tokenizer_path: str,
    base_output_path: Path,
    em_mode: str = "soft",
    unk_token: str = None,
) -> None:
    """
    Train a single shared base Unigram tokenizer from all language corpora.

    This creates the shared vocabulary that will be used for all language-specific
    re-estimations. The base tokenizer is initialized from the model tokenizer
    and trained on the combined corpora of all languages.

    Always trains from scratch - does not load existing tokenizers.
    """

    logging.info(f"Training base Unigram tokenizer on {len(corpus_info)} languages")
    train_files = []
    for _, train_file in corpus_info.items():
        train_files.append(train_file)

    if not train_files:
        logging.error("No training files found")
        return None

    # Create and train the tokenizer

    base_tokenizer = StandardUnigramLMTokenizer(
        em_mode=em_mode, from_base_tokenizer=model_tokenizer_path, unk_token=unk_token
    )
    base_tokenizer.train(train_files, base_output_path, corpus_info=corpus_info)


# ---------------------- Corpus script filtering ----------------------

# Code languages: accept all lines (code is inherently mixed-script)
_CODE_LANGS = {"python", "javascript", "java", "go", "c", "cpp", "rust",
               "typescript", "ruby", "php", "shell", "bash"}


# ISO 15924 script code -> list of (lo, hi) Unicode codepoint ranges (inclusive)
# that belong to that script. Punctuation / symbols / separators / numbers are
# handled separately as universally script-neutral via the Unicode General
# Category check in `_is_in_target_script`. ASCII letters (A-Z, a-z) are
# deliberately only accepted for Latn — otherwise a mostly-English line in a
# Hangul corpus would pass the filter.
SCRIPT_RANGES = {
    "Latn": [
        (0x0000, 0x007F),  # Basic Latin (ASCII letters + digits + punctuation)
        (0x0080, 0x00FF),  # Latin-1 Supplement (ñ, ü, ß, ...)
        (0x0100, 0x017F),  # Latin Extended-A (ğ, ı, ş, ő, ű, ...)
        (0x0180, 0x024F),  # Latin Extended-B
        (0x0300, 0x036F),  # Combining Diacritical Marks
        (0x1E00, 0x1EFF),  # Latin Extended Additional
        (0x2000, 0x206F),  # General Punctuation
        (0x20A0, 0x20CF),  # Currency Symbols
        (0x2100, 0x214F),  # Letterlike Symbols
        (0x2C60, 0x2C7F),  # Latin Extended-C
        (0xA720, 0xA7FF),  # Latin Extended-D
    ],
    "Hang": [
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0xA960, 0xA97F),  # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
    ],
    "Arab": [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0870, 0x089F),  # Arabic Extended-B
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ],
    # Added for F15 (hin_Deva).
    "Deva": [
        (0x0900, 0x097F),  # Devanagari
        (0xA8E0, 0xA8FF),  # Devanagari Extended
    ],
    # Added for the 7-language x 11-family MorphScore sweep (tam_Taml);
    # see scripts/sweep_intrinsic_v2.py. Purely additive: no existing script
    # key's ranges are touched, so every previously-fit corpus is unaffected.
    "Taml": [
        (0x0B80, 0x0BFF),  # Tamil
        (0x11FC0, 0x11FFF),  # Tamil Supplement
    ],
}


def _resolve_target_script(lang_code: str) -> str:
    """Extract the ISO 15924 script suffix from a lang code like `tur_Latn`.

    Raises ValueError if the lang_code is not in `xxx_Yyyy` form or if the
    resulting script has no entry in SCRIPT_RANGES. Hard-fail semantics per
    project policy (no silent fallback to Latn)."""
    parts = lang_code.rsplit("_", 1)
    if len(parts) != 2 or len(parts[1]) != 4 or not parts[1].isalpha() or not parts[1][0].isupper():
        raise ValueError(
            f"Cannot infer target script from lang code {lang_code!r}: expected "
            f"'xxx_Yyyy' form with a four-letter ISO 15924 script suffix "
            f"(e.g. 'tur_Latn', 'kor_Hang', 'arb_Arab'). Code langs should be "
            f"registered in _CODE_LANGS and will bypass this filter."
        )
    target_script = parts[1]
    if target_script not in SCRIPT_RANGES:
        raise ValueError(
            f"No Unicode codepoint ranges configured for script {target_script!r} "
            f"(from lang {lang_code!r}). Known scripts: {sorted(SCRIPT_RANGES.keys())}. "
            f"Add ranges to SCRIPT_RANGES in "
            f"las_experiment/tokenizer_training/training.py."
        )
    return target_script


def _is_in_target_script(ch: str, target_script: str) -> bool:
    """Return True if `ch` is in `target_script` or is a script-neutral glyph
    (digit, punctuation, symbol, separator). Letters outside the target script
    return False, including ASCII letters when target_script != "Latn"."""
    # Script-neutral Unicode categories: always accept.
    cat = unicodedata.category(ch)
    if cat.startswith(("P", "S", "Z", "N")):
        return True
    cp = ord(ch)
    for (lo, hi) in SCRIPT_RANGES[target_script]:
        if lo <= cp <= hi:
            return True
    return False


def filter_line_by_script(text: str, lang_code: str, threshold: float = 0.10) -> bool:
    """
    Return True if the line should be KEPT (enough target-script characters).

    Target script is resolved from the lang_code suffix (ISO 15924). Keeps lines
    where at most `threshold` of non-whitespace characters are OUTSIDE the target
    script (i.e. at least (1 - threshold) are target-script or script-neutral).
    For code languages, keeps all lines.

    Raises on unknown script codes — no silent fallback.
    """
    base_lang = lang_code.replace("_Latn", "")
    if base_lang in _CODE_LANGS:
        return True

    target_script = _resolve_target_script(lang_code)

    non_ws_chars = [ch for ch in text if not ch.isspace()]
    if len(non_ws_chars) == 0:
        return False

    outside_count = sum(1 for ch in non_ws_chars if not _is_in_target_script(ch, target_script))
    return outside_count / len(non_ws_chars) <= threshold


# ---------------------- FineWeb2-10K streaming / reading ----------------------
def prepare_language_corpora(langs, lines_per_lang=50000, output_dir="./data/parallel",
                            local_data_dir=None, text_column="text"):
    """
    Build per-language text files either by:
    - reading from a local parquet directory (if local_data_dir is set), or
    - streaming FineWeb2-10K for each language from HuggingFace Hub.

    Returns a dict mapping language code -> local text file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_info = {}

    for lang in langs:
        out_file = output_dir / f"{lang}.txt"

        if local_data_dir:
            lang_dir = Path(local_data_dir) / lang
            logging.info(f"Loading {lang} from local parquet: {lang_dir}")
            try:
                ds = load_dataset("parquet", data_dir=str(lang_dir), split="train")
                lines = []
                filtered_count = 0
                for ex in ds:
                    text = ex.get(text_column, "").strip()
                    if not text:
                        continue
                    if not filter_line_by_script(text, lang):
                        filtered_count += 1
                        continue
                    lines.append(text)
                    if len(lines) >= lines_per_lang:
                        break
                if filtered_count > 0:
                    logging.info(
                        f"Filtered {filtered_count} non-target-script lines for {lang}"
                    )
            except Exception as e:
                logging.warning(f"Skipping {lang}: {e}")
                continue
        else:
            logging.info(f"Streaming {lang} from FineWeb2-10K...")
            try:
                ds_stream = load_dataset(
                    "MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True
                )
            except Exception as e:
                logging.warning(f"Skipping {lang}: {e}")
                continue

            lines = []
            filtered_count = 0
            for ex in ds_stream:
                text = ex.get(text_column, "").strip()
                if not text:
                    continue
                if not filter_line_by_script(text, lang):
                    filtered_count += 1
                    continue
                lines.append(text)
                if len(lines) >= lines_per_lang:
                    break
            if filtered_count > 0:
                logging.info(
                    f"Filtered {filtered_count} non-target-script lines for {lang}"
                )

        # Fail loud if corpus prep produced no lines — otherwise downstream
        # SentencePiece training gets a 0-byte input and dies with a cryptic
        # "returned non-zero exit status 255" without revealing the real cause.
        if len(lines) == 0:
            raise RuntimeError(
                f"Corpus for {lang!r} is EMPTY after script filtering "
                f"(filtered_count={filtered_count}). The script filter is "
                f"rejecting every line — check that SCRIPT_RANGES covers the "
                f"target script suffix for {lang!r}. Source: "
                f"{(Path(local_data_dir) / lang) if local_data_dir else 'HF:FineWeb2-10K'}."
            )

        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logging.info(f"Wrote {len(lines)} lines for {lang} to {out_file}")
        corpus_info[lang] = str(out_file)

    return corpus_info


# ---------------------- Main pipeline ----------------------
def run_tokenizer_training(
    langs,
    output_dir,
    lines_per_lang=50000,
    train_new_base_tokenizer=True,
    hf_username="MultilingualUnigramLM",
    model_name="allenai/Olmo-3-1025-7B",
    training_cfg=None,
    is_push_to_hf=True,
    local_data_dir=None,
    text_column="text",
):
    """
    End-to-end pipeline for training a family of language-specific tokenizers.

    Steps:
    1. Load model tokenizer as initialization seed
    2. Prepare per-language corpora
    3. Create final tokenizer directory (named after HF repo)
    4. Train ONE shared base Unigram tokenizer on all corpora combined
    5. For each language, re-estimate token probabilities from the base (in parallel)
    6. Validate tokenizers (pre/post-tokenizer consistency, token order preservation)
    7. Push all tokenizers to HuggingFace Hub (if enabled)

    Output structure:
    {output_dir}/las-tokenizers-{model}-{lang_codes}/
        ├── base_tokenizer.json
        ├── langspec_eng.json
        ├── langspec_yor.json
        └── ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = training_cfg or {}

    # 1) Load model tokenizer as the initialization seed
    logging.info(f"Loading model tokenizer from {model_name}...")
    model_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    seed_tokenizer_dir = output_dir / "model_tokenizer"
    seed_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_tokenizer.save_pretrained(seed_tokenizer_dir)
    model_tokenizer_path = seed_tokenizer_dir / "tokenizer.json"

    # Detect unk_token from model tokenizer, fallback to endoftext token
    unk_token = (
        model_tokenizer.unk_token
        if model_tokenizer.unk_token
        else model_tokenizer.eos_token
    )
    logging.info(f"Using unk_token={unk_token} from model tokenizer")

    # 2) Prepare per-language corpora (stream from FineWeb2-10K or read from disk)
    # Corpora are write-once, regenerable artifacts — never read after training
    # completes. Home dir has a tight quota, so we redirect to scratch when
    # LAS_TOK_CORPORA_ROOT is set (via slurm/env.sh). Maintain a symlink at the
    # conventional `<output_dir>/corpora` location so any path-expectation
    # downstream code still resolves.
    corpora_root_env = os.environ.get("LAS_TOK_CORPORA_ROOT")
    conv_corpora_dir = output_dir / "corpora"
    if corpora_root_env:
        scratch_corpora = Path(corpora_root_env) / f"{output_dir.name}_corpora"
        scratch_corpora.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        if conv_corpora_dir.exists() or conv_corpora_dir.is_symlink():
            # If a symlink exists pointing elsewhere, or a stale dir exists,
            # stop — do not fallback. Caller must clean up manually.
            if not (conv_corpora_dir.is_symlink() and Path(os.readlink(conv_corpora_dir)) == scratch_corpora):
                raise RuntimeError(
                    f"{conv_corpora_dir} already exists and does not point to "
                    f"{scratch_corpora}. Cannot redirect corpora without "
                    f"losing data. Remove or fix manually."
                )
        else:
            os.symlink(scratch_corpora, conv_corpora_dir)
        corpora_dir = scratch_corpora
        logging.info(f"Corpora redirected to scratch: {scratch_corpora} (symlink at {conv_corpora_dir})")
    else:
        corpora_dir = conv_corpora_dir
    logging.info(f"Preparing corpora for {len(langs)} languages...")
    corpus_info = prepare_language_corpora(
        langs,
        lines_per_lang=lines_per_lang,
        output_dir=corpora_dir,
        local_data_dir=local_data_dir,
        text_column=text_column,
    )

    if not corpus_info:
        logging.error("No corpora prepared. Exiting.")
        return

    # 3) Create final tokenizer directory
    # Write directly to output_dir (all languages share the same directory).
    tokenizers_dir = output_dir
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    # 4) Train the shared base Unigram tokenizer on ALL language corpora
    if train_new_base_tokenizer:
        base_tokenizer_path = tokenizers_dir / "base_tokenizer.json"
        train_base_tokenizer(
            corpus_info=corpus_info,
            model_tokenizer_path=str(model_tokenizer_path),
            base_output_path=str(base_tokenizer_path),
            em_mode=str(training_cfg.get("em_mode", "soft")),
            unk_token=unk_token,  # Use auto-detected value
        )
        validate_tokenizers(model_tokenizer_path, base_tokenizer_path)
    else:
        base_tokenizer_path = model_tokenizer_path

    # 5) Train language-specific tokenizers in parallel
    # Each re-estimates token probabilities from the shared base
    logging.info(
        f"Training {len(corpus_info)} language-specific tokenizers in parallel..."
    )
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        reestimation_em_mode=str(training_cfg.get("em_mode", "soft")),
        unk_token=unk_token,
    )
    las_tokenizer.train(
        corpus_info=corpus_info,
        base_tokenizer_path=str(base_tokenizer_path),
        tokenizer_path_format=str(tokenizers_dir / "langspec_{lang_code}.json"),
        load_base_if_exists=True,  # Always load the base we just trained
        load_tokenizers_if_exists=False,  # Always retrain
    )

    # 5b) Generate PathPiece tokenizer (shortest segmentation baseline).
    # Setting all Unigram scores to -1 makes Viterbi prefer the fewest tokens
    # (each token costs -1, so fewer tokens = higher total score = longest match).
    pathpiece_path = tokenizers_dir / "pathpiece_tokenizer.json"
    logging.info(f"Generating PathPiece tokenizer: {pathpiece_path}")
    _generate_pathpiece_tokenizer(str(base_tokenizer_path), str(pathpiece_path))

    logging.info("All tokenizers trained successfully!")
    logging.info(f"Tokenizers directory: {tokenizers_dir}")
    logging.info("  - Base tokenizer: base_tokenizer.json")
    logging.info("  - PathPiece tokenizer: pathpiece_tokenizer.json")
    logging.info("  - Language-specific tokenizers: langspec_<lang>.json")

    # 6) Validate tokenizers
    lang_tokenizer_paths = {
        lang: tokenizers_dir / f"langspec_{lang}.json" for lang in corpus_info.keys()
    }
    validate_tokenizers(model_tokenizer_path, base_tokenizer_path, lang_tokenizer_paths)

    # 7) Push all tokenizers to HuggingFace Hub if requested
    if is_push_to_hf:
        lang_codes = "_".join(sorted(corpus_info.keys())[:3])
        if len(corpus_info) > 3:
            lang_codes += f"_plus{len(corpus_info) - 3}"
        repo_name = f"las-tokenizers-{model_name.split('/')[-1]}-{lang_codes}"

        logging.info(f"Pushing all tokenizers to HuggingFace Hub as {hf_username}/{repo_name}...")
        push_to_hf(tokenizers_dir, hf_username, repo_name)
        logging.info(
            f"All tokenizers pushed to HuggingFace Hub: {hf_username}/{repo_name}"
        )
