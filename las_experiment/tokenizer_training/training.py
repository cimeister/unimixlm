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
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from las_experiment.tokenizer_training.validation import validate_tokenizers
from las_experiment.tokenizer_training.utils import (
    push_to_hf,
)


def train_base_tokenizer(
    *,
    corpus_info: dict,
    model_tokenizer_path: str,
    base_output_path: Path,
    vocab_size: int,
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

    # Create temporary directory for training
    temp_output_dir = base_output_path.parent / "_temp_base"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all corpus files into a single base corpus file
    combined_corpus_path = temp_output_dir / "combined_corpus.txt"
    logging.info(
        f"Combining {len(corpus_info)} corpus files into {combined_corpus_path}"
    )

    with open(combined_corpus_path, "w", encoding="utf-8") as outfile:
        for lang, corpus_file in corpus_info.items():
            logging.info(f"  Adding corpus from {lang}: {corpus_file}")
            with open(corpus_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    logging.info(f"Combined corpus created with data from {len(corpus_info)} languages")

    # Train base tokenizer with combined corpus
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        reestimation_em_mode=em_mode,
        unk_token=unk_token,
    )

    las_tokenizer.train(
        corpus_info={"base": str(combined_corpus_path)},
        base_tokenizer_path=str(model_tokenizer_path),
        output_dir=str(temp_output_dir),
        load_base_if_exists=True,
        load_tokenizers_if_exists=False,  # Always retrain
    )

    # Move the trained tokenizer to the final location with correct name
    # The file is saved as: langspec_{em_mode}_base.tokenizer.json
    temp_tokenizer_path = temp_output_dir / f"langspec_{em_mode}_base.tokenizer.json"
    if temp_tokenizer_path.exists():
        shutil.move(str(temp_tokenizer_path), str(base_output_path))
        logging.info(f"Base tokenizer saved to {base_output_path}")
    else:
        logging.error(f"Expected base tokenizer file not found: {temp_tokenizer_path}")
        if temp_output_dir.exists():
            logging.error(
                f"Files in {temp_output_dir}: {list(temp_output_dir.glob('*'))}"
            )
        raise FileNotFoundError(
            f"Base tokenizer training failed, file not found: {temp_tokenizer_path}"
        )

    # Clean up temporary directory
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def train_language_specific_tokenizer(
    *,
    lang: str,
    corpus_file: str,
    base_tokenizer_path: Path,
    vocab_size: int,
    lang_output_path: Path,
    reestimation_em_mode: str = "soft",
    unk_token: str = None,
) -> None:
    """
    Train a language-specific tokenizer by re-estimating token probabilities
    from the shared base tokenizer for a single language.

    The resulting tokenizer is saved to `lang_output_path`.

    Always trains from scratch - does not load existing tokenizers.
    """
    logging.info(f"Training language-specific tokenizer for {lang}")

    # Train the language-specific LAS tokenizer using the base.
    # This re-estimates token probabilities based on the single-language corpus.
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        reestimation_em_mode=reestimation_em_mode,
        unk_token=unk_token,
    )

    # Create temporary directory for training output
    temp_output_dir = lang_output_path.parent / f"_temp_{lang}"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    las_tokenizer.train(
        corpus_info={lang: corpus_file},
        base_tokenizer_path=str(base_tokenizer_path),
        output_dir=str(temp_output_dir),
        load_base_if_exists=True,  # Always load the base we just trained
        load_tokenizers_if_exists=False,  # Always retrain
    )

    # Find and move the language-specific tokenizer to the final location
    # The file is saved as: langspec_{em_mode}_{lang}.tokenizer.json
    temp_tokenizer_path = (
        temp_output_dir / f"langspec_{reestimation_em_mode}_{lang}.tokenizer.json"
    )
    if temp_tokenizer_path.exists():
        shutil.move(str(temp_tokenizer_path), str(lang_output_path))
        logging.info(f"Saved {lang} tokenizer to {lang_output_path}")
    else:
        logging.error(f"Expected tokenizer file not found: {temp_tokenizer_path}")
        # List files in temp directory for debugging
        if temp_output_dir.exists():
            logging.error(
                f"Files in {temp_output_dir}: {list(temp_output_dir.glob('*'))}"
            )
        raise FileNotFoundError(
            f"Language-specific tokenizer training failed for {lang}, file not found: {temp_tokenizer_path}"
        )

    # Clean up temporary directory
    shutil.rmtree(temp_output_dir, ignore_errors=True)

    logging.info(f"Finished training {lang}")


# ---------------------- FineWeb2-10K streaming / reading ----------------------
def prepare_language_corpora(langs, lines_per_lang=10000, output_dir="./data/parallel"):
    """
    Build per-language text files either by:
    - streaming FineWeb2-10K for each language.

    Returns a dict mapping language code -> local text file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_info = {}

    for lang in langs:
        out_file = output_dir / f"{lang}.txt"

        logging.info(f"Streaming {lang} from FineWeb2-10K...")
        try:
            ds_stream = load_dataset(
                "MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True
            )
        except Exception as e:
            logging.warning(f"Skipping {lang}: {e}")
            continue

        lines = []
        for ex in itertools.islice(ds_stream, lines_per_lang):
            text = ex.get("text", "").strip()
            if text:
                lines.append(text)

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
    lines_per_lang=10000,
    hf_username="MultilingualUnigramLM",
    model_name="allenai/Olmo-3-1025-7B",
    max_workers=8,
    training_cfg=None,
    is_push_to_hf=True,
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
    logging.info(f"Preparing corpora for {len(langs)} languages...")
    corpus_info = prepare_language_corpora(
        langs,
        lines_per_lang=lines_per_lang,
        output_dir=output_dir / "corpora",
    )

    if not corpus_info:
        logging.error("No corpora prepared. Exiting.")
        return

    # 3) Create final tokenizer directory named after the HF repo
    # Determine repository name based on languages (truncate to 3 letters)
    truncated_langs = [lang[:3] for lang in sorted(corpus_info.keys())[:3]]
    lang_codes = "_".join(truncated_langs)
    if len(corpus_info) > 3:
        lang_codes += f"_plus{len(corpus_info) - 3}"

    repo_name = f"las-tokenizers-{model_name.split('/')[-1]}-{lang_codes}"
    tokenizers_dir = output_dir / repo_name
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    # 4) Train the shared base Unigram tokenizer on ALL language corpora
    base_tokenizer_path = tokenizers_dir / "base_tokenizer.json"
    train_base_tokenizer(
        corpus_info=corpus_info,
        model_tokenizer_path=model_tokenizer_path,
        base_output_path=base_tokenizer_path,
        vocab_size=len(model_tokenizer),
        em_mode=str(training_cfg.get("em_mode", "soft")),
        unk_token=unk_token,  # Use auto-detected value
    )

    # 5) Train language-specific tokenizers in parallel
    # Each re-estimates token probabilities from the shared base
    logging.info(
        f"Training {len(corpus_info)} language-specific tokenizers in parallel..."
    )
    futures_by_lang = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)

    for lang, corpus_file in corpus_info.items():
        # Save language-specific tokenizer in the same flat directory
        lang_tokenizer_path = tokenizers_dir / f"langspec_{lang}.json"

        future = executor.submit(
            train_language_specific_tokenizer,
            lang=lang,
            corpus_file=corpus_file,
            base_tokenizer_path=base_tokenizer_path,
            lang_output_path=lang_tokenizer_path,
            vocab_size=len(model_tokenizer),
            reestimation_em_mode=str(training_cfg.get("reestimation_em_mode", "soft")),
            unk_token=unk_token,  # Use auto-detected value
        )
        futures_by_lang[lang] = future

    # Wait for all language-specific training to complete and collect failures
    failures = []
    for lang, future in futures_by_lang.items():
        try:
            future.result()
        except Exception as e:
            error_msg = f"Language-specific training failed for {lang}: {e}"
            logging.error(error_msg)
            failures.append((lang, e))

    executor.shutdown(wait=True)

    # Only proceed if all trainings succeeded
    if failures:
        failure_summary = "\n".join([f"  - {lang}: {str(e)}" for lang, e in failures])
        error_message = f"Failed to train {len(failures)} language-specific tokenizer(s):\n{failure_summary}"
        logging.error(error_message)
        raise RuntimeError(error_message)

    logging.info("All tokenizers trained successfully!")
    logging.info(f"Tokenizers directory: {tokenizers_dir}")
    logging.info("  - Base tokenizer: base_tokenizer.json")
    logging.info("  - Language-specific tokenizers: langspec_<lang>.json")

    # 6) Validate tokenizers
    lang_tokenizer_paths = {
        lang: tokenizers_dir / f"langspec_{lang}.json" for lang in corpus_info.keys()
    }
    validate_tokenizers(model_tokenizer_path, base_tokenizer_path, lang_tokenizer_paths)

    # 7) Push all tokenizers to HuggingFace Hub if requested
    if is_push_to_hf:
        logging.info("Pushing all tokenizers to HuggingFace Hub...")

        # Push the tokenizers directory (base + all language-specific tokenizers) to single repo
        push_to_hf(tokenizers_dir, hf_username, repo_name)
        logging.info(
            f"All tokenizers pushed to HuggingFace Hub: {hf_username}/{repo_name}"
        )
