#!/usr/bin/env python3
"""
tokenizer_training.py
End-to-end, in-process pipeline for training language-specific LAS UnigramLM tokenizers.
For each language, train a LanguageSpecificUnigramLMTokenizer (LAS) seeded from the model tokenizer
on the language corpus. Languages are processed in parallel via ThreadPoolExecutor, and
the resulting tokenizer JSON artifacts can optionally be pushed to the HF Hub.
"""

import itertools
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


def push_to_hf(output_dir: Path, hf_username: str, repo_name: str) -> None:
    """Create/update a public HF repo and upload tokenizer JSON artifacts."""
    from huggingface_hub import HfApi, HfFolder

    api = HfApi()
    token = HfFolder.get_token()
    repo_id = f"{hf_username}/{repo_name}"
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        token=token,
        allow_patterns=["*.json"],  # save only .json tokenizer files.
        ignore_patterns=[".git/*"],
    )
    logging.info("Pushed to HF repo: https://huggingface.co/%s", repo_id)


def train_language_tokenizer(
    *,
    lang: str,
    corpus_file: str,
    seed_tokenizer_path: str,
    hf_username: str,
    vocab_size: int,
    byte_level: bool = True,
    lang_output_dir: Path,
    reestimation_em_mode: str = "soft",
    load_base_if_exists: bool = True,
    load_tokenizers_if_exists: bool = False,
    push_to_hf_flag: bool = True,
    unk_token: str = None,
) -> None:
    """
    Train a tokenizer for a single language:
    A language-specific LAS Unigram tokenizer re-estimated from the base.

    The resulting artifacts are written under `output_dir` and can optionally
    be pushed to a Hugging Face Hub repo.
    """

    # Path where the base Unigram tokenizer JSON will be stored.

    # Train the language-specific LAS tokenizer using the base.
    # This re-estimates token probabilities based on the single-language corpus.
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        reestimation_em_mode=reestimation_em_mode,
        unk_token=unk_token,
    )
    las_tokenizer.train(
        corpus_info={lang: corpus_file},
        base_tokenizer_path=str(seed_tokenizer_path),
        output_dir=str(lang_output_dir),
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists,
    )

    # Optional: upload JSON artifacts to the HF Hub.
    if push_to_hf_flag:
        push_to_hf(lang_output_dir, hf_username, f"las-tok-olmo-3-fineweb-{lang}")

    logging.info("Finished training and pushing %s", lang)


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
    push_to_hf=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load Olmo-3 tokenizer as the base vocabulary seed.
    logging.info("Loading Olmo-3 tokenizer as base...")
    model_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    seed_tokenizer_dir = output_dir / "model_tokenizer"
    seed_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_tokenizer.save_pretrained(seed_tokenizer_dir)
    seed_tokenizer_path = seed_tokenizer_dir / "tokenizer.json"

    # 2) Prepare per-language corpora (stream or read from disk).
    corpus_info = prepare_language_corpora(
        langs,
        lines_per_lang=lines_per_lang,
        output_dir=output_dir / "corpora",
    )

    # 3) Train & push in parallel threads.
    futures = []
    push_executor = ThreadPoolExecutor(max_workers=max_workers)
    training_cfg = training_cfg or {}

    for lang, corpus_file in corpus_info.items():
        # Dispatch per-language training using the in-process function.

        # Ensure the output directory exists for this language.
        lang_output_dir = Path(output_dir) / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        # save the seed tokenizer under the name of base tokenizer for consistency
        base_tokenizer_path = lang_output_dir / "base.tokenizer.json"
        shutil.copy2(seed_tokenizer_path, base_tokenizer_path)

        futures.append(
            push_executor.submit(
                train_language_tokenizer,
                lang=lang,
                corpus_file=corpus_file,
                seed_tokenizer_path=str(seed_tokenizer_path),
                lang_output_dir=lang_output_dir,
                hf_username=hf_username,
                vocab_size=len(model_tokenizer),
                byte_level=bool(training_cfg.get("byte_level", True)),
                reestimation_em_mode=str(
                    training_cfg.get("reestimation_em_mode", "soft")
                ),
                load_base_if_exists=bool(training_cfg.get("load_base_if_exists", True)),
                load_tokenizers_if_exists=bool(
                    training_cfg.get("load_tokenizers_if_exists", False)
                ),
                push_to_hf_flag=bool(push_to_hf),
                unk_token="<|endoftext|>",
            )
        )

    # Wait for all tasks
    for future in futures:
        future.result()

    logging.info("All language-specific tokenizers trained and pushed asynchronously!")
