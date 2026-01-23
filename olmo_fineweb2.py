#!/usr/bin/env python3
"""
new_olmo.py
End-to-end pipeline for language-specific LAS-UnigramLM tokenizers using Olmo-3 as base.
Each language trains in its own subprocess to avoid daemonic process issues on macOS.
HF push happens asynchronously immediately after each tokenizer finishes training.
"""

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.standard_unigramlm import StandardUnigramLMTokenizer


# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("Pushed to HF repo: https://huggingface.co/%s", repo_id)


def train_language_tokenizer(
    *,
    lang: str,
    corpus_file: str,
    seed_tokenizer_path: str,
    output_dir: str,
    hf_username: str,
    vocab_size: int,
    byte_level: bool = True,
    em_mode: str = "soft",
    reestimation_em_mode: str = "soft",
    use_sp_seed_vocab: bool = False,
    load_base_if_exists: bool = True,
    load_tokenizers_if_exists: bool = False,
    push_to_hf_flag: bool = True,
) -> None:
    """
    Train a two-stage tokenizer for a single language:
    1) A base Unigram tokenizer seeded from the base model's tokenizer vocab.
    2) A language-specific LAS Unigram tokenizer re-estimated from the base.

    The resulting artifacts are written under `output_dir` and can optionally
    be pushed to a Hugging Face Hub repo.
    """
    # Ensure the output directory exists for this language.
    lang_output_dir = Path(output_dir)
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    # Path where the base Unigram tokenizer JSON will be stored.
    base_tokenizer_path = lang_output_dir / "base.tokenizer.json"

    # Stage 1: Train the base Unigram tokenizer, seeded from the model vocab.
    # - `initial_vocab_tokens` points at the base model's tokenizer.json.
    # - `use_sp_seed_vocab=False` means we do not override the seed vocab.
    base_tok = StandardUnigramLMTokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        em_mode=em_mode,
        initial_vocab_tokens=str(Path(seed_tokenizer_path)),
        use_sp_seed_vocab=use_sp_seed_vocab,
    )
    base_tok.train(
        [corpus_file],
        output_path=str(base_tokenizer_path),
        load_if_exists=False,
    )

    # Stage 2: Train the language-specific LAS tokenizer using the base.
    # This re-estimates token probabilities based on the single-language corpus.
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        reestimation_em_mode=reestimation_em_mode,
    )
    las_tokenizer.train(
        corpus_info={lang: corpus_file},
        base_tokenizer_path=str(base_tokenizer_path),
        output_dir=str(lang_output_dir),
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists,
    )

    # Optional: upload JSON artifacts to the HF Hub.
    if push_to_hf_flag:
        push_to_hf(lang_output_dir, hf_username, f"las-tok-olmo-3-fineweb-{lang}")

    logger.info("Finished training and pushing %s", lang)


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

        logger.info(f"Streaming {lang} from FineWeb2-10K...")
        try:
            ds_stream = load_dataset(
                "MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True
            )
        except Exception as e:
            logger.warning(f"Skipping {lang}: {e}")
            continue

        lines = []
        for ex in itertools.islice(ds_stream, lines_per_lang):
            text = ex.get("text", "").strip()
            if text:
                lines.append(text)

        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logger.info(f"Wrote {len(lines)} lines for {lang} to {out_file}")
        corpus_info[lang] = str(out_file)

    return corpus_info


# ---------------------- Main pipeline ----------------------
def main(
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
    logger.info("Loading Olmo-3 tokenizer as base...")
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
        futures.append(
            push_executor.submit(
                train_language_tokenizer,
                lang=lang,
                corpus_file=corpus_file,
                seed_tokenizer_path=str(seed_tokenizer_path),
                output_dir=str(output_dir / lang),
                hf_username=hf_username,
                vocab_size=len(model_tokenizer),
                byte_level=bool(training_cfg.get("byte_level", True)),
                em_mode=str(training_cfg.get("em_mode", "soft")),
                reestimation_em_mode=str(
                    training_cfg.get("reestimation_em_mode", "soft")
                ),
                use_sp_seed_vocab=bool(training_cfg.get("use_sp_seed_vocab", False)),
                load_base_if_exists=bool(training_cfg.get("load_base_if_exists", True)),
                load_tokenizers_if_exists=bool(
                    training_cfg.get("load_tokenizers_if_exists", False)
                ),
                push_to_hf_flag=bool(push_to_hf),
            )
        )

    # Wait for all tasks
    for future in futures:
        future.result()

    logger.info("All language-specific tokenizers trained and pushed asynchronously!")


# ---------------------- Entry point ----------------------
@hydra.main(
    config_path="config/tokenizer_training",
    config_name="default",
    version_base="1.3",
)
def hydra_main(cfg: DictConfig) -> None:
    langs = list(cfg.languages.langs)
    main(
        langs,
        output_dir=cfg.data.output_dir,
        lines_per_lang=cfg.data.lines_per_lang,
        hf_username=cfg.hf.username,
        model_name=cfg.model.model_name,
        max_workers=cfg.execution.max_workers,
        training_cfg=cfg.training,
        push_to_hf=cfg.hf.push,
    )


if __name__ == "__main__":
    hydra_main()
