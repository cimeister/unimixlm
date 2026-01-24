#!/usr/bin/env python3
"""
tokenizer_training.py
End-to-end, in-process pipeline for training language-specific LAS UnigramLM tokenizers.
Modifications: 
- Supports capturing the OLMo HF tokenizer pipeline and enforcing it in both
"""

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.standard_unigramlm import StandardUnigramLMTokenizer


def push_to_hf(output_dir: Path, hf_username: str, repo_name: str) -> None:
    from huggingface_hub import HfApi, HfFolder

    api = HfApi()
    token = HfFolder.get_token()
    repo_id = f"{hf_username}/{repo_name}"
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        token=token,
        allow_patterns=["*.json"],
        ignore_patterns=[".git/*"],
    )
    logging.info("Pushed to HF repo: https://huggingface.co/%s", repo_id)


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
    load_tokenizers_if_exists: bool = False,
    push_to_hf_flag: bool = True,
) -> None:
    """
    Train a language-specific LAS tokenizer **without altering OLMo's vocab**:
    - Pre-tokenization is captured from OLMo and reused.
    - No new tokens are added; existing token probabilities are re-estimated.
    """
    from tokenizers import Tokenizer as HFTokenizer

    lang_output_dir = Path(output_dir)
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    base_tokenizer_path = lang_output_dir / "base.tokenizer.json"

    # ---------------------- Load OLMo tokenizer ----------------------
    olmo_tok = HFTokenizer.from_file(seed_tokenizer_path)
    olmo_normalizer = olmo_tok.normalizer
    olmo_pretokenizer = olmo_tok.pre_tokenizer

    logging.info(
        "OLMo pipeline captured for '%s': normalizer=%s pre_tokenizer=%s",
        lang,
        olmo_normalizer.__class__.__name__ if olmo_normalizer else None,
        olmo_pretokenizer.__class__.__name__ if olmo_pretokenizer else None,
    )

    # ---------------------- Base Unigram: preserve vocab ----------------------
    base_tok = StandardUnigramLMTokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        em_mode=em_mode,
        initial_vocab_tokens=str(seed_tokenizer_path),
        use_sp_seed_vocab=True,  # <- enforce Olmo vocab exactly
    )

    # Inject OLMo normalization + pretokenization
    if hasattr(base_tok, "external_normalizer"):
        base_tok.external_normalizer = olmo_normalizer
        base_tok.external_pretokenizer = olmo_pretokenizer
    else:
        base_tok._external_normalizer = olmo_normalizer
        base_tok._external_pretokenizer = olmo_pretokenizer

    # Train **only re-estimating probabilities**; do not add/remove tokens
    base_tok.train(
        [corpus_file],
        output_path=str(base_tokenizer_path),
        load_if_exists=False,
        freeze_vocab=True,  # <- crucial to preserve Olmo IDs
    )

    # ---------------------- LAS tokenizer: re-estimate probabilities ----------------------
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        reestimation_em_mode=reestimation_em_mode,
    )

    if hasattr(las_tokenizer, "external_normalizer"):
        las_tokenizer.external_normalizer = olmo_normalizer
        las_tokenizer.external_pretokenizer = olmo_pretokenizer
    else:
        las_tokenizer._external_normalizer = olmo_normalizer
        las_tokenizer._external_pretokenizer = olmo_pretokenizer

    # Train LAS tokenizer, **vocab is fixed**, only probabilities are updated
    las_tokenizer.train(
        corpus_info={lang: corpus_file},
        base_tokenizer_path=str(base_tokenizer_path),
        output_dir=str(lang_output_dir),
        load_base_if_exists=True,
        load_tokenizers_if_exists=load_tokenizers_if_exists,
        freeze_vocab=True,
    )

    # ---------------------- Post-LAS sanity check ----------------------
    las_tok_json = lang_output_dir / f"langspec_{reestimation_em_mode}_{lang}.tokenizer.json"
    las_tok_hf = HFTokenizer.from_file(str(las_tok_json))

    assert las_tok_hf.normalizer.to_str() == olmo_tok.normalizer.to_str(), "Normalizer drift in LAS tokenizer"
    assert las_tok_hf.pre_tokenizer.to_str() == olmo_tok.pre_tokenizer.to_str(), "Pretokenizer drift in LAS tokenizer"

    logging.info(
        "✅ LAS tokenizer probabilities re-estimated for '%s' with Olmo vocab preserved", lang
    )

    if push_to_hf_flag:
        push_to_hf(lang_output_dir, hf_username, f"las-tok-olmo-3-fineweb-{lang}")



# ---------------------- FineWeb2-10K streaming ----------------------
def prepare_language_corpora(langs, lines_per_lang=10000, output_dir="./data/parallel"):
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

        lines = [ex.get("text", "").strip() for ex in itertools.islice(ds_stream, lines_per_lang) if ex.get("text", "").strip()]

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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

    # Save Olmo tokenizer locally as seed
    model_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    seed_tokenizer_dir = output_dir / "model_tokenizer"
    seed_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_tokenizer.save_pretrained(seed_tokenizer_dir)
    seed_tokenizer_path = seed_tokenizer_dir / "tokenizer.json"

    # Prepare corpora
    corpus_info = prepare_language_corpora(
        langs,
        lines_per_lang=lines_per_lang,
        output_dir=output_dir / "corpora",
    )

    futures = []
    push_executor = ThreadPoolExecutor(max_workers=max_workers)
    training_cfg = training_cfg or {}

    for lang, corpus_file in corpus_info.items():
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
                reestimation_em_mode=str(training_cfg.get("reestimation_em_mode", "soft")),
                use_sp_seed_vocab=bool(training_cfg.get("use_sp_seed_vocab", False)),
                load_base_if_exists=bool(training_cfg.get("load_base_if_exists", True)),
                load_tokenizers_if_exists=bool(training_cfg.get("load_tokenizers_if_exists", False)),
                push_to_hf_flag=bool(push_to_hf),
            )
        )

    for future in futures:
        future.result()

    logging.info("All language-specific tokenizers trained and pushed asynchronously!")

