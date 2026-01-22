#!/usr/bin/env python3
"""
new_olmo.py
End-to-end pipeline for language-specific LAS-UnigramLM tokenizers using Olmo-3 as base.
Each language trains in its own subprocess to avoid daemonic process issues on macOS.
HF push happens asynchronously immediately after each tokenizer finishes training.
"""

import logging
import subprocess
import sys
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Languages ----------------------
all_langs = [
    "afr_Latn",
    "als_Latn",
    "azj_Latn",
    "bos_Latn",
    "cat_Latn",
    "cos_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "fao_Latn",
    "fil_Latn",
    "fin_Latn",
    "fry_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "gmh_Latn",
    "hat_Latn",
    "hau_Latn",
    "haw_Latn",
    "hif_Latn",
    "hin_Latn",
    "hrv_Latn",
    "hun_Latn",
    "ibo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "kin_Latn",
    "kmr_Latn",
    "lit_Latn",
    "ltz_Latn",
    "lus_Latn",
    "lvs_Latn",
    "mlt_Latn",
    "mww_Latn",
    "nob_Latn",
    "plt_Latn",
    "por_Latn",
    "ron_Latn",
    "slv_Latn",
    "smo_Latn",
    "sot_Latn",
    "srp_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "tuk_Latn",
    "uzn_Latn",
    "vie_Latn",
    "yor_Latn",
    "zul_Latn",
    "sna_Latn",
    "nya_Latn",
    "zsm_Latn",
    "som_Latn",
]


def push_to_hf_new(output_dir, hf_username, repo_name):
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
    logger.info(f"Pushed to HF repo: https://huggingface.co/{repo_id}")


# ---------------------- FineWeb2-10K streaming / reading ----------------------
def prepare_language_corpora(
    langs, lines_per_lang=10000, input_dir=None, output_dir="./data/parallel"
):
    """
    Either read existing txt files from input_dir or stream FineWeb2-10K.
    Returns dict {lang_code: path_to_file}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_info = {}

    for lang in langs:
        out_file = output_dir / f"{lang}.txt"
        if input_dir and (Path(input_dir) / f"{lang}.txt").exists():
            logger.info(f"Copying existing file for {lang}")
            out_file.write_text(
                (Path(input_dir) / f"{lang}.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            corpus_info[lang] = str(out_file)
            continue

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
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Olmo-3 tokenizer as base
    logger.info("Loading Olmo-3 tokenizer as base...")
    model_tokenizer = AutoTokenizer.from_pretrained(
        "allenai/Olmo-3-1025-7B", trust_remote_code=True
    )
    seed_tokenizer_dir = output_dir / "model_tokenizer"
    seed_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_tokenizer.save_pretrained(seed_tokenizer_dir)
    seed_tokenizer_path = seed_tokenizer_dir / "tokenizer.json"

    # 2. Prepare corpora
    corpus_info = prepare_language_corpora(
        langs, lines_per_lang=lines_per_lang, output_dir=output_dir / "corpora"
    )

    # 3. Train & push in separate subprocesses
    futures = []
    push_executor = ThreadPoolExecutor(max_workers=8)

    for lang, corpus_file in corpus_info.items():
        train_cmd = [
            sys.executable,
            "-c",
            f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path(__file__).parent}/code')
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from standard_unigramlm import StandardUnigramLMTokenizer
from olmo_fineweb2 import push_to_hf_new

lang = '{lang}'
corpus_file = '{corpus_file}'
seed_tokenizer_path = '{seed_tokenizer_path}'
base_tokenizer_path = '{output_dir / lang / "base.tokenizer.json"}'
vocab_size = {len(model_tokenizer)}
lang_output_dir = Path('{output_dir / lang}')
lang_output_dir.mkdir(parents=True, exist_ok=True)

# Train base Unigram tokenizer initialized from the model tokenizer vocab
base_tok = StandardUnigramLMTokenizer(
    vocab_size=vocab_size,
    byte_level=True,
    em_mode="soft",
    initial_vocab_tokens=str(seed_tokenizer_path),
    use_sp_seed_vocab=False,
)
base_tok.train([corpus_file], output_path=str(base_tokenizer_path), load_if_exists=False)

# Train language-specific tokenizer from the base Unigram tokenizer
las_tokenizer = LanguageSpecificUnigramLMTokenizer(
    vocab_size=vocab_size,
    byte_level=True,
    reestimation_em_mode='soft'
)
las_tokenizer.train(
    corpus_info={{lang: corpus_file}},
    base_tokenizer_path=str(base_tokenizer_path),
    output_dir=str(lang_output_dir),
    load_base_if_exists=True,
    load_tokenizers_if_exists=False
)
push_to_hf_new(lang_output_dir, '{hf_username}', f'olmo-3-fineweb-{{lang}}')
print('Finished training and pushing {lang}')
            """,
        ]
        futures.append(push_executor.submit(subprocess.run, train_cmd, check=True))

    # Wait for all tasks
    for future in futures:
        future.result()

    logger.info("All language-specific tokenizers trained and pushed asynchronously!")


# ---------------------- Entry point ----------------------
if __name__ == "__main__":
    main(
        all_langs,
        output_dir="./olmo3_las_tokenizers_new",
        lines_per_lang=10000,
        hf_username="MultilingualUnigramLM",
    )
