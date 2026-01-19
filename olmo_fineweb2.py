#!/usr/bin/env python3
"""
Run olmo_fineweb2.py to:
- Download Olmo-3 tokenizer if not present
- Stream FineWeb2-10K dataset
- Save N lines per language locally
- Train LAS-UnigramLM tokenizers per language using Olmo-3 as base
- Push resulting tokenizers to Hugging Face
"""

import logging, os, subprocess, sys, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Add code/ to sys.path ----------------------
sys.path.insert(0, str(Path(__file__).parent / "code"))
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

# ---------------------- Languages ----------------------
all_langs = [
    "afr_Latn","als_Latn","azj_Latn","bos_Latn","cat_Latn","cos_Latn","cym_Latn",
    "dan_Latn","deu_Latn","fao_Latn","fil_Latn","fin_Latn","fry_Latn","gla_Latn",
    "gle_Latn","glg_Latn","gmh_Latn","hat_Latn","hau_Latn","haw_Latn","hif_Latn",
    "hin_Latn","hrv_Latn","hun_Latn","ibo_Latn","ind_Latn","isl_Latn","ita_Latn",
    "jav_Latn","kin_Latn","kmr_Latn","lit_Latn","ltz_Latn","lus_Latn","lvs_Latn",
    "mlt_Latn","mww_Latn","nob_Latn","plt_Latn","por_Latn","ron_Latn","slv_Latn",
    "smo_Latn","sot_Latn","srp_Latn","sun_Latn","swe_Latn","swh_Latn","tuk_Latn",
    "uzn_Latn","vie_Latn","yor_Latn","zul_Latn","sna_Latn","nya_Latn","zsm_Latn",
    "som_Latn"
]

# ---------------------- HuggingFace push utility ----------------------
def push_to_hf(output_dir, hf_username, repo_name):
    from huggingface_hub import HfApi, HfFolder
    api = HfApi()
    token = HfFolder.get_token()
    repo_id = f"{hf_username}/{repo_name}"
    hf_url = f"https://huggingface.co/{repo_id}"
    try:
        api.repo_info(repo_id, token=token)
    except:
        logger.info(f"Creating HF repo: {hf_url}")
        api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

    if not (output_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(output_dir), check=True)
        subprocess.run(["git", "checkout", "-b", "main"], cwd=str(output_dir), check=True)
        subprocess.run(["git", "remote", "add", "origin", hf_url], cwd=str(output_dir), check=True)

    subprocess.run(["git", "add", "."], cwd=str(output_dir), check=True)
    try:
        subprocess.run(["git", "commit", "-m", f"Add {repo_name} files"], cwd=str(output_dir), check=True)
    except subprocess.CalledProcessError:
        logger.warning("Nothing to commit, skipping git commit.")
    subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=str(output_dir), check=True)
    logger.info(f"Pushed to HF repo: {hf_url}")

# ---------------------- FineWeb2-10K streaming / reading ----------------------
def prepare_language_corpora(langs, lines_per_lang=10000, input_dir=None, output_dir="./data/parallel"):
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
            out_file.write_text((Path(input_dir) / f"{lang}.txt").read_text(encoding="utf-8"), encoding="utf-8")
            corpus_info[lang] = str(out_file)
            continue

        logger.info(f"Streaming {lang} from FineWeb2-10K...")
        try:
            ds_stream = load_dataset("MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True)
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

# ---------------------- Single language training ----------------------
def train_and_push(lang, corpus_file, base_tokenizer_path, output_root, hf_username):
    """
    Train one-language LAS-UnigramLM tokenizer and push asynchronously.
    """
    logger.info(f"Training LAS-UnigramLM tokenizer for {lang}...")
    lang_output_dir = output_root / "tokenizers" / lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(vocab_size=8000, byte_level=True, reestimation_em_mode="soft")
    las_tokenizer.train(
        corpus_info={lang: corpus_file},
        base_tokenizer_path=str(base_tokenizer_path),
        output_dir=str(lang_output_dir),
        load_base_if_exists=True,
        load_tokenizers_if_exists=False
    )

    # Async HF push
    push_to_hf(lang_output_dir, hf_username, f"olmo-3-fineweb-{lang}")
    logger.info(f"Finished training and pushing {lang}")

# ---------------------- Main pipeline ----------------------
def main(langs, lines_per_lang=10000, output_dir="./olmo3_las_tokenizers", hf_username="MultilingualUnigramLM"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Olmo-3 tokenizer as base
    logger.info("Loading Olmo-3 tokenizer as base...")
    base_tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)

    base_path = output_dir / "olmo3_base_tokenizer.json"
    if not base_path.exists():
        logger.info(f"Saving base tokenizer to {base_path}")
        base_tokenizer.save_pretrained(output_dir)

    # 2. Prepare corpora
    corpus_info = prepare_language_corpora(langs, lines_per_lang=lines_per_lang, output_dir=output_dir / "corpora")

    # 3. Train & push in separate subprocesses
    futures = []
    push_executor = ThreadPoolExecutor(max_workers=8)

    for lang, corpus_file in corpus_info.items():
        train_cmd = [
            sys.executable, "-c",
            f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path(__file__).parent}/code')
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from new_olmo import push_to_hf

lang = '{lang}'
corpus_file = '{corpus_file}'
base_path = '{base_path}'
output_dir = Path('{output_dir}') / 'tokenizers' / lang
output_dir.mkdir(parents=True, exist_ok=True)

las_tokenizer = LanguageSpecificUnigramLMTokenizer(vocab_size=8000, byte_level=True, reestimation_em_mode='soft')
las_tokenizer.train(
    corpus_info={{lang: corpus_file}},
    base_tokenizer_path=str(base_path),
    output_dir=str(output_dir),
    load_base_if_exists=True,
    load_tokenizers_if_exists=False
)
push_to_hf(output_dir, '{hf_username}', f'olmo-3-fineweb-{{lang}}')
print('Finished training and pushing {lang}')
            """
        ]
        futures.append(push_executor.submit(subprocess.run, train_cmd, check=True))

    # Wait for all tasks
    for future in futures:
        future.result()

    logger.info("All language-specific tokenizers trained and pushed asynchronously!")

# ---------------------- Entry point ----------------------
if __name__ == "__main__":
    main(all_langs, lines_per_lang=10000, output_dir="./olmo3_las_tokenizers", hf_username="MultilingualUnigramLM")

