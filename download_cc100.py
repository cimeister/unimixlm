#!/usr/bin/env python3
"""
CC100-only multilingual corpus downloader (HF Hub).

python download_cc100.py \
  --lang_set L60 \
  --output_dir ./cc100_10k_lines \
  --max_lines_per_lang 10000 \
  --push_to_hf MultilingualUnigramLM/cc100_10k_lines

Features:
- CC100 streaming only. No FLORES Download, since we use this for evaluation. 
- Language sets: L20 / L60
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import Repository

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_MAIN_TEMP = 5.0
MAX_LANG_CAP = 0.5  # fraction of total corpus per language

# ---------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------
LANGUAGE_METADATA = {
    "en": {"cc100": "en", "flores": "eng_Latn", "est_gb": 82.0},
    "de": {"cc100": "de", "flores": "deu_Latn", "est_gb": 18.0},
    "fr": {"cc100": "fr", "flores": "fra_Latn", "est_gb": 14.0},
    "es": {"cc100": "es", "flores": "spa_Latn", "est_gb": 14.0},
    "ru": {"cc100": "ru", "flores": "rus_Cyrl", "est_gb": 46.0},
    "pt": {"cc100": "pt", "flores": "por_Latn", "est_gb": 13.0},
    "it": {"cc100": "it", "flores": "ita_Latn", "est_gb": 7.8},
    "nl": {"cc100": "nl", "flores": "nld_Latn", "est_gb": 7.9},
    "pl": {"cc100": "pl", "flores": "pol_Latn", "est_gb": 12.0},
    "sv": {"cc100": "sv", "flores": "swe_Latn", "est_gb": 2.1},
    "fi": {"cc100": "fi", "flores": "fin_Latn", "est_gb": 1.5},
    "da": {"cc100": "da", "flores": "dan_Latn", "est_gb": 1.2},
    "no": {"cc100": "no", "flores": "nob_Latn", "est_gb": 1.3},
    "cs": {"cc100": "cs", "flores": "ces_Latn", "est_gb": 4.4},
    "hu": {"cc100": "hu", "flores": "hun_Latn", "est_gb": 1.5},
    "ro": {"cc100": "ro", "flores": "ron_Latn", "est_gb": 1.6},
    "uk": {"cc100": "uk", "flores": "ukr_Cyrl", "est_gb": 1.4},
    "el": {"cc100": "el", "flores": "ell_Grek", "est_gb": 7.4},
    "he": {"cc100": "he", "flores": "heb_Hebr", "est_gb": 6.1},
    "ar": {"cc100": "ar", "flores": "arb_Arab", "est_gb": 5.4},
    "hi": {"cc100": "hi", "flores": "hin_Deva", "est_gb": 2.5},
    "bn": {"cc100": "bn", "flores": "ben_Beng", "est_gb": 0.86},
    "ta": {"cc100": "ta", "flores": "tam_Taml", "est_gb": 0.13},
    "te": {"cc100": "te", "flores": "tel_Telu", "est_gb": 0.05},
    "ml": {"cc100": "ml", "flores": "mal_Mlym", "est_gb": 0.083},
    "mr": {"cc100": "mr", "flores": "mar_Deva", "est_gb": 0.033},
    "ur": {"cc100": "ur", "flores": "urd_Arab", "est_gb": 0.088},
    "zh": {"cc100": "zh-Hans", "flores": "zho_Hans", "est_gb": 14.0},
    "ja": {"cc100": "ja", "flores": "jpn_Jpan", "est_gb": 15.0},
    "ko": {"cc100": "ko", "flores": "kor_Hang", "est_gb": 14.0},
    "id": {"cc100": "id", "flores": "ind_Latn", "est_gb": 3.6},
    "vi": {"cc100": "vi", "flores": "vie_Latn", "est_gb": 2.8},
    "tr": {"cc100": "tr", "flores": "tur_Latn", "est_gb": 5.4},
    "sw": {"cc100": "sw", "flores": "swh_Latn", "est_gb": 0.33},
    "fa": {"cc100": "fa", "flores": "fas_Arab", "est_gb": 2.0},
}

# ---------------------------------------------------------------------
# Language sets
# ---------------------------------------------------------------------
LANG_SET_20 = [
    "en","zh","es","ar","ru","de","fr","ja","hi","pt",
    "id","vi","tr","pl","el","sw","fa","bn","ta","ko",
]

LANG_SET_60 = LANG_SET_20 + [
    "it","nl","uk","ro","cs","hu","sv","fi","da","no",
    "he","ml","mr","ur","te",
]

def resolve_lang_set(name: str) -> List[str]:
    if name == "L20":
        return LANG_SET_20
    if name == "L60":
        return LANG_SET_60
    raise ValueError(f"Unknown lang_set: {name}")

# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------
def temperature_sampling(sizes: Dict[str, float], temp: float) -> Dict[str, float]:
    powered = {k: v ** (1.0 / temp) for k, v in sizes.items()}
    total = sum(powered.values())
    return {k: v / total for k, v in powered.items()}

# ---------------------------------------------------------------------
# CC100 Downloader
# ---------------------------------------------------------------------
class CC100Downloader:
    def __init__(
        self,
        lang: str,
        out_dir: Path,
        cap_bytes: int | None,
        min_lines: int,
        max_lines: int | None,
    ):
        self.lang = lang
        self.out_dir = out_dir
        self.cap_bytes = cap_bytes
        self.min_lines = min_lines
        self.max_lines = max_lines

    def run(self):
        meta = self.out_dir / "meta.json"
        if meta.exists():
            log.info(f"[{self.lang}] exists, skipping")
            return

        log.info(f"[{self.lang}] streaming from HF Hub")
        ds = load_dataset("SEACrowd/cc100", self.lang, streaming=True)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.out_dir / "train.txt"

        written = 0
        lines = 0
        pbar = tqdm(unit="lines", desc=self.lang)

        with open(out_file, "w", encoding="utf-8") as f:
            for ex in ds["train"]:
                if self.max_lines is not None and lines >= self.max_lines:
                    break

                txt = ex.get("text")
                if not txt:
                    continue
                if not txt.endswith("\n"):
                    txt += "\n"

                b = len(txt.encode("utf-8"))
                if self.cap_bytes is not None and written + b > self.cap_bytes and lines >= self.min_lines:
                    break

                f.write(txt)
                written += b
                lines += 1
                pbar.update(1)

        pbar.close()

        with open(meta, "w") as mf:
            json.dump(
                {
                    "language": self.lang,
                    "lines": lines,
                    "bytes": written,
                    "max_lines_cap": self.max_lines,
                },
                mf,
                indent=2,
            )

        log.info(f"[{self.lang}] done ({lines} lines)")

# ---------------------------------------------------------------------
# FLORES (optional)
# ---------------------------------------------------------------------
def download_flores(out_dir: Path, langs: List[str]):
    log.info("Downloading FLORES+ dev/devtest")
    for split in ["dev", "devtest"]:
        ds = load_dataset("openlanguagedata/flores_plus", split=split)
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        files = {lg: open(split_dir / f"{lg}.txt", "w", encoding="utf-8") for lg in langs}

        for ex in ds:
            for lg in langs:
                iso, script = LANGUAGE_METADATA[lg]["flores"].split("_")
                if ex["iso_639_3"] == iso and ex["iso_15924"] == script:
                    files[lg].write(ex["text"] + "\n")

        for f in files.values():
            f.close()

# ---------------------------------------------------------------------
# Push to HF
# ---------------------------------------------------------------------
def push_to_hf(local_dir: Path, repo_id: str, datasetdict_layout: bool = False):
    if datasetdict_layout:
        # Build DatasetDict
        data_dict = {}
        for lang_dir in local_dir.iterdir():
            if lang_dir.is_dir():
                txt_file = lang_dir / "train.txt"
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = [{"text": line.strip()} for line in f]
                data_dict[lang_dir.name] = Dataset.from_list(lines)
        dataset = DatasetDict(data_dict)
        dataset.push_to_hub(repo_id)
        log.info(f"Pushed DatasetDict to HF: {repo_id}")
    else:
        # Folder-per-language
        repo = Repository(local_dir, clone_from=repo_id, use_auth_token=True)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Upload CC100 corpus")
        repo.git_push()
        log.info(f"Pushed folder-per-language to HF: {repo_id}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_set", default="L60")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--total_size_gb", type=float, default=None)
    ap.add_argument("--max_lines_per_lang", type=int, default=None)
    ap.add_argument("--min_lines_floor", type=int, default=0)
    ap.add_argument("--main_sampling_temp", type=float, default=DEFAULT_MAIN_TEMP)
    ap.add_argument("--download_flores", type=str, default="false")
    ap.add_argument("--push_to_hf", type=str, default=None, help="HF repo ID (org/repo_name)")
    ap.add_argument("--datasetdict_layout", action="store_true", help="Use DatasetDict layout")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    # -----------------------
    # Validation
    # -----------------------
    if args.total_size_gb is None and args.max_lines_per_lang is None:
        raise ValueError("Must provide either --total_size_gb or --max_lines_per_lang")

    random.seed(args.seed)
    download_flores_flag = args.download_flores.lower() == "true"

    langs = resolve_lang_set(args.lang_set)
    sizes = {lg: LANGUAGE_METADATA[lg]["est_gb"] for lg in langs}
    props = temperature_sampling(sizes, args.main_sampling_temp)

    root = Path(args.output_dir)
    main_dir = root / "main_corpus"

    for lg in langs:
        if args.max_lines_per_lang is not None:
            cap_bytes = None
        else:
            cap_gb = min(props[lg] * args.total_size_gb, MAX_LANG_CAP * args.total_size_gb)
            cap_bytes = int(cap_gb * (1024 ** 3))

        CC100Downloader(
            lg,
            main_dir / lg,
            cap_bytes,
            args.min_lines_floor,
            args.max_lines_per_lang,
        ).run()

    if download_flores_flag:
        download_flores(root / "parallel_data", langs)
    else:
        log.info("Skipping FLORES download (--download_flores false)")

    if args.push_to_hf:
        push_to_hf(main_dir, args.push_to_hf, args.datasetdict_layout)

if __name__ == "__main__":
    main()
