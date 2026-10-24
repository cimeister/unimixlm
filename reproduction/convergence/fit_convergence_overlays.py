#!/usr/bin/env python3
"""Fit the per-(language, N) LangMAP overlays for the convergence table,
with the same protocol as the F15 production overlays
(scripts/prep_f15_tokenizers.py): the shared Qwen2.5-0.5B F15
base_tokenizer.json (not refit), per-language EM seeded from the base
tokenizer's own trained log-score for each piece (BaseScoreSeededLASTokenizer,
imported unmodified from prep_f15_tokenizers.py), K=10 EM iterations
(code/constants.py NUM_FINETUNE_EM_ITERATIONS, asserted below), and the same
script filter (already applied when the corpora were built by
build_convergence_corpora.py).

Only the estimation-corpus size varies across cells; everything else is held
fixed. Reads the per-language corpora_manifest_<lang>.json files written by
build_convergence_corpora.py and writes the overlays plus fit_manifest.json.

CPU-only. Each cell's EM re-estimation is an external `spm_train` call
capped at --num_threads=20 (see BaseScoreSeededLASTokenizer).

Environment:
  CONVERGENCE_WORK_DIR   directory holding the corpora manifests; overlays and
                         fit_manifest.json are written there (required)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from code.constants import NUM_FINETUNE_EM_ITERATIONS  # noqa: E402
from las_experiment.tokenizer_training.validation import validate_tokenizers  # noqa: E402
from prep_f15_tokenizers import (  # noqa: E402
    BaseScoreSeededLASTokenizer,
    encode_probe,
    verify_and_patch_specials,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fit_convergence_overlays")


def _env_dir(var: str, what: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set; point it at {what}.")
    return Path(value)


OUT_DIR = _env_dir("CONVERGENCE_WORK_DIR", "the directory holding the convergence corpora manifests")
OVERLAY_DIR = OUT_DIR / "overlays"
BASE_TOKENIZER_PATH = REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/base_tokenizer.json"
MODEL_TOKENIZER_DIR = REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/model_tokenizer"
F15_MANIFEST_PATH = REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/manifest.json"

LANGS = ["deu", "fin", "tur"]
POINT_LABELS = ["N125", "N500", "N1000", "N4000", "N16000", "N32000", "prod25M"]

assert NUM_FINETUNE_EM_ITERATIONS == 10, (
    f"code/constants.py NUM_FINETUNE_EM_ITERATIONS={NUM_FINETUNE_EM_ITERATIONS}, "
    f"expected 10 (same protocol as scripts/prep_f15_tokenizers.py). Aborting "
    f"rather than silently fitting with a different EM iteration count."
)


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def fit_one(lang: str, lang_code: str, point_label: str, corpus_path: str,
            model_tok, model_tok_json: dict, embed_rows: int, unk_token: str) -> dict:
    lang_dir = OVERLAY_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = lang_dir / f"langspec_{lang_code}_{point_label}.json"

    t0 = time.time()
    if overlay_path.exists():
        logger.info(f"[{lang}/{point_label}] {overlay_path} already exists, reusing")
    else:
        trainer = BaseScoreSeededLASTokenizer(
            reestimation_em_mode="soft",
            unk_token=unk_token,
        )
        assert trainer.num_reestimation_iterations == NUM_FINETUNE_EM_ITERATIONS
        trainer.train(
            corpus_info={lang_code: corpus_path},
            base_tokenizer_path=str(BASE_TOKENIZER_PATH),
            tokenizer_path_format=str(overlay_path),
            load_base_if_exists=True,
            load_tokenizers_if_exists=False,
            use_sentencepiece=True,
            max_workers=1,
        )
    fit_time_s = time.time() - t0

    # Same validation gates as prep_f15_tokenizers.py: the base tokenizer and
    # every per-language overlay must agree on vocabulary and special tokens.
    validate_tokenizers(
        MODEL_TOKENIZER_DIR / "tokenizer.json", BASE_TOKENIZER_PATH,
        {lang_code: overlay_path},
    )
    specials = verify_and_patch_specials(overlay_path, model_tok, model_tok_json, embed_rows)
    tok = Tokenizer.from_file(str(overlay_path))
    encode_probe(tok, Path(corpus_path), embed_rows)

    result = {
        "path": str(overlay_path),
        "sha256": sha256_path(overlay_path),
        "fit_time_s": fit_time_s,
        **specials,
    }
    logger.info(f"[{lang}/{point_label}] fit OK in {fit_time_s:.1f}s: {result}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, default=None, help="Restrict to one language")
    ap.add_argument("--points", nargs="*", default=None,
                    help=f"Restrict to specific point labels (default all: {POINT_LABELS})")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: fin, N125 only, ignores --lang/--points")
    args = ap.parse_args()

    f15_manifest = json.loads(F15_MANIFEST_PATH.read_text())
    embed_rows = f15_manifest["embed_rows"]

    model_tok = AutoTokenizer.from_pretrained(str(MODEL_TOKENIZER_DIR))
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token
    unk_token = model_tok.unk_token if model_tok.unk_token else model_tok.eos_token
    model_tok_json = json.loads((MODEL_TOKENIZER_DIR / "tokenizer.json").read_text())

    if args.smoke:
        jobs = [("fin", "N125")]
    else:
        langs = [args.lang] if args.lang else LANGS
        points = args.points if args.points else POINT_LABELS
        jobs = [(lang, pt) for lang in langs for pt in points]

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    t_start = time.time()
    for lang, point_label in jobs:
        corpora_manifest = json.loads((OUT_DIR / f"corpora_manifest_{lang}.json").read_text())
        lang_code = corpora_manifest["lang_code"]
        point_info = corpora_manifest["points"][point_label]
        corpus_path = point_info["path"]

        result = fit_one(
            lang, lang_code, point_label, corpus_path,
            model_tok, model_tok_json, embed_rows, unk_token,
        )
        result["corpus"] = point_info
        all_results.setdefault(lang, {})[point_label] = result

        out_path = OUT_DIR / "fit_manifest.json"
        if out_path.exists():
            merged = json.loads(out_path.read_text())
        else:
            merged = {}
        merged.setdefault(lang, {})[point_label] = result
        out_path.write_text(json.dumps(merged, indent=2))

    total_time = time.time() - t_start
    logger.info(f"=== Done: {len(jobs)} cell(s) fit in {total_time:.1f}s total ===")


if __name__ == "__main__":
    main()
