#!/usr/bin/env python3
"""Evaluate a single from-scratch GPT-2 checkpoint on the morphological /
code downstream task set (MultiBLiMP, TurBLiMP, Global-PIQA, Belebele, +
HumanEval/MBPP for code track).

Unlike P4d's `run_eval.py` (which multiplexes 4 tokenizer modes against one
pretrained base + adapters), from-scratch trains a SEPARATE model per
tokenizer — so each model gets its own eval job. We wrap the appropriate
custom-tokenizer HFLM (BaseUnigramHFLM for bpe/base/pathpiece; LasTokenizerHFLM
for langspec) and run lm-eval-harness tasks directly.

Reuses:
  - `las_experiment.bpb_core.load_eval_model` (full-model checkpoint loader).
  - `las_experiment.downstream_eval` wrappers (BaseUnigramHFLM, LasTokenizerHFLM,
    task builders, provenance guard).
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.utils import handle_non_serializable

from las_experiment.bpb_core import load_eval_model
from las_experiment.downstream_eval import (
    BaseUnigramHFLM,
    LasTokenizerHFLM,
    _assert_checkpoint_provenance,
    build_multiblimp_tasks,
    build_turblimp_tasks,
    build_piqa_tasks,
    build_belebele_tasks,
)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Map from-scratch tokenizer_type → (las_mode for provenance, wrapper class).
LAS_MODE_BY_TYPE = {
    "bpe":       "original",
    "base":      "base",
    "pathpiece": "pathpiece",
    "langspec":  "langspec",
}
# Default tokenizer filename within `tokenizer_dir` for each non-langspec type.
TOK_FILE_BY_TYPE = {
    "bpe":       "bpe_tokenizer.json",
    "base":      "base_tokenizer.json",       # canonical (matches LAS dir naming)
    "pathpiece": "pathpiece_tokenizer.json",  # canonical
}


def _build_lm(model_path: str, tokenizer_dir: str, tokenizer_type: str,
               las_lang: str | None, lang_suffix: str):
    """Construct the HFLM-wrapped from-scratch model."""
    # Provenance gate: schema bridge from train_fromscratch_v2 must be intact.
    expected_las_dir = (None if tokenizer_type == "bpe"
                        else str(Path(tokenizer_dir).resolve()))
    _assert_checkpoint_provenance(
        ckpt_path=model_path,
        base_pretrained=str(Path(model_path).resolve()),
        expected_las_mode=LAS_MODE_BY_TYPE[tokenizer_type],
        expected_las_dir=expected_las_dir,
    )

    # Load the trained full-model checkpoint (no adapter).
    model, device = load_eval_model(
        base_model=str(Path(model_path).resolve()),
        peft_path=None,  # plain full-model save
    )

    # Wrap with the appropriate tokenizer-aware HFLM.
    common_kwargs = dict(
        pretrained=model,
        tokenizer=str(Path(model_path).resolve()),  # any path; native unused
    )
    if tokenizer_type == "langspec":
        if las_lang is None:
            raise ValueError("--las_lang is required for langspec mode")
        lm = LasTokenizerHFLM(
            las_tokenizer_main_path=str(Path(tokenizer_dir).resolve()),
            las_lang=las_lang,
            lang_suffix=lang_suffix,
            **common_kwargs,
        )
    else:
        tok_file = Path(tokenizer_dir) / TOK_FILE_BY_TYPE[tokenizer_type]
        if not tok_file.exists():
            raise FileNotFoundError(f"Tokenizer file missing: {tok_file}")
        lm = BaseUnigramHFLM(
            base_tokenizer_path=str(tok_file.resolve()),
            **common_kwargs,
        )
    return lm


# From-scratch models have a 512-token context (n_positions). HumanEval/MBPP
# default to max_gen_toks=1024, which lm-eval rejects (must be < model context).
# Cap generation so prompt(≤512-MAX_GEN_TOKS_CODE) + completion fit the context.
# 256 leaves 256 tokens for the prompt; HumanEval/MBPP completions are short, so
# the binding limit is prompt context — a known constraint of evaluating these
# short-context from-scratch models on code-gen (numbers are NOT comparable to
# long-context models, but ARE comparable across our 4 tokenizers). Note: a more
# efficient tokenizer fits more prompt text in the same 256 tokens — that is part
# of the tokenizer effect, not a confound to remove.
MAX_GEN_TOKS_CODE = 256
_CODE_FAMILIES = {"humaneval", "mbpp"}


def _run_tasks(lm, langs: list[str], tasks: list[str], output_path: Path,
                batch_size: str = "auto"):
    """Run each task family; lm-eval handles per-task task-name resolution."""
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict[str, object] = {}

    # Map task-family name -> (builder, num_fewshot).
    FAMILIES = {
        "multiblimp": (build_multiblimp_tasks, 0),
        "turblimp":   (lambda _ls: build_turblimp_tasks(), 0),
        "piqa":       (build_piqa_tasks, 0),
        "belebele":   (build_belebele_tasks, 5),
        # Code: HumanEval / MBPP are direct task names (no per-lang builder).
        "humaneval":  (lambda _ls: ["humaneval"], 0),
        "mbpp":       (lambda _ls: ["mbpp"], 3),
    }
    for fam in tasks:
        if fam not in FAMILIES:
            raise KeyError(f"Unknown task family: {fam}")
        builder, fs = FAMILIES[fam]
        # turblimp only fires when tur is in langs (the task is tur-only).
        if fam == "turblimp" and "tur" not in langs:
            results[fam] = None; continue
        task_list = builder(langs)
        if not task_list:
            results[fam] = None; continue
        logger.info(f"[{fam}] tasks={task_list} fewshot={fs}")
        # humaneval/mbpp execute generated code; lm-eval gates them behind
        # confirm_run_unsafe_code=True (in addition to HF_ALLOW_CODE_EVAL=1).
        # Harmless for the non-code (safe) task families. The user has accepted
        # the code-execution caveat for HumanEval/MBPP.
        # Cap generation length for code-gen families (512-ctx models).
        gen_kwargs = (f"max_gen_toks={MAX_GEN_TOKS_CODE}"
                      if fam in _CODE_FAMILIES else None)
        r = lm_eval.simple_evaluate(
            model=lm, model_args=None, tasks=task_list,
            num_fewshot=fs, batch_size=batch_size, log_samples=False,
            confirm_run_unsafe_code=True, gen_kwargs=gen_kwargs,
        )
        results[fam] = r

    out_json = output_path / f"eval_fromscratch_{ts}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=handle_non_serializable)
    logger.info(f"Wrote {out_json}")
    return out_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="From-scratch checkpoint dir (with run_metadata.json).")
    p.add_argument("--tokenizer_dir", required=True,
                   help="Dir containing bpe_tokenizer/unigram_base/pathpiece/langspec_*.json.")
    p.add_argument("--tokenizer_type", required=True,
                   choices=list(LAS_MODE_BY_TYPE))
    p.add_argument("--las_lang", default=None,
                   help="Primary language for langspec routing (required if langspec).")
    p.add_argument("--lang_suffix", default="_Latn",
                   help="Script suffix for the primary lang (_Latn/_Hang/_Arab/'').")
    p.add_argument("--langs", required=True,
                   help="Comma-separated eval_langs (e.g. eng,fin or eng,python).")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated task families (multiblimp,turblimp,"
                        "piqa,belebele,humaneval,mbpp).")
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", default="auto")
    args = p.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    lm = _build_lm(
        args.model_path, args.tokenizer_dir, args.tokenizer_type,
        args.las_lang, args.lang_suffix,
    )
    _run_tasks(lm, langs, tasks, Path(args.output_path), args.batch_size)


if __name__ == "__main__":
    main()
