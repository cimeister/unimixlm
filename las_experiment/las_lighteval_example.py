#!/usr/bin/env python3
"""
Small helper for LAS v* preprocessing and a minimal lighteval-style usage example.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydra.utils import get_original_cwd

from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from las_experiment.utils import build_las_preprocess


def example_lighteval_usage(cfg):
    """Minimal example showing how to plug LAS preprocessing into lighteval."""
    base_dir = Path(get_original_cwd())

    high = cfg.high
    low = cfg.low
    eval_base_model = bool(cfg.get("eval_base_model", True))
    model_id = cfg.base_model if eval_base_model else cfg.finetuned_model
    tasks = list(cfg.get("tasks", []))

    def _resolve_model_id(value: str | None) -> str | None:
        if value is None:
            return None
        path_value = Path(value)
        if path_value.is_absolute() and path_value.exists():
            return str(path_value)
        candidate = base_dir / path_value
        if candidate.exists():
            return str(candidate)
        return value

    model_id = _resolve_model_id(model_id)

    las_tokenizer_path = cfg.get("las_tokenizer_path", None)
    tokenizer_source = model_id
    if not las_tokenizer_path:
        tokenizer_source = (
            _resolve_model_id(cfg.get("finetuned_model", None)) or model_id
        )

    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )

    las_preprocess = None
    if las_tokenizer_path:
        las_dir = Path(las_tokenizer_path)
        if not las_dir.is_absolute():
            las_dir = base_dir / las_dir

        base_tokenizer_path = las_dir / "shared_base/tokenizer.json"
        language_paths = {
            high: str(las_dir / f"langspec_soft_{high}.tokenizer.json"),
            low: str(las_dir / f"langspec_soft_{low}.tokenizer.json"),
        }

        if not base_tokenizer_path.exists():
            raise FileNotFoundError(
                f"Base LAS tokenizer not found: {base_tokenizer_path}"
            )
        for lang, path in language_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"LAS tokenizer for {lang} not found: {path}")

        shared_las = LanguageSpecificUnigramLMTokenizer(
            vocab_size=len(hf_tokenizer), byte_level=True
        )
        shared_las.load(str(base_tokenizer_path), language_paths)

        las_preprocess = build_las_preprocess(
            shared_las=shared_las,
            hf_tokenizer=hf_tokenizer,
            length_prior_weights=cfg.get("length_prior_weights", None),
        )
    else:
        las_preprocess = (
            lambda x: x
            if getattr(hf_tokenizer, "byte_level", False)
            else x.strip().lower()
        )

    class LASEvalTokenizer:
        def __init__(self, base_tokenizer, preprocess_fn):
            self.base_tokenizer = base_tokenizer
            self.preprocess_fn = preprocess_fn

        def __call__(self, text, **kwargs):
            return self.base_tokenizer(self.preprocess_fn(text), **kwargs)

        @property
        def pad_token_id(self):
            return self.base_tokenizer.pad_token_id

        @property
        def eos_token_id(self):
            return self.base_tokenizer.eos_token_id

    las_eval_tokenizer = LASEvalTokenizer(hf_tokenizer, las_preprocess)

    # Example lighteval usage (API may vary with version):
    # from lighteval import evaluator
    # evaluator.run(
    #     model_name_or_path=str(model_id),
    #     tasks=tasks,
    #     tokenizer=las_eval_tokenizer,
    # )

    _ = las_eval_tokenizer("Hello world!", return_tensors="pt")
