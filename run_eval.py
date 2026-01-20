#!/usr/bin/env python3
"""Run lm-eval-harness with the settings from eval.sh via the Python API."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import hydra
import lm_eval
import pycountry
from lm_eval.models.huggingface import HFLM
from omegaconf import DictConfig

import logging

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


class LasTokenizerHFLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        las_tokenizer_main_path: str,
        las_lang: str,
        las_byte_level: bool = True,
        las_lowercase_if_not_byte_level: bool = True,
        **kwargs,
    ) -> None:
        self._las_tokenizer_path = Path(las_tokenizer_main_path)
        self._las_lang = las_lang
        self._las_byte_level = las_byte_level
        self._las_lowercase_if_not_byte_level = las_lowercase_if_not_byte_level
        self._las = None

        super().__init__(pretrained=pretrained, **kwargs)
        self._init_las_tokenizer()

    def _init_las_tokenizer(self) -> None:
        if not self._las_tokenizer_path:
            raise ValueError(
                "las_tokenizer_path must be set when custom tokenizer is enabled."
            )

        base_path = self._las_tokenizer_path / "base.tokenizer.json"
        language_path = (
            self._las_tokenizer_path
            / f"langspec_soft_{self._las_lang}_Latn.tokenizer.json"
        )

        if not base_path.exists():
            raise FileNotFoundError(f"LAS base tokenizer not found: {base_path}")
        if not language_path.exists():
            raise FileNotFoundError(
                f"LAS language-specific tokenizer not found: {language_path}"
            )

        las = LanguageSpecificUnigramLMTokenizer(
            vocab_size=len(self.tokenizer),
            byte_level=self._las_byte_level,
        )
        las.load(str(base_path), [str(language_path)])
        self._las = las

    def _base_preprocess(self, text: str) -> str:
        if self._las_lowercase_if_not_byte_level and not getattr(
            self.tokenizer, "byte_level", False
        ):
            return text.strip().lower()
        return text

    def _las_preprocess(self, text: str) -> str:
        if self._las is None:
            return text
        text = self._base_preprocess(text)
        _, tokens = self._las.best_language_encode(text)
        return " ".join(tokens)

    def _las_preprocess_batch(self, texts: list[str]) -> list[str]:
        if self._las is None:
            return texts
        preprocessed = [self._base_preprocess(t) for t in texts]
        _, encs = self._las.best_language_encode_batch(preprocessed)
        rendered = []
        for text, enc in zip(preprocessed, encs):
            if enc is None:
                _, tokens = self._las.best_language_encode(text)
                rendered.append(" ".join(tokens))
            else:
                rendered.append(" ".join(enc.tokens))
        return rendered

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        return super().tok_encode(
            self._las_preprocess(string),
            add_special_tokens=add_special_tokens,
            left_truncate_len=left_truncate_len,
            **kwargs,
        )

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self._las_preprocess_batch(strings)
        return super().tok_batch_encode(
            processed,
            padding_side=padding_side,
            left_truncate_len=left_truncate_len,
            truncation=truncation,
        )


def to_iso639_3(code: str) -> str:
    code = code.lower()
    entry = pycountry.languages.get(alpha_2=code)
    return entry.alpha_3


def build_mmlu_tasks(langs: list[str]) -> list[str]:
    return [f"global_mmlu_full_{lang}" for lang in langs]


def build_belebele_tasks(langs: list[str]) -> list[str]:
    return [f"belebele_{to_iso639_3(lang)}_Latn" for lang in langs]


def build_piqa_tasks(langs: list[str]) -> list[str]:
    return [f"global_piqa_completions_{to_iso639_3(lang)}_latn" for lang in langs]


def run_eval_one_model(model, model_args, cfg: DictConfig) -> None:
    langs = list(cfg.eval_langs)
    mmlu_tasks = build_mmlu_tasks(langs)
    belebele_tasks = build_belebele_tasks(langs)
    piqa_tasks = build_piqa_tasks(langs)

    logging.info(f"Tasks to run: {mmlu_tasks + belebele_tasks + piqa_tasks}")

    # run belebele and mmlu with configured few-shot
    results1 = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=mmlu_tasks + belebele_tasks,
        num_fewshot=5,
        batch_size=str(cfg.batch_size),
        log_samples=bool(cfg.log_samples),
        limit=10,  # TODO remove after testing
    )

    # run piqa with configured few-shot
    results2 = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=piqa_tasks,
        num_fewshot=0,
        batch_size=str(cfg.batch_size),
        log_samples=bool(cfg.log_samples),
        limit=10,  # TODO remove after testing
    )

    return results1, results2


@hydra.main(config_path="config/eval", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    eval_results = {"config": cfg}

    if cfg.get("cuda_visible_devices"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.cuda_visible_devices))

    model = cfg.model
    base_model_args = f"pretrained={cfg.model_args.pretrained}"

    logging.info(f"Evaluating base model: {model} with args: {base_model_args}")
    r1, r2 = run_eval_one_model(model, base_model_args, cfg)
    eval_results["base_model"] = {"mmlu_belebele": r1, "piqa": r2}

    finetuned_model_args = f"pretrained={cfg.model_args.pretrained},peft={cfg.model_args.peft_original_tok}"
    logging.info(
        f"Evaluating finetuned model with original tokenizer: {model} with args: {finetuned_model_args}"
    )
    r1, r2 = run_eval_one_model(model, finetuned_model_args, cfg=cfg)
    eval_results["finetuned_model_original_tokenizer"] = {
        "mmlu_belebele": r1,
        "piqa": r2,
    }

    finetuned_model_las_args = (
        f"pretrained={cfg.model_args.pretrained},peft={cfg.model_args.peft_las_tok}"
    )

    if cfg.get("custom_tokenizer") and cfg.custom_tokenizer.enabled:
        lm = LasTokenizerHFLM.create_from_arg_string(
            finetuned_model_las_args,
            additional_config={
                "las_tokenizer_main_path": str(
                    cfg.custom_tokenizer.las_tokenizer_main_path
                ),
                "las_lang": list(cfg.custom_tokenizer.las_lang),
                "las_byte_level": bool(cfg.custom_tokenizer.byte_level),
                "las_lowercase_if_not_byte_level": bool(
                    cfg.custom_tokenizer.lowercase_if_not_byte_level
                ),
            },
        )
        model = lm
        model_args = None

        logging.info(
            f"Evaluating finetuned model with LAS tokenizer: {model} with args: {model_args}"
        )
        r1, r2 = run_eval_one_model(model, model_args, cfg)
        eval_results["finetuned_model_las_tokenizer"] = {
            "mmlu_belebele": r1,
            "piqa": r2,
        }

    # save results with date and time stamp
    output_dir = Path(getattr(cfg, "output_path", "./eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_results_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
