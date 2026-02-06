#!/usr/bin/env python3
"""Run lm-eval-harness with the settings from eval.sh via the Python API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import lm_eval
import pycountry
from lm_eval.models.huggingface import HFLM
from omegaconf import DictConfig
from lm_eval.utils import handle_non_serializable


import logging

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


class LasTokenizerHFLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        las_tokenizer_main_path: str,
        las_lang: str,
        **kwargs,
    ) -> None:
        self._las_tokenizer_path = Path(las_tokenizer_main_path)
        self._las_lang = las_lang
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
        )
        las.load(str(base_path), {self._las_lang: str(language_path)})
        self._las = las

    def _las_preprocess(self, text: str) -> str:
        if self._las is None:
            return text
        _, tokens = self._las.best_language_encode(text)
        return tokens

    def _las_preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        if self._las is None:
            return texts
        _, encs = self._las.best_language_encode_batch(texts)
        rendered: list[list[str]] = []
        for text, enc in zip(texts, encs):
            if enc is None:
                _, tokens = self._las.best_language_encode(text)
                rendered.append(tokens)
            else:
                rendered.append(enc.tokens)
        return rendered

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        tokens = self._las_preprocess(string)
        if isinstance(tokens, str):
            return super().tok_encode(
                tokens,
                add_special_tokens=add_special_tokens,
                left_truncate_len=left_truncate_len,
                **kwargs,
            )
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(ids, int):
            ids = [ids]
        if left_truncate_len is not None:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self._las_preprocess_batch(strings)
        if processed and isinstance(processed[0], str):
            return super().tok_batch_encode(
                processed,
                padding_side=padding_side,
                left_truncate_len=left_truncate_len,
                truncation=truncation,
            )
        token_ids = []
        for tokens in processed:
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if isinstance(ids, int):
                ids = [ids]
            if left_truncate_len is not None:
                ids = ids[-left_truncate_len:]
            token_ids.append(ids)
        if truncation and left_truncate_len is None:
            max_len = getattr(self.tokenizer, "model_max_length", None)
            if max_len and max_len > 0:
                token_ids = [ids[-max_len:] for ids in token_ids]

        self.tokenizer.padding_side = padding_side
        batch = self.tokenizer.pad(
            {"input_ids": token_ids},
            padding=True,
            return_tensors="pt",
        )
        return batch["input_ids"], batch["attention_mask"]


def to_iso639_2(code: str) -> str:
    code = code.lower()
    entry = pycountry.languages.get(alpha_3=code)
    if not hasattr(entry, "alpha_2"):
        if code == "swh":
            return "sw"
        elif code == "zsm":
            return "ms"
        elif code == "fil":
            return "fil"  # Filipino uses 'fil' as alpha_2 code in global MMLU
        else:
            raise ValueError(f"Cannot convert language code: {code}")
    return entry.alpha_2


def build_mmlu_tasks(langs: list[str]) -> list[str]:
    return [f"global_mmlu_full_{to_iso639_2(lang)}" for lang in langs]


def build_belebele_tasks(langs: list[str]) -> list[str]:
    if "fil" in langs:
        langs.remove("fil")
        langs.append("tgl")  # belebele uses 'tgl' for Filipino
    return [f"belebele_{lang}_Latn" for lang in langs]


def build_piqa_tasks(langs: list[str]) -> list[str]:
    return [f"global_piqa_completions_{lang}_latn" for lang in langs]


def run_eval_one_model(model, model_args, cfg: DictConfig) -> None:
    langs = list(cfg.eval_langs)

    tasks_list = list(cfg.tasks)

    logging.info(f"Tasks to run: {tasks_list}")

    # run belebele and mmlu with configured few-shot
    results1 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_mmlu_tasks(langs),
            num_fewshot=5,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "mmlu" in tasks_list
        else None
    )

    results2 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_belebele_tasks(langs),
            num_fewshot=5,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "belebele" in tasks_list
        else None
    )

    # run piqa with configured few-shot
    results3 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_piqa_tasks(langs),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "piqa" in tasks_list
        else None
    )

    return results1, results2, results3


def run_eval(cfg: DictConfig) -> None:
    output_dir = Path(getattr(cfg, "output_path", "./eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = cfg.model
    if bool(cfg.eval_base.enabled):
        base_model_args = f"pretrained={cfg.model_args.pretrained}"

        logging.info(f"Evaluating base model: {model} with args: {base_model_args}")
        r1, r2, r3 = run_eval_one_model(model, base_model_args, cfg)

        output_path = output_dir / f"eval_results_base_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {"mmlu": r1, "belebele": r2, "piqa": r3},
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if bool(cfg.eval_ft.enabled):
        finetuned_model_args = f"pretrained={cfg.model_args.pretrained},peft={cfg.model_args.peft_original_tok}"
        logging.info(
            f"Evaluating finetuned model with original tokenizer: {model} with args: {finetuned_model_args}"
        )
        r1, r2, r3 = run_eval_one_model(model, finetuned_model_args, cfg=cfg)

        output_path = output_dir / f"eval_results_ft_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

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
                "las_lang": str(cfg.custom_tokenizer.las_lang),
            },
        )
        model = lm
        model_args = None

        logging.info(
            f"Evaluating finetuned model with LAS tokenizer: {model} with args: {model_args}"
        )
        r1, r2, r3 = run_eval_one_model(model, model_args, cfg)

        output_path = output_dir / f"eval_results_ft_las_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    logging.info("Evaluation complete.")
