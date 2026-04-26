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

from tokenizers import Tokenizer


class LasTokenizerHFLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        las_tokenizer_main_path: str,
        las_lang: str,
        las_mode: str = "langspec",
        las_domain: str = "natural_language",
        **kwargs,
    ) -> None:
        self._las_tokenizer_path = Path(las_tokenizer_main_path)
        self._las_lang = las_lang
        self._las_mode = las_mode
        self._las_domain = las_domain
        self._las_tokenizer = None

        super().__init__(pretrained=pretrained, **kwargs)
        self._init_las_tokenizer()

    def _init_las_tokenizer(self) -> None:
        if self._las_mode == "original":
            self._las_tokenizer = None
            return

        if not self._las_tokenizer_path:
            raise ValueError(
                "las_tokenizer_path must be set when las_mode is not 'original'."
            )

        if self._las_mode == "base":
            path = self._las_tokenizer_path / "base_tokenizer.json"
            if not path.exists():
                raise FileNotFoundError(f"LAS base tokenizer not found: {path}")
            self._las_tokenizer = Tokenizer.from_file(str(path))

        elif self._las_mode == "langspec":
            if self._las_domain != "code" and self._las_lang == "eng":
                self._las_tokenizer = None
                return
            if self._las_domain == "code":
                path = self._las_tokenizer_path / f"langspec_{self._las_lang}.json"
            else:
                path = self._las_tokenizer_path / f"langspec_{self._las_lang}_Latn.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"LAS language-specific tokenizer not found: {path}"
                )
            self._las_tokenizer = Tokenizer.from_file(str(path))

        else:
            raise ValueError(
                f"Unknown las_mode: {self._las_mode}. Use 'original', 'base', or 'langspec'."
            )

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        if self._las_tokenizer is None:
            return super().tok_encode(
                string,
                add_special_tokens=add_special_tokens,
                left_truncate_len=left_truncate_len,
                **kwargs,
            )
        ids = self._las_tokenizer.encode(string, add_special_tokens=False).ids
        if left_truncate_len is not None:
            ids = ids[-left_truncate_len:]
        return list(ids)

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._las_tokenizer is None:
            return super().tok_batch_encode(
                strings,
                padding_side=padding_side,
                left_truncate_len=left_truncate_len,
                truncation=truncation,
            )
        encodings = self._las_tokenizer.encode_batch(strings, add_special_tokens=False)
        token_ids = [list(enc.ids) for enc in encodings]
        if left_truncate_len is not None:
            token_ids = [ids[-left_truncate_len:] for ids in token_ids]
        if truncation:
            max_len = getattr(self.tokenizer, "model_max_length", None)
            if max_len and max_len > 0:
                token_ids = [ids[-max_len:] for ids in token_ids]
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        batch = self.tokenizer.pad(
            {"input_ids": token_ids},
            padding=True,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = old_padding_side
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
    langs = [
        ("tgl" if la == "fil" else la) for la in langs
    ]  # belebele uses 'tgl' for Filipino
    return [f"belebele_{lang}_Latn" for lang in langs]


def build_piqa_tasks(langs: list[str]) -> list[str]:
    langs = [
        ("tgl" if la == "fil" else la) for la in langs
    ]  # global_piqa uses 'tgl' for Filipino
    return [f"global_piqa_completions_{lang}_latn" for lang in langs]


# ----------------------------------------------------------------------------
# EMNLP main-paper compositional / NER evals.
#
# All three tasks are wired through lm-eval-harness. Task IDs follow these
# canonical conventions (verify availability per harness version):
#
#   XNLI    : xnli_<iso639-1>           e.g. xnli_tr
#   WikiANN : wikiann_<iso639-1>        e.g. wikiann_fi
#   MultiBLiMP : multiblimp_<iso639-3>  e.g. multiblimp_hun
#
# Only XNLI(Turkish), WikiANN(Turkish/Finnish/Hungarian), and MultiBLiMP for
# the three Tier-2 languages are guaranteed by the harness. For any (task,
# language) the harness does not implement, the call returns an error which
# we surface to the caller — driver scripts should filter unsupported pairs
# upstream rather than hardcoding a whitelist here.
# ----------------------------------------------------------------------------


def build_xnli_tasks(langs: list[str]) -> list[str]:
    """XNLI uses 2-letter codes (tr, fi, hu, etc.)."""
    return [f"xnli_{to_iso639_2(lang)}" for lang in langs]


def build_wikiann_tasks(langs: list[str]) -> list[str]:
    """WikiANN NER uses 2-letter codes."""
    return [f"wikiann_{to_iso639_2(lang)}" for lang in langs]


def build_multiblimp_tasks(langs: list[str]) -> list[str]:
    """MultiBLiMP uses 3-letter (ISO 639-3) codes; pass through verbatim."""
    return [f"multiblimp_{lang.lower()}" for lang in langs]


def run_eval_one_model(model, model_args, cfg: DictConfig) -> tuple:
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

    # XNLI — 0-shot natural-language inference (Turkish in our suite).
    results_xnli = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_xnli_tasks(langs),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "xnli" in tasks_list
        else None
    )

    # WikiANN — span-level NER, 0-shot.
    results_wikiann = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_wikiann_tasks(langs),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "wikiann" in tasks_list
        else None
    )

    # MultiBLiMP — minimal-pair grammaticality, 0-shot.
    results_multiblimp = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_multiblimp_tasks(langs),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "multiblimp" in tasks_list
        else None
    )

    return (
        results1,
        results2,
        results3,
        results_xnli,
        results_wikiann,
        results_multiblimp,
    )


def run_eval(cfg: DictConfig) -> None:
    output_dir = Path(getattr(cfg, "output_path", "./eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = cfg.model
    dtype = cfg.get("dtype", "auto")
    if bool(cfg.eval_base.enabled):
        base_model_args = f"pretrained={cfg.model_args.pretrained},dtype={dtype}"

        logging.info(f"Evaluating base model: {model} with args: {base_model_args}")
        r1, r2, r3, r_xnli, r_wikiann, r_mblimp = run_eval_one_model(
            model, base_model_args, cfg
        )

        output_path = output_dir / f"eval_results_base_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                    "xnli": r_xnli,
                    "wikiann": r_wikiann,
                    "multiblimp": r_mblimp,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if bool(cfg.eval_ft.enabled):
        finetuned_model_args = f"pretrained={cfg.model_args.pretrained},peft={cfg.model_args.peft_original_tok},dtype={dtype}"
        logging.info(
            f"Evaluating finetuned model with original tokenizer: {model} with args: {finetuned_model_args}"
        )
        r1, r2, r3, r_xnli, r_wikiann, r_mblimp = run_eval_one_model(
            model, finetuned_model_args, cfg=cfg
        )

        output_path = output_dir / f"eval_results_ft_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                    "xnli": r_xnli,
                    "wikiann": r_wikiann,
                    "multiblimp": r_mblimp,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    finetuned_model_las_args = f"pretrained={cfg.model_args.pretrained},peft={cfg.model_args.peft_las_tok},dtype={dtype}"

    if cfg.get("custom_tokenizer") and cfg.custom_tokenizer.enabled:
        lm = LasTokenizerHFLM.create_from_arg_string(
            finetuned_model_las_args,
            additional_config={
                "las_tokenizer_main_path": str(
                    cfg.custom_tokenizer.las_tokenizer_main_path
                ),
                "las_lang": str(cfg.custom_tokenizer.las_lang),
                "las_mode": str(cfg.custom_tokenizer.get("las_mode", "langspec")),
            },
        )
        model = lm
        model_args = None

        logging.info(
            f"Evaluating finetuned model with LAS tokenizer: {model} with args: {model_args}"
        )
        r1, r2, r3, r_xnli, r_wikiann, r_mblimp = run_eval_one_model(
            model, model_args, cfg
        )

        output_path = output_dir / f"eval_results_ft_las_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                    "xnli": r_xnli,
                    "wikiann": r_wikiann,
                    "multiblimp": r_mblimp,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    logging.info("Evaluation complete.")
