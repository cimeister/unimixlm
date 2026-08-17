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
import os

from tokenizers import Tokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


def _is_peft_checkpoint(ckpt_path: str) -> bool:
    """Return True if the checkpoint is a PEFT adapter (has adapter_config.json)."""
    return os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))


def _find_run_metadata(ckpt_path: str):
    """run_metadata.json lives in the ft* dir; the eval may point at the
    ft* dir itself or a checkpoint-N subdir. Search both. None if absent."""
    for cand in (
        os.path.join(ckpt_path, "run_metadata.json"),
        os.path.join(os.path.dirname(ckpt_path.rstrip("/")), "run_metadata.json"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _assert_checkpoint_provenance(
    ckpt_path: str,
    base_pretrained: str,
    expected_las_mode: str,
    expected_las_dir: str | None = None,
) -> None:
    """CRITICAL-2 + eos-bug guard (2026-05-17). Refuse to evaluate a checkpoint
    whose run_metadata.json does not match the eval config. No silent trust of
    the Hydra YAML — a stale pairing produces plausible but meaningless numbers.

    Also auto-refuses eos-bugged checkpoints: `model_tokenizer_path != model.name`
    means eos/pad were resolved from a foreign tokenizer
    (see memory project_qwen3_eos_model_tokenizer_bug). The invariant
    model.name == model_tokenizer_path holds for every clean cell and ONLY
    differs for the confounded ones.
    """
    meta_path = _find_run_metadata(ckpt_path)
    if meta_path is None:
        raise FileNotFoundError(
            f"No run_metadata.json near {ckpt_path} — cannot verify checkpoint "
            f"provenance. Refusing to evaluate (no silent trust of the config)."
        )
    meta = json.load(open(meta_path))
    m = meta.get("model", {})
    t = meta.get("tokenizer", {})
    errs = []
    if m.get("name") != base_pretrained:
        errs.append(
            f"base model: meta={m.get('name')!r} vs eval={base_pretrained!r}"
        )
    if t.get("las_mode") != expected_las_mode:
        errs.append(
            f"las_mode: meta={t.get('las_mode')!r} vs eval={expected_las_mode!r}"
        )
    if expected_las_dir is not None:
        a = str(Path(t.get("las_tokenizer_path", "")).resolve())
        b = str(Path(expected_las_dir).resolve())
        if a != b:
            errs.append(f"las_tokenizer_path: meta={a!r} vs eval={b!r}")
    mtp = t.get("model_tokenizer_path")
    if mtp is not None and m.get("name") is not None and mtp != m.get("name"):
        errs.append(
            f"EOS-BUG: checkpoint trained with model_tokenizer_path={mtp!r} != "
            f"model.name={m.get('name')!r} — eos/pad resolved from a foreign "
            f"tokenizer (project_qwen3_eos_model_tokenizer_bug). REFUSING."
        )
    if errs:
        raise ValueError(
            f"Checkpoint provenance mismatch for {ckpt_path} "
            f"(metadata={meta_path}): " + "; ".join(errs)
        )


def _assert_ids_valid(tokens, ids) -> None:
    """HIGH-1: a routed token string absent from the base HF vocab makes
    `convert_tokens_to_ids` return the unk id (None for Qwen3 → unk_token=None),
    silently corrupting the id stream. Fail loud naming the offenders rather
    than feeding None / a length-mismatched list into lm-eval."""
    if len(ids) != len(tokens):
        raise ValueError(
            f"convert_tokens_to_ids length mismatch: {len(tokens)} tokens -> "
            f"{len(ids)} ids (routed-token / base-vocab misalignment)."
        )
    if any(i is None for i in ids):
        bad = [t for t, i in zip(tokens, ids) if i is None][:10]
        raise ValueError(
            f"convert_tokens_to_ids returned None for routed token(s) {bad!r} "
            f"— not in the base HF vocab. Refusing to feed unk/None to lm-eval."
        )


def _build_eval_lm_kwargs(
    base_pretrained: str,
    ckpt_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    expected_las_mode: str | None = None,
    expected_las_dir: str | None = None,
) -> dict:
    """Build HFLM kwargs by *pre-loading* the model and tokenizer ourselves.

    This avoids two lm-eval auto-loading bugs that affect our checkpoints:

    1. PEFT shape mismatch (LoRA+embed unfrozen): when the base model's
       config vocab_size > tokenizer vocab_size (e.g. Qwen3-0.6B has
       151936 model slots vs 151669 tokenizer entries), HFLM resizes
       embeddings *down* before loading the PEFT adapter — but the saved
       adapter has `embed_tokens=[151936, hidden]` and `lm_head=[151936,
       hidden]` because we trained at the original model vocab size, so
       the load fails with a size mismatch.

    2. Embed-only checkpoints contain only `model.safetensors` and
       `config.json` — no tokenizer files. lm-eval's auto-tokenizer-from-
       checkpoint-path then crashes inside `Qwen2Tokenizer.__init__` with
       `expected str, ..., not NoneType`.

    Both are fixed by passing a pre-built `PreTrainedModel` instance as
    `pretrained=...` (HFLM detects this and skips `_create_model`,
    huggingface.py:115-125) along with a `tokenizer=<base_pretrained>`
    string so lm-eval loads the tokenizer from the *base* model rather
    than the checkpoint dir.
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # CRITICAL-2 + eos-bug guard: verify checkpoint provenance before loading.
    if expected_las_mode is not None:
        _assert_checkpoint_provenance(
            ckpt_path, base_pretrained, expected_las_mode, expected_las_dir
        )

    torch_dtype = getattr(_torch, dtype) if isinstance(dtype, str) else dtype

    # A4 (las_mode="vocab_extend"): the checkpoint was trained after
    # `resize_token_embeddings`, so its saved `embed_tokens` (and `lm_head`)
    # have MORE rows than the base model. The base model must be resized to the
    # same size before the adapter loads, otherwise PEFT raises a shape
    # mismatch. The target size comes from the run's own provenance record —
    # never from the eval config, so it cannot disagree with what trained.
    resize_to = None
    if expected_las_mode == "vocab_extend":
        meta_path = _find_run_metadata(ckpt_path)
        if meta_path is None:
            raise FileNotFoundError(
                f"las_mode='vocab_extend' eval needs run_metadata.json near "
                f"{ckpt_path} to recover the extended vocabulary size."
            )
        _ve = (json.load(open(meta_path)).get("tokenizer") or {}).get("vocab_extend")
        if not _ve or _ve.get("n_total_vocab") is None:
            raise ValueError(
                f"{meta_path} has no tokenizer.vocab_extend.n_total_vocab — the "
                f"checkpoint was not produced by the vocab_extend path. Refusing "
                f"to guess the resize target."
            )
        resize_to = int(_ve["n_total_vocab"])

    if _is_peft_checkpoint(ckpt_path):
        model = AutoModelForCausalLM.from_pretrained(
            base_pretrained, dtype=torch_dtype
        )
        if resize_to is not None:
            model.resize_token_embeddings(resize_to, mean_resizing=False)
            if model.get_input_embeddings().weight.shape[0] != resize_to:
                raise RuntimeError(
                    f"resize to {resize_to} rows failed before adapter load."
                )
        model = PeftModel.from_pretrained(model, ckpt_path)

        # Two adapter shapes are produced by training:
        #   (A) Legacy bug-4 untied runs: both `embed_tokens` and `lm_head` are
        #       in `modules_to_save`. They were trained as independent copies.
        #   (B) Tied-fix runs (post-2026-04-13 fix for tied-embedding models
        #       like Qwen3/Gemma3): only `embed_tokens` is in `modules_to_save`.
        #       The trained input embedding should ALSO feed the output
        #       projection via the tie.
        #
        # For (B), PEFT wraps `embed_tokens` in `ModulesToSaveWrapper` holding
        # the trained copy at `modules_to_save["default"].weight`. But `lm_head`
        # is unwrapped — its `weight` still points at the ORIGINAL (un-trained)
        # input embedding via the natural tie. Without a manual fix, eval would
        # use trained embeddings for input lookup and untrained weights for
        # output projection — garbage logits.
        #
        # Detect (B) and re-tie lm_head onto the trained copy. (A) is unchanged
        # because lm_head has its own trained ModulesToSaveWrapper that's used
        # directly by the forward pass.
        inner = getattr(getattr(model, "base_model", None), "model", None)
        if inner is not None:
            emb_module = inner.get_input_embeddings()
            is_emb_wrapped = hasattr(emb_module, "modules_to_save")
            lm_head_module = getattr(inner, "lm_head", None)
            is_lm_head_wrapped = (
                lm_head_module is not None
                and hasattr(lm_head_module, "modules_to_save")
            )
            # Guard against natively-untied models (e.g., OLMo) accidentally
            # getting re-tied: only re-tie when the base model's config flag
            # says the model was originally tied.
            inner_was_tied = bool(getattr(inner.config, "tie_word_embeddings", False))
            if (
                is_emb_wrapped
                and not is_lm_head_wrapped
                and lm_head_module is not None
                and inner_was_tied
            ):
                # Tied-fix run — share the trained embedding Parameter with lm_head.
                active = getattr(model, "active_adapter", "default")
                if isinstance(active, list):
                    active = active[0] if active else "default"
                adapter_key = active if active in emb_module.modules_to_save else next(
                    iter(emb_module.modules_to_save)
                )
                trained_param = emb_module.modules_to_save[adapter_key].weight
                lm_head_module.weight = trained_param

        # PEFT issue #2777: HFLM unconditionally calls `model.tie_weights()`
        # after init, which crashes inside `_tie_or_clone_weights` because
        # PEFT's `ModulesToSaveWrapper` exposes a `weight` attribute that the
        # tie logic mis-handles. Disable the auto-tie path on every relevant
        # config. For tied-fix runs we've already shared the trained Parameter
        # with `lm_head` above; for legacy untied runs `lm_head` has its own
        # trained ModulesToSaveWrapper. Either way, the forward pass uses the
        # right trained weights regardless of `tie_word_embeddings`.
        for cfg_obj in (
            getattr(model, "config", None),
            getattr(getattr(model, "base_model", None), "config", None),
            getattr(
                getattr(getattr(model, "base_model", None), "model", None),
                "config",
                None,
            ),
        ):
            if cfg_obj is not None and hasattr(cfg_obj, "tie_word_embeddings"):
                cfg_obj.tie_word_embeddings = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch_dtype
        )
        if resize_to is not None and (
            model.get_input_embeddings().weight.shape[0] != resize_to
        ):
            raise ValueError(
                f"Full checkpoint {ckpt_path} has "
                f"{model.get_input_embeddings().weight.shape[0]} embedding rows "
                f"but run_metadata says the extended vocabulary is {resize_to}."
            )

    model.to(device)
    model.eval()

    return {"pretrained": model, "tokenizer": base_pretrained}


class LasTokenizerHFLM(HFLM):
    def __init__(
        self,
        pretrained,  # str | transformers.PreTrainedModel
        las_tokenizer_main_path: str,
        las_lang: str,
        lang_suffix: str = "_Latn",
        **kwargs,
    ) -> None:
        self._las_tokenizer_path = Path(las_tokenizer_main_path)
        self._las_lang = las_lang
        self._lang_suffix = lang_suffix
        self._las = None

        super().__init__(pretrained=pretrained, **kwargs)
        self._init_las_tokenizer()

    def _init_las_tokenizer(self) -> None:
        if not self._las_tokenizer_path:
            raise ValueError(
                "las_tokenizer_path must be set when custom tokenizer is enabled."
            )

        base_path = self._las_tokenizer_path / "base_tokenizer.json"
        if not base_path.exists():
            raise FileNotFoundError(f"LAS base tokenizer not found: {base_path}")

        # Load EVERY langspec_*.json file in the tokenizer directory so that
        # `best_language_encode_batch` actually performs routing. Loading only
        # one langspec file (the prior behavior) made the LAS inference scheme
        # an oracle eval — routing degenerates to "pick the one loaded lang".
        # The paper's central claim is about LAS-at-inference, so the routing
        # pool must be the multi-lang set the tokenizer dir was built for.
        langspec_files = sorted(self._las_tokenizer_path.glob("langspec_*.json"))
        if not langspec_files:
            raise FileNotFoundError(
                f"No langspec_*.json files in {self._las_tokenizer_path}"
            )
        language_paths: dict[str, str] = {}
        for f in langspec_files:
            # filename: langspec_<lang_key>.json — lang_key may include script
            # suffix (e.g. "fin_Latn", "arb_Arab", "python").
            lang_key = f.stem[len("langspec_"):]
            language_paths[lang_key] = str(f)

        # Verify the primary lang is in the routing pool — fail loudly if the
        # tokenizer dir doesn't cover the language the caller wanted to route
        # toward (this is a config error, not a recoverable miss).
        primary_key_with_suffix = f"{self._las_lang}{self._lang_suffix}"
        if (
            primary_key_with_suffix not in language_paths
            and self._las_lang not in language_paths
        ):
            raise FileNotFoundError(
                f"Primary LAS lang {self._las_lang!r}(+suffix {self._lang_suffix!r}) "
                f"not in routing pool. Available: {sorted(language_paths.keys())}"
            )

        # CRITICAL-3: enforce base↔per-lang tokenizer consistency on the
        # ACTUALLY-loaded routing pool (vocab size + special-token ids), not a
        # hardcoded list in a standalone script that may not cover this dir.
        from scripts.verify_tokenizer_consistency import (
            _assert_tokenizers_consistent,
        )

        _specials = [
            s
            for s in (
                getattr(self.tokenizer, "eos_token", None),
                getattr(self.tokenizer, "pad_token", None),
                getattr(self.tokenizer, "bos_token", None),
                getattr(self.tokenizer, "unk_token", None),
            )
            if s
        ]
        _assert_tokenizers_consistent(
            str(base_path), list(language_paths.values()), _specials
        )

        las = LanguageSpecificUnigramLMTokenizer(
            vocab_size=len(self.tokenizer),
        )
        las.load(str(base_path), language_paths)
        self._las = las
        # Tally of routing decisions made during eval — useful for diagnostics
        # ("when evaluating Finnish text, what % of the time did routing pick fin?").
        self._routing_counts: dict[str, int] = {}

    def _las_preprocess(self, text: str) -> str:
        if self._las is None:
            return text
        lang, tokens = self._las.best_language_encode(text)
        if lang is not None:
            self._routing_counts[lang] = self._routing_counts.get(lang, 0) + 1
        return tokens

    def _las_preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        if self._las is None:
            return texts
        langs, encs = self._las.best_language_encode_batch(texts)
        rendered: list[list[str]] = []
        for text, lang, enc in zip(texts, langs, encs):
            if enc is None:
                lang, tokens = self._las.best_language_encode(text)
                rendered.append(tokens)
            else:
                rendered.append(enc.tokens)
            if lang is not None:
                self._routing_counts[lang] = self._routing_counts.get(lang, 0) + 1
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
        _assert_ids_valid(tokens, ids)
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
            _assert_ids_valid(tokens, ids)
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


class BaseUnigramHFLM(HFLM):
    """HFLM that uses a base Unigram tokenizer (language-agnostic) for encoding."""

    def __init__(
        self,
        pretrained,  # str | transformers.PreTrainedModel
        base_tokenizer_path: str,
        **kwargs,
    ) -> None:
        self._base_tok_path = Path(base_tokenizer_path)
        super().__init__(pretrained=pretrained, **kwargs)

        if not self._base_tok_path.exists():
            raise FileNotFoundError(
                f"Base Unigram tokenizer not found: {self._base_tok_path}"
            )
        self._base_tok = Tokenizer.from_file(str(self._base_tok_path))

        # CRITICAL-3 (base/pathpiece): the model was trained on this tokenizer's
        # id space; a wrong-vocab tokenizer dir silently misaligns every id.
        # Assert vocab size matches the model tokenizer + key specials resolve.
        _bt_vocab = self._base_tok.get_vocab_size()
        _model_vocab = len(self.tokenizer)
        if _bt_vocab != _model_vocab:
            raise AssertionError(
                f"Base/pathpiece tokenizer vocab {_bt_vocab} != model tokenizer "
                f"vocab {_model_vocab} ({self._base_tok_path}) — wrong tokenizer "
                f"dir for this model. Refusing (no silent id misalignment)."
            )
        for _s in (
            getattr(self.tokenizer, "eos_token", None),
            getattr(self.tokenizer, "pad_token", None),
        ):
            if _s and self._base_tok.token_to_id(_s) is None:
                raise AssertionError(
                    f"Special token {_s!r} absent from {self._base_tok_path} — "
                    f"model/tokenizer mismatch. Refusing."
                )

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        ids = self._base_tok.encode(
            string, add_special_tokens=bool(add_special_tokens or False)
        ).ids
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
        encodings = self._base_tok.encode_batch(strings, add_special_tokens=False)
        token_ids = [enc.ids for enc in encodings]
        if left_truncate_len is not None:
            token_ids = [ids[-left_truncate_len:] for ids in token_ids]
        if truncation:
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


class ExtendedVocabHFLM(HFLM):
    """HFLM for A4 (F15 `las_mode="vocab_extend"`) checkpoints.

    Encoding uses the extended tokenizer file (the model's own BPE plus the
    appended monolingual pieces). Decoding goes through
    `vocab_extend.ExtendedTokenizer`, whose byte-exact `decode` is required:
    `tokenizers.Tokenizer.decode` corrupts added tokens whose surface
    characters all sit in the GPT-2 byte<->char table.

    Interface contract for eval drivers:
        ExtendedVocabHFLM(pretrained, extended_tokenizer_path, **kwargs)
    where `pretrained` / the remaining kwargs come from
    `_build_eval_lm_kwargs(..., expected_las_mode="vocab_extend")`, which also
    resizes the base model to the extended vocabulary before loading the
    adapter. `self.tokenizer` stays the BASE HF tokenizer (used only for
    padding and for the eos/pad ids, which the extension leaves at their native
    values), so its length is deliberately SMALLER than the model's embedding
    rows — do not assert equality against it.
    """

    def __init__(
        self,
        pretrained,  # str | transformers.PreTrainedModel
        extended_tokenizer_path: str,
        **kwargs,
    ) -> None:
        from las_experiment.vocab_extend import ExtendedTokenizer

        self._ext_tok_path = Path(extended_tokenizer_path)
        if not self._ext_tok_path.exists():
            raise FileNotFoundError(
                f"Extended tokenizer not found: {self._ext_tok_path}"
            )
        super().__init__(pretrained=pretrained, **kwargs)
        self._ext_tok = ExtendedTokenizer.from_file(self._ext_tok_path)

        # The model was resized to this vocabulary; every id the extended
        # tokenizer emits must index a real embedding row, and every row must
        # have a piece. Compare against the MODEL, not the base HF tokenizer.
        _ext_vocab = self._ext_tok.get_vocab_size()
        _emb = self.model.get_input_embeddings()
        _emb_w = (
            _emb.modules_to_save[
                next(iter(_emb.modules_to_save))
            ].weight
            if hasattr(_emb, "modules_to_save")
            else _emb.weight
        )
        _rows = int(_emb_w.shape[0])
        if _ext_vocab != _rows:
            raise AssertionError(
                f"Extended tokenizer vocab {_ext_vocab} != model embedding rows "
                f"{_rows} ({self._ext_tok_path}) — wrong tokenizer file for this "
                f"checkpoint. Refusing (no silent id misalignment)."
            )
        for _s in (
            getattr(self.tokenizer, "eos_token", None),
            getattr(self.tokenizer, "pad_token", None),
        ):
            if _s and self._ext_tok.token_to_id(_s) is None:
                raise AssertionError(
                    f"Special token {_s!r} absent from {self._ext_tok_path} — "
                    f"model/tokenizer mismatch. Refusing."
                )
        for _s, _id in (
            ("eos", getattr(self.tokenizer, "eos_token", None)),
            ("pad", getattr(self.tokenizer, "pad_token", None)),
        ):
            if _id is None:
                continue
            _base_id = self.tokenizer.convert_tokens_to_ids(_id)
            _ext_id = self._ext_tok.token_to_id(_id)
            if _base_id != _ext_id:
                raise AssertionError(
                    f"{_s}_token {_id!r} is id {_base_id} in the base tokenizer "
                    f"but {_ext_id} in {self._ext_tok_path} — the extension moved "
                    f"a special token. Refusing."
                )

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        ids = self._ext_tok.encode(
            string, add_special_tokens=bool(add_special_tokens or False)
        ).ids
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
        encodings = self._ext_tok.encode_batch(strings, add_special_tokens=False)
        token_ids = [enc.ids for enc in encodings]
        if left_truncate_len is not None:
            token_ids = [ids[-left_truncate_len:] for ids in token_ids]
        if truncation:
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

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        # HFLM's default would decode with the BASE tokenizer, which has no
        # entry for the appended ids at all.
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._ext_tok.decode(tokens, skip_special_tokens=skip_special_tokens)


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
    mapped = ["tgl" if lang == "fil" else lang for lang in langs]
    return [f"belebele_{lang}_Latn" for lang in mapped]


def build_piqa_tasks(langs: list[str]) -> list[str]:
    return [f"global_piqa_completions_{lang}_latn" for lang in langs]


def build_multiblimp_tasks(langs: list[str]) -> list[str]:
    return [f"multiblimp_{lang}" for lang in langs]


def build_turblimp_tasks() -> list[str]:
    return ["turblimp_core"]


def build_codexglue_tasks(langs: list[str]) -> list[str]:
    """CodeXGLUE code-to-text tasks (code summarization)."""
    # Available: python, javascript, java, go, ruby, php
    valid = {"python", "javascript", "java", "go", "ruby", "php"}
    return [f"codexglue_code_to_text-{lang}" for lang in langs if lang in valid]


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

    # Code evaluation benchmarks (Python only)
    results4 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=["humaneval"],
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
            confirm_run_unsafe_code=True,
        )
        if "humaneval" in tasks_list
        else None
    )

    results5 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=["mbpp"],
            num_fewshot=3,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
            confirm_run_unsafe_code=True,
        )
        if "mbpp" in tasks_list
        else None
    )

    results6 = (
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

    # TurBLiMP is Turkish-specific; skip for other languages
    results7 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_turblimp_tasks(),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "turblimp" in tasks_list and "tur" in langs
        else None
    )

    results8 = (
        lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=build_codexglue_tasks(langs),
            num_fewshot=0,
            batch_size=str(cfg.batch_size),
            log_samples=bool(cfg.log_samples),
        )
        if "codexglue" in tasks_list and build_codexglue_tasks(langs)
        else None
    )

    return results1, results2, results3, results4, results5, results6, results7, results8


def run_eval(cfg: DictConfig) -> None:
    output_dir = Path(getattr(cfg, "output_path", "./eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = cfg.model
    if bool(cfg.eval_base.enabled):
        base_model_args = f"pretrained={cfg.model_args.pretrained}"

        logging.info(f"Evaluating base model: {model} with args: {base_model_args}")
        r1, r2, r3, r4, r5, r6, r7, r8 = run_eval_one_model(model, base_model_args, cfg)

        output_path = output_dir / f"eval_results_base_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {"mmlu": r1, "belebele": r2, "piqa": r3, "humaneval": r4, "mbpp": r5, "multiblimp": r6, "turblimp": r7, "codexglue": r8},
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if bool(cfg.eval_ft.enabled):
        ft_kwargs = _build_eval_lm_kwargs(
            cfg.model_args.pretrained, cfg.model_args.peft_original_tok,
            expected_las_mode="original", expected_las_dir=None,
        )
        logging.info(
            f"Evaluating finetuned model with original tokenizer "
            f"(ckpt={cfg.model_args.peft_original_tok})"
        )
        lm = HFLM(**ft_kwargs)
        r1, r2, r3, r4, r5, r6, r7, r8 = run_eval_one_model(lm, None, cfg=cfg)

        output_path = output_dir / f"eval_results_ft_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                    "humaneval": r4,
                    "mbpp": r5,
                    "multiblimp": r6,
                    "turblimp": r7,
                    "codexglue": r8,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if cfg.get("eval_ft_base") and bool(cfg.eval_ft_base.enabled):
        base_tok_path = str(cfg.eval_ft_base.base_tokenizer_path)
        base_kw = _build_eval_lm_kwargs(
            cfg.model_args.pretrained, cfg.model_args.peft_base_tok,
            expected_las_mode="base",
            expected_las_dir=os.path.dirname(base_tok_path),
        )

        lm = BaseUnigramHFLM(
            base_tokenizer_path=base_tok_path,
            **base_kw,
        )

        logging.info(
            f"Evaluating finetuned model with base Unigram tokenizer: {base_tok_path}"
        )
        r1, r2, r3, r4, r5, r6, r7, r8 = run_eval_one_model(lm, None, cfg)

        output_path = output_dir / f"eval_results_ft_base_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {"mmlu": r1, "belebele": r2, "piqa": r3, "humaneval": r4, "mbpp": r5, "multiblimp": r6, "turblimp": r7, "codexglue": r8},
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if cfg.get("eval_ft_pathpiece") and bool(cfg.eval_ft_pathpiece.enabled):
        pp_tok_path = str(cfg.eval_ft_pathpiece.pathpiece_tokenizer_path)
        pp_kw = _build_eval_lm_kwargs(
            cfg.model_args.pretrained, cfg.model_args.peft_pathpiece_tok,
            expected_las_mode="pathpiece",
            expected_las_dir=os.path.dirname(pp_tok_path),
        )

        lm = BaseUnigramHFLM(
            base_tokenizer_path=pp_tok_path,
            **pp_kw,
        )

        logging.info(
            f"Evaluating finetuned model with PathPiece tokenizer: {pp_tok_path}"
        )
        r1, r2, r3, r4, r5, r6, r7, r8 = run_eval_one_model(lm, None, cfg)

        output_path = output_dir / f"eval_results_ft_pathpiece_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {"mmlu": r1, "belebele": r2, "piqa": r3, "humaneval": r4, "mbpp": r5, "multiblimp": r6, "turblimp": r7, "codexglue": r8},
                f,
                indent=2,
                default=handle_non_serializable,
            )

    if cfg.get("custom_tokenizer") and cfg.custom_tokenizer.enabled:
        las_kw = _build_eval_lm_kwargs(
            cfg.model_args.pretrained, cfg.model_args.peft_las_tok,
            expected_las_mode="langspec",
            expected_las_dir=str(cfg.custom_tokenizer.las_tokenizer_main_path),
        )
        lang_suffix = str(cfg.custom_tokenizer.get("lang_suffix", "_Latn"))
        lm = LasTokenizerHFLM(
            las_tokenizer_main_path=str(cfg.custom_tokenizer.las_tokenizer_main_path),
            las_lang=str(cfg.custom_tokenizer.las_lang),
            lang_suffix=lang_suffix,
            **las_kw,
        )

        logging.info(
            f"Evaluating finetuned model with LAS tokenizer"
        )
        r1, r2, r3, r4, r5, r6, r7, r8 = run_eval_one_model(lm, None, cfg)

        output_path = output_dir / f"eval_results_ft_las_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "mmlu": r1,
                    "belebele": r2,
                    "piqa": r3,
                    "humaneval": r4,
                    "mbpp": r5,
                    "multiblimp": r6,
                    "turblimp": r7,
                    "codexglue": r8,
                },
                f,
                indent=2,
                default=handle_non_serializable,
            )

    logging.info("Evaluation complete.")
