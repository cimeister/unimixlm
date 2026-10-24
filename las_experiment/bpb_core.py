"""Shared BpB computation + model-loading utilities for offline eval scripts.

Factored out of `scripts/eval_zeroshot_bpb.py` so the three eval entry points
(zero-shot, held-out in-domain, FLORES+) share a single implementation and
can't silently diverge.

Public API:
  - `compute_bpb(...)` — numerical core: NLL / (bytes * ln 2).
  - `tokenize_texts(...)` — encode a text list with byte counts.
  - `load_eval_model(...)` — PEFT-aware loader that reproduces the tie-weights
    workaround from `las_experiment/downstream_eval.py`.
  - `pick_tokenizer(...)` — select the right tokenizer file for a mode/lang.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm


# ---- Numerical core ----------------------------------------------------

def compute_bpb(model, token_ids_list, num_bytes_list, device,
                batch_size=4, max_seq_len=1024, pad_id=0):
    """Compute bits-per-byte on a list of tokenized sequences.

    BpB formula: sum(NLL across non-pad positions) / (total_bytes * ln 2).

    Args:
      model: a causal LM (already moved to `device`, in eval mode).
      token_ids_list: list[list[int]] — one tokenized doc per entry.
      num_bytes_list: list[int] — byte count of the decoded tokens per doc.
      pad_id: token id to pad with (model's pad_token_id). Loss is masked on
        attention_mask, so the exact value is numerically irrelevant as long
        as padded positions don't explode the logits; defaults to 0.

    Returns:
      dict(bpb, nll, n_tokens, n_bytes).
    """
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(token_ids_list), batch_size), desc="BpB"):
            batch_ids = token_ids_list[i : i + batch_size]
            batch_bytes = num_bytes_list[i : i + batch_size]
            if not batch_ids:
                continue
            max_len = min(max(len(ids) for ids in batch_ids), max_seq_len)
            padded = []
            masks = []
            for ids in batch_ids:
                ids = ids[:max_len]
                pad_len = max_len - len(ids)
                padded.append(ids + [pad_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
            input_ids = torch.tensor(padded, device=device)
            attention_mask = torch.tensor(masks, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.shape)
            masked_losses = token_losses * shift_mask
            total_nll += masked_losses.sum().item()
            total_tokens += int(shift_mask.sum().item())
            total_bytes += sum(batch_bytes)

    bpb = total_nll / (total_bytes * math.log(2)) if total_bytes > 0 else float("inf")
    return {
        "bpb": bpb,
        "nll": total_nll,
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
    }


def tokenize_texts(texts, tokenizer: Tokenizer, max_seq_len: int = 1024):
    """Tokenize a list of texts; return (token_ids_list, num_bytes_list).

    `num_bytes` is computed from the *decoded* truncated token stream so it
    matches the subset the model actually scores — matches the accounting in
    `las_experiment.finetune.TokenizedTextDataset`.
    Docs that tokenize to zero tokens or decode to zero bytes are dropped.
    """
    token_ids_list: list[list[int]] = []
    num_bytes_list: list[int] = []
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids = enc.ids[:max_seq_len]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        n_bytes = len(decoded.encode("utf-8"))
        if ids and n_bytes:
            token_ids_list.append(ids)
            num_bytes_list.append(n_bytes)
    return token_ids_list, num_bytes_list


def tokenize_texts_las_routed(texts, las_tokenizer, max_seq_len: int = 1024):
    """Tokenize via LAS routing: each text segments under every loaded language
    and the language with the highest log-prob wins.

    Required to actually exercise the LAS inference scheme at eval time.
    Loading a single language and using `tokenize_texts` directly is an oracle
    eval — the per-language tokenizer is picked from the file label, bypassing
    the routing argmax that defines LAS.

    Returns: (token_ids_list, num_bytes_list, picked_langs, all_routed_langs, n_dropped)
        token_ids_list / num_bytes_list / picked_langs: parallel lists over KEPT
            texts (non-empty segmentation, non-zero bytes). picked_langs[i] is
            the lang token_ids_list[i] was tokenized under.
        all_routed_langs: parallel list to the input `texts` for which routing
            produced a non-None lang (i.e., excludes only completely-failed
            routing decisions). Used to compute routing accuracy without bias
            from dropped-but-routed texts.
        n_dropped: int — total docs not in token_ids_list (failed routing OR
            empty segmentation OR zero decoded bytes).
    """
    if not getattr(las_tokenizer, "per_lang_tok", None):
        raise ValueError(
            "tokenize_texts_las_routed requires a LanguageSpecificUnigramLMTokenizer "
            "with per_lang_tok populated. Did you forget to call .load(...)?"
        )

    best_langs, best_encs = las_tokenizer.best_language_encode_batch(texts)
    token_ids_list: list[list[int]] = []
    num_bytes_list: list[int] = []
    picked_langs: list[str] = []
    # Track ALL routing decisions (not just kept docs) so routing_dist reflects
    # actual model behavior. Dropped docs (empty segmentation / zero bytes) are
    # excluded from BpB but the routing decision they triggered is still
    # reported. Otherwise `routing_top_lang_frac` is biased upward by however
    # many docs got silently filtered.
    all_routed_langs: list[str] = []
    n_dropped = 0
    for lang, enc in zip(best_langs, best_encs):
        if enc is None or lang is None:
            n_dropped += 1
            continue
        all_routed_langs.append(lang)
        ids = enc.ids[:max_seq_len]
        decoded = las_tokenizer.per_lang_tok[lang]["tokenizer"].decode(
            ids, skip_special_tokens=True
        )
        n_bytes = len(decoded.encode("utf-8"))
        if ids and n_bytes:
            token_ids_list.append(ids)
            num_bytes_list.append(n_bytes)
            picked_langs.append(lang)
        else:
            n_dropped += 1
    return token_ids_list, num_bytes_list, picked_langs, all_routed_langs, n_dropped


def load_las_routing_tokenizer(tokenizer_dir, vocab_size: int,
                                model_tokenizer=None):
    """Construct a LAS routing-capable tokenizer from a directory.

    Loads `base_tokenizer.json` + every `langspec_*.json` in the dir. The lang
    keys come from filenames (e.g. `langspec_fin_Latn.json` -> key `fin_Latn`).

    `vocab_size` is the LAS constructor argument; pass `len(model_tokenizer)`.

    `model_tokenizer` (optional but strongly recommended): when provided, asserts
    that every loaded per-lang LAS tokenizer's vocab order matches the model
    tokenizer's. This is the `validate_tokenizers` invariant enforced at LAS
    training time. Re-checking at eval load prevents silent UNK substitution
    if a langspec file is corrupted / swapped / regenerated with drift.
    """
    from pathlib import Path as _Path
    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

    tokenizer_dir = _Path(tokenizer_dir)
    base_path = tokenizer_dir / "base_tokenizer.json"
    if not base_path.exists():
        raise FileNotFoundError(f"LAS base tokenizer not found: {base_path}")
    langspec_files = sorted(tokenizer_dir.glob("langspec_*.json"))
    if not langspec_files:
        raise FileNotFoundError(
            f"No langspec_*.json files in {tokenizer_dir}"
        )
    language_paths: dict[str, str] = {}
    for f in langspec_files:
        lang_key = f.stem[len("langspec_"):]
        if not lang_key:
            raise ValueError(
                f"Empty lang_key from filename {f.name} — refusing to load."
            )
        language_paths[lang_key] = str(f)
    las = LanguageSpecificUnigramLMTokenizer(vocab_size=vocab_size)
    las.load(str(base_path), language_paths)

    # Runtime vocab-order assertion: every per-lang LAS tokenizer must agree on
    # token order with the model tokenizer. validate_tokenizers enforces this
    # at training time; re-check here so a swapped/corrupted langspec fails
    # loudly instead of producing silent UNKs after routing.
    if model_tokenizer is not None:
        model_vocab = model_tokenizer.get_vocab()  # {token: id}
        # Reverse to {id: token} for ordered comparison.
        model_id_to_tok = {v: k for k, v in model_vocab.items()}
        for lang_key, info in las.per_lang_tok.items():
            tok = info["tokenizer"]
            las_vocab_size = tok.get_vocab_size()
            for i in range(min(las_vocab_size, len(model_id_to_tok))):
                las_tok = tok.id_to_token(i)
                model_tok = model_id_to_tok.get(i)
                if las_tok != model_tok:
                    raise RuntimeError(
                        f"LAS langspec '{lang_key}' vocab drift at id={i}: "
                        f"LAS has {las_tok!r}, model tokenizer has {model_tok!r}. "
                        f"Routing would silently UNK-out mismatched tokens. "
                        f"Tokenizer dir: {tokenizer_dir}"
                    )
    return las


# ---- Model loader ------------------------------------------------------

def load_eval_model(base_model: str, peft_path: str | None = None,
                    dtype=torch.bfloat16, device: str | None = None):
    """Load a causal LM for eval. Three code paths:

      1. `peft_path=None`: load the untrained `base_model` (for diagnostics).
      2. `peft_path` points at a PEFT adapter dir (has `adapter_config.json`):
         load base_model + apply adapter. For tied-fix PEFT runs (Qwen3/Gemma3
         LoRA+embed where only `embed_tokens` is in `modules_to_save`),
         manually re-share the trained input embedding Parameter with
         `lm_head.weight` BEFORE disabling auto-tie. Otherwise lm_head would
         use the ORIGINAL (un-trained) weights while embed_tokens uses the
         trained copy → silently invalid logits. Then `tie_word_embeddings=
         False` on every config to skip HFLM's `tie_weights()` crash path
         (PEFT issue #2777).
      3. `peft_path` points at a full-model save (no adapter_config.json, e.g.
         embed-only runs that wrote `model.safetensors` directly): load the
         model directly from that dir. The trained weights are already in
         place — no PEFT machinery, no tie-weight hazard.

    For (1) and (3) the model is loaded from the path as-is. Only (2) needs
    the tie-weight workaround.
    """
    import os as _os
    from transformers import AutoModelForCausalLM
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if peft_path is None:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    elif _os.path.isfile(_os.path.join(peft_path, "adapter_config.json")):
        # PEFT adapter path
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        model = PeftModel.from_pretrained(model, peft_path)

        # Tied-fix re-share: trained embed_tokens.weight -> lm_head.weight.
        inner = getattr(getattr(model, "base_model", None), "model", None)
        if inner is not None:
            emb_module = inner.get_input_embeddings()
            is_emb_wrapped = hasattr(emb_module, "modules_to_save")
            lm_head_module = getattr(inner, "lm_head", None)
            is_lm_head_wrapped = (
                lm_head_module is not None
                and hasattr(lm_head_module, "modules_to_save")
            )
            inner_was_tied = bool(getattr(inner.config, "tie_word_embeddings", False))
            if (
                is_emb_wrapped
                and not is_lm_head_wrapped
                and lm_head_module is not None
                and inner_was_tied
            ):
                active = getattr(model, "active_adapter", "default")
                if isinstance(active, list):
                    active = active[0] if active else "default"
                adapter_key = active if active in emb_module.modules_to_save else next(
                    iter(emb_module.modules_to_save)
                )
                trained_param = emb_module.modules_to_save[adapter_key].weight
                lm_head_module.weight = trained_param

        for cfg_obj in (
            getattr(model, "config", None),
            getattr(getattr(model, "base_model", None), "config", None),
            getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        ):
            if cfg_obj is not None and hasattr(cfg_obj, "tie_word_embeddings"):
                cfg_obj.tie_word_embeddings = False
    else:
        # Full-model save (embed-only / from-scratch): load directly from the
        # checkpoint dir. No PEFT, no tie-weight workaround needed.
        model = AutoModelForCausalLM.from_pretrained(peft_path, dtype=dtype)

    model.to(device).eval()
    return model, device


# ---- Tokenizer selection ----------------------------------------------

def pick_tokenizer(tokenizer_mode: str, tokenizer_dir: Path | None,
                   lang: str, lang_suffix: str = "_Latn",
                   model_tokenizer=None) -> Tokenizer:
    """Resolve a `Tokenizer` object for a given mode and language.

    Modes:
      - `"original"` — use the model's native tokenizer (requires model_tokenizer).
      - `"base"` — `{tokenizer_dir}/base_tokenizer.json`.
      - `"pathpiece"` — `{tokenizer_dir}/pathpiece_tokenizer.json`.
      - `"langspec"` — `{tokenizer_dir}/langspec_{lang}{lang_suffix}.json`.

    Raises loudly on missing files — no silent fallback to `original`.
    """
    if tokenizer_mode == "original":
        if model_tokenizer is None:
            raise ValueError("pick_tokenizer(mode=original) requires model_tokenizer")
        return model_tokenizer.backend_tokenizer
    if tokenizer_dir is None:
        raise ValueError(
            f"pick_tokenizer(mode={tokenizer_mode!r}) requires tokenizer_dir"
        )
    tokenizer_dir = Path(tokenizer_dir)
    filename_map = {
        "base": "base_tokenizer.json",
        "pathpiece": "pathpiece_tokenizer.json",
        "langspec": f"langspec_{lang}{lang_suffix}.json",
    }
    if tokenizer_mode not in filename_map:
        raise ValueError(
            f"Unknown tokenizer_mode={tokenizer_mode!r}. "
            f"Expected one of: original, base, pathpiece, langspec."
        )
    path = tokenizer_dir / filename_map[tokenizer_mode]
    if not path.exists():
        raise FileNotFoundError(
            f"No {tokenizer_mode} tokenizer at {path}. "
            f"No fallback — fix the path or mode."
        )
    logging.info(f"Loading {tokenizer_mode} tokenizer: {path}")
    return Tokenizer.from_file(str(path))
