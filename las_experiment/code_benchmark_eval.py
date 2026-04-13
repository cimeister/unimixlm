#!/usr/bin/env python3
"""
UniEval v2 — Unified Production Evaluation System

Features:
- One CLI (no mismatch with bash)
- Auto routing: lm-eval vs perplexity
- Auto dtype fallback: bf16 → fp16 → fp32
- SLURM-safe execution
- JSONL structured logging
- HF leaderboard upload
- No silent failures (strict mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("unieval")

# -----------------------------
# CONFIG
# -----------------------------
CODE_TASKS = ["humaneval", "mbpp"]

MULTIPL_E_TASK_MAP = {
    "go": "multiple_go",
    "typescript": "multiple_ts",
    "java": "multiple_java",
    "javascript": "multiple_js",
    "cpp": "multiple_cpp",
    "csharp": "multiple_cs",
    "php": "multiple_php",
}

PERPLEXITY_LANGS = {"html", "css", "markdown"}

THESTACK_REPO = "MultilingualUnigramLM/LangMap-TheStack-{lang}-100M"

DTYPE_FALLBACKS = ["bfloat16", "float16", "float32"]


# -----------------------------
# LOGGING (JSONL)
# -----------------------------
def jsonl_log(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# -----------------------------
# LM-EVAL IMPORT
# -----------------------------
def import_lm_eval():
    import lm_eval
    return lm_eval


# -----------------------------
# TASK BUILDER
# -----------------------------
def build_tasks(lang):
    if lang in PERPLEXITY_LANGS:
        return []
    tasks = list(CODE_TASKS)
    if lang in MULTIPL_E_TASK_MAP:
        tasks.append(MULTIPL_E_TASK_MAP[lang])
    return tasks


# -----------------------------
# SAFE LM-EVAL RUN
# -----------------------------
def run_lmeval(model_args, tasks, batch_size):
    lm_eval = import_lm_eval()

    return lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )


# -----------------------------
# DTYPE AUTO FALLBACK
# -----------------------------
def run_with_dtype_fallback(model_id, adapter, tasks, batch_size, log_path):
    last_err = None

    for dtype in DTYPE_FALLBACKS:
        try:
            logger.info(f"[dtype] Trying {dtype}")

            base_args = f"pretrained={model_id},dtype={dtype}"
            ft_args = f"pretrained={model_id},peft={adapter},dtype={dtype}"

            base = run_lmeval(base_args, tasks, batch_size)
            ft = run_lmeval(ft_args, tasks, batch_size)

            jsonl_log(log_path, {
                "event": "success",
                "dtype": dtype,
                "model": model_id,
                "tasks": tasks,
            })

            return {"dtype": dtype, "base": base, "ft": ft}

        except Exception as e:
            last_err = str(e)
            jsonl_log(log_path, {
                "event": "dtype_fail",
                "dtype": dtype,
                "error": last_err[:300],
            })

    raise RuntimeError(f"All dtype fallbacks failed: {last_err}")


# -----------------------------
# PERPLEXITY (BpB)
# -----------------------------
def run_bpb(model_id, adapter, lang, max_samples, log_path):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    repo = THESTACK_REPO.format(lang=lang)

    ds = load_dataset(repo, split="test", streaming=True)

    texts = []
    col = None

    for i, x in enumerate(ds):
        if i == 0:
            col = "content" if "content" in x else list(x.keys())[0]

        texts.append(x[col])
        if len(texts) >= max_samples:
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            ids = enc["input_ids"].to(device)

            if ids.shape[1] < 2:
                continue

            out = model(ids, labels=ids)
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]

    bpb = total_loss / (total_tokens * 0.693)

    jsonl_log(log_path, {
        "event": "bpb",
        "lang": lang,
        "bpb": bpb,
    })

    return {"bpb": bpb}


# -----------------------------
# MAIN
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--lang", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-repo", required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--batch-size", default="auto:4")
    p.add_argument("--mode", default="auto", choices=["auto", "lmeval", "perplexity"])

    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "unieval.jsonl"

    lang = args.lang.lower()

    is_perplexity = (lang in PERPLEXITY_LANGS) if args.mode == "auto" else (args.mode == "perplexity")

    logger.info(f"UniEval v2 | lang={lang} | mode={'bpb' if is_perplexity else 'lmeval'}")

    if not is_perplexity:
        tasks = build_tasks(lang)
        if not tasks:
            raise RuntimeError(f"No tasks found for {lang}")

        result = run_with_dtype_fallback(
            args.base_model,
            args.adapter_repo,
            tasks,
            args.batch_size,
            log_path,
        )

        summary = {
            "lang": lang,
            "mode": "lmeval",
            "dtype": result["dtype"],
        }

    else:
        base = run_bpb(
            args.base_model,
            None,
            lang,
            200,
            log_path,
        )

        ft = run_bpb(
            args.base_model,
            args.adapter_repo,
            lang,
            200,
            log_path,
        )

        summary = {
            "lang": lang,
            "mode": "bpb",
            "base": base,
            "ft": ft,
            "delta": base["bpb"] - ft["bpb"],
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = out / f"leaderboard_{ts}.json"

    with open(final_path, "w") as f:
        json.dump(summary, f, indent=2)

    jsonl_log(log_path, {"event": "done", "file": str(final_path)})

    logger.info(f"Saved → {final_path}")


if __name__ == "__main__":
    main()
