#!/usr/bin/env python3
"""
Per-language code benchmark evaluation.

For each language, evaluates:
  - Base model (allenai/OLMo-3-1025-7B, no LoRA)
  - LAS finetuned model (base + LoRA adapter from HuggingFace)

For CODE_LANGS: lm_eval.simple_evaluate with humaneval, mbpp, multiple_{lang}
For PERPLEXITY_LANGS (html, css, markdown): BpB on TheStack test splits

Usage:
    python3 las_experiment/code_benchmark_eval.py \\
        --lang java \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-java \\
        --output-dir ./eval_results/code_java \\
        --batch-size auto:4

    # Perplexity mode (html/css/markdown):
    python3 las_experiment/code_benchmark_eval.py \\
        --lang html \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html \\
        --output-dir ./eval_results/code_html \\
        --eval-mode perplexity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Language → lm-eval MultiPL-E task name mapping ---
MULTIPL_E_TASK_MAP: dict[str, str] = {
    "go":         "multiple_go",
    "typescript": "multiple_ts",
    "java":       "multiple_java",
    "javascript": "multiple_js",
    "cpp":        "multiple_cpp",
    "csharp":     "multiple_cs",
    "php":        "multiple_php",
    # "c" intentionally omitted - multiple_c is not consistently available
}

# Languages with no standard functional benchmark → perplexity (BpB) only
PERPLEXITY_LANGS: set[str] = {"html", "css", "markdown"}

# Fallback: if multiple_{lang} task fails to load, drop it silently
TASK_FALLBACK_ONLY_PYTHON: set[str] = {"c"}

# HF dataset repo for perplexity test data
THESTACK_REPO_TEMPLATE = "MultilingualUnigramLM/LangMap-TheStack-{lang}-100M"

# Number of samples to use for perplexity evaluation (capped to avoid OOM)
PERPLEXITY_MAX_SAMPLES = 200
PERPLEXITY_MAX_SEQ_LEN = 512


def build_task_list(lang: str) -> list[str]:
    """Return the lm-eval task list for a given language."""
    if lang in PERPLEXITY_LANGS:
        raise ValueError(f"Language '{lang}' uses perplexity mode, not lm-eval tasks.")
    tasks = ["humaneval", "mbpp"]
    if lang in MULTIPL_E_TASK_MAP:
        tasks.append(MULTIPL_E_TASK_MAP[lang])
    # For TASK_FALLBACK_ONLY_PYTHON (e.g. 'c'), just humaneval+mbpp
    return tasks


def _try_import_lm_eval():
    try:
        import lm_eval
        from lm_eval.utils import handle_non_serializable
        return lm_eval, handle_non_serializable
    except ImportError as e:
        raise ImportError(
            "lm_eval is not installed. Install via: "
            "pip install lm-eval or from requirements.txt"
        ) from e


def run_lmeval(
    model_args_str: str,
    tasks: list[str],
    batch_size: str,
    output_dir: Path,
    run_label: str,
) -> dict:
    """Run lm-eval-harness and save results JSON. Returns the results dict."""
    lm_eval, handle_non_serializable = _try_import_lm_eval()

    logger.info(f"[lm-eval] {run_label}: tasks={tasks}, model_args={model_args_str}")

    # Try to run all tasks; if a task fails to register, drop it and retry
    active_tasks = list(tasks)
    results = None
    while active_tasks:
        try:
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=model_args_str,
                tasks=active_tasks,
                num_fewshot=0,
                batch_size=batch_size,
                log_samples=False,
            )
            break
        except Exception as e:
            err_str = str(e)
            # If a task isn't registered, remove it and retry
            removed = False
            for task in list(active_tasks):
                if task in err_str or "multiple_" in task:
                    logger.warning(
                        f"Task '{task}' failed ({err_str[:80]}...). Dropping and retrying."
                    )
                    active_tasks.remove(task)
                    removed = True
                    break
            if not removed:
                raise

    if results is None:
        raise RuntimeError("All lm-eval tasks failed.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_results_{run_label}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=handle_non_serializable)

    logger.info(f"[lm-eval] Saved {run_label} results to {out_file}")
    return results


def _load_model_and_tokenizer(base_model_id: str, adapter_repo: str | None, dtype: str):
    """Load base model + optional LoRA adapter for perplexity evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    if adapter_repo:
        logger.info(f"Loading LoRA adapter: {adapter_repo}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_repo)

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def _compute_bpb(model, tokenizer, texts: list[str], device: str, max_seq_len: int) -> float:
    """Compute bits-per-byte on a list of texts. Lower is better."""
    total_nll = 0.0
    total_bytes = 0

    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue
            byte_count = len(text.encode("utf-8"))
            if byte_count == 0:
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            nll = outputs.loss.item()  # mean token NLL
            n_tokens = input_ids.shape[1]
            total_nll += nll * n_tokens
            total_bytes += byte_count

    if total_bytes == 0:
        return float("nan")

    bpb = total_nll / (total_bytes * math.log(2))
    return bpb


def run_perplexity(
    base_model_id: str,
    adapter_repo: str | None,
    lang: str,
    output_dir: Path,
    run_label: str,
    dtype: str = "bfloat16",
    max_samples: int = PERPLEXITY_MAX_SAMPLES,
    max_seq_len: int = PERPLEXITY_MAX_SEQ_LEN,
) -> dict:
    """Compute BpB on TheStack test split and save results JSON."""
    from datasets import load_dataset

    repo_name = THESTACK_REPO_TEMPLATE.format(lang=lang)
    logger.info(f"[perplexity] {run_label}: Loading dataset {repo_name}")

    try:
        dataset = load_dataset(repo_name, split="test", streaming=True, trust_remote_code=True)
    except Exception:
        # Fallback: try train split with last 5% as test proxy
        logger.warning(f"No test split in {repo_name}, using tail of train split.")
        dataset = load_dataset(repo_name, split="train", streaming=True, trust_remote_code=True)

    texts = []
    content_col = None
    for i, sample in enumerate(dataset):
        if i == 0:
            # Auto-detect content column
            for col in ("content", "text", "code", "whole_func_string"):
                if col in sample:
                    content_col = col
                    break
            if content_col is None:
                content_col = list(sample.keys())[0]
            logger.info(f"Using column '{content_col}' for perplexity text.")
        texts.append(sample[content_col])
        if len(texts) >= max_samples:
            break

    if not texts:
        raise ValueError(f"No samples found in {repo_name}")

    logger.info(f"[perplexity] {run_label}: {len(texts)} samples loaded, computing BpB...")

    model, tokenizer, device = _load_model_and_tokenizer(base_model_id, adapter_repo, dtype)
    bpb = _compute_bpb(model, tokenizer, texts, device, max_seq_len)
    logger.info(f"[perplexity] {run_label}: BpB = {bpb:.4f}")

    result = {
        "lang": lang,
        "eval_mode": "perplexity",
        "run_label": run_label,
        "model_args": f"pretrained={base_model_id}" + (f",peft={adapter_repo}" if adapter_repo else ""),
        "n_samples": len(texts),
        "results": {"bpb": bpb},
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_results_{run_label}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"[perplexity] Saved {run_label} results to {out_file}")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Code benchmark evaluation (lm-eval + BpB)")
    parser.add_argument("--lang", required=True, help="Programming language to evaluate")
    parser.add_argument("--base-model", default="allenai/OLMo-3-1025-7B",
                        help="HuggingFace base model ID")
    parser.add_argument("--adapter-repo", required=True,
                        help="HuggingFace LoRA adapter repo for finetuned model")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save eval results JSON files")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                        help="Model dtype for inference")
    parser.add_argument("--batch-size", default="auto:4",
                        help="lm-eval batch size string (e.g. 'auto:4', '8')")
    parser.add_argument(
        "--eval-mode",
        default="auto",
        choices=["auto", "lmeval", "perplexity"],
        help=(
            "Evaluation mode. 'auto' selects perplexity for html/css/markdown, "
            "lmeval for all others."
        ),
    )
    parser.add_argument("--max-perplexity-samples", type=int, default=PERPLEXITY_MAX_SAMPLES,
                        help="Max samples for perplexity evaluation")
    args = parser.parse_args()

    lang = args.lang.lower()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine effective eval mode
    if args.eval_mode == "auto":
        effective_mode = "perplexity" if lang in PERPLEXITY_LANGS else "lmeval"
    else:
        effective_mode = args.eval_mode

    logger.info(f"Evaluating language: {lang} | mode: {effective_mode}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter: {args.adapter_repo}")
    logger.info(f"Output dir: {output_dir}")

    if effective_mode == "lmeval":
        tasks = build_task_list(lang)
        logger.info(f"Tasks: {tasks}")

        base_args = f"pretrained={args.base_model},dtype={args.dtype}"
        ft_args = f"pretrained={args.base_model},peft={args.adapter_repo},dtype={args.dtype}"

        # Evaluate base model
        run_lmeval(base_args, tasks, args.batch_size, output_dir, run_label="base")
        # Evaluate finetuned model
        run_lmeval(ft_args, tasks, args.batch_size, output_dir, run_label="ft")

    elif effective_mode == "perplexity":
        # Evaluate base model
        run_perplexity(
            args.base_model,
            adapter_repo=None,
            lang=lang,
            output_dir=output_dir,
            run_label="base",
            dtype=args.dtype,
            max_samples=args.max_perplexity_samples,
        )
        # Evaluate finetuned model
        run_perplexity(
            args.base_model,
            adapter_repo=args.adapter_repo,
            lang=lang,
            output_dir=output_dir,
            run_label="ft",
            dtype=args.dtype,
            max_samples=args.max_perplexity_samples,
        )
    else:
        raise ValueError(f"Unknown eval mode: {effective_mode}")

    logger.info(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
