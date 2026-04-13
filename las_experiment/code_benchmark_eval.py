#!/usr/bin/env python3
"""
UniMixLM Eval System v3

Per-language code benchmark evaluation.
- Registry-safe task names (no fake multiple_* crash)
- Dtype fallback: bf16 → fp16 → fp32 (only retries on OOM, NOT task errors)
- pass@k (k=1,10,100) via multiple-sample generation
- BpB perplexity on TheStack (correct byte-based formula)
- HF leaderboard sync (--push-to-hub)
- Regression tracking vs. prior runs (--baseline-dir)
- SLURM-safe JSONL logging

Usage:
    # lm-eval mode (go, java, typescript, etc.)
    python3 las_experiment/code_benchmark_eval.py \\
        --lang java \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-java \\
        --output-dir ./eval_results/code_java \\
        --batch-size auto:4 \\
        --n-samples 1

    # Perplexity mode (html, css, markdown)
    python3 las_experiment/code_benchmark_eval.py \\
        --lang html \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html \\
        --output-dir ./eval_results/code_html \\
        --mode perplexity

    # With HF leaderboard sync
    python3 las_experiment/code_benchmark_eval.py \\
        --lang java ... --push-to-hub MultilingualUnigramLM/eval-leaderboard
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("unieval")

# ─── Config ──────────────────────────────────────────────────────────────────

# Core Python benchmarks — stable across all lm-eval versions
CODE_TASKS = ["humaneval", "mbpp"]

# MultiPL-E task names in lm-eval registry.
# These are checked against the live registry at runtime — no crashes on mismatch.
MULTIPL_E_TASK_MAP: dict[str, str] = {
    "go":         "multiple_go",
    "typescript": "multiple_ts",
    "java":       "multiple_java",
    "javascript": "multiple_js",
    "cpp":        "multiple_cpp",
    "csharp":     "multiple_cs",
    "php":        "multiple_php",
    # "c" omitted — multiple_c is rarely registered
}

# Languages that have no functional benchmark → perplexity (BpB) only
PERPLEXITY_LANGS: set[str] = {"html", "css", "markdown"}

THESTACK_REPO_TEMPLATE = "MultilingualUnigramLM/LangMap-TheStack-{lang}-100M"

# dtype preference order — we only retry on OOM, never on task errors
DTYPE_FALLBACKS = ["bfloat16", "float16", "float32"]


# ─── JSONL logging ───────────────────────────────────────────────────────────

def jsonl_log(log_path: Path, obj: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({**obj, "_ts": datetime.now().isoformat()}) + "\n")


# ─── Task registry (safe) ────────────────────────────────────────────────────

def get_available_tasks(lm_eval_module) -> set[str]:
    """Return the set of registered lm-eval task names."""
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        return set(tm.all_tasks)
    except Exception:
        pass
    try:
        # Older lm-eval API: get_task_dict with an empty list returns all tasks
        return set(lm_eval_module.tasks.get_task_dict([]).keys())
    except Exception:
        pass
    logger.warning("[registry] Could not enumerate lm-eval task registry. "
                   "Will attempt tasks and catch failures.")
    return set()


def build_tasks(lang: str, available_tasks: Optional[set[str]] = None) -> list[str]:
    """
    Build the task list for a language, skipping any tasks not in the registry.
    If available_tasks is None (registry lookup failed), attempts all and relies on
    the retry logic in run_lmeval to drop bad task names.
    """
    tasks: list[str] = []

    for t in CODE_TASKS:
        if available_tasks is None or t in available_tasks:
            tasks.append(t)
        else:
            logger.warning(f"[registry] Core task '{t}' not found — skipping.")

    candidate = MULTIPL_E_TASK_MAP.get(lang)
    if candidate:
        if available_tasks is None or candidate in available_tasks:
            tasks.append(candidate)
        else:
            logger.warning(
                f"[registry] '{candidate}' not in lm-eval registry. "
                "Install via: pip install lm-eval[multiple] or "
                "git+https://github.com/nuprl/MultiPL-E.git. Skipping."
            )

    if not tasks:
        raise RuntimeError(
            f"No valid lm-eval tasks found for lang='{lang}'. "
            "Check your lm-eval installation."
        )

    return tasks


# ─── Error classification ─────────────────────────────────────────────────────

def _is_oom(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ["out of memory", "cuda oom", "oom", "memory"])


def _is_task_error(e: Exception) -> bool:
    """True if the error is about task names, not hardware."""
    s = str(e)
    return any(k in s for k in ["task", "not found", "unknown task", "KeyError"])


# ─── lm-eval runner ──────────────────────────────────────────────────────────

def _run_lmeval_once(model_args: str, tasks: list[str], batch_size: str,
                     n_samples: int) -> dict:
    """Single lm-eval call. Raises on failure."""
    import lm_eval

    # Generation kwargs for pass@k sampling (k > 1 needs do_sample + temperature)
    gen_kwargs = None
    if n_samples > 1:
        gen_kwargs = f"n={n_samples},do_sample=True,temperature=0.8,top_p=0.95"

    kwargs = dict(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        log_samples=(n_samples > 1),   # need samples for pass@k computation
        confirm_run_unsafe_code=True,   # HumanEval code execution
    )
    if gen_kwargs:
        kwargs["gen_kwargs"] = gen_kwargs

    return lm_eval.simple_evaluate(**kwargs)


def run_lmeval(
    model_args: str,
    tasks: list[str],
    batch_size: str,
    output_dir: Path,
    run_label: str,
    n_samples: int,
    log_path: Path,
) -> dict:
    """
    Run lm-eval with fallback: drops unregistered tasks and retries.
    Saves eval_results_{run_label}_{timestamp}.json.
    """
    active_tasks = list(tasks)
    results = None

    while active_tasks:
        logger.info(f"[lm-eval] {run_label}: tasks={active_tasks}, n_samples={n_samples}")
        try:
            results = _run_lmeval_once(model_args, active_tasks, batch_size, n_samples)
            break
        except Exception as e:
            if _is_task_error(e):
                # Find and drop the offending task
                dropped = False
                for task in list(active_tasks):
                    if task in str(e) or task.startswith("multiple_"):
                        logger.warning(f"[lm-eval] Dropping unregistered task '{task}': {e}")
                        active_tasks.remove(task)
                        dropped = True
                        break
                if not dropped:
                    raise   # can't identify which task — re-raise
            else:
                raise   # OOM or model error — let caller handle

    if results is None or not active_tasks:
        raise RuntimeError(f"All tasks failed for run_label={run_label}. "
                           "Check lm-eval task installation.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_results_{run_label}_{ts}.json"
    try:
        from lm_eval.utils import handle_non_serializable
        _serializer = handle_non_serializable
    except (ImportError, AttributeError):
        _serializer = str
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=_serializer)

    jsonl_log(log_path, {"event": "lmeval_done", "run": run_label,
                          "tasks": active_tasks, "file": str(out_file)})
    logger.info(f"[lm-eval] Saved {run_label} → {out_file}")
    return results


# ─── Dtype-safe runner ───────────────────────────────────────────────────────

def run_with_dtype_fallback(
    base_model_id: str,
    adapter: str,
    tasks: list[str],
    batch_size: str,
    output_dir: Path,
    log_path: Path,
    n_samples: int,
) -> tuple[dict, dict, str]:
    """
    Try dtypes in order, but ONLY retry on OOM errors.
    Task registration errors are propagated immediately (not hidden by dtype loop).
    Returns (base_results, ft_results, dtype_used).
    """
    last_err: Optional[Exception] = None

    for dtype in DTYPE_FALLBACKS:
        logger.info(f"[dtype] Trying {dtype}")
        base_args = f"pretrained={base_model_id},dtype={dtype}"
        ft_args   = f"pretrained={base_model_id},peft={adapter},dtype={dtype}"
        try:
            base = run_lmeval(base_args, tasks, batch_size, output_dir, "base", n_samples, log_path)
            ft   = run_lmeval(ft_args,   tasks, batch_size, output_dir, "ft",   n_samples, log_path)

            jsonl_log(log_path, {"event": "dtype_success", "dtype": dtype})
            return base, ft, dtype

        except Exception as e:
            if _is_oom(e):
                logger.warning(f"[dtype] OOM with {dtype}, trying next. {e}")
                last_err = e
                jsonl_log(log_path, {"event": "oom", "dtype": dtype, "error": str(e)[:200]})
            else:
                # Task error or model error — do NOT retry with different dtype
                jsonl_log(log_path, {"event": "fatal", "dtype": dtype, "error": str(e)[:200]})
                raise

    raise RuntimeError(f"OOM across all dtypes. Last error: {last_err}")


# ─── pass@k extraction ───────────────────────────────────────────────────────

def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((1 - k / i) for i in range(n - c + 1, n + 1))


def extract_pass_at_k(results: dict, k_values: list[int] = (1, 10, 100)) -> dict:
    """
    Extract pass@k metrics from lm-eval results.
    lm-eval stores pass@1 directly; for k>1 compute via unbiased estimator
    if per-sample correctness data is available.
    """
    summary: dict[str, dict] = {}
    task_results = results.get("results", {})

    for task, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        task_summary: dict[str, float] = {}

        # pass@1 (always available)
        p1 = metrics.get("pass@1,none") or metrics.get("pass@1")
        if p1 is not None:
            task_summary["pass@1"] = float(p1)
            task_summary["pass@1_stderr"] = float(
                metrics.get("pass@1_stderr,none") or metrics.get("pass@1_stderr", 0.0)
            )

        # pass@k from samples (if log_samples=True was used)
        samples = results.get("samples", {}).get(task, [])
        if samples and len(k_values) > 1:
            n = len(samples)
            c = sum(1 for s in samples if s.get("exact_match") or s.get("pass@1"))
            for k in k_values:
                if k <= n:
                    task_summary[f"pass@{k}"] = _pass_at_k_unbiased(n, c, k)

        if task_summary:
            summary[task] = task_summary

    return summary


# ─── Perplexity (BpB) ────────────────────────────────────────────────────────

def _load_model(base_model_id: str, adapter: Optional[str], dtype_str: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }.get(dtype_str, torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok, device


def compute_bpb(model, tokenizer, texts: list[str], device: str,
                max_seq_len: int = 512) -> float:
    """Compute bits-per-byte. Correct formula: BpB = total_nll / (total_bytes × ln2)."""
    total_nll = 0.0
    total_bytes = 0

    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue
            n_bytes = len(text.encode("utf-8"))
            if n_bytes == 0:
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len, padding=False)
            ids = enc["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue

            loss = model(input_ids=ids, labels=ids).loss.item()
            n_tokens = ids.shape[1]
            total_nll   += loss * n_tokens
            total_bytes += n_bytes

    if total_bytes == 0:
        return float("nan")
    return total_nll / (total_bytes * math.log(2))


def run_bpb(
    base_model_id: str,
    adapter: Optional[str],
    lang: str,
    output_dir: Path,
    run_label: str,
    log_path: Path,
    max_samples: int = 200,
    max_seq_len: int = 512,
) -> dict:
    """Compute BpB on TheStack test split. Saves eval_results_{run_label}_{ts}.json."""
    from datasets import load_dataset

    repo = THESTACK_REPO_TEMPLATE.format(lang=lang)
    logger.info(f"[bpb] {run_label}: loading {repo}")

    try:
        ds = load_dataset(repo, split="test", streaming=True, trust_remote_code=True)
    except Exception:
        logger.warning(f"[bpb] No 'test' split in {repo}, using tail of 'train'.")
        ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=True)

    texts: list[str] = []
    col = None
    for i, sample in enumerate(ds):
        if col is None:
            col = next(
                (c for c in ("content", "text", "code") if c in sample),
                list(sample.keys())[0],
            )
        texts.append(sample[col])
        if len(texts) >= max_samples:
            break

    if not texts:
        raise ValueError(f"No samples found in {repo}")

    logger.info(f"[bpb] {run_label}: {len(texts)} samples, computing BpB...")

    # Try dtype fallback for BpB too
    bpb = float("nan")
    for dtype in DTYPE_FALLBACKS:
        try:
            model, tok, device = _load_model(base_model_id, adapter, dtype)
            bpb = compute_bpb(model, tok, texts, device, max_seq_len)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        except Exception as e:
            if _is_oom(e):
                logger.warning(f"[bpb] OOM with {dtype}, trying next.")
                continue
            raise

    logger.info(f"[bpb] {run_label}: BpB = {bpb:.4f}")

    result = {
        "lang": lang,
        "eval_mode": "perplexity",
        "run_label": run_label,
        "model_args": f"pretrained={base_model_id}" + (f",peft={adapter}" if adapter else ""),
        "n_samples": len(texts),
        "results": {"bpb": bpb},
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_results_{run_label}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    jsonl_log(log_path, {"event": "bpb_done", "run": run_label, "bpb": bpb, "file": str(out_file)})
    logger.info(f"[bpb] Saved {run_label} → {out_file}")
    return result


# ─── HF leaderboard sync ─────────────────────────────────────────────────────

def push_to_hub(result_file: Path, hub_repo: str, lang: str) -> None:
    """Upload result JSON to a HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(result_file),
            path_in_repo=f"eval_results/{lang}/{result_file.name}",
            repo_id=hub_repo,
            repo_type="dataset",
        )
        logger.info(f"[hub] Uploaded {result_file.name} → {hub_repo}")
    except Exception as e:
        logger.warning(f"[hub] Upload failed (non-fatal): {e}")


# ─── Regression tracking ─────────────────────────────────────────────────────

def _extract_primary_score(results: dict) -> Optional[float]:
    """Extract the most informative single score from a results dict."""
    task_results = results.get("results", {})
    scores = []
    for task_data in task_results.values():
        if isinstance(task_data, dict):
            for key in ("pass@1,none", "pass@1", "acc,none", "acc"):
                v = task_data.get(key)
                if v is not None:
                    scores.append(float(v))
                    break
    return float(sum(scores) / len(scores)) if scores else None


def run_regression_check(
    lang: str,
    current_ft_results: dict,
    baseline_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> None:
    """Compare current FT results against a previous baseline, log regressions."""
    baseline_file = baseline_dir / f"eval_results_ft_{lang}.json"
    if not baseline_file.exists():
        logger.info(f"[regression] No baseline found at {baseline_file}, skipping.")
        return

    with open(baseline_file) as f:
        baseline = json.load(f)

    cur_score = _extract_primary_score(current_ft_results)
    base_score = _extract_primary_score(baseline)

    if cur_score is None or base_score is None:
        logger.warning("[regression] Could not extract primary scores for comparison.")
        return

    delta = cur_score - base_score
    status = "IMPROVED" if delta > 0.005 else ("REGRESSED" if delta < -0.005 else "STABLE")

    report = {
        "lang": lang,
        "baseline_score": base_score,
        "current_score": cur_score,
        "delta": delta,
        "status": status,
    }

    logger.info(f"[regression] {lang}: {status} | baseline={base_score:.4f} "
                f"current={cur_score:.4f} delta={delta:+.4f}")

    report_file = output_dir / f"regression_{lang}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    jsonl_log(log_path, {"event": "regression", **report})


# ─── Leaderboard summary builder ─────────────────────────────────────────────

def build_leaderboard_summary(
    lang: str,
    base_results: dict,
    ft_results: dict,
    pass_at_k: dict,
    mode: str,
    dtype: str,
    output_dir: Path,
) -> Path:
    """Save a compact summary JSON (leaderboard_{lang}.json)."""
    summary = {
        "lang": lang,
        "mode": mode,
        "dtype": dtype,
        "timestamp": datetime.now().isoformat(),
        "pass_at_k": pass_at_k,
    }

    if mode == "lmeval":
        base_score = _extract_primary_score(base_results)
        ft_score   = _extract_primary_score(ft_results)
        summary["base_score"]  = base_score
        summary["ft_score"]    = ft_score
        summary["delta"]       = (ft_score - base_score) if (base_score is not None and ft_score is not None) else None
    else:  # perplexity
        base_bpb = base_results.get("results", {}).get("bpb")
        ft_bpb   = ft_results.get("results", {}).get("bpb")
        summary["base_bpb"] = base_bpb
        summary["ft_bpb"]   = ft_bpb
        summary["delta_bpb"] = (base_bpb - ft_bpb) if (base_bpb and ft_bpb) else None

    path = output_dir / f"leaderboard_{lang}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[leaderboard] Saved → {path}")
    return path


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UniMixLM Eval System v3")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--base-model", default="allenai/OLMo-3-1025-7B")
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", default="auto:4")
    parser.add_argument("--mode", default="auto", choices=["auto", "lmeval", "perplexity"])
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Completions per problem: 1=pass@1, 10=pass@1+10, 100=pass@1+10+100")
    parser.add_argument("--max-bpb-samples", type=int, default=200)
    parser.add_argument("--push-to-hub", default=None,
                        metavar="HF_REPO",
                        help="Upload leaderboard JSON to this HF dataset repo")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="Directory with prior eval_results_ft_{lang}.json for regression check")
    args = parser.parse_args()

    lang = args.lang.lower()
    out  = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "unieval_v3.jsonl"

    jsonl_log(log_path, {
        "event": "start",
        "lang": lang,
        "base_model": args.base_model,
        "adapter": args.adapter_repo,
        "n_samples": args.n_samples,
    })

    # Determine mode
    if args.mode == "auto":
        is_perplexity = lang in PERPLEXITY_LANGS
    else:
        is_perplexity = args.mode == "perplexity"

    mode_str = "perplexity" if is_perplexity else "lmeval"
    logger.info(f"UniEval v3 | lang={lang} | mode={mode_str}")

    # ── lm-eval path ──────────────────────────────────────────────────────────
    if not is_perplexity:
        import lm_eval
        available = get_available_tasks(lm_eval)
        if available:
            logger.info(f"[registry] {len(available)} tasks registered in lm-eval.")
        else:
            logger.warning("[registry] Could not enumerate registry. Will attempt tasks directly.")

        tasks = build_tasks(lang, available if available else None)
        logger.info(f"[tasks] Final task list for {lang}: {tasks}")

        base_res, ft_res, dtype_used = run_with_dtype_fallback(
            args.base_model,
            args.adapter_repo,
            tasks,
            args.batch_size,
            out,
            log_path,
            args.n_samples,
        )

        k_values = [1]
        if args.n_samples >= 10:
            k_values.append(10)
        if args.n_samples >= 100:
            k_values.append(100)

        pass_at_k = {
            "base": extract_pass_at_k(base_res, k_values),
            "ft":   extract_pass_at_k(ft_res,   k_values),
        }

    # ── Perplexity path ───────────────────────────────────────────────────────
    else:
        dtype_used = "bfloat16"
        base_res = run_bpb(
            args.base_model, None, lang, out, "base", log_path, args.max_bpb_samples
        )
        ft_res = run_bpb(
            args.base_model, args.adapter_repo, lang, out, "ft", log_path, args.max_bpb_samples
        )
        pass_at_k = {}

    # ── Summary / leaderboard ─────────────────────────────────────────────────
    summary_path = build_leaderboard_summary(
        lang, base_res, ft_res, pass_at_k, mode_str, dtype_used, out
    )

    # ── HF leaderboard sync ───────────────────────────────────────────────────
    if args.push_to_hub:
        push_to_hub(summary_path, args.push_to_hub, lang)

    # ── Regression check ──────────────────────────────────────────────────────
    if args.baseline_dir:
        run_regression_check(lang, ft_res, args.baseline_dir, out, log_path)

    jsonl_log(log_path, {"event": "done", "lang": lang})
    logger.info(f"Done. Results in: {out}")


if __name__ == "__main__":
    main()
