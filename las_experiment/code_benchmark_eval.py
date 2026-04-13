#!/usr/bin/env python3
"""
UniMixLM Eval System v4 — bigcode-evaluation-harness

Per-language code benchmark evaluation using bigcode-evaluation-harness.
Each language runs ONLY its language-specific benchmarks (no cross-language
Python retention tests).

- Python:     HumanEval, MBPP, HumanEval+
- Go/Java/JS/TS/C++/C#/PHP: MultiPL-E + HumanEvalPack where available
- C/HTML/CSS/Markdown:       BpB perplexity (no functional benchmark)
- pass@k (k=1,10,100) via multiple-sample generation
- BpB perplexity on TheStack (correct byte-based formula)
- HF leaderboard sync (--push-to-hub)
- Regression tracking vs. prior runs (--baseline-dir)
- SLURM-safe JSONL logging

Usage:
    # bigcode-eval mode (python, go, java, typescript, etc.)
    python3 las_experiment/code_benchmark_eval.py \\
        --lang python \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-python \\
        --output-dir ./eval_results/code_python \\
        --n-samples 1

    # Perplexity mode (c, html, css, markdown)
    python3 las_experiment/code_benchmark_eval.py \\
        --lang html \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html \\
        --output-dir ./eval_results/code_html \\
        --mode perplexity

    # With HF leaderboard sync
    python3 las_experiment/code_benchmark_eval.py \\
        --lang python ... --push-to-hub MultilingualUnigramLM/eval-leaderboard
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
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

# Language -> list of bigcode-eval task names
BIGCODE_TASK_MAP: dict[str, list[str]] = {
    "python":     ["humaneval", "mbpp", "humanevalplus"],
    "go":         ["multiple-go", "humanevalsynthesize-go"],
    "typescript": ["multiple-ts"],
    "java":       ["multiple-java", "humanevalsynthesize-java"],
    "javascript": ["multiple-js", "humanevalsynthesize-javascript"],
    "cpp":        ["multiple-cpp", "humanevalsynthesize-cpp"],
    "csharp":     ["multiple-cs"],
    "php":        ["multiple-php"],
}

# Languages that have no functional benchmark → perplexity (BpB) only
PERPLEXITY_LANGS: set[str] = {"c", "html", "css", "markdown"}

THESTACK_REPO_TEMPLATE = "MultilingualUnigramLM/LangMap-TheStack-{lang}-100M"

# Precision fallback order — only retry on OOM, never on task errors
PRECISION_FALLBACKS = ["bf16", "fp16", "fp32"]


# ─── JSONL logging ───────────────────────────────────────────────────────────

def jsonl_log(log_path: Path, obj: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({**obj, "_ts": datetime.now().isoformat()}) + "\n")


# ─── Error classification ─────────────────────────────────────────────────────

def _is_oom(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in ["out of memory", "cuda oom", "oom", "memory"])


# ─── bigcode-eval runner ────────────────────────────────────────────────────

def _find_bigcode_main(override: Optional[str] = None) -> str:
    """Locate bigcode-evaluation-harness main.py."""
    if override:
        p = Path(override)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"bigcode-eval main.py not found at: {override}")

    # Try env var
    env_path = os.environ.get("BIGCODE_EVAL_MAIN")
    if env_path and Path(env_path).exists():
        return env_path

    # Try to find via the installed package
    try:
        import bigcode_eval
        pkg_dir = Path(bigcode_eval.__file__).parent.parent
        main_py = pkg_dir / "main.py"
        if main_py.exists():
            return str(main_py)
    except ImportError:
        pass

    # Try common locations
    for candidate in [
        Path("bigcode-evaluation-harness/main.py"),
        Path.home() / "bigcode-evaluation-harness" / "main.py",
    ]:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Cannot find bigcode-evaluation-harness main.py. "
        "Set BIGCODE_EVAL_MAIN env var or --bigcode-eval-path."
    )


def run_bigcode_eval(
    base_model_id: str,
    peft_model: Optional[str],
    tasks: list[str],
    output_dir: Path,
    run_label: str,
    log_path: Path,
    n_samples: int = 1,
    batch_size: int = 1,
    max_length_generation: int = 512,
    temperature: float = 0.2,
    precision: str = "bf16",
    bigcode_main: str = "",
) -> dict:
    """
    Run bigcode-evaluation-harness via subprocess.
    Returns the parsed metric output JSON.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_path = output_dir / f"bigcode_metrics_{run_label}_{ts}.json"
    gen_path = output_dir / f"generations_{run_label}_{ts}.json"

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        bigcode_main,
        "--model", base_model_id,
        "--tasks", ",".join(tasks),
        "--max_length_generation", str(max_length_generation),
        "--temperature", str(temperature),
        "--n_samples", str(n_samples),
        "--batch_size", str(batch_size),
        "--precision", precision,
        "--allow_code_execution",
        "--metric_output_path", str(metric_path),
        "--save_generations",
        "--save_generations_path", str(gen_path),
    ]

    if peft_model:
        cmd.extend(["--peft_model", peft_model])

    if n_samples > 1:
        cmd.extend(["--do_sample", "True"])

    logger.info(
        f"[bigcode-eval] {run_label}: tasks={tasks}, "
        f"n_samples={n_samples}, precision={precision}"
    )
    logger.info(f"[bigcode-eval] cmd: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_ALLOW_CODE_EVAL": "1"},
    )

    if result.returncode != 0:
        logger.error(f"[bigcode-eval] stderr:\n{result.stderr[-2000:]}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    if result.stdout:
        logger.info(f"[bigcode-eval] stdout (last 500 chars):\n{result.stdout[-500:]}")

    # Read metric output
    if not metric_path.exists():
        raise FileNotFoundError(
            f"bigcode-eval did not produce metric output at {metric_path}"
        )

    with open(metric_path) as f:
        metrics = json.load(f)

    # Save a copy in the standard naming convention for plot compatibility
    compat_file = output_dir / f"eval_results_{run_label}_{ts}.json"
    compat_data = {
        "results": {k: v for k, v in metrics.items() if k != "config"},
        "config": metrics.get("config", {}),
        "eval_engine": "bigcode-eval",
        "lang": run_label.split("_")[0] if "_" in run_label else "",
        "run_label": run_label,
    }
    with open(compat_file, "w") as f:
        json.dump(compat_data, f, indent=2)

    jsonl_log(log_path, {
        "event": "bigcode_eval_done",
        "run": run_label,
        "tasks": tasks,
        "metric_file": str(metric_path),
        "compat_file": str(compat_file),
    })
    logger.info(f"[bigcode-eval] Saved {run_label} → {compat_file}")
    return compat_data


def run_with_precision_fallback(
    base_model_id: str,
    peft_model: Optional[str],
    tasks: list[str],
    output_dir: Path,
    run_label: str,
    log_path: Path,
    n_samples: int,
    batch_size: int,
    max_length_generation: int,
    temperature: float,
    bigcode_main: str,
) -> dict:
    """Try precisions in order, but ONLY retry on OOM errors."""
    last_err: Optional[Exception] = None

    for precision in PRECISION_FALLBACKS:
        logger.info(f"[precision] Trying {precision} for {run_label}")
        try:
            return run_bigcode_eval(
                base_model_id=base_model_id,
                peft_model=peft_model,
                tasks=tasks,
                output_dir=output_dir,
                run_label=run_label,
                log_path=log_path,
                n_samples=n_samples,
                batch_size=batch_size,
                max_length_generation=max_length_generation,
                temperature=temperature,
                precision=precision,
                bigcode_main=bigcode_main,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if _is_oom(stderr):
                logger.warning(f"[precision] OOM with {precision}, trying next.")
                last_err = e
                jsonl_log(log_path, {
                    "event": "oom", "precision": precision,
                    "error": stderr[:200],
                })
            else:
                jsonl_log(log_path, {
                    "event": "fatal", "precision": precision,
                    "error": stderr[:200],
                })
                raise

    raise RuntimeError(f"OOM across all precisions. Last error: {last_err}")


# ─── Metric extraction ──────────────────────────────────────────────────────

def extract_bigcode_metrics(results: dict) -> dict[str, dict[str, float]]:
    """
    Extract metrics from bigcode-eval output.

    Input format (compat_data):
        {"results": {"humaneval": {"pass@1": 0.35}, ...}, "config": {...}}

    Returns:
        {"humaneval": {"pass@1": 0.35}, ...}
    """
    metrics: dict[str, dict[str, float]] = {}
    results_block = results.get("results", {})

    for task_name, task_data in results_block.items():
        if task_name == "config":
            continue
        if isinstance(task_data, dict):
            task_metrics = {}
            for k, v in task_data.items():
                if isinstance(v, (int, float)):
                    task_metrics[k] = float(v)
            if task_metrics:
                metrics[task_name] = task_metrics

    return metrics


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
    """Compute bits-per-byte. Correct formula: BpB = total_nll / (total_bytes * ln2)."""
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


DTYPE_FALLBACKS = ["bfloat16", "float16", "float32"]


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
            if _is_oom(str(e)):
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
    """Extract the most informative single score from a results dict.

    Handles both bigcode-eval and legacy lm-eval formats.
    """
    results_block = results.get("results", results)
    scores = []
    for task_name, task_data in results_block.items():
        if task_name in ("config", "eval_engine", "lang", "run_label"):
            continue
        if isinstance(task_data, dict):
            for key in ("pass@1", "pass@1,none", "acc,none", "acc"):
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
    precision: str,
    output_dir: Path,
) -> Path:
    """Save a compact summary JSON (leaderboard_{lang}.json)."""
    summary = {
        "lang": lang,
        "mode": mode,
        "precision": precision,
        "eval_engine": "bigcode-eval" if mode == "bigcode" else "perplexity",
        "timestamp": datetime.now().isoformat(),
        "pass_at_k": pass_at_k,
    }

    if mode == "bigcode":
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
    parser = argparse.ArgumentParser(description="UniMixLM Eval System v4 (bigcode-eval)")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--base-model", default="allenai/OLMo-3-1025-7B")
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", default="auto", choices=["auto", "bigcode", "perplexity"])
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Completions per problem: 1=pass@1, 10=pass@1+10, 100=pass@1+10+100")
    parser.add_argument("--max-length-generation", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-bpb-samples", type=int, default=200)
    parser.add_argument("--bigcode-eval-path", default=None,
                        help="Path to bigcode-evaluation-harness main.py")
    parser.add_argument("--push-to-hub", default=None,
                        metavar="HF_REPO",
                        help="Upload leaderboard JSON to this HF dataset repo")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="Directory with prior eval_results_ft_{lang}.json for regression check")
    args = parser.parse_args()

    lang = args.lang.lower()
    out  = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "unieval_v4.jsonl"

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

    mode_str = "perplexity" if is_perplexity else "bigcode"
    logger.info(f"UniEval v4 | lang={lang} | mode={mode_str}")

    # ── bigcode-eval path ─────────────────────────────────────────────────────
    if not is_perplexity:
        tasks = BIGCODE_TASK_MAP.get(lang)
        if not tasks:
            raise ValueError(
                f"No bigcode-eval tasks defined for lang='{lang}'. "
                f"Known languages: {sorted(BIGCODE_TASK_MAP.keys())}"
            )
        logger.info(f"[tasks] bigcode-eval tasks for {lang}: {tasks}")

        bigcode_main = _find_bigcode_main(args.bigcode_eval_path)
        logger.info(f"[bigcode-eval] Using main.py at: {bigcode_main}")

        # Run base model
        base_res = run_with_precision_fallback(
            base_model_id=args.base_model,
            peft_model=None,
            tasks=tasks,
            output_dir=out,
            run_label="base",
            log_path=log_path,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            max_length_generation=args.max_length_generation,
            temperature=args.temperature,
            bigcode_main=bigcode_main,
        )

        # Run finetuned model
        ft_res = run_with_precision_fallback(
            base_model_id=args.base_model,
            peft_model=args.adapter_repo,
            tasks=tasks,
            output_dir=out,
            run_label="ft",
            log_path=log_path,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            max_length_generation=args.max_length_generation,
            temperature=args.temperature,
            bigcode_main=bigcode_main,
        )

        pass_at_k = {
            "base": extract_bigcode_metrics(base_res),
            "ft":   extract_bigcode_metrics(ft_res),
        }

    # ── Perplexity path ───────────────────────────────────────────────────────
    else:
        base_res = run_bpb(
            args.base_model, None, lang, out, "base", log_path, args.max_bpb_samples
        )
        ft_res = run_bpb(
            args.base_model, args.adapter_repo, lang, out, "ft", log_path, args.max_bpb_samples
        )
        pass_at_k = {}

    # ── Summary / leaderboard ─────────────────────────────────────────────────
    precision_used = args.precision if not is_perplexity else "bfloat16"
    summary_path = build_leaderboard_summary(
        lang, base_res, ft_res, pass_at_k, mode_str, precision_used, out
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
