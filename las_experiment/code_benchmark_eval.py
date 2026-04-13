#!/usr/bin/env python3
"""
UniMixLM Eval System v4 — bigcode-evaluation-harness

Per-language code benchmark evaluation using bigcode-evaluation-harness.
Each language runs ONLY its language-specific benchmarks.

- Python:     HumanEval, MBPP, HumanEval+
- Go/Java/JS/TS/C++/C#/PHP/C: MultiPL-E + HumanEvalPack where available
- HTML/CSS/Markdown:           BpB perplexity (no functional benchmark)
- pass@k (k=1,10,100) via multiple-sample generation
- BpB perplexity on TheStack (correct byte-based formula)
- LAS tokenizer support (critical for correct evaluation)
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
        --las-tokenizer-dir ./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B \\
        --n-samples 10

    # Perplexity mode (html, css, markdown)
    python3 las_experiment/code_benchmark_eval.py \\
        --lang html \\
        --base-model allenai/OLMo-3-1025-7B \\
        --adapter-repo MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html \\
        --output-dir ./eval_results/code_html \\
        --las-tokenizer-dir ./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B \\
        --mode perplexity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# ─── Environment (must be set before any HF imports) ─────────────────────────
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("unieval")

# ─── Config ──────────────────────────────────────────────────────────────────

# Language -> list of bigcode-eval task names
# Each language evaluated ONLY with its own benchmarks
BIGCODE_TASK_MAP: dict[str, list[str]] = {
    "python":     ["humaneval", "mbpp", "humanevalplus"],
    "go":         ["multiple-go", "humanevalsynthesize-go"],
    "typescript": ["multiple-ts"],
    "java":       ["multiple-java", "humanevalsynthesize-java"],
    "javascript": ["multiple-js", "humanevalsynthesize-javascript"],
    "cpp":        ["humanevalsynthesize-cpp"],
    "csharp":     ["multiple-cs"],
    "php":        ["multiple-php"],
}

# HumanEvalPack tasks require --prompt; run separately from standard tasks
PROMPT_REQUIRED_PREFIXES = ("humanevalsynthesize-", "humanevalfix-", "humanevalexplain-")

# Languages that have no functional benchmark → perplexity (BpB) only
PERPLEXITY_LANGS: set[str] = {"html", "css", "markdown", "c"}

# Exact HF model names for each language adapter
ADAPTER_MAP: dict[str, str] = {
    "python":     "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-python",
    "go":         "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-go",
    "typescript": "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-typescript",
    "java":       "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-java",
    "javascript": "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-javascript",
    "cpp":        "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-cpp",
    "csharp":     "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-csharp",
    "php":        "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-php",
    "c":          "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-c",
    "html":       "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html",
    "css":        "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-css",
    "markdown":   "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-markdown",
}

THESTACK_REPO_TEMPLATE = "MultilingualUnigramLM/LangMap-TheStack-{lang}-100M"

# Precision fallback order — only retry on OOM, never on task errors
PRECISION_FALLBACKS = ["bf16", "fp16", "fp32"]
DTYPE_FALLBACKS = ["bfloat16", "float16", "float32"]


# ─── JSONL logging ───────────────────────────────────────────────────────────

def jsonl_log(log_path: Path, obj: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({**obj, "_ts": datetime.now().isoformat()}) + "\n")


# ─── Evaluation timer ──────────────────────────────────────────────────────────

class EvalTimer:
    """Track and log evaluation phase timings with ETA estimates and tqdm bar."""

    def __init__(self, lang: str, total_phases: int = 0):
        self.lang = lang
        self.start_time = time.monotonic()
        self.phases: list[tuple[str, float]] = []
        self.total_phases = total_phases
        self._phase_start: float | None = None
        self._phase_name: str = ""
        # Top-level tqdm bar tracking phases
        self._pbar: tqdm | None = None
        if total_phases > 0:
            self._pbar = tqdm(
                total=total_phases,
                desc=f"[{lang}] Evaluation phases",
                unit="phase",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                leave=True,
            )

    def start_phase(self, name: str) -> None:
        if self._phase_start is not None:
            self._end_current_phase()
        self._phase_name = name
        self._phase_start = time.monotonic()
        elapsed_total = self._format_duration(time.monotonic() - self.start_time)
        completed = len(self.phases)
        remaining = self.total_phases - completed if self.total_phases else "?"
        logger.info(
            f"[timer] [{self.lang}] Phase {completed + 1}"
            f"{'/' + str(self.total_phases) if self.total_phases else ''}"
            f": Starting '{name}' | Total elapsed: {elapsed_total}"
            f" | Phases remaining after this: {remaining}"
        )
        if self._pbar is not None:
            self._pbar.set_postfix_str(name[:60], refresh=True)
        if completed > 0 and self.total_phases > 0:
            avg_phase = (time.monotonic() - self.start_time) / completed
            est_remaining = avg_phase * (self.total_phases - completed)
            logger.info(
                f"[timer] [{self.lang}] Estimated time remaining: "
                f"~{self._format_duration(est_remaining)} "
                f"(based on avg {self._format_duration(avg_phase)}/phase)"
            )

    def _end_current_phase(self) -> None:
        if self._phase_start is not None:
            elapsed = time.monotonic() - self._phase_start
            self.phases.append((self._phase_name, elapsed))
            logger.info(
                f"[timer] [{self.lang}] Completed '{self._phase_name}' "
                f"in {self._format_duration(elapsed)}"
            )
            if self._pbar is not None:
                self._pbar.update(1)
            self._phase_start = None

    def end_phase(self) -> None:
        self._end_current_phase()

    def total_elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def summary(self) -> str:
        self._end_current_phase()
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        total = self.total_elapsed()
        lines = [
            "",
            "=" * 70,
            f"  TIMING SUMMARY — {self.lang}",
            "=" * 70,
        ]
        for i, (name, dur) in enumerate(self.phases, 1):
            pct = (dur / total * 100) if total > 0 else 0
            lines.append(
                f"  {i:2d}. {name:42s} "
                f"{self._format_duration(dur):>12s}  ({pct:5.1f}%)"
            )
        lines.append("-" * 70)
        lines.append(f"      {'TOTAL':42s} {self._format_duration(total):>12s}")
        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m {int(s)}s"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)}h {int(m)}m {int(s)}s"


# ─── Error classification ─────────────────────────────────────────────────────

def _is_oom(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in [
        "out of memory", "cuda out of memory", "cuda oom",
        "torch.cuda.outofmemoryerror", "runtime error: cuda",
        "allocator", "oom killed",
    ])


# ─── Robust dataset loading (fixes HF cache + config mismatch) ──────────────

def _parse_cached_configs(error_msg: str) -> list[str]:
    """Extract cached config names from HF datasets error messages.

    Errors look like: "Available configs in the cache: ['css-540ede6bde1cefae']"
    """
    import re
    m = re.search(r"Available configs in the cache:\s*\[([^\]]+)\]", str(error_msg))
    if not m:
        return []
    return [c.strip().strip("'\"") for c in m.group(1).split(",")]


def safe_load_dataset(repo: str, split: str = "test", max_retries: int = 3):
    """
    Robust dataset loading with fallbacks for:
    - Missing repo on HF
    - Broken cache / config name mismatch
    - Missing test split
    """
    from datasets import load_dataset, get_dataset_config_names

    # Attempt 1: direct load
    first_err = None
    try:
        return load_dataset(repo, split=split, streaming=True)
    except Exception as e:
        first_err = e
        logger.warning(f"[dataset] primary load failed for {repo}: {e}")

    # Attempt 2: try cached config names parsed from the error message
    cached_configs = _parse_cached_configs(str(first_err))
    if cached_configs:
        logger.info(f"[dataset] trying cached configs: {cached_configs}")
        for cfg in cached_configs:
            for sp in (split, "train"):
                try:
                    return load_dataset(repo, name=cfg, split=sp, streaming=True)
                except Exception:
                    pass
        logger.warning(f"[dataset] cached config fallback failed for {repo}")

    # Attempt 3: enumerate configs from HF, pick first valid one
    try:
        configs = get_dataset_config_names(repo)
        logger.warning(f"[dataset] available configs for {repo}: {configs}")
        if configs:
            config = configs[0]
            try:
                return load_dataset(
                    repo, name=config, split=split,
                    streaming=True,
                )
            except Exception:
                # Try train split if test doesn't exist
                return load_dataset(
                    repo, name=config, split="train",
                    streaming=True,
                )
    except Exception as e2:
        logger.warning(f"[dataset] config-based load failed: {e2}")

    # Attempt 4: fallback to train split
    try:
        return load_dataset(repo, split="train", streaming=True)
    except Exception as e3:
        logger.warning(f"[dataset] train split fallback failed: {e3}")

    raise RuntimeError(f"Could not load dataset {repo} after all fallback attempts")


# ─── LAS tokenizer preparation ──────────────────────────────────────────────

def prepare_las_tokenizer_dir(
    base_model_id: str,
    lang: str,
    las_tokenizer_dir: str,
) -> Optional[str]:
    """
    Create an HF-compatible tokenizer directory with the LAS tokenizer.

    Steps:
      1. Load base model tokenizer → save to temp dir (gets all config files)
      2. Overwrite tokenizer.json with the LAS langspec tokenizer
      3. Return temp dir path

    Returns None if LAS tokenizer file not found.
    """
    las_tok_file = Path(las_tokenizer_dir) / f"langspec_{lang}.json"
    if not las_tok_file.exists():
        logger.warning(
            f"[las-tok] LAS tokenizer not found: {las_tok_file}. "
            f"Falling back to base model tokenizer."
        )
        return None

    from transformers import AutoTokenizer

    tmp_dir = tempfile.mkdtemp(prefix=f"las_tok_{lang}_")

    # Save base tokenizer to get all config files
    base_tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_tok.save_pretrained(tmp_dir)

    # Overwrite tokenizer.json with LAS tokenizer
    shutil.copy2(str(las_tok_file), os.path.join(tmp_dir, "tokenizer.json"))

    logger.info(f"[las-tok] Prepared LAS tokenizer dir: {tmp_dir} (from {las_tok_file})")
    return tmp_dir


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
    las_tokenizer_path: Optional[str] = None,
    prompt_type: Optional[str] = None,
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

    # HumanEvalPack tasks require --prompt
    if prompt_type:
        cmd.extend(["--prompt", prompt_type])

    # Use LAS tokenizer if available
    if las_tokenizer_path:
        cmd.extend(["--tokenizer", las_tokenizer_path])

    if n_samples > 1:
        cmd.extend(["--do_sample", "True"])

    logger.info(
        f"[bigcode-eval] {run_label}: tasks={tasks}, "
        f"n_samples={n_samples}, precision={precision}, "
        f"batch_size={batch_size}, max_len={max_length_generation}, "
        f"temperature={temperature}"
    )
    if peft_model:
        logger.info(f"[bigcode-eval] {run_label}: PEFT adapter = {peft_model}")
    if las_tokenizer_path:
        logger.info(f"[bigcode-eval] {run_label}: LAS tokenizer = {las_tokenizer_path}")
    if prompt_type:
        logger.info(f"[bigcode-eval] {run_label}: prompt_type = {prompt_type}")
    logger.info(f"[bigcode-eval] {run_label}: cmd: {' '.join(cmd)}")

    logger.info(f"[bigcode-eval] {run_label}: running subprocess (this may take a while)...")
    eval_start = time.monotonic()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_ALLOW_CODE_EVAL": "1",
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )

    eval_elapsed = time.monotonic() - eval_start
    logger.info(
        f"[bigcode-eval] {run_label}: subprocess finished in "
        f"{EvalTimer._format_duration(eval_elapsed)} (returncode={result.returncode})"
    )

    if result.returncode != 0:
        logger.error(f"[bigcode-eval] {run_label}: FAILED — stderr:\n{result.stderr[-2000:]}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    if result.stdout:
        logger.info(f"[bigcode-eval] {run_label}: stdout (last 500 chars):\n{result.stdout[-500:]}")

    # Read metric output
    if not metric_path.exists():
        raise FileNotFoundError(
            f"bigcode-eval did not produce metric output at {metric_path}"
        )

    with open(metric_path) as f:
        metrics = json.load(f)

    # Log per-task metric breakdown
    for task_name, task_data in metrics.items():
        if task_name == "config":
            continue
        if isinstance(task_data, dict):
            metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in task_data.items()]
            logger.info(
                f"[bigcode-eval] {run_label} | task={task_name} | "
                + ", ".join(metric_strs)
            )

    # Save a copy in the standard naming convention for plot compatibility
    compat_file = output_dir / f"eval_results_{run_label}_{ts}.json"
    compat_data = {
        "results": {k: v for k, v in metrics.items() if k != "config"},
        "config": metrics.get("config", {}),
        "eval_engine": "bigcode-eval",
        "lang": "",
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
    logger.info(f"[bigcode-eval] Saved {run_label} -> {compat_file}")
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
    las_tokenizer_path: Optional[str] = None,
    prompt_type: Optional[str] = None,
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
                prompt_type=prompt_type,
                las_tokenizer_path=las_tokenizer_path,
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

def _load_model(
    base_model_id: str,
    adapter: Optional[str],
    dtype_str: str,
    tokenizer_path: Optional[str] = None,
):
    """
    Load model + optional PEFT adapter + optional LAS tokenizer.
    Includes OLMo rope_scaling fix.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }.get(dtype_str, torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix OLMo rope_scaling config (beta_fast/beta_slow must be float)
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if isinstance(config.rope_scaling, dict):
            for key in ("beta_fast", "beta_slow"):
                if key in config.rope_scaling:
                    config.rope_scaling[key] = float(config.rope_scaling[key])

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    # Use LAS tokenizer if provided, otherwise base model tokenizer
    tok_source = tokenizer_path if tokenizer_path else base_model_id
    tok = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    return model, tok, device


def compute_bpb(model, tokenizer, texts: list[str], device: str,
                max_seq_len: int = 512, lang: str = "") -> float:
    """Compute bits-per-byte. Correct formula: BpB = total_nll / (total_bytes * ln2)."""
    total_nll = 0.0
    total_bytes = 0
    skipped = 0

    with torch.no_grad():
        pbar = tqdm(
            texts,
            desc=f"[{lang}] Computing BpB" if lang else "Computing BpB",
            unit="sample",
            leave=False,
        )
        for text in pbar:
            if not text or not text.strip():
                skipped += 1
                continue
            n_bytes = len(text.encode("utf-8"))
            if n_bytes == 0:
                skipped += 1
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len, padding=False)
            ids = enc["input_ids"].to(device)
            if ids.shape[1] < 2:
                skipped += 1
                continue

            loss = model(input_ids=ids, labels=ids).loss.item()
            n_tokens = ids.shape[1]
            total_nll   += loss * n_tokens
            total_bytes += n_bytes

            # Live BpB estimate in progress bar
            running_bpb = total_nll / (total_bytes * math.log(2))
            pbar.set_postfix(bpb=f"{running_bpb:.4f}", skipped=skipped)
        pbar.close()

    logger.info(
        f"[bpb] [{lang}] Processed {len(texts) - skipped}/{len(texts)} samples "
        f"({skipped} skipped), total_bytes={total_bytes:,}"
    )

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
    tokenizer_path: Optional[str] = None,
) -> dict:
    """Compute BpB on TheStack test split. Saves eval_results_{run_label}_{ts}.json."""
    repo = THESTACK_REPO_TEMPLATE.format(lang=lang)
    logger.info(f"[bpb] [{lang}] {run_label}: loading dataset {repo}")

    # Use robust dataset loading (fixes config mismatch / cache issues)
    ds = safe_load_dataset(repo)
    logger.info(f"[bpb] [{lang}] {run_label}: dataset loaded, collecting up to {max_samples} samples")

    texts: list[str] = []
    col = None
    pbar = tqdm(
        total=max_samples,
        desc=f"[{lang}] Loading samples ({run_label})",
        unit="sample",
        leave=False,
    )
    for i, sample in enumerate(ds):
        if col is None:
            col = next(
                (c for c in ("content", "text", "code") if c in sample),
                list(sample.keys())[0],
            )
            logger.info(f"[bpb] [{lang}] Using column '{col}' from dataset")
        texts.append(sample[col])
        pbar.update(1)
        if len(texts) >= max_samples:
            break
    pbar.close()

    if not texts:
        raise ValueError(f"No samples found in {repo}")

    avg_len = sum(len(t) for t in texts) / len(texts)
    logger.info(
        f"[bpb] [{lang}] {run_label}: collected {len(texts)} samples "
        f"(avg {avg_len:.0f} chars), computing BpB..."
    )

    # Try dtype fallback for BpB too
    bpb = float("nan")
    for dtype in DTYPE_FALLBACKS:
        try:
            logger.info(f"[bpb] [{lang}] {run_label}: loading model (dtype={dtype})")
            model, tok, device = _load_model(
                base_model_id, adapter, dtype,
                tokenizer_path=tokenizer_path,
            )
            logger.info(
                f"[bpb] [{lang}] {run_label}: model loaded on {device}, "
                f"vocab_size={len(tok)}, dtype={dtype}"
            )
            bpb = compute_bpb(model, tok, texts, device, max_seq_len, lang=lang)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        except Exception as e:
            if _is_oom(str(e)):
                logger.warning(f"[bpb] [{lang}] OOM with {dtype}, trying next.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    logger.info(f"[bpb] [{lang}] {run_label}: BpB = {bpb:.4f}")

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
    logger.info(f"[bpb] Saved {run_label} -> {out_file}")
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
        logger.info(f"[hub] Uploaded {result_file.name} -> {hub_repo}")
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
    base_bpb_override: float | None = None,
    ft_bpb_override: float | None = None,
) -> Path:
    """Save a compact summary JSON (leaderboard_{lang}.json).

    For bigcode-mode languages, pass base_bpb_override / ft_bpb_override
    to include BpB alongside pass@k metrics.
    """
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
        # Include BpB if computed for this language
        if base_bpb_override is not None:
            summary["base_bpb"] = base_bpb_override
        if ft_bpb_override is not None:
            summary["ft_bpb"] = ft_bpb_override
        if base_bpb_override is not None and ft_bpb_override is not None:
            summary["delta_bpb"] = base_bpb_override - ft_bpb_override
    else:  # perplexity
        base_bpb = base_results.get("results", {}).get("bpb")
        ft_bpb   = ft_results.get("results", {}).get("bpb")
        summary["base_bpb"] = base_bpb
        summary["ft_bpb"]   = ft_bpb
        summary["delta_bpb"] = (base_bpb - ft_bpb) if (base_bpb and ft_bpb) else None

    path = output_dir / f"leaderboard_{lang}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[leaderboard] Saved -> {path}")
    return path


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UniMixLM Eval System v4 (bigcode-eval)")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--base-model", default="allenai/OLMo-3-1025-7B")
    parser.add_argument("--adapter-repo", required=True,
                        help="Exact HF repo name for the PEFT adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", default="auto", choices=["auto", "bigcode", "perplexity"])
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Completions per problem: 1=pass@1, 10=pass@1+10, 100=pass@1+10+100")
    parser.add_argument("--max-length-generation", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-bpb-samples", type=int, default=200)
    parser.add_argument("--las-tokenizer-dir", default=None,
                        help="Path to LAS tokenizer dir containing langspec_{lang}.json files")
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

    # Validate adapter repo matches expected pattern
    expected_adapter = ADAPTER_MAP.get(lang)
    if expected_adapter and args.adapter_repo != expected_adapter:
        logger.warning(
            f"[adapter] Provided adapter '{args.adapter_repo}' differs from "
            f"expected '{expected_adapter}'. Using provided value."
        )

    jsonl_log(log_path, {
        "event": "start",
        "lang": lang,
        "base_model": args.base_model,
        "adapter": args.adapter_repo,
        "n_samples": args.n_samples,
    })

    # Prepare LAS tokenizer if available
    las_tok_dir = None
    if args.las_tokenizer_dir:
        las_tok_dir = prepare_las_tokenizer_dir(
            args.base_model, lang, args.las_tokenizer_dir,
        )
        if las_tok_dir:
            logger.info(f"[las-tok] Using LAS tokenizer: {las_tok_dir}")
        else:
            logger.warning("[las-tok] LAS tokenizer not available, using base model tokenizer")

    # Determine mode
    if args.mode == "auto":
        is_perplexity = lang in PERPLEXITY_LANGS
    else:
        is_perplexity = args.mode == "perplexity"

    mode_str = "perplexity" if is_perplexity else "bigcode"

    # ── Startup banner ───────────────────────────────────────────────────────
    banner = "\n".join([
        "",
        "=" * 70,
        f"  UniMixLM Eval System v4 — {lang.upper()}",
        "=" * 70,
        f"  Language       : {lang}",
        f"  Mode           : {mode_str}",
        f"  Base model     : {args.base_model}",
        f"  Adapter        : {args.adapter_repo}",
        f"  LAS tokenizer  : {las_tok_dir or 'None (using base)'}",
        f"  n_samples      : {args.n_samples}",
        f"  batch_size     : {args.batch_size}",
        f"  temperature    : {args.temperature}",
        f"  precision      : {args.precision}",
        f"  output_dir     : {out}",
        f"  GPU available  : {torch.cuda.is_available()}",
    ])
    if torch.cuda.is_available():
        banner += f"\n  GPU device     : {torch.cuda.get_device_name(0)}"
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        banner += f"\n  GPU memory     : {gpu_mem:.1f} GB"
    banner += "\n" + "=" * 70
    logger.info(banner)

    # ── Initialise evaluation timer ──────────────────────────────────────────
    if is_perplexity:
        n_phases = 2  # BpB base + BpB ft
    else:
        all_tasks_preview = BIGCODE_TASK_MAP.get(lang, [])
        has_prompt = any(t.startswith(PROMPT_REQUIRED_PREFIXES) for t in all_tasks_preview)
        # std base, std ft, [prompt base, prompt ft], BpB base, BpB ft
        n_phases = 4 + (2 if has_prompt else 0)
    timer = EvalTimer(lang, total_phases=n_phases)

    # ── bigcode-eval path ─────────────────────────────────────────────────────
    if not is_perplexity:
        all_tasks = BIGCODE_TASK_MAP.get(lang)
        if not all_tasks:
            raise ValueError(
                f"No bigcode-eval tasks defined for lang='{lang}'. "
                f"Known languages: {sorted(BIGCODE_TASK_MAP.keys())}"
            )
        logger.info(f"[tasks] [{lang}] bigcode-eval tasks: {all_tasks}")

        # Split into standard tasks and tasks requiring --prompt
        std_tasks = [t for t in all_tasks
                     if not t.startswith(PROMPT_REQUIRED_PREFIXES)]
        prompt_tasks = [t for t in all_tasks
                        if t.startswith(PROMPT_REQUIRED_PREFIXES)]

        logger.info(f"[tasks] [{lang}] Standard tasks ({len(std_tasks)}): {std_tasks}")
        if prompt_tasks:
            logger.info(f"[tasks] [{lang}] HumanEvalPack tasks ({len(prompt_tasks)}): {prompt_tasks}")
        else:
            logger.info(f"[tasks] [{lang}] No HumanEvalPack tasks for this language")

        bigcode_main = _find_bigcode_main(args.bigcode_eval_path)
        logger.info(f"[bigcode-eval] Using main.py at: {bigcode_main}")

        def _run_task_group(tasks, run_label, peft_model, las_tokenizer_path,
                           prompt_type=None):
            return run_with_precision_fallback(
                base_model_id=args.base_model,
                peft_model=peft_model,
                tasks=tasks,
                output_dir=out,
                run_label=run_label,
                log_path=log_path,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                max_length_generation=args.max_length_generation,
                temperature=args.temperature,
                bigcode_main=bigcode_main,
                las_tokenizer_path=las_tokenizer_path,
                prompt_type=prompt_type,
            )

        def _merge_results(a: dict, b: dict) -> dict:
            """Merge two bigcode-eval compat_data dicts."""
            merged = dict(a)
            merged["results"] = {**a.get("results", {}), **b.get("results", {})}
            return merged

        # ── Run standard tasks ───────────────────────────────────────────
        timer.start_phase(f"bigcode-eval base model ({', '.join(std_tasks)})")
        base_res = _run_task_group(
            std_tasks, "base", peft_model=None, las_tokenizer_path=None,
        )
        timer.end_phase()

        timer.start_phase(f"bigcode-eval finetuned model ({', '.join(std_tasks)})")
        ft_res = _run_task_group(
            std_tasks, "ft", peft_model=args.adapter_repo,
            las_tokenizer_path=las_tok_dir,
        )
        timer.end_phase()

        # ── Run prompt-required tasks (non-fatal if they fail) ───────────
        if prompt_tasks:
            logger.info(
                f"[tasks] Running HumanEvalPack tasks with --prompt continue: "
                f"{prompt_tasks}"
            )
            timer.start_phase(f"HumanEvalPack base ({', '.join(prompt_tasks)})")
            try:
                prompt_base = _run_task_group(
                    prompt_tasks, "base_synth", peft_model=None,
                    las_tokenizer_path=None, prompt_type="continue",
                )
                base_res = _merge_results(base_res, prompt_base)
            except Exception as e:
                logger.warning(
                    f"[tasks] HumanEvalPack base run failed (non-fatal): {e}"
                )
            timer.end_phase()

            timer.start_phase(f"HumanEvalPack finetuned ({', '.join(prompt_tasks)})")
            try:
                prompt_ft = _run_task_group(
                    prompt_tasks, "ft_synth",
                    peft_model=args.adapter_repo,
                    las_tokenizer_path=las_tok_dir, prompt_type="continue",
                )
                ft_res = _merge_results(ft_res, prompt_ft)
            except Exception as e:
                logger.warning(
                    f"[tasks] HumanEvalPack ft run failed (non-fatal): {e}"
                )
            timer.end_phase()

        pass_at_k = {
            "base": extract_bigcode_metrics(base_res),
            "ft":   extract_bigcode_metrics(ft_res),
        }

        # ── BpB for this language (computed for ALL languages) ───────────
        logger.info(f"[bpb] Computing BpB for {lang} (all languages get BpB)")
        base_bpb = None
        ft_bpb = None

        timer.start_phase(f"BpB computation — base model ({lang})")
        try:
            bpb_base_res = run_bpb(
                args.base_model, None, lang, out, "base_bpb", log_path,
                args.max_bpb_samples, tokenizer_path=None,
            )
            base_bpb = bpb_base_res.get("results", {}).get("bpb")
            logger.info(f"[bpb] Base BpB for {lang}: {base_bpb:.4f}")
        except Exception as e:
            logger.warning(f"[bpb] Base BpB computation failed (non-fatal): {e}")
        timer.end_phase()

        timer.start_phase(f"BpB computation — finetuned model ({lang})")
        try:
            bpb_ft_res = run_bpb(
                args.base_model, args.adapter_repo, lang, out, "ft_bpb", log_path,
                args.max_bpb_samples, tokenizer_path=las_tok_dir,
            )
            ft_bpb = bpb_ft_res.get("results", {}).get("bpb")
            logger.info(f"[bpb] FT BpB for {lang}: {ft_bpb:.4f}")
        except Exception as e:
            logger.warning(f"[bpb] FT BpB computation failed (non-fatal): {e}")
        timer.end_phase()

        if base_bpb is not None and ft_bpb is not None:
            delta = base_bpb - ft_bpb
            logger.info(
                f"[bpb] {lang} BpB delta: {delta:+.4f} "
                f"({'improved' if delta > 0 else 'regressed'})"
            )

    # ── Perplexity path ───────────────────────────────────────────────────────
    else:
        logger.info(f"[{lang}] Perplexity-only mode (no functional benchmarks)")
        base_bpb = None
        ft_bpb = None

        timer.start_phase(f"BpB computation — base model ({lang})")
        base_res = run_bpb(
            args.base_model, None, lang, out, "base", log_path,
            args.max_bpb_samples, tokenizer_path=None,
        )
        base_bpb = base_res.get("results", {}).get("bpb")
        logger.info(f"[{lang}] Base model BpB = {base_bpb:.4f}" if base_bpb else f"[{lang}] Base BpB unavailable")
        timer.end_phase()

        timer.start_phase(f"BpB computation — finetuned model ({lang})")
        ft_res = run_bpb(
            args.base_model, args.adapter_repo, lang, out, "ft", log_path,
            args.max_bpb_samples, tokenizer_path=las_tok_dir,
        )
        ft_bpb = ft_res.get("results", {}).get("bpb")
        logger.info(f"[{lang}] Finetuned model BpB = {ft_bpb:.4f}" if ft_bpb else f"[{lang}] FT BpB unavailable")
        timer.end_phase()

        pass_at_k = {}

    # ── Summary / leaderboard ─────────────────────────────────────────────────
    precision_used = args.precision if not is_perplexity else "bfloat16"
    summary_path = build_leaderboard_summary(
        lang, base_res, ft_res, pass_at_k, mode_str, precision_used, out,
        base_bpb_override=base_bpb if not is_perplexity else None,
        ft_bpb_override=ft_bpb if not is_perplexity else None,
    )

    # ── HF leaderboard sync ───────────────────────────────────────────────────
    if args.push_to_hub:
        push_to_hub(summary_path, args.push_to_hub, lang)

    # ── Regression check ──────────────────────────────────────────────────────
    if args.baseline_dir:
        run_regression_check(lang, ft_res, args.baseline_dir, out, log_path)

    # ── Cleanup temp LAS tokenizer dir ────────────────────────────────────────
    if las_tok_dir and os.path.exists(las_tok_dir):
        shutil.rmtree(las_tok_dir, ignore_errors=True)
        logger.info(f"[las-tok] Cleaned up temp dir: {las_tok_dir}")

    # ── Results summary ────────────────────────────────────────────────────────
    results_banner = [
        "",
        "=" * 70,
        f"  RESULTS SUMMARY — {lang.upper()}",
        "=" * 70,
    ]
    if pass_at_k:
        for model_type, task_metrics in pass_at_k.items():
            for task_name, metrics in task_metrics.items():
                metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
                results_banner.append(
                    f"  [{model_type:>4s}] {task_name:30s}  {', '.join(metric_strs)}"
                )
    if base_bpb is not None:
        results_banner.append(f"  [base] BpB: {base_bpb:.4f}")
    if ft_bpb is not None:
        results_banner.append(f"  [  ft] BpB: {ft_bpb:.4f}")
    if base_bpb is not None and ft_bpb is not None:
        delta = base_bpb - ft_bpb
        results_banner.append(
            f"  [diff] BpB delta: {delta:+.4f} "
            f"({'improved' if delta > 0 else 'regressed'})"
        )
    results_banner.append("=" * 70)
    logger.info("\n".join(results_banner))

    # ── Timing summary ───────────────────────────────────────────────────────
    timing_summary = timer.summary()
    logger.info(timing_summary)
    jsonl_log(log_path, {
        "event": "done",
        "lang": lang,
        "total_elapsed_seconds": round(timer.total_elapsed(), 1),
        "phase_timings": {name: round(dur, 1) for name, dur in timer.phases},
    })
    logger.info(f"[{lang}] Done. Results in: {out}")


if __name__ == "__main__":
    main()
