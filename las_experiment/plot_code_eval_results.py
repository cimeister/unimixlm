#!/usr/bin/env python3
"""
Generate evaluation plots for UniMixLM code benchmarks.

Reads eval_results/code_{lang}/ directories, produces:
  1. Per-language grouped bar charts  (per_lang/bar_{lang}.png)
  2. Cross-language score heatmap     (heatmap_ft_scores.png)
  3. Improvement delta heatmap        (heatmap_delta.png)
  4. Summary improvement bar chart    (summary_delta.png)
  5. CSV summary table                (results_summary.csv)

Supports both bigcode-evaluation-harness (v4) and legacy lm-eval (v3) output formats.

Usage:
    python3 las_experiment/plot_code_eval_results.py \\
        --results-dir eval_results \\
        --output-dir eval_results/code_summary \\
        --langs python go typescript java javascript cpp csharp php c html css markdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False
    warnings.warn("seaborn not found; using matplotlib fallback for heatmaps.")

# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Constants ---
ALL_LANGS = ["python", "go", "typescript", "java", "javascript", "cpp", "csharp",
             "php", "c", "html", "css", "markdown"]

PERPLEXITY_LANGS = {"html", "css", "markdown"}

# Display-friendly task labels (bigcode-eval task names use hyphens)
TASK_LABELS = {
    # Python-native benchmarks
    "humaneval":                       "HumanEval\n(pass@1)",
    "mbpp":                            "MBPP\n(pass@1)",
    "humanevalplus":                   "HumanEval+\n(pass@1)",
    # MultiPL-E (bigcode-eval naming: hyphens)
    "multiple-go":                     "MultiPL-E Go\n(pass@1)",
    "multiple-ts":                     "MultiPL-E TS\n(pass@1)",
    "multiple-java":                   "MultiPL-E Java\n(pass@1)",
    "multiple-js":                     "MultiPL-E JS\n(pass@1)",
    "multiple-cpp":                    "MultiPL-E C++\n(pass@1)",
    "multiple-cs":                     "MultiPL-E C#\n(pass@1)",
    "multiple-c":                      "MultiPL-E C\n(pass@1)",
    "multiple-php":                    "MultiPL-E PHP\n(pass@1)",
    # HumanEvalPack synthesis
    "humanevalsynthesize-go":          "HEPack Go\n(pass@1)",
    "humanevalsynthesize-java":        "HEPack Java\n(pass@1)",
    "humanevalsynthesize-javascript":  "HEPack JS\n(pass@1)",
    "humanevalsynthesize-js":          "HEPack JS\n(pass@1)",
    "humanevalsynthesize-cpp":         "HEPack C++\n(pass@1)",
    # Legacy lm-eval MultiPL-E (underscore naming)
    "multiple_go":                     "MultiPL-E Go\n(pass@1)",
    "multiple_ts":                     "MultiPL-E TS\n(pass@1)",
    "multiple_java":                   "MultiPL-E Java\n(pass@1)",
    "multiple_js":                     "MultiPL-E JS\n(pass@1)",
    "multiple_cpp":                    "MultiPL-E C++\n(pass@1)",
    "multiple_cs":                     "MultiPL-E C#\n(pass@1)",
    "multiple_php":                    "MultiPL-E PHP\n(pass@1)",
    # Perplexity
    "bpb":                             "BpB\n(lower better)",
}

# Publication-quality matplotlib style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colour palette: base = steel blue, ft = coral
COLORS = {"base": "#4878CF", "ft": "#E87722"}


# ─── Data loading ────────────────────────────────────────────────────────────

def _find_latest(out_dir: Path, run_label: str) -> Optional[Path]:
    """Find the most recent eval_results_{run_label}_*.json in out_dir."""
    pattern = re.compile(rf"eval_results_{re.escape(run_label)}_(\d{{8}}_\d{{6}})\.json$")
    candidates = []
    for p in out_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def _find_leaderboard(out_dir: Path, lang: str) -> Optional[Path]:
    """Find leaderboard_{lang}.json in out_dir (compact summary format)."""
    p = out_dir / f"leaderboard_{lang}.json"
    return p if p.exists() else None


def _extract_from_leaderboard(lb: dict, run_label: str) -> dict[str, tuple[float, float]]:
    """
    Extract metrics from a leaderboard_{lang}.json.
    Supports both bigcode mode and perplexity mode.
    """
    metrics: dict[str, tuple[float, float]] = {}
    mode = lb.get("mode", "")

    if mode == "perplexity":
        bpb_key = "base_bpb" if run_label == "base" else "ft_bpb"
        bpb = lb.get(bpb_key)
        if bpb is not None:
            metrics["bpb"] = (float(bpb), 0.0)

    elif mode in ("lmeval", "bigcode"):
        pak = lb.get("pass_at_k", {}).get(run_label, {})
        for task, task_metrics in pak.items():
            if not isinstance(task_metrics, dict):
                continue
            score = task_metrics.get("pass@1")
            if score is not None:
                metrics[task] = (float(score),
                                 float(task_metrics.get("pass@1_stderr", 0.0)))
        # Also extract BpB if present (computed for all languages)
        bpb_key = "base_bpb" if run_label == "base" else "ft_bpb"
        bpb = lb.get(bpb_key)
        if bpb is not None:
            metrics["bpb"] = (float(bpb), 0.0)

    return metrics


def _extract_code_metrics(data: dict) -> dict[str, tuple[float, float]]:
    """Extract {task: (score, stderr)} from a bigcode-eval or lm-eval results JSON.

    Handles both formats:
      - bigcode-eval: {"results": {"humaneval": {"pass@1": 0.35}}}
      - lm-eval:      {"results": {"humaneval": {"pass@1,none": 0.35, "pass@1_stderr,none": 0.03}}}
    """
    metrics: dict[str, tuple[float, float]] = {}
    results_block = data.get("results", {})
    for task_name, task_data in results_block.items():
        if not isinstance(task_data, dict):
            continue
        if task_name in ("config", "eval_engine", "lang", "run_label"):
            continue

        # Try bigcode-eval format first: {"pass@1": 0.35}
        score = task_data.get("pass@1")
        stderr = 0.0

        # Fallback to lm-eval format: {"pass@1,none": 0.35}
        if score is None:
            score = task_data.get("pass@1,none") or task_data.get("pass@1,create_test")
            stderr = float(task_data.get("pass@1_stderr,none") or
                          task_data.get("pass@1_stderr,create_test") or
                          task_data.get("pass@1_stderr") or 0.0)

        # Fallback to acc (classification tasks)
        if score is None:
            score = task_data.get("acc,none") or task_data.get("acc")
            stderr = float(task_data.get("acc_stderr,none") or task_data.get("acc_stderr", 0.0))

        # Fallback: first float value
        if score is None:
            for v in task_data.values():
                if isinstance(v, (int, float)):
                    score = float(v)
                    stderr = 0.0
                    break

        if score is not None:
            # Normalise task name: strip group prefixes
            short = task_name.split(",")[0]
            metrics[short] = (float(score), float(stderr or 0.0))
    return metrics


def _extract_perplexity_metrics(data: dict) -> dict[str, tuple[float, float]]:
    """Extract {bpb: (value, 0.0)} from a perplexity results JSON."""
    bpb = data.get("results", {}).get("bpb")
    if bpb is None:
        return {}
    return {"bpb": (float(bpb), 0.0)}


def load_all_results(
    results_dir: Path,
    langs: list[str],
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """
    Returns:
        { lang: { "base": {task: (score, stderr)}, "ft": {task: (score, stderr)} } }

    Supports result formats:
      - v4 bigcode-eval output (eval_results_{base|ft}_{ts}.json with "eval_engine": "bigcode-eval")
      - v3 leaderboard_{lang}.json (compact, both runs in one file)
      - v2/original lm-eval output (separate files per run)
    """
    out: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}

    for lang in langs:
        lang_dir = results_dir / f"code_{lang}"
        if not lang_dir.exists():
            continue

        lang_data: dict[str, dict[str, tuple[float, float]]] = {}

        # ── Try leaderboard format first ──────────────────────────────────
        lb_path = _find_leaderboard(lang_dir, lang)
        if lb_path is not None:
            with open(lb_path) as f:
                lb = json.load(f)
            for run_label in ("base", "ft"):
                extracted = _extract_from_leaderboard(lb, run_label)
                if extracted:
                    lang_data[run_label] = extracted

        # ── Fall back to per-run eval_results_*.json ──────────────────────
        # (also merges in any tasks present in the per-file format but not the leaderboard)
        for run_label in ("base", "ft"):
            path = _find_latest(lang_dir, run_label)
            if path is None:
                continue
            with open(path) as f:
                data = json.load(f)

            if "eval_mode" in data and data.get("eval_mode") == "perplexity":
                file_metrics = _extract_perplexity_metrics(data)
            elif "results" in data and isinstance(data["results"], dict):
                first_val = next(iter(data["results"].values()), None)
                if isinstance(first_val, (int, float)):
                    file_metrics = _extract_perplexity_metrics(data)
                else:
                    file_metrics = _extract_code_metrics(data)
            else:
                file_metrics = {}

            if file_metrics:
                # Merge: leaderboard takes priority if key already present
                merged = {**file_metrics, **lang_data.get(run_label, {})}
                lang_data[run_label] = merged

        # ── Also load BpB results for code languages (base_bpb / ft_bpb) ─
        for bpb_label, target_label in [("base_bpb", "base"), ("ft_bpb", "ft")]:
            bpb_path = _find_latest(lang_dir, bpb_label)
            if bpb_path is None:
                continue
            with open(bpb_path) as f:
                bpb_data = json.load(f)
            bpb_metrics = _extract_perplexity_metrics(bpb_data)
            if bpb_metrics:
                existing = lang_data.get(target_label, {})
                # Only add BpB if not already present
                if "bpb" not in existing:
                    lang_data[target_label] = {**existing, **bpb_metrics}

        if lang_data:
            out[lang] = lang_data

    return out


# ─── Plot 1: Per-language grouped bar charts ─────────────────────────────────

def plot_per_language(
    lang: str,
    lang_data: dict[str, dict[str, tuple[float, float]]],
    output_dir: Path,
) -> None:
    base_metrics = lang_data.get("base", {})
    ft_metrics   = lang_data.get("ft", {})

    all_tasks = sorted(set(base_metrics) | set(ft_metrics))
    if not all_tasks:
        return

    # For BpB: invert so higher = better for visual consistency (show as reduction)
    is_bpb = all_tasks == ["bpb"] or (len(all_tasks) == 1 and "bpb" in all_tasks)

    x = np.arange(len(all_tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(all_tasks) * 2.0), 5.5))

    bars_list = []
    for i, (run_label, metrics, color) in enumerate([
        ("Base OLMo-3-7B", base_metrics, COLORS["base"]),
        ("LAS Finetuned", ft_metrics, COLORS["ft"]),
    ]):
        scores = [metrics.get(t, (0.0, 0.0))[0] for t in all_tasks]
        stderrs = [metrics.get(t, (0.0, 0.0))[1] for t in all_tasks]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, scores, width,
            yerr=stderrs, capsize=4,
            color=color, alpha=0.88, label=run_label,
            edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        bars_list.append((bars, scores))

    # Add value labels on top of bars
    for bars, scores in bars_list:
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="medium",
                )

    tick_labels = [TASK_LABELS.get(t, t) for t in all_tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("BpB (lower = better)" if is_bpb else "Score (pass@1)")
    ax.set_title(f"Base vs. LAS Finetuned — {lang.upper()}")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    per_lang_dir = output_dir / "per_lang"
    per_lang_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(per_lang_dir / f"bar_{lang}.png", bbox_inches="tight")
    fig.savefig(per_lang_dir / f"bar_{lang}.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Plot 2: Cross-language score heatmap ────────────────────────────────────

def _build_pivot(
    all_data: dict[str, dict[str, dict[str, tuple[float, float]]]],
    langs: list[str],
    run_label: str,
    negate_bpb: bool = True,
) -> pd.DataFrame:
    """Build a tasks x langs DataFrame of scores.

    negate_bpb: if True, store -BpB so that higher = better throughout.
    """
    # Collect all tasks across all languages
    all_tasks: set[str] = set()
    for lang in langs:
        all_tasks.update(all_data.get(lang, {}).get(run_label, {}).keys())

    # Ordered: functional tasks first, then bpb
    functional = [t for t in sorted(all_tasks) if t != "bpb"]
    has_bpb    = "bpb" in all_tasks
    ordered_tasks = functional + (["bpb"] if has_bpb else [])

    rows = {}
    for task in ordered_tasks:
        row = {}
        for lang in langs:
            val = all_data.get(lang, {}).get(run_label, {}).get(task)
            if val is None:
                row[lang] = float("nan")
            else:
                score = val[0]
                if task == "bpb" and negate_bpb:
                    score = -score  # invert so higher = better
                row[lang] = score
        rows[task] = row

    return pd.DataFrame.from_dict(rows, orient="index", columns=langs)


def plot_heatmap_scores(
    all_data: dict[str, dict[str, dict[str, tuple[float, float]]]],
    langs: list[str],
    output_dir: Path,
) -> None:
    df = _build_pivot(all_data, langs, run_label="ft", negate_bpb=True)
    if df.empty:
        return

    task_labels = [TASK_LABELS.get(t, t).replace("\n", " ") for t in df.index]
    lang_labels = [l.upper() for l in df.columns]

    fig, ax = plt.subplots(figsize=(max(11, len(langs) * 1.2), max(4.5, len(df) * 0.75)))

    if _SEABORN:
        sns.heatmap(
            df.values.astype(float),
            ax=ax,
            cmap="YlOrRd",
            annot=True, fmt=".3f", annot_kws={"size": 9, "fontweight": "medium"},
            xticklabels=lang_labels,
            yticklabels=task_labels,
            linewidths=0.6, linecolor="white",
            cbar_kws={"label": "Score (higher better; BpB shown as -BpB)", "shrink": 0.8},
            square=False,
        )
    else:
        data = df.values.astype(float)
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="YlOrRd", aspect="auto")
        fig.colorbar(im, ax=ax, label="Score (higher better; BpB shown as -BpB)", shrink=0.8)
        ax.set_xticks(range(len(lang_labels)))
        ax.set_yticks(range(len(task_labels)))
        ax.set_xticklabels(lang_labels, fontsize=10)
        ax.set_yticklabels(task_labels, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            fontweight="medium")

    ax.set_title("LAS Finetuned Scores Across Languages", pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_ft_scores.png", bbox_inches="tight")
    fig.savefig(output_dir / "heatmap_ft_scores.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Plot 3: Delta heatmap ───────────────────────────────────────────────────

def plot_heatmap_delta(
    all_data: dict[str, dict[str, dict[str, tuple[float, float]]]],
    langs: list[str],
    output_dir: Path,
) -> None:
    df_base = _build_pivot(all_data, langs, run_label="base", negate_bpb=True)
    df_ft   = _build_pivot(all_data, langs, run_label="ft",   negate_bpb=True)

    if df_base.empty or df_ft.empty:
        return

    # Align on common tasks
    common_tasks = df_base.index.intersection(df_ft.index)
    if len(common_tasks) == 0:
        return

    delta = df_ft.loc[common_tasks] - df_base.loc[common_tasks]
    task_labels = [TASK_LABELS.get(t, t).replace("\n", " ") for t in delta.index]
    lang_labels = [l.upper() for l in delta.columns]

    fig, ax = plt.subplots(figsize=(max(11, len(langs) * 1.2), max(4.5, len(delta) * 0.75)))

    data = delta.values.astype(float)
    vmax = max(abs(np.nanmax(data)), abs(np.nanmin(data)), 0.01)

    if _SEABORN:
        sns.heatmap(
            data,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax, vmax=vmax,
            annot=True, fmt="+.3f", annot_kws={"size": 9, "fontweight": "medium"},
            xticklabels=lang_labels,
            yticklabels=task_labels,
            linewidths=0.6, linecolor="white",
            cbar_kws={"label": "Delta Score (FT - Base, higher better)", "shrink": 0.8},
            square=False,
        )
    else:
        masked = np.ma.masked_invalid(data)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(masked, cmap="RdBu_r", norm=norm, aspect="auto")
        fig.colorbar(im, ax=ax, label="Delta Score (FT - Base, higher better)", shrink=0.8)
        ax.set_xticks(range(len(lang_labels)))
        ax.set_yticks(range(len(task_labels)))
        ax.set_xticklabels(lang_labels, fontsize=10)
        ax.set_yticklabels(task_labels, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                            fontweight="medium")

    ax.set_title("Improvement: LAS Finetuned - Base OLMo-3-7B", pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_delta.png", bbox_inches="tight")
    fig.savefig(output_dir / "heatmap_delta.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Plot 4: Summary improvement bar chart ───────────────────────────────────

def plot_summary_delta(
    all_data: dict[str, dict[str, dict[str, tuple[float, float]]]],
    langs: list[str],
    output_dir: Path,
) -> None:
    lang_means: list[tuple[str, float, float]] = []  # (lang, mean_delta, std_delta)

    for lang in langs:
        base_m = all_data.get(lang, {}).get("base", {})
        ft_m   = all_data.get(lang, {}).get("ft", {})
        common = set(base_m) & set(ft_m)
        if not common:
            continue

        deltas = []
        for task in common:
            base_score = base_m[task][0]
            ft_score   = ft_m[task][0]
            if task == "bpb":
                # For BpB: improvement = base - ft (lower BpB = better)
                delta = base_score - ft_score
            else:
                delta = ft_score - base_score
            deltas.append(delta)

        if deltas:
            lang_means.append((lang, float(np.mean(deltas)), float(np.std(deltas))))

    if not lang_means:
        return

    labels = [l.upper() for l, _, _ in lang_means]
    means  = [m for _, m, _ in lang_means]
    stds   = [s for _, _, s in lang_means]
    colors = [COLORS["ft"] if m >= 0 else "#999999" for m in means]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.0), 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.0, "capthick": 1.0})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Value labels
    for bar, mean in zip(bars, means):
        va = "bottom" if mean >= 0 else "top"
        offset = 0.003 if mean >= 0 else -0.003
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{mean:+.3f}", ha="center", va=va, fontsize=8.5, fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Improvement (LAS-FT - Base)")
    ax.set_title("Average Improvement per Language")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_delta.png", bbox_inches="tight")
    fig.savefig(output_dir / "summary_delta.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── CSV export ──────────────────────────────────────────────────────────────

def export_csv(
    all_data: dict[str, dict[str, dict[str, tuple[float, float]]]],
    langs: list[str],
    output_dir: Path,
) -> None:
    rows = []
    for lang in langs:
        base_m = all_data.get(lang, {}).get("base", {})
        ft_m   = all_data.get(lang, {}).get("ft", {})
        all_tasks = sorted(set(base_m) | set(ft_m))
        for task in all_tasks:
            base_score, base_se = base_m.get(task, (float("nan"), float("nan")))
            ft_score,   ft_se   = ft_m.get(task,   (float("nan"), float("nan")))

            if task == "bpb":
                # delta = reduction in BpB (positive = improvement)
                delta = base_score - ft_score if not (np.isnan(base_score) or np.isnan(ft_score)) else float("nan")
                metric_type = "bpb (lower is better)"
            else:
                delta = ft_score - base_score if not (np.isnan(base_score) or np.isnan(ft_score)) else float("nan")
                metric_type = "pass@1"

            rows.append({
                "lang": lang,
                "task": task,
                "metric_type": metric_type,
                "base_score": round(base_score, 5) if not np.isnan(base_score) else "",
                "base_stderr": round(base_se, 5) if not np.isnan(base_se) else "",
                "ft_score": round(ft_score, 5) if not np.isnan(ft_score) else "",
                "ft_stderr": round(ft_se, 5) if not np.isnan(ft_se) else "",
                "delta": round(delta, 5) if not np.isnan(delta) else "",
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
    print(df.to_string(index=False))


# ─── Reusable plot generation ────────────────────────────────────────────────

def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
    langs: list[str],
) -> bool:
    """Generate all plots from current results. Returns True if any data was found."""
    all_data = load_all_results(results_dir, langs)
    if not all_data:
        return False

    found_langs = sorted(all_data.keys())
    print(f"Found results for: {found_langs}")

    for lang in found_langs:
        plot_per_language(lang, all_data[lang], output_dir)

    plot_heatmap_scores(all_data, found_langs, output_dir)
    plot_heatmap_delta(all_data, found_langs, output_dir)
    plot_summary_delta(all_data, found_langs, output_dir)
    export_csv(all_data, found_langs, output_dir)
    return True


def _compute_results_fingerprint(results_dir: Path, langs: list[str]) -> str:
    """Hash based on result files and their modification times."""
    h = hashlib.md5()
    for lang in langs:
        lang_dir = results_dir / f"code_{lang}"
        if lang_dir.exists():
            for f in sorted(lang_dir.glob("*.json")):
                h.update(f"{f.name}:{f.stat().st_mtime}".encode())
    return h.hexdigest()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UniMixLM code evaluation results")
    parser.add_argument("--results-dir", type=Path, default=Path("eval_results"),
                        help="Directory containing code_{lang}/ subdirectories")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/code_summary"),
                        help="Directory to save all plots and CSV")
    parser.add_argument("--langs", nargs="+", default=ALL_LANGS,
                        help="Languages to include in plots")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster output (default: 300)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch for new results and regenerate plots incrementally")
    parser.add_argument("--watch-interval", type=int, default=60,
                        help="Seconds between checks in watch mode (default: 60)")
    args = parser.parse_args()

    # Apply DPI override
    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi

    langs = [l.lower() for l in args.langs]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    print(f"Languages: {langs}")

    # Initial plot generation
    if not generate_all_plots(args.results_dir, output_dir, langs):
        print(f"No evaluation results found in {args.results_dir}/code_{{lang}}/")
        if not args.watch:
            print("Run 'bash scripts/run_code_eval.sh' first to generate results.")
            return
        print("Waiting for results to appear...")

    print(f"\nAll outputs saved to: {output_dir}")
    print("  per_lang/bar_{lang}.{png,pdf}  — per-language grouped bar charts")
    print("  heatmap_ft_scores.{png,pdf}    — cross-language score heatmap")
    print("  heatmap_delta.{png,pdf}        — improvement delta heatmap")
    print("  summary_delta.{png,pdf}        — mean improvement per language")
    print("  results_summary.csv            — full numeric results table")

    # Watch mode: poll for changes and regenerate
    if args.watch:
        print(f"\nWatching for new results every {args.watch_interval}s (Ctrl+C to stop)...")
        prev_fp = _compute_results_fingerprint(args.results_dir, langs)
        try:
            while True:
                time.sleep(args.watch_interval)
                cur_fp = _compute_results_fingerprint(args.results_dir, langs)
                if cur_fp != prev_fp:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] Results changed, regenerating plots...")
                    generate_all_plots(args.results_dir, output_dir, langs)
                    prev_fp = cur_fp
                    print(f"[{ts}] Plots updated.")
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")


if __name__ == "__main__":
    main()
