#!/usr/bin/env python3
"""Combine evaluation results for base, ft, and ft_las runs.

This script:
  - Finds the latest eval_results_base_*.json, eval_results_ft_*.json,
    and eval_results_ft_las_*.json files in the eval_results directory.
  - Extracts MMLU (overall + categories), PIQA (overall), and BELEBELE (overall)
    for English and a downstream language (default: Yoruba).
  - Prints nicely formatted tables per language with accuracy ± stderr.
  - Plots grouped bar charts per language for each benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RUN_PATTERNS = {
    "base": re.compile(r"eval_results_base_(\d{8}_\d{6})\.json$"),
    "ft": re.compile(r"eval_results_ft_(\d{8}_\d{6})\.json$"),
    "ft_las": re.compile(r"eval_results_ft_las_(\d{8}_\d{6})\.json$"),
}

MMLU_GENERAL_SUFFIXES = {"stem", "humanities", "social_sciences", "other"}
LANG_ALIAS_TO_THREE = {
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "yo": "yor",
    "yor": "yor",
    "yoruba": "yor",
    "ig": "ibo",
    "ibo": "ibo",
    "fil": "fil",
    "sw": "swh",
    "swh": "swh",
    "ms": "zsm",
    "zsm": "zsm",
    "ro": "ron",
    "ron": "ron",
    "li": "lit",
    "lit": "lit",
}
THREE_TO_MMLU = {
    "eng": "en",
    "yor": "yo",
    "ibo": "ig",
    "fil": "fil",
    "swh": "sw",
    "zsm": "ms",
    "ron": "ro",
    "lit": "li",
    "ind": "id",
}


@dataclass(frozen=True)
class Metric:
    acc: float
    stderr: float


def _find_latest_by_pattern(results_dir: Path, pattern: re.Pattern) -> Optional[Path]:
    candidates: List[Tuple[str, Path]] = []
    for path in results_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric(entry: dict) -> Optional[Metric]:
    acc = entry.get("acc,none")
    stderr = entry.get("acc_stderr,none")
    if acc is None or stderr is None:
        return None
    return Metric(acc=float(acc), stderr=float(stderr))


def _normalize_lang(lang: str) -> str:
    lang = lang.strip().lower()
    if lang in LANG_ALIAS_TO_THREE:
        return LANG_ALIAS_TO_THREE[lang]
    if re.fullmatch(r"[a-z]{2,3}", lang):
        return lang
    raise ValueError(f"Unsupported language: {lang}")


def _language_key(lang: str) -> Dict[str, str]:
    three = _normalize_lang(lang)
    mmlu_code = THREE_TO_MMLU.get(three, three)
    if three == "fil":
        three = "tgl"
    return {
        "mmlu": f"global_mmlu_full_{mmlu_code}",
        "piqa": f"global_piqa_completions_{three}_latn",
        "belebele": f"belebele_{three}_Latn",
    }


def _parse_model_name(model_args: Optional[str]) -> str:
    if not model_args:
        return "unknown"
    parts = [p.strip() for p in model_args.split(",") if p.strip()]
    kv = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key.strip()] = value.strip()
    pretrained = kv.get("pretrained", "unknown")
    peft = kv.get("peft") or kv.get("peft_original_tok") or kv.get("peft_las_tok")
    if peft:
        return f"{pretrained} + peft:{Path(peft).name}"
    return pretrained


def _extract_model_name(data: dict) -> str:
    for section in data.keys():
        if data[section] is not None:
            cfg = data.get(section, {}).get("config", {})
            if "model_args" in cfg:
                return _parse_model_name(cfg.get("model_args"))
    return "unknown"


def _collect_mmlu_keys(results: dict, mmlu_prefix: str) -> List[str]:
    keys: List[str] = []
    for key in results.keys():
        if key == mmlu_prefix:
            keys.append(key)
            continue
        if key.startswith(mmlu_prefix + "_"):
            suffix = key[len(mmlu_prefix) + 1 :]
            if suffix in MMLU_GENERAL_SUFFIXES:
                keys.append(key)

    def key_sort(k: str) -> Tuple[int, str]:
        if k == mmlu_prefix:
            return (0, "")
        return (1, k)

    return sorted(keys, key=key_sort)


def _label_for_benchmark(mmlu_prefix: str, key: str) -> str:
    if key == mmlu_prefix:
        return "mmlu_overall"
    suffix = key[len(mmlu_prefix) + 1 :]
    return f"mmlu_{suffix}"


def _extract_language_metrics(data: dict, lang: str) -> Dict[str, Metric]:
    lang_keys = _language_key(lang)
    metrics: Dict[str, Metric] = {}

    # MMLU overall + categories
    if data["mmlu"] is not None:
        mmlu_results = data["mmlu"]["results"]
        for key in _collect_mmlu_keys(mmlu_results, lang_keys["mmlu"]):
            metric = _extract_metric(mmlu_results.get(key, {}))
            if metric is not None:
                metrics[_label_for_benchmark(lang_keys["mmlu"], key)] = metric

    # BELEBELE
    if data["belebele"] is not None:
        belebele_results = data["belebele"]["results"]
        bele = belebele_results.get(lang_keys["belebele"], {})
        bele_metric = _extract_metric(bele)
        if bele_metric is not None:
            metrics["belebele"] = bele_metric

    # PIQA
    if data["piqa"] is not None:
        piqa_results = data["piqa"]["results"]
        piqa = piqa_results.get(lang_keys["piqa"], {})
        piqa_metric = _extract_metric(piqa)
        if piqa_metric is not None:
            metrics["piqa"] = piqa_metric

    return metrics


def _format_metric(metric: Optional[Metric]) -> str:
    if metric is None:
        return "N/A"
    return f"{metric.acc:.4f} ± {metric.stderr:.4f}"


def _metric_to_dict(metric: Optional[Metric]) -> Optional[dict]:
    if metric is None:
        return None
    return {"acc": float(metric.acc), "stderr": float(metric.stderr)}


def _build_table(
    per_run_metrics: Dict[str, Dict[str, Metric]],
    benchmarks: Iterable[str],
    runs: List[str],
) -> pd.DataFrame:
    rows = {}
    for benchmark in benchmarks:
        row = {}
        for run in runs:
            metric = per_run_metrics.get(run, {}).get(benchmark)
            row[run] = _format_metric(metric)
        rows[benchmark] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "benchmark"
    return df


def _plot_language(
    lang: str,
    per_run_metrics: Dict[str, Dict[str, Metric]],
    benchmarks: List[str],
    runs: List[str],
    output_dir: Path,
    run_models: Dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data = []
    for benchmark in benchmarks:
        for run in runs:
            metric = per_run_metrics.get(run, {}).get(benchmark)
            if metric is None:
                continue
            plot_data.append(
                {
                    "benchmark": benchmark,
                    "run": run,
                    "acc": metric.acc,
                    "stderr": metric.stderr,
                }
            )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    runs_present = [r for r in runs if r in df["run"].unique().tolist()]
    x = range(len(benchmarks))
    width = 0.25
    offsets = {
        run: (i - (len(runs_present) - 1) / 2) * width
        for i, run in enumerate(runs_present)
    }

    fig_width = max(12, len(benchmarks) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for run in runs_present:
        run_df = df[df["run"] == run].set_index("benchmark").reindex(benchmarks)
        y = run_df["acc"].values
        yerr = run_df["stderr"].values
        ax.bar(
            [i + offsets[run] for i in x],
            y,
            width=width,
            yerr=yerr,
            capsize=3,
            label=run,
            alpha=0.9,
        )

    ax.set_title(f"Benchmark comparison ({lang})")
    subtitle_parts = [f"{run}={run_models.get(run, 'unknown')}" for run in runs_present]
    subtitle = "Models: " + "; ".join(subtitle_parts)
    fig.text(0.5, 0.98, subtitle, ha="center", va="top", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(benchmarks, rotation=60, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / f"benchmark_comparison_{lang}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _ordered_benchmarks(benchmarks: Iterable[str]) -> List[str]:
    benchmarks = list(benchmarks)
    mmlu = sorted([b for b in benchmarks if b.startswith("mmlu_")])
    mmlu_overall = [b for b in mmlu if b == "mmlu_overall"]
    mmlu_rest = [b for b in mmlu if b != "mmlu_overall"]
    tail = [b for b in benchmarks if b in {"piqa", "belebele"}]
    tail_sorted = [b for b in ["piqa", "belebele"] if b in tail]
    others = [b for b in benchmarks if b not in set(mmlu + tail)]
    return mmlu_overall + mmlu_rest + tail_sorted + sorted(others)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine eval results and plot tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory containing eval_results_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to results dir).",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir or results_dir
    combined_dir = output_dir / "combined"

    run_files: Dict[str, Optional[Path]] = {
        run: _find_latest_by_pattern(results_dir, pattern)
        for run, pattern in RUN_PATTERNS.items()
    }

    available_runs = {run: path for run, path in run_files.items() if path is not None}
    if not available_runs:
        print(f"No eval_results_*.json files found in {results_dir}")
        return 1

    print("Using result files:")
    for run, path in available_runs.items():
        print(f"  {run}: {path.name}")

    loaded = {run: _load_json(path) for run, path in available_runs.items()}
    run_models = {run: _extract_model_name(data) for run, data in loaded.items()}

    combined_results: Dict[str, dict] = {}

    target_lang = str(results_dir).split("/")[-1]
    for lang in ["en", target_lang]:
        per_run_metrics = {
            run: _extract_language_metrics(data, lang) for run, data in loaded.items()
        }
        all_benchmarks = set()
        for metrics in per_run_metrics.values():
            all_benchmarks.update(metrics.keys())

        benchmarks = _ordered_benchmarks(all_benchmarks)
        runs = [r for r in ["base", "ft", "ft_las"] if r in available_runs]

        print("\n" + "=" * 80)
        print(f"Language: {lang}")
        print("=" * 80)
        table = _build_table(per_run_metrics, benchmarks, runs)
        print(table.to_string())

        _plot_language(
            lang, per_run_metrics, benchmarks, runs, combined_dir, run_models
        )

        combined_results[lang] = {
            "benchmarks": benchmarks,
            "runs": {
                run: {
                    benchmark: _metric_to_dict(
                        per_run_metrics.get(run, {}).get(benchmark)
                    )
                    for benchmark in benchmarks
                }
                for run in runs
            },
        }

    combined_payload = {
        "source_files": {run: path.name for run, path in available_runs.items()},
        "models": run_models,
        "results": combined_results,
    }

    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "combined_results.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)

    print(f"\nOutputs saved to: {combined_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
