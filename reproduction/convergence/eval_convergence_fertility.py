#!/usr/bin/env python3
"""Compute held-out FLORES-200 devtest fertility for every fitted convergence
overlay, plus the two F15 production overlays (fin, tur) as a self-check, and
assemble convergence_v2.json.

Metric definition (the quantity of the paper's convergence table): the
fertility of one held-out document is (number of overlay-tokenizer output
pieces) / (number of characters in the raw text), not tokens per word. The
character count is the CHARACTERS unit of
tokenizer_analysis.config.NormalizationMethod from the TokEval library
(tokenizer_analysis/metrics/basic.py:_compute_fertility_stats), obtained
through TextMeasurer. The word count is also reported per document
(n_heldout_words) as auxiliary metadata but does not enter the fertility.

Aggregation: mean of per-document ratios (AGGREGATION_MEAN_OF_RATIOS in
tokenizer_analysis/metrics/basic.py, an unweighted mean over documents, not a
pooled total-tokens/total-chars ratio), matching that library's documented
convention. A document producing zero tokens is excluded from the mean and
counted separately (zero_token_documents), also matching that library.

Held-out set: FLORES-200 devtest, one sentence per line, 1012 lines per
language for deu/fin/tur, read from ``$FLORES200_DEVTEST_DIR/<lang>_Latn.devtest``.
FLORES is professionally translated Wikinews/Wikijunior text and FineWeb2 is
web-crawl text, so the held-out set is disjoint from every estimation corpus.

Encoding uses add_special_tokens=False (the convention of encode_probe in
scripts/prep_f15_tokenizers.py).

Environment:
  CONVERGENCE_WORK_DIR    directory holding fit_manifest.json; convergence_v2.json
                          is written there (required)
  FLORES200_DEVTEST_DIR   directory of FLORES-200 devtest files (required)
  TOKEVAL_REPO            checkout of tokenizer-intrinsic-evals, which provides
                          tokenizer_analysis (required)
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _env_dir(var: str, what: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set; point it at {what}.")
    return Path(value)


TOKEVAL_REPO = _env_dir("TOKEVAL_REPO", "a checkout of tokenizer-intrinsic-evals")
if not (TOKEVAL_REPO / "tokenizer_analysis" / "config.py").is_file():
    raise RuntimeError(
        f"{TOKEVAL_REPO} does not contain tokenizer_analysis/config.py; "
        f"TOKEVAL_REPO must point at a checkout of tokenizer-intrinsic-evals."
    )
if str(TOKEVAL_REPO) not in sys.path:
    sys.path.insert(0, str(TOKEVAL_REPO))

from tokenizers import Tokenizer  # noqa: E402
from tokenizer_analysis.config import create_default_configs, TextMeasurer  # noqa: E402

OUT_DIR = _env_dir("CONVERGENCE_WORK_DIR", "the directory holding fit_manifest.json")
FLORES_DIR = _env_dir("FLORES200_DEVTEST_DIR", "the FLORES-200 devtest directory")
LANG_CODES = {"deu": "deu_Latn", "fin": "fin_Latn", "tur": "tur_Latn"}
POINT_LABELS = ["N125", "N500", "N1000", "N4000", "N16000", "N32000", "prod25M"]
FULL_REFERENCE_LABEL = "N32000"  # the paper's "full (~32k-example)" reference row

EXISTING_F15_OVERLAYS = {
    "fin": REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/fin/langspec_fin_Latn.json",
    "tur": REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/tur/langspec_tur_Latn.json",
    # deu: no F15 overlay exists, so the self-check covers fin/tur only.
}

CHAR_MEASURER = TextMeasurer(create_default_configs()["characters"])


def load_flores_lines(lang_code: str) -> list[str]:
    path = FLORES_DIR / f"{lang_code}.devtest"
    if not path.exists():
        raise RuntimeError(f"FLORES devtest file not found: {path}")
    lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        raise RuntimeError(f"No non-empty lines in {path}")
    return lines


def compute_fertility(tok: Tokenizer, lines: list[str]) -> dict:
    """Mirrors tokenizer_analysis.metrics.basic.BasicTokenizationMetrics.
    _compute_fertility_stats with normalization_unit='characters': mean of
    per-document (tokens / chars) ratios; zero-token documents excluded from
    the mean and counted separately.
    """
    fertilities = []
    zero_token_documents = 0
    total_words = 0
    for line in lines:
        n_chars = CHAR_MEASURER.get_unit_count(line)
        if n_chars <= 0:
            continue
        ids = tok.encode(line, add_special_tokens=False).ids
        n_tokens = len(ids)
        total_words += len(line.split())
        if n_tokens == 0:
            zero_token_documents += 1
            continue
        fertilities.append(n_tokens / n_chars)

    if not fertilities:
        raise RuntimeError("No documents produced a non-zero fertility ratio")

    mean_fertility = sum(fertilities) / len(fertilities)
    return {
        "mean": mean_fertility,
        "n_heldout_documents": len(fertilities),
        "n_heldout_words": total_words,
        "zero_token_documents": zero_token_documents,
        "n_lines_total": len(lines),
    }


def main():
    fit_manifest = json.loads((OUT_DIR / "fit_manifest.json").read_text())

    flores_cache = {lang: load_flores_lines(code) for lang, code in LANG_CODES.items()}

    fertility_results = {}
    for lang in LANG_CODES:
        fertility_results[lang] = {}
        lines = flores_cache[lang]
        for point_label in POINT_LABELS:
            entry = fit_manifest.get(lang, {}).get(point_label)
            if entry is None:
                continue
            tok = Tokenizer.from_file(entry["path"])
            fert = compute_fertility(tok, lines)
            fert["overlay_path"] = entry["path"]
            fert["overlay_sha256"] = entry["sha256"]
            fert["corpus"] = entry["corpus"]
            fertility_results[lang][point_label] = fert

    # Differences from the "full" (N32000) reference, laid out like the
    # paper's convergence table.
    deltas = {}
    for lang, points in fertility_results.items():
        if FULL_REFERENCE_LABEL not in points:
            continue
        full_fert = points[FULL_REFERENCE_LABEL]["mean"]
        deltas[lang] = {}
        for point_label, fert in points.items():
            abs_delta = abs(fert["mean"] - full_fert)
            deltas[lang][point_label] = {
                "fertility": fert["mean"],
                "abs_delta_vs_full": abs_delta,
                "ratio_vs_full": fert["mean"] / full_fert,
                "pct_delta_vs_full": 100.0 * abs_delta / full_fert,
            }

    # Self-check: fertility of the F15 production overlay (fin/tur only) on the
    # same held-out set, compared to this study's N32000 fit.
    self_check_a = {}
    for lang, overlay_path in EXISTING_F15_OVERLAYS.items():
        tok = Tokenizer.from_file(str(overlay_path))
        existing_fert = compute_fertility(tok, flores_cache[lang])
        this_study_full = fertility_results[lang][FULL_REFERENCE_LABEL]
        self_check_a[lang] = {
            "existing_f15_overlay_path": str(overlay_path),
            "existing_f15_overlay_fertility": existing_fert["mean"],
            "existing_f15_overlay_n_docs": "50000 (see las_tokenizers/f15/qwen2_5-0_5b/manifest.json)",
            "this_study_N32000_fertility": this_study_full["mean"],
            "this_study_N32000_n_docs": this_study_full["corpus"]["n_docs"],
            "abs_delta": abs(existing_fert["mean"] - this_study_full["mean"]),
            "pct_delta": 100.0 * abs(existing_fert["mean"] - this_study_full["mean"])
                         / existing_fert["mean"],
        }

    final = {
        "metric": {
            "definition": "fertility = output pieces / characters (the paper's convergence table)",
            "aggregation": "mean of per-document ratios",
            "held_out_set": "FLORES-200 devtest, 1 sentence/line",
            "held_out_disjoint_note": (
                "FLORES-200 (professionally translated Wikinews/Wikijunior/etc.) "
                "is disjoint from every FineWeb2 web-crawl estimation corpus by "
                "construction: different source corpora, no overlap mechanism."
            ),
            "encoding": "add_special_tokens=False",
        },
        "grid_docs": [125, 500, 1000, 4000, 16000, 32000],
        "prod_token_target": 25_000_000,
        "full_reference_label": FULL_REFERENCE_LABEL,
        "fertility": fertility_results,
        "deltas_vs_full": deltas,
        "self_check_a_existing_f15_overlay": self_check_a,
    }

    out_path = OUT_DIR / "convergence_v2.json"
    out_path.write_text(json.dumps(final, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
