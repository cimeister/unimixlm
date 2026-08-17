#!/usr/bin/env python3
"""F15 statistics/aggregation tool (plan: "the-paper-has-not-stateful-bear.md",
section "Statistical protocol"). Builds cell means, paired seed contrasts vs a
reference arm with percentile-bootstrap CIs, Holm-Bonferroni correction, and
the pre-registered intrinsic-vs-downstream directional test — from per-cell
eval JSONs only. No hand-entered numbers anywhere; every table cell carries
the count and paths of the files it was computed from.

INPUT FORMAT
------------
One JSON file per (family, lang, arm, seed) cell, as produced by
scripts/eval_f14_multiblimp.py. Required top-level keys (read directly from
the JSON, never re-derived from the filename):
    tag                 str,  e.g. "qwen2_5-0_5b_tur_langspec_s42"
    lang                str,  3-letter code, e.g. "tur"
    arm                 str,  e.g. "original" | "langspec" | "placebo" |
                               "base" | "vocab_extend" (F15 additions)
    seed                int
    reinit_embeddings   bool
    truetie             bool
    head_wiring         str,  e.g. "training" | "reshared"
    results             dict, {task_name: {"<metric>,none": float, ...}} —
                         exactly one task_name key per file
Optional:
    n_samples           dict, {task_name: {"effective": int, ...}}. Read via
                         `.get()` with no default key requirement --
                         n_samples_effective is None in the output when absent
                         rather than the file being skipped.
Any *.json in --eval-dir lacking the REQUIRED keys above is skipped (reported, not
silently dropped) rather than erroring the whole run, since the eval
directory may accumulate unrelated JSON (e.g. this script's own --out-json).
A file that has these keys but is internally inconsistent (e.g. a tag that
disagrees with its own lang/arm/seed/reinit/truetie fields) is a hard error:
refusing to guess is safer than guessing wrong for a stats pipeline.

GROUPING KEYS ("family" made explicit)
---------------------------------------
Two axes get folded together in the raw files: the model (encoded only in
the tag prefix, e.g. "qwen2_5-0_5b") and the *wiring variant* — the
reinit_embeddings / truetie / head_wiring booleans and strings. Averaging
across wiring variants would silently mix numbers that are not comparable
(F14 as-run vs F14c true-tie vs F14b true-tie+reinit are different training
protocols entirely — see EXPERIMENTS_RESULTS.md F14/F14b/F14c). This script
therefore defines
    family = "<model>[<wiring-bits>]"
where <model> is the tag with the "_<lang>_<arm>[_reinitemb][_truetie]_s<seed>"
suffix stripped (validated against the record's own fields — see
`derive_model_and_family`), and <wiring-bits> is "truetie"/"reinit" (present
only if true) plus "head-<head_wiring>". E.g.:
    qwen2_5-0_5b[head-training]                    F14 as-run
    qwen2_5-0_5b[truetie+head-training]             F14c true-tie pretrained
    qwen2_5-0_5b[truetie+reinit+head-training]      F14b true-tie reinit
"arm" is kept as the raw JSON field (original/langspec/placebo/base/
vocab_extend/...) so contrasts stay comparable across families that share an
arm vocabulary, and the family string carries the model + wiring context.
When F15 lands a second model (Llama-3.2-1B), its tag prefix differs, so it
becomes a distinct family automatically — no code change needed.

OPTIONAL PER-ITEM FILES
------------------------
For item-level paired bootstrap (plan: "pairing over items, seeds as
replicates"), a cell's JSON `<tag>__head<wiring>.json` may have a sibling
    <tag>__head<wiring>__items.jsonl
(one JSON object per line: {"doc_id": <int|str>, "<metric>": <float>}, one
line per eval item, `doc_id` stable across arms/seeds for the same task).
Naming matches the results JSON's own keying (pre-launch review fix: a bare
`<tag>__items.jsonl`, with no head_wiring, would let a training- and a
reshared-wiring eval of the same cell silently clobber each other's items
file). `load_items` falls back to the pre-fix `<tag>__items.jsonl` name when
the new one is absent, for any file an older driver already wrote. The
loader falls back cleanly to seed-only bootstrap when neither file exists,
labeling the CI method used in every contrast row so nothing is silently
under-powered without saying so.

STATISTICAL METHODS (all implemented by hand; stdlib + numpy only)
--------------------------------------------------------------------
- Percentile bootstrap: resample seeds with replacement (B iterations, fixed
  RNG seed for reproducibility); when per-item files are available for every
  matched seed of both arms in a contrast, use a nested seed+item bootstrap
  (resample seeds, then resample items within each resampled seed, matching
  the plan's "pairing over items, seeds as replicates"); otherwise resample
  seeds only, using each seed's already-computed paired delta.
- Bootstrap p-value: p = 2 * min(P(delta<=0), P(delta>=0)) over the bootstrap
  distribution, capped at 1 (standard percentile sign-flip p-value).
- Holm-Bonferroni: standard step-down procedure, applied across all contrasts
  within a family (metric is fixed per run via --metric, so "per (family,
  metric) table" in the plan reduces to "per family" for a single invocation).
- Spearman rho: Pearson correlation of average ranks (ties handled by
  averaging tied ranks).
- Exact sign test: exact two-sided binomial test at p=0.5 via math.comb
  (2 * min(lower tail, upper tail), capped at 1).

Magic numbers (flagged per project convention): BOOTSTRAP_ITERS=10000,
BOOTSTRAP_RNG_SEED=0, ALPHA=0.05 (the plan's pre-registered threshold). All
three are CLI-overridable; defaults are recorded in --out-json metadata.

CLI
---
    --eval-dir       directory of per-cell JSONs (+ optional __items.jsonl)
    --reference-arm  arm every contrast is computed against (default: original)
    --metric         lm-eval metric name, without ",none" (default: acc)
    --intrinsic-csv  optional CSV, header "lang,delta", for the directional test
    --out-json       path to write the machine-readable summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Magic numbers (see module docstring "Magic numbers" section)
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_RNG_SEED = 0
ALPHA = 0.05
DIRECTIONAL_TARGET_ARM = "langspec"  # the plan's pre-registered method arm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"tag", "lang", "arm", "seed", "reinit_embeddings", "truetie",
                  "head_wiring", "results"}


def derive_model_and_family(rec: dict) -> tuple[str, str]:
    """Strip the "_<lang>_<arm>[_reinitemb][_truetie]_s<seed>" suffix from
    `tag`, built from the record's OWN fields, to recover the model prefix.
    Raises if the tag does not match — a mismatch means the filename and the
    JSON body disagree, which is a mislabeled-cell bug, not something to
    guess past."""
    tag, lang, arm = rec["tag"], rec["lang"], rec["arm"]
    seed, reinit, truetie = rec["seed"], rec["reinit_embeddings"], rec["truetie"]
    suffix = f"_{lang}_{arm}"
    if reinit:
        suffix += "_reinitemb"
    if truetie:
        suffix += "_truetie"
    suffix += f"_s{seed}"
    if not tag.endswith(suffix):
        raise ValueError(
            f"tag {tag!r} does not end with the suffix built from its own "
            f"fields ({suffix!r}); refusing to guess the model prefix "
            f"(lang={lang} arm={arm} seed={seed} reinit={reinit} "
            f"truetie={truetie})."
        )
    model = tag[: -len(suffix)]
    if not model:
        raise ValueError(f"tag {tag!r}: empty model prefix after stripping {suffix!r}")

    wiring_bits = []
    if truetie:
        wiring_bits.append("truetie")
    if reinit:
        wiring_bits.append("reinit")
    wiring_bits.append(f"head-{rec['head_wiring']}")
    family = f"{model}[{'+'.join(wiring_bits)}]"
    return model, family


def extract_metric(rec: dict, metric: str) -> tuple[float, str]:
    """Pull the single task's "<metric>,none" value. Requires exactly one
    task key in `results` — a cell JSON is defined as one (cell, task) pair
    by eval_f14_multiblimp.py, so >1 or 0 keys is a schema violation."""
    results = rec["results"]
    if len(results) != 1:
        raise ValueError(
            f"{rec['tag']}: results dict has {len(results)} task keys "
            f"{list(results)}, expected exactly 1."
        )
    task, task_results = next(iter(results.items()))
    key = f"{metric},none"
    if key not in task_results:
        raise KeyError(
            f"{rec['tag']}: task {task!r} results missing key {key!r}; "
            f"available keys: {list(task_results)}"
        )
    return float(task_results[key]), task


def load_items(eval_dir: Path, tag: str, head_wiring: str, metric: str) -> Optional[dict]:
    """Optional sibling `<tag>__head<wiring>__items.jsonl` (current
    eval_f14_multiblimp.py naming, matching the results JSON's own keying).
    Falls back to the pre-rename `<tag>__items.jsonl` for files an older
    driver already wrote. Returns {doc_id: value} or None if neither exists.
    See module docstring for the format."""
    path = eval_dir / f"{tag}__head{head_wiring}__items.jsonl"
    if not path.exists():
        legacy_path = eval_dir / f"{tag}__items.jsonl"
        if not legacy_path.exists():
            return None
        path = legacy_path
    items: dict = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "doc_id" not in rec:
            raise KeyError(f"{path}:{lineno}: missing 'doc_id'")
        if metric not in rec:
            raise KeyError(
                f"{path}:{lineno}: missing metric key {metric!r}; "
                f"available keys: {list(rec)}"
            )
        doc_id = rec["doc_id"]
        if doc_id in items:
            raise ValueError(f"{path}:{lineno}: duplicate doc_id {doc_id!r}")
        items[doc_id] = float(rec[metric])
    return items


def load_cells(eval_dir: Path, metric: str) -> tuple[list[dict], list[dict]]:
    """Scan --eval-dir for per-cell JSONs. Returns (cells, skipped) where
    `skipped` records every *.json that was not a cell record, with why."""
    cells: list[dict] = []
    skipped: list[dict] = []
    seen_keys: dict[tuple, Path] = {}
    for path in sorted(eval_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            skipped.append({"path": str(path), "reason": f"invalid JSON: {e}"})
            continue
        if not isinstance(raw, dict) or not REQUIRED_KEYS.issubset(raw):
            missing = REQUIRED_KEYS - (raw.keys() if isinstance(raw, dict) else set())
            skipped.append({
                "path": str(path),
                "reason": f"not a per-cell record (missing keys: {sorted(missing)})",
            })
            continue

        model, family = derive_model_and_family(raw)
        value, task = extract_metric(raw, metric)
        n_eff = raw.get("n_samples", {}).get(task, {}).get("effective")
        items = load_items(eval_dir, raw["tag"], raw["head_wiring"], metric)

        key = (family, raw["lang"], raw["arm"], raw["seed"])
        if key in seen_keys:
            raise ValueError(
                f"duplicate cell {key}: both {seen_keys[key]} and {path} map "
                f"to it. Refusing to silently pick one or average them."
            )
        seen_keys[key] = path

        cells.append({
            "path": str(path),
            "tag": raw["tag"],
            "model": model,
            "family": family,
            "lang": raw["lang"],
            "arm": raw["arm"],
            "seed": raw["seed"],
            "reinit_embeddings": raw["reinit_embeddings"],
            "truetie": raw["truetie"],
            "head_wiring": raw["head_wiring"],
            "task": task,
            "value": value,
            "n_samples_effective": n_eff,
            "items": items,
        })
    return cells, skipped


# ---------------------------------------------------------------------------
# Cell aggregation
# ---------------------------------------------------------------------------

def aggregate_cells(cells: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["family"], c["lang"], c["arm"])].append(c)

    out: dict[tuple, dict] = {}
    for key, recs in groups.items():
        seeds = [r["seed"] for r in recs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed within cell {key}: seeds={seeds}")
        values = np.array([r["value"] for r in recs], dtype=float)
        n_eff = {r["seed"]: r["n_samples_effective"] for r in recs}
        out[key] = {
            "family": key[0], "lang": key[1], "arm": key[2],
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "n_seeds": len(recs),
            "seeds": sorted(seeds),
            "n_files": len(recs),
            "files": sorted(r["path"] for r in recs),
            "n_samples_effective": n_eff,
        }
    return out


# ---------------------------------------------------------------------------
# Paired contrasts + bootstrap
# ---------------------------------------------------------------------------

def seed_bootstrap(deltas: np.ndarray, rng: np.random.Generator, iters: int) -> np.ndarray:
    n = deltas.shape[0]
    idx = rng.integers(0, n, size=(iters, n))
    return deltas[idx].mean(axis=1)


def nested_item_seed_bootstrap(
    seed_item_deltas: list[np.ndarray], rng: np.random.Generator, iters: int
) -> np.ndarray:
    """Outer: resample seeds with replacement. Inner: for each resampled
    seed, resample its paired per-item deltas with replacement and take the
    mean. Bootstrap replicate = mean over the (resampled) seeds of those
    per-seed item-resampled means. Implements the plan's "pairing over
    items, seeds as replicates"."""
    n_seeds = len(seed_item_deltas)
    out = np.empty(iters)
    for b in range(iters):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = np.empty(n_seeds)
        for j, si in enumerate(seed_idx):
            d = seed_item_deltas[si]
            item_idx = rng.integers(0, d.shape[0], size=d.shape[0])
            seed_means[j] = d[item_idx].mean()
        out[b] = seed_means.mean()
    return out


def bootstrap_p(boot: np.ndarray) -> float:
    p_pos = float(np.mean(boot >= 0))
    p_neg = float(np.mean(boot <= 0))
    return min(1.0, 2.0 * min(p_pos, p_neg))


def paired_contrast(
    family: str, lang: str, arm: str, reference_arm: str,
    ref_recs: list[dict], arm_recs: list[dict],
    rng: np.random.Generator, iters: int,
) -> dict:
    ref_vals = {r["seed"]: r["value"] for r in ref_recs}
    arm_vals = {r["seed"]: r["value"] for r in arm_recs}
    matched_seeds = sorted(set(ref_vals) & set(arm_vals))
    if not matched_seeds:
        return {
            "family": family, "lang": lang, "arm": arm, "reference_arm": reference_arm,
            "skipped_reason": (
                f"no seed present in both {arm!r} (seeds={sorted(arm_vals)}) and "
                f"{reference_arm!r} (seeds={sorted(ref_vals)})"
            ),
        }

    deltas = np.array([arm_vals[s] - ref_vals[s] for s in matched_seeds])
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")

    ref_items = {r["seed"]: r["items"] for r in ref_recs if r["items"] is not None}
    arm_items = {r["seed"]: r["items"] for r in arm_recs if r["items"] is not None}
    item_seeds = [s for s in matched_seeds if s in ref_items and s in arm_items]

    note = None
    if item_seeds and item_seeds == matched_seeds:
        seed_item_deltas = []
        for s in matched_seeds:
            ri, ai = ref_items[s], arm_items[s]
            if set(ri) != set(ai):
                raise ValueError(
                    f"{family}/{lang}/{arm} vs {reference_arm} seed={s}: item id "
                    f"sets differ between arms ({len(ri)} vs {len(ai)} items, "
                    f"symmetric diff {len(set(ri) ^ set(ai))}) — cannot pair."
                )
            common = sorted(ri)
            seed_item_deltas.append(np.array([ai[i] - ri[i] for i in common]))
        boot = nested_item_seed_bootstrap(seed_item_deltas, rng, iters)
        ci_method = "item+seed percentile bootstrap"
    else:
        boot = seed_bootstrap(deltas, rng, iters)
        ci_method = "seed-only percentile bootstrap"
        if item_seeds and item_seeds != matched_seeds:
            note = (
                f"per-item files present for only {len(item_seeds)}/"
                f"{len(matched_seeds)} matched seeds; falling back to "
                f"seed-only bootstrap rather than mixing item- and "
                f"non-item-backed seeds."
            )

    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    p = bootstrap_p(boot)

    result = {
        "family": family, "lang": lang, "arm": arm, "reference_arm": reference_arm,
        "mean_delta": mean_delta, "sd_delta": sd_delta,
        "n_seeds_matched": len(matched_seeds), "matched_seeds": matched_seeds,
        "ci_low": lo, "ci_high": hi, "bootstrap_p": p, "ci_method": ci_method,
        "n_files": len(ref_recs) + len(arm_recs),
        "files": {
            "reference": sorted(r["path"] for r in ref_recs if r["seed"] in matched_seeds),
            "arm": sorted(r["path"] for r in arm_recs if r["seed"] in matched_seeds),
        },
    }
    if note:
        result["note"] = note
    return result


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: dict, alpha: float) -> dict:
    """Standard step-down Holm procedure. `pvals`: {key: p}. Returns
    {key: {"holm_p": adjusted, "significant_holm": bool}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items, start=1):
        adj = (m - i + 1) * p
        running_max = max(running_max, adj)
        adjusted[key] = min(running_max, 1.0)
    return {
        key: {"holm_p": adjusted[key], "significant_holm": adjusted[key] <= alpha}
        for key in pvals
    }


# ---------------------------------------------------------------------------
# Spearman rho + exact sign test
# ---------------------------------------------------------------------------

def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values))
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank over the tie block
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = average_ranks(x) - average_ranks(x).mean()
    ry = average_ranks(y) - average_ranks(y).mean()
    denom = math.sqrt(float(np.sum(rx ** 2)) * float(np.sum(ry ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry)) / denom


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def exact_sign_test(n_concordant: int, n_trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test at p=0.5: p = 2 * min(lower tail,
    upper tail), capped at 1. Standard sign-test p-value."""
    if n_trials == 0:
        return float("nan")
    lower = sum(binom_pmf(k, n_trials, p) for k in range(0, n_concordant + 1))
    upper = sum(binom_pmf(k, n_trials, p) for k in range(n_concordant, n_trials + 1))
    return min(1.0, 2.0 * min(lower, upper))


def load_intrinsic_csv(path: Path) -> dict:
    out: dict = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"lang", "delta"} - set(reader.fieldnames):
            raise ValueError(
                f"{path}: expected header 'lang,delta', got {reader.fieldnames}"
            )
        for row in reader:
            lang = row["lang"].strip()
            if lang in out:
                raise ValueError(f"{path}: duplicate lang {lang!r}")
            out[lang] = float(row["delta"])
    return out


def directional_test(
    intrinsic: dict, contrasts: dict, reference_arm: str, target_arm: str,
) -> dict:
    pairs = []
    skipped = []
    for (family, lang, arm), c in sorted(contrasts.items()):
        if arm != target_arm:
            continue
        if lang not in intrinsic:
            skipped.append({
                "family": family, "lang": lang,
                "reason": f"lang {lang!r} not in --intrinsic-csv",
            })
            continue
        if "skipped_reason" in c:
            skipped.append({
                "family": family, "lang": lang, "reason": c["skipped_reason"],
            })
            continue
        pairs.append({
            "family": family, "lang": lang,
            "intrinsic_delta": intrinsic[lang],
            "downstream_delta": c["mean_delta"],
        })

    if len(pairs) < 2:
        return {
            "enabled": True, "target_arm": target_arm, "reference_arm": reference_arm,
            "n_pairs": len(pairs), "spearman_rho": None, "sign_test": None,
            "pairs": pairs, "skipped": skipped,
            "note": "fewer than 2 (lang, family) pairs available; correlation/sign test not computed",
        }

    x = np.array([p["intrinsic_delta"] for p in pairs])
    y = np.array([p["downstream_delta"] for p in pairs])
    rho = spearman_rho(x, y)

    n_concordant = 0
    n_discordant = 0
    n_zero = 0
    for p in pairs:
        sx, sy = np.sign(p["intrinsic_delta"]), np.sign(p["downstream_delta"])
        if sx == 0 or sy == 0:
            n_zero += 1
        elif sx == sy:
            n_concordant += 1
        else:
            n_discordant += 1
    n_trials = n_concordant + n_discordant
    sign_p = exact_sign_test(n_concordant, n_trials) if n_trials > 0 else float("nan")

    return {
        "enabled": True, "target_arm": target_arm, "reference_arm": reference_arm,
        "n_pairs": len(pairs), "spearman_rho": rho,
        "sign_test": {
            "n_concordant": n_concordant, "n_discordant": n_discordant,
            "n_zero_excluded": n_zero, "n_trials": n_trials, "p_exact": sign_p,
        },
        "pairs": pairs, "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_cells_table(cells_agg: dict, metric: str) -> str:
    lines = []
    families = sorted({k[0] for k in cells_agg})
    for family in families:
        lines.append(f"\n## Cell means — family `{family}` (metric = {metric})\n")
        lines.append("| lang | arm | mean | sd | n_seeds | seeds | n_files |")
        lines.append("|---|---|---|---|---|---|---|")
        rows = sorted(k for k in cells_agg if k[0] == family)
        for key in rows:
            c = cells_agg[key]
            lines.append(
                f"| {c['lang']} | {c['arm']} | {fmt(c['mean'])} | {fmt(c['sd'])} "
                f"| {c['n_seeds']} | {','.join(str(s) for s in c['seeds'])} "
                f"| {c['n_files']} |"
            )
    return "\n".join(lines)


def render_contrasts_table(contrasts: dict, holm: dict, reference_arm: str,
                            metric: str, alpha: float) -> str:
    lines = []
    families = sorted({k[0] for k in contrasts})
    for family in families:
        lines.append(
            f"\n## Paired contrasts vs `{reference_arm}` — family `{family}` "
            f"(metric = {metric}, alpha = {alpha})\n"
        )
        lines.append(
            "| lang | arm | mean_delta | sd_delta | ci95_lo | ci95_hi | "
            "raw_p | holm_p | sig(raw) | sig(holm) | method | n_seeds | n_files |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = sorted(k for k in contrasts if k[0] == family)
        for key in rows:
            c = contrasts[key]
            if "skipped_reason" in c:
                lines.append(
                    f"| {key[1]} | {key[2]} | -- | -- | -- | -- | -- | -- | -- | -- "
                    f"| SKIPPED: {c['skipped_reason']} | -- | -- |"
                )
                continue
            h = holm[key]
            sig_raw = c["bootstrap_p"] <= alpha
            lines.append(
                f"| {c['lang']} | {c['arm']} | {fmt(c['mean_delta'])} "
                f"| {fmt(c['sd_delta'])} | {fmt(c['ci_low'])} | {fmt(c['ci_high'])} "
                f"| {fmt(c['bootstrap_p'])} | {fmt(h['holm_p'])} "
                f"| {'yes' if sig_raw else 'no'} "
                f"| {'yes' if h['significant_holm'] else 'no'} "
                f"| {c['ci_method']} | {c['n_seeds_matched']} | {c['n_files']} |"
            )
    return "\n".join(lines)


def render_directional_test(dt: dict) -> str:
    lines = ["\n## Pre-registered directional test\n"]
    if not dt["enabled"]:
        lines.append(dt.get("note", "disabled: no --intrinsic-csv supplied."))
        return "\n".join(lines)
    lines.append(
        f"Target arm: `{dt['target_arm']}` vs reference `{dt['reference_arm']}`. "
        f"n_pairs = {dt['n_pairs']} (lang x family)."
    )
    if dt.get("note"):
        lines.append(dt["note"])
    if dt["spearman_rho"] is not None:
        lines.append(f"Spearman rho = {fmt(dt['spearman_rho'])}")
        st = dt["sign_test"]
        lines.append(
            f"Sign test: {st['n_concordant']} concordant / {st['n_trials']} trials "
            f"({st['n_zero_excluded']} zero-sign pairs excluded), "
            f"exact two-sided p = {fmt(st['p_exact'])}"
        )
    lines.append("\n| lang | family | intrinsic_delta | downstream_delta | concordant |")
    lines.append("|---|---|---|---|---|")
    for p in dt["pairs"]:
        sx, sy = np.sign(p["intrinsic_delta"]), np.sign(p["downstream_delta"])
        conc = "yes" if (sx != 0 and sy != 0 and sx == sy) else (
            "no" if (sx != 0 and sy != 0) else "zero-sign"
        )
        lines.append(
            f"| {p['lang']} | {p['family']} | {fmt(p['intrinsic_delta'])} "
            f"| {fmt(p['downstream_delta'])} | {conc} |"
        )
    if dt["skipped"]:
        lines.append("\nSkipped pairs:")
        for s in dt["skipped"]:
            lines.append(f"- {s['family']}/{s['lang']}: {s['reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="F15 stats/aggregation: cell means, paired bootstrap "
                     "contrasts, Holm-Bonferroni, directional test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="directory of per-cell eval JSONs (+ optional __items.jsonl)")
    ap.add_argument("--reference-arm", default="original",
                    help="arm every contrast is computed against (default: original)")
    ap.add_argument("--metric", default="acc",
                    help="lm-eval metric name without ',none' suffix (default: acc)")
    ap.add_argument("--intrinsic-csv", type=Path, default=None,
                    help="optional CSV with header 'lang,delta' for the directional test")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="path to write the machine-readable summary JSON")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"significance threshold (default: {ALPHA})")
    ap.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS,
                    help=f"bootstrap resample count (default: {BOOTSTRAP_ITERS})")
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_RNG_SEED,
                    help=f"RNG seed for the bootstrap (default: {BOOTSTRAP_RNG_SEED})")
    args = ap.parse_args()

    if not args.eval_dir.is_dir():
        print(f"ERROR: --eval-dir {args.eval_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    cells, skipped = load_cells(args.eval_dir, args.metric)
    if not cells:
        print(f"ERROR: no per-cell JSONs found under {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"# F15 stats — {args.eval_dir}")
    print(f"\nLoaded {len(cells)} cell files, skipped {len(skipped)} non-cell files.")
    if skipped:
        print("\nSkipped files:")
        for s in skipped:
            print(f"- {s['path']}: {s['reason']}")

    cells_agg = aggregate_cells(cells)
    print(render_cells_table(cells_agg, args.metric))

    # Paired contrasts, per family, per lang, per non-reference arm present.
    rng = np.random.default_rng(args.bootstrap_seed)
    cells_by_group: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        cells_by_group[(c["family"], c["lang"], c["arm"])].append(c)

    contrasts: dict[tuple, dict] = {}
    for (family, lang, arm) in sorted(cells_by_group):
        if arm == args.reference_arm:
            continue
        ref_key = (family, lang, args.reference_arm)
        key = (family, lang, arm)
        if ref_key not in cells_by_group:
            contrasts[key] = {
                "family": family, "lang": lang, "arm": arm,
                "reference_arm": args.reference_arm,
                "skipped_reason": (
                    f"no {args.reference_arm!r} cell for family={family!r} lang={lang!r}"
                ),
            }
            continue
        contrasts[key] = paired_contrast(
            family, lang, arm, args.reference_arm,
            cells_by_group[ref_key], cells_by_group[key],
            rng, args.bootstrap_iters,
        )

    # Holm-Bonferroni within each family (metric fixed per run).
    holm_by_family: dict = {}
    for family in sorted({k[0] for k in contrasts}):
        pvals = {
            k: contrasts[k]["bootstrap_p"]
            for k in contrasts if k[0] == family and "skipped_reason" not in contrasts[k]
        }
        holm_by_family.update(holm_bonferroni(pvals, args.alpha))

    print(render_contrasts_table(contrasts, holm_by_family, args.reference_arm,
                                  args.metric, args.alpha))

    # Directional test.
    if args.intrinsic_csv is not None:
        intrinsic = load_intrinsic_csv(args.intrinsic_csv)
        dt = directional_test(intrinsic, contrasts, args.reference_arm,
                               DIRECTIONAL_TARGET_ARM)
    else:
        dt = {"enabled": False, "note": "no --intrinsic-csv supplied; directional test skipped."}
    print(render_directional_test(dt))

    # Machine-readable summary.
    repo = Path(__file__).resolve().parent.parent
    summary = {
        "metadata": {
            "eval_dir": str(args.eval_dir),
            "reference_arm": args.reference_arm,
            "metric": args.metric,
            "alpha": args.alpha,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "intrinsic_csv": str(args.intrinsic_csv) if args.intrinsic_csv else None,
            "intrinsic_csv_sha256": sha256(args.intrinsic_csv) if args.intrinsic_csv else None,
            "n_input_files_loaded": len(cells),
            "n_input_files_skipped": len(skipped),
            "skipped_files": skipped,
            "script_git_commit": git_commit(repo),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "cells": {
            "|".join(k): v for k, v in sorted(cells_agg.items())
        },
        "contrasts": {
            "|".join(k) + f"_vs_{args.reference_arm}": {
                **v,
                **(
                    {"holm_p": holm_by_family[k]["holm_p"],
                     "significant_raw": v["bootstrap_p"] <= args.alpha,
                     "significant_holm": holm_by_family[k]["significant_holm"]}
                    if "skipped_reason" not in v
                    # Explicit nulls (not absent keys) for skipped contrasts,
                    # so a consumer can key into every contrast record the
                    # same way without a KeyError/.get() branch.
                    else {"holm_p": None, "significant_raw": None, "significant_holm": None}
                ),
            }
            for k, v in sorted(contrasts.items())
        },
        "directional_test": dt,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote machine-readable summary to {args.out_json}")


if __name__ == "__main__":
    main()
