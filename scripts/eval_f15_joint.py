#!/usr/bin/env python3
"""Routed-eval mode for F15 wave-2 joint-multilingual checkpoints.

F15 wave 2 trains ONE checkpoint per (arm, seed) on a JOINT mix of 5 target
languages (tur/fin/hun/arb/hin) plus eng, all at proportion 0.2 (cfg_snapshot.
langs; see a wave-2 run_metadata.json). For the `langspec` arm this differs
from every earlier F14/F14b/F15-wave-1 cell: `tokenizer.las_tokenizer_path`
points at a shared ROUTING POOL directory (base_tokenizer.json + one
langspec_<lang><suffix>.json per language, see pool_manifest.json), not a
single per-language overlay. At deployment/eval time there is no oracle label
telling the model which language a given sentence is in, so the realistic
question is: how well does per-item LAS routing
(`LasTokenizerHFLM`/`best_language_encode_batch`, which scores a sentence
under every pooled overlay and picks the highest-likelihood one) do against
an oracle that is simply handed the correct overlay outright?

For each cell matching `qwen2_5-0_5b_joint5_(original|langspec)_truetie_s\\d+`
under --cells-root:
  1. Provenance (loud failure, no skip-and-continue): run_metadata.json
     status == "completed", tokenizer.las_mode == arm, model.true_tie is
     True; plus `_assert_checkpoint_provenance` (las_experiment.
     downstream_eval) for the base-model-id and eos/pad-tokenizer-mismatch
     guards (project_qwen3_eos_model_tokenizer_bug). The las_tokenizer_path
     half of that guard is self-referential here (las_dir is READ from the
     same metadata file), so it can never fire on its own; the las_mode and
     eos-bug halves are the live checks. NO ASSERT FROM THE IMPORTED HELPERS
     IS WEAKENED OR BYPASSED.
  2. Model: PEFT adapter loaded with the TRAINING wiring (frozen-original
     output head; `load_model_training_wiring`, copied verbatim from
     scripts/eval_f14_multiblimp.py) -- no lm_head re-share. F15 arms are
     true_tie throughout (verified per-cell in step 1), so the re-shared
     wiring would load the SAME Parameter anyway; training wiring is used
     for consistency with the rest of the F15 eval suite. Loaded ONCE per
     cell and reused (same pattern as eval_f14_multiblimp.py's
     build_eng_retention_lm: multiple HFLM-family wrappers around one
     `pretrained=` model object) across every language and condition for
     that cell.
  3. Per language in {tur,fin,hun,arb,hin}, multiblimp_<lang> (0-shot,
     batch 16):
       original  : one condition, `model_tokenizer` -- plain HFLM.
       langspec  : two conditions --
         routed : LasTokenizerHFLM over the pool dir, las_lang=<lang>. Every
                  item is scored under EVERY pooled overlay and routed to
                  the argmax at encode time (`best_language_encode[_batch]`)
                  -- exactly the deployment question, since MultiBLiMP items
                  are single sentences and routing operates per item. A
                  FRESH instance is created per (cell, lang), so
                  `_routing_counts` starts empty each time (naturally
                  "reset between languages" -- no manual reset needed) and
                  is recorded verbatim after the task run.
         oracle : BaseUnigramHFLM pointed directly at that language's own
                  langspec_<lang><suffix>.json in the pool -- the same
                  overlay file the routed condition may or may not pick.
       routed_vs_oracle_delta_acc = routed acc(acc,none) - oracle acc.
  4. Routing accuracy standalone (no model, no GPU): once per UNIQUE pool
     dir referenced by the matched langspec cells (a pool is shared across
     all seeds of a family, so this is not repeated per seed). Loads the
     pool directly via LanguageSpecificUnigramLMTokenizer (not through any
     HFLM wrapper) and runs `best_language_encode_batch` over the FLORES+
     parallel sentences for each of the 5 languages
     (.../flores_parallel_data/<lang><suffix>.txt), reporting the 5x5
     routing confusion matrix and per-language routing accuracy. Written to
     routing_accuracy_<family>.json (family from pool_manifest.json).
  5. Provenance recorded per (cell, lang) JSON: checkpoint dir, training
     git commit, pool_manifest.json's own recorded per-file sha256
     (RE-VERIFIED against the live pool files before use -- an additional
     assert, not a replacement for LasTokenizerHFLM's own internal
     `_assert_tokenizers_consistent`, which is untouched), eval-repo
     commit+dirty state, lm_eval version.

Output: one `<tag>__<lang>.json` per (cell, lang) in --out, plus one
`routing_accuracy_<family>.json` per unique pool.

Usage:
  python scripts/eval_f15_joint.py \\
      [--cells-root /capstor/scratch/cscs/cmeister747/f15_wave2] \\
      [--out eval_results/f15_joint] [--langs tur,fin,hun,arb,hin] \\
      [--batch-size 16] [--device cuda] [--only-cell TAG] [--force]
"""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import lm_eval
from lm_eval.utils import handle_non_serializable
from lm_eval.models.huggingface import HFLM

from las_experiment.downstream_eval import (
    BaseUnigramHFLM,
    LasTokenizerHFLM,
    _assert_checkpoint_provenance,
)

# ---------------------------------------------------------------------------
# Cell-tag schema. Wave-2 joint cells are ONE model family (qwen2_5-0_5b) so,
# unlike eval_f14_multiblimp.py's MODEL_MAP, the base model id is a constant.
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
CELL_RE = re.compile(
    r"^qwen2_5-0_5b_joint5_(?P<arm>original|langspec)_truetie_s(?P<seed>\d+)$"
)

# Script-aware overlay-filename suffixes, matching pool_manifest.json's
# "langs" field and scripts/eval_f14_multiblimp.py's LANG_SUFFIX.
LANG_SUFFIX = {
    "tur": "_Latn",
    "fin": "_Latn",
    "hun": "_Latn",
    "arb": "_Arab",
    "hin": "_Deva",
}
DEFAULT_LANGS = list(LANG_SUFFIX.keys())

FLORES_DIR = Path(
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_state(repo: Path) -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def find_derived(cell_dir: Path) -> Path:
    """The ft*/ dir directly under the cell holding run_metadata.json and
    the adapter files. Copied from scripts/eval_f14_multiblimp.py."""
    cands = [d for d in cell_dir.iterdir()
             if d.is_dir() and d.name.startswith("ft") and
             (d / "run_metadata.json").exists()]
    if len(cands) != 1:
        raise RuntimeError(
            f"{cell_dir}: expected exactly 1 derived run dir with "
            f"run_metadata.json, found {[d.name for d in cands]}"
        )
    return cands[0]


def check_joint_provenance(tag: str, arm: str, meta: dict) -> None:
    """Loud, fail-fast provenance gate specific to this driver (spec item
    1): completed status, las_mode == arm, true_tie == True. Distinct from
    (and in addition to) `_assert_checkpoint_provenance`'s base-model-id and
    eos-bug checks, which are called separately -- this function does not
    replace or weaken those."""
    errs = []
    status = meta.get("status")
    if status != "completed":
        errs.append(f"status={status!r} != 'completed'")
    las_mode = meta.get("tokenizer", {}).get("las_mode")
    if las_mode != arm:
        errs.append(f"tokenizer.las_mode={las_mode!r} != arm {arm!r}")
    true_tie = meta.get("model", {}).get("true_tie")
    if true_tie is not True:
        errs.append(f"model.true_tie={true_tie!r} != True")
    if errs:
        raise ValueError(f"{tag}: provenance check failed: " + "; ".join(errs))


def load_model_training_wiring(base_model_id: str, ckpt_path: str, device: str):
    """PEFT load WITHOUT the lm_head re-share -- copied verbatim from
    scripts/eval_f14_multiblimp.py (see that module's docstring for the
    modules_to_save-tie rationale). Input lookup uses the trained
    modules_to_save copy, output head stays the frozen original; F15 arms
    are true_tie throughout (checked in check_joint_provenance above) so
    this is numerically identical to the re-shared wiring for these cells."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ckpt_path)
    for cfg_obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ):
        if cfg_obj is not None and hasattr(cfg_obj, "tie_word_embeddings"):
            cfg_obj.tie_word_embeddings = False
    model.to(device)
    model.eval()
    return model


def extract_metric(res: dict, task_name: str, metric_key: str) -> float:
    task_results = res.get("results", {}).get(task_name)
    if task_results is None or metric_key not in task_results:
        raise KeyError(
            f"{task_name}: missing metric {metric_key!r}; available: "
            f"{sorted((task_results or {}).keys())}"
        )
    return float(task_results[metric_key])


def load_and_verify_pool_manifest(pool_dir: Path) -> dict:
    """Read pool_manifest.json and RE-VERIFY every entry's recorded sha256
    against the live file it resolves to (following the pool's relative
    symlinks). This is an ADDITIONAL guard on top of LasTokenizerHFLM's own
    `_assert_tokenizers_consistent` (untouched) -- it catches drift between
    what the manifest recorded at pool-build time and what is on disk now,
    which vocab/special-token consistency alone would not catch (e.g. a
    same-shaped but re-fit overlay silently swapped in later)."""
    manifest_path = pool_dir / "pool_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{pool_dir}: pool_manifest.json missing")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("family"):
        raise ValueError(f"{manifest_path}: no 'family' field")
    for name, entry in manifest.get("entries", {}).items():
        live_path = pool_dir / name
        if not live_path.exists():
            raise FileNotFoundError(
                f"{pool_dir}: manifest entry {name!r} not found on disk "
                f"({live_path})"
            )
        live_sha = sha256(live_path)
        if live_sha != entry.get("sha256"):
            raise ValueError(
                f"{pool_dir}: sha256 drift for {name!r} -- manifest="
                f"{entry.get('sha256')!r} live={live_sha!r}. Refusing to "
                f"evaluate against a pool that no longer matches its own "
                f"manifest."
            )
    return manifest


def compute_routing_accuracy(pool_dir: Path, langs: list, manifest: dict) -> dict:
    """Standalone routing accuracy (spec item 4): no model, no GPU. Loads
    the pool's base + langspec tokenizers directly through
    LanguageSpecificUnigramLMTokenizer and routes the FLORES+ parallel
    sentences for each language, reporting the 5x5 confusion matrix (rows =
    true language, columns = routed-to pool key) and per-language accuracy.

    `vocab_size` passed to the constructor is a placeholder: `.load()`
    unconditionally overwrites `self.vocab_size` from the base tokenizer
    file (code/standard_unigramlm.py:462), so no HF AutoTokenizer / model
    needs to be loaded to get this right.
    """
    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

    base_path = pool_dir / "base_tokenizer.json"
    if not base_path.exists():
        raise FileNotFoundError(f"{pool_dir}: base_tokenizer.json missing")
    langspec_files = sorted(pool_dir.glob("langspec_*.json"))
    if not langspec_files:
        raise FileNotFoundError(f"{pool_dir}: no langspec_*.json files")
    language_paths = {f.stem[len("langspec_"):]: str(f) for f in langspec_files}

    las = LanguageSpecificUnigramLMTokenizer(vocab_size=8_000)
    las.load(str(base_path), language_paths)

    confusion = {true_lang: Counter() for true_lang in langs}
    n_sentences = {}
    for true_lang in langs:
        suffix = LANG_SUFFIX[true_lang]
        target_key = f"{true_lang}{suffix}"
        if target_key not in language_paths:
            raise KeyError(
                f"{pool_dir}: routing target {target_key!r} not in pool "
                f"languages {sorted(language_paths)}"
            )
        flores_file = FLORES_DIR / f"{true_lang}{suffix}.txt"
        if not flores_file.exists():
            raise FileNotFoundError(
                f"FLORES+ parallel file missing for {true_lang!r}: {flores_file}"
            )
        sentences = [l for l in flores_file.read_text().splitlines() if l.strip()]
        if not sentences:
            raise ValueError(f"{flores_file}: no non-empty sentences")
        best_langs, _ = las.best_language_encode_batch(sentences)
        for routed in best_langs:
            confusion[true_lang][routed] += 1
        n_sentences[true_lang] = len(sentences)

    routing_accuracy = {}
    for true_lang in langs:
        target_key = f"{true_lang}{LANG_SUFFIX[true_lang]}"
        correct = confusion[true_lang].get(target_key, 0)
        routing_accuracy[true_lang] = correct / n_sentences[true_lang]

    return {
        "pool_dir": str(pool_dir),
        "family": manifest["family"],
        "pool_manifest_git_commit": manifest.get("git_commit"),
        "langs": langs,
        "n_sentences": n_sentences,
        "confusion_counts": {k: dict(v) for k, v in confusion.items()},
        "routing_accuracy": routing_accuracy,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells-root", type=Path,
        default=Path("/capstor/scratch/cscs/cmeister747/f15_wave2"),
    )
    ap.add_argument("--out", type=Path, default=Path("eval_results/f15_joint"))
    ap.add_argument(
        "--langs", default=",".join(DEFAULT_LANGS),
        help=f"comma list from {sorted(LANG_SUFFIX)}",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--only-cell", default=None,
        help="restrict to a single cell dir name under --cells-root "
             "(validation runs).",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    bad = set(langs) - set(LANG_SUFFIX)
    if bad:
        raise ValueError(
            f"--langs: unknown language(s) {sorted(bad)}; choose from "
            f"{sorted(LANG_SUFFIX)}"
        )
    if not langs:
        raise ValueError("--langs: empty language list")

    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    gstate = git_state(repo)

    matched = []
    for cell in sorted(args.cells_root.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        if args.only_cell and cell.name != args.only_cell:
            continue
        matched.append((cell, m))

    print(
        f"discovered {len(matched)} joint cells matching {CELL_RE.pattern!r} "
        f"under {args.cells_root}"
    )
    if not matched:
        raise RuntimeError(
            f"No cells matching {CELL_RE.pattern!r} under {args.cells_root} "
            f"(only_cell={args.only_cell!r}). Refusing to run an empty batch."
        )

    # -----------------------------------------------------------------
    # Item 4: standalone routing accuracy, once per UNIQUE pool dir (a pool
    # is shared across every seed of a family -- see module docstring).
    # Also builds the pool-manifest cache reused by the per-cell loop below.
    # -----------------------------------------------------------------
    pool_cells = {}
    for cell, m in matched:
        if m["arm"] != "langspec":
            continue
        derived = find_derived(cell)
        meta = json.loads((derived / "run_metadata.json").read_text())
        las_dir = meta.get("tokenizer", {}).get("las_tokenizer_path")
        if las_dir is None:
            raise ValueError(f"{cell.name}: langspec arm but no las_tokenizer_path")
        pool_cells.setdefault(las_dir, []).append(cell.name)

    manifest_cache = {}
    for pool_dir_str, cells_using in pool_cells.items():
        pool_dir = Path(pool_dir_str)
        manifest = load_and_verify_pool_manifest(pool_dir)
        manifest_cache[pool_dir_str] = manifest

        routing_out = args.out / f"routing_accuracy_{manifest['family']}.json"
        if routing_out.exists() and not args.force:
            print(f"[skip] routing accuracy exists: {routing_out}")
            continue
        print(f"[routing-accuracy] {manifest['family']} <- {pool_dir}")
        acc_record = compute_routing_accuracy(pool_dir, langs, manifest)
        acc_record["cells_using_pool"] = sorted(cells_using)
        acc_record["eval_git"] = gstate
        acc_record["timestamp"] = datetime.now().isoformat()
        routing_out.write_text(
            json.dumps(acc_record, indent=2, default=handle_non_serializable)
        )
        print(f"[done] routing accuracy -> {routing_out}")
        print(f"  routing_accuracy: {acc_record['routing_accuracy']}")

    # -----------------------------------------------------------------
    # Items 1-3: per-(cell, language) model eval.
    # -----------------------------------------------------------------
    for cell, m in matched:
        tag = cell.name
        arm, seed = m["arm"], int(m["seed"])

        derived = find_derived(cell)
        meta = json.loads((derived / "run_metadata.json").read_text())

        check_joint_provenance(tag, arm, meta)
        las_dir = meta.get("tokenizer", {}).get("las_tokenizer_path")
        _assert_checkpoint_provenance(str(derived), BASE_MODEL_ID, arm, las_dir)

        pool_dir = None
        manifest = None
        if arm == "langspec":
            pool_dir = Path(las_dir)
            manifest = manifest_cache.get(las_dir) or load_and_verify_pool_manifest(pool_dir)
            manifest_cache[las_dir] = manifest

        out_paths = {lang: args.out / f"{tag}__{lang}.json" for lang in langs}
        pending = [lang for lang in langs if args.force or not out_paths[lang].exists()]
        if not pending:
            print(f"[skip] {tag} (all languages already evaluated)")
            continue

        print(f"[load] {tag} ({arm})")
        model_obj = load_model_training_wiring(BASE_MODEL_ID, str(derived), args.device)
        kw = {"pretrained": model_obj, "tokenizer": BASE_MODEL_ID}

        for lang in pending:
            suffix = LANG_SUFFIX[lang]
            eval_task_name = f"multiblimp_{lang}"

            record = {
                "tag": tag,
                "arm": arm,
                "seed": seed,
                "lang": lang,
                "model": BASE_MODEL_ID,
                "checkpoint": str(derived),
                "train_git_commit": meta.get("git_commit"),
                "training_langs": meta.get("cfg_snapshot", {}).get("langs"),
                "eval_git": gstate,
                "lm_eval_version": lm_eval.__version__,
                "batch_size": args.batch_size,
                "num_fewshot": 0,
                "task": eval_task_name,
                "timestamp": datetime.now().isoformat(),
            }

            if arm == "original":
                print(f"[eval] {tag} lang={lang} condition=model_tokenizer")
                lm = HFLM(**kw, batch_size=args.batch_size)
                res = lm_eval.simple_evaluate(
                    model=lm, tasks=[eval_task_name], num_fewshot=0
                )
                record["conditions"] = {
                    "model_tokenizer": {
                        "results": res["results"],
                        "n_samples": res.get("n-samples"),
                    }
                }
                del lm

            elif arm == "langspec":
                overlay_path = pool_dir / f"langspec_{lang}{suffix}.json"
                if not overlay_path.exists():
                    raise FileNotFoundError(f"{tag}: oracle overlay {overlay_path} missing")

                print(f"[eval] {tag} lang={lang} condition=routed")
                lm_routed = LasTokenizerHFLM(
                    las_tokenizer_main_path=str(pool_dir),
                    las_lang=lang,
                    lang_suffix=suffix,
                    **kw,
                    batch_size=args.batch_size,
                )
                res_routed = lm_eval.simple_evaluate(
                    model=lm_routed, tasks=[eval_task_name], num_fewshot=0
                )
                routing_distribution = dict(lm_routed._routing_counts)
                del lm_routed

                print(f"[eval] {tag} lang={lang} condition=oracle")
                lm_oracle = BaseUnigramHFLM(
                    base_tokenizer_path=str(overlay_path), **kw, batch_size=args.batch_size
                )
                res_oracle = lm_eval.simple_evaluate(
                    model=lm_oracle, tasks=[eval_task_name], num_fewshot=0
                )
                del lm_oracle

                routed_acc = extract_metric(res_routed, eval_task_name, "acc,none")
                oracle_acc = extract_metric(res_oracle, eval_task_name, "acc,none")

                record["pool_dir"] = str(pool_dir)
                record["pool_manifest_sha256"] = {
                    k: v["sha256"] for k, v in manifest["entries"].items()
                }
                record["pool_manifest_git_commit"] = manifest.get("git_commit")
                record["conditions"] = {
                    "routed": {
                        "results": res_routed["results"],
                        "n_samples": res_routed.get("n-samples"),
                        "routing_distribution": routing_distribution,
                    },
                    "oracle": {
                        "results": res_oracle["results"],
                        "n_samples": res_oracle.get("n-samples"),
                        "overlay_tokenizer": str(overlay_path),
                        "overlay_sha256": sha256(overlay_path),
                    },
                }
                record["routed_acc"] = routed_acc
                record["oracle_acc"] = oracle_acc
                record["routed_vs_oracle_delta_acc"] = routed_acc - oracle_acc
                print(
                    f"  routed_acc={routed_acc:.4f} oracle_acc={oracle_acc:.4f} "
                    f"delta={record['routed_vs_oracle_delta_acc']:+.4f} "
                    f"routing={routing_distribution}"
                )

            else:
                raise ValueError(f"{tag}: unhandled arm {arm!r}")

            out_paths[lang].write_text(
                json.dumps(record, indent=2, default=handle_non_serializable)
            )
            print(f"[done] {tag} lang={lang} -> {out_paths[lang]}")

        del model_obj, kw
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
