#!/usr/bin/env python3
"""MultiBLiMP + Belebele-eng eval for F14/F14b/F15 cells.

F14/F14b: original PR#21 tab:extrinsic replication. F15: LangMAP downstream
campaign (plan: `the-paper-has-not-stateful-bear.md`, "Evaluation suite" item
5) -- adds the base-Unigram (A5) and vocab-extension (A4) arms, the
Llama-3.2-1B family, script-aware overlay filenames for arb/hin, and a
5-shot English-retention task (belebele_eng) routed per the arm's OWN
training-time English tokenizer.

For each completed cell under --cells-root (family/model-filtered):
  - provenance checks: run_metadata status/las_mode/eos-guard, overlay dir
    matches the cell language, cfg_snapshot.reinit_embeddings matches the tag
  - model loading per --head-wiring:
      training  : PEFT adapter loaded, NO lm_head re-share -- the output head
                  stays the frozen original the model actually trained
                  against (modules_to_save does not preserve the tie; see
                  memory modules-to-save-tie-not-preserved)
      reshared  : downstream_eval._build_eval_lm_kwargs re-share -- the
                  F12/P4d-lineage wiring (tied trained matrix; a
                  configuration training never saw)
    Both avoid lm-eval's resize-before-adapter-load by pre-building the model.
    SHORT-CIRCUIT: F15 arms are true_tie throughout (plan Design section), so
    for F15 cells the two wirings load the SAME shared trained Parameter and
    are numerically identical -- the flag only distinguishes pre-true-tie
    F14/F14b cells.
  - encoding with the arm's own tokenizer:
      original                : model tokenizer (HFLM)
      langspec / placebo      : single overlay file (BaseUnigramHFLM),
                                 matching the single-overlay training
                                 protocol (no routing pool)
      base                    : `base_tokenizer.json` (BaseUnigramHFLM) --
                                 language-AGNOSTIC Unigram rescoring of the
                                 same vocab (F15 arm A5); the overlay dir is
                                 shared across languages for a family, so
                                 (unlike langspec/placebo/vocab_extend) it is
                                 NOT required to be named after the cell's
                                 language
      vocab_extend            : `vocabext_<lang><suffix>.json`
                                 (ExtendedVocabHFLM, `las_experiment.
                                 downstream_eval`, imported lazily since it is
                                 landing in a parallel worktree). The
                                 checkpoint's embedding matrix is RESIZED
                                 (base vocab + new per-language BPE pieces);
                                 neither downstream_eval._build_eval_lm_kwargs
                                 nor this script's load_model_training_wiring
                                 resize before loading the adapter, so both
                                 would shape-mismatch. `load_model_vocab_extend`
                                 below resizes first, reading the target row
                                 count from run_metadata.json's
                                 tokenizer.vocab_extend.n_total_vocab (see
                                 `_resolve_extended_vocab_size` -- the same
                                 field the landed A4 downstream_eval.py code
                                 uses for the identical resize).
  - lm_eval multiblimp_<lang> (0-shot) and/or belebele_eng_Latn (5-shot, see
    `build_eng_retention_lm` for the English-routing rule); one JSON per
    (cell, wiring, task) with full provenance. multiblimp also writes an
    optional `<tag>__items.jsonl` sibling (per-item accuracy) for
    scripts/compile_f15_stats.py's item-level paired bootstrap.

Sharding is by sha1(tag) % N, so shard membership is independent of what
other cell dirs exist (no drift when new families land in the same root).

Usage:
  python scripts/eval_f14_multiblimp.py --cells-root <dir> --out <dir> \
      --family f14 --head-wiring training [--tasks multiblimp,belebele_eng] \
      [--model qwen2_5-0_5b] [--shard K/N] [--batch-size 16]
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.utils import handle_non_serializable
from lm_eval.models.huggingface import HFLM

from las_experiment.downstream_eval import (
    BaseUnigramHFLM,
    _assert_checkpoint_provenance,
    _build_eval_lm_kwargs,
)

# ---------------------------------------------------------------------------
# Tag schema. Model and language are regex groups built FROM these maps (not
# hardcoded alternations) so the two stay a single source of truth.
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "qwen2_5-0_5b": "Qwen/Qwen2.5-0.5B",
    "llama3_2-1b": "meta-llama/Llama-3.2-1B",
}
# Script-aware overlay-filename suffixes (F15 adds arb/Arabic, hin/Devanagari
# to the tur/fin/hun Latin-script set).
LANG_SUFFIX = {
    "tur": "_Latn",
    "fin": "_Latn",
    "hun": "_Latn",
    "arb": "_Arab",
    "hin": "_Deva",
}
_MODEL_ALT = "|".join(re.escape(k) for k in MODEL_MAP)
_LANG_ALT = "|".join(re.escape(k) for k in LANG_SUFFIX)
TAG_RE = re.compile(
    rf"^(?P<model>{_MODEL_ALT})_(?P<lang>{_LANG_ALT})_"
    r"(?P<arm>original|langspec|placebo|base|vocab_extend)"
    r"(?P<reinit>_reinitemb)?(?P<truetie>_truetie)?_s(?P<seed>\d+)$"
)

TASKS_AVAILABLE = {"multiblimp", "belebele_eng"}
# multiblimp keeps the ORIGINAL (no task suffix) filename so F14/F14b re-runs
# against the existing --out dir keep matching already-completed files;
# belebele_eng is new and gets its own suffix so it never collides with (or
# forces re-eval of) those files.
TASK_FILENAME_SUFFIX = {"multiblimp": "", "belebele_eng": "__belebele_eng"}
TASK_NUM_FEWSHOT = {"multiblimp": 0, "belebele_eng": 5}


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
    cands = [d for d in cell_dir.iterdir()
             if d.is_dir() and d.name.startswith("ft") and
             (d / "run_metadata.json").exists()]
    if len(cands) != 1:
        raise RuntimeError(
            f"{cell_dir}: expected exactly 1 derived run dir with "
            f"run_metadata.json, found {[d.name for d in cands]}"
        )
    return cands[0]


def load_model_training_wiring(base_model_id: str, ckpt_path: str, device: str):
    """PEFT load WITHOUT the lm_head re-share: input lookup uses the trained
    modules_to_save copy (wrapper forward), output head stays the frozen
    original -- exactly the training-time wiring of modules_to_save-only
    checkpoints. tie_word_embeddings is still disabled on the configs to
    dodge HFLM's tie_weights() crash (PEFT issue #2777); with the re-share
    skipped this does NOT change any weight, it only prevents a re-tie from
    clobbering the wiring."""
    import torch
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


def _resolve_extended_vocab_size(meta: dict, meta_path_hint: str) -> int:
    """Row count the checkpoint's embedding matrix was resized to for a
    vocab_extend run, read from the training run's OWN provenance record --
    `run_metadata.json`'s `tokenizer.vocab_extend.n_total_vocab`. This
    mirrors the authoritative source `downstream_eval._build_eval_lm_kwargs`
    uses for the identical resize (confirmed against the landed A4
    implementation, las_experiment/downstream_eval.py, 2026-08-16): "the
    target size comes from the run's own provenance record -- never from the
    eval config, so it cannot disagree with what trained." Deliberately the
    ONLY source used (no silent fallback to a value reverse-engineered from
    the saved adapter tensor shape, which was this function's first-pass
    design before the A4 class landed and exposed the real field)."""
    ve = (meta.get("tokenizer") or {}).get("vocab_extend")
    if not ve or ve.get("n_total_vocab") is None:
        raise ValueError(
            f"run_metadata for {meta_path_hint} has no "
            f"tokenizer.vocab_extend.n_total_vocab -- the checkpoint was not "
            f"produced by the vocab_extend path. Refusing to guess the "
            f"resize target."
        )
    return int(ve["n_total_vocab"])


def load_model_vocab_extend(base_model_id: str, ckpt_path: str, device: str,
                             head_wiring: str, meta: dict):
    """vocab_extend checkpoints trained with a RESIZED embedding matrix (base
    vocab + new per-language BPE pieces, FOCUS-init). Resize BEFORE
    PeftModel.from_pretrained -- loading at the base model's native vocab
    size first (as both downstream_eval._build_eval_lm_kwargs's non-A4 path
    and this script's load_model_training_wiring do) shape-mismatches
    against the saved adapter. `mean_resizing=False` matches the landed A4
    implementation -- the new rows are immediately overwritten by the
    adapter's trained embed_tokens copy, so this only avoids an unnecessary
    full-matrix mean computation, not a correctness difference.

    head_wiring 'training' vs 'reshared' is kept for interface parity with
    the F14/F14b path (see module docstring SHORT-CIRCUIT note): F15 arms
    are true_tie throughout, so 'reshared' below re-ties lm_head onto the
    SAME trained Parameter 'training' already uses -- for F15 cells the two
    produce identical weights.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.bfloat16)
    resize_to = _resolve_extended_vocab_size(meta, ckpt_path)
    orig_vocab = model.get_input_embeddings().weight.shape[0]
    if resize_to < orig_vocab:
        raise ValueError(
            f"vocab_extend: run_metadata n_total_vocab={resize_to}, fewer "
            f"than the base model's {orig_vocab} -- extension should only "
            f"ever GROW the vocab. Refusing (wrong checkpoint/base model "
            f"pairing, not a resize target)."
        )
    if resize_to != orig_vocab:
        model.resize_token_embeddings(resize_to, mean_resizing=False)
        if model.get_input_embeddings().weight.shape[0] != resize_to:
            raise RuntimeError(f"resize to {resize_to} rows failed before adapter load.")
    model = PeftModel.from_pretrained(model, ckpt_path)

    inner = getattr(getattr(model, "base_model", None), "model", None)
    if head_wiring == "reshared" and inner is not None:
        # Mirrors downstream_eval._build_eval_lm_kwargs's tied-fix re-share
        # (case B): embed_tokens is modules_to_save-wrapped, lm_head is not,
        # so lm_head still points at the untrained (but now-resized) tied
        # embedding unless re-pointed at the trained copy.
        emb_module = inner.get_input_embeddings()
        is_emb_wrapped = hasattr(emb_module, "modules_to_save")
        lm_head_module = getattr(inner, "lm_head", None)
        is_lm_head_wrapped = (
            lm_head_module is not None and hasattr(lm_head_module, "modules_to_save")
        )
        inner_was_tied = bool(getattr(inner.config, "tie_word_embeddings", False))
        if is_emb_wrapped and not is_lm_head_wrapped and lm_head_module is not None and inner_was_tied:
            active = getattr(model, "active_adapter", "default")
            if isinstance(active, list):
                active = active[0] if active else "default"
            adapter_key = active if active in emb_module.modules_to_save else next(
                iter(emb_module.modules_to_save)
            )
            trained_param = emb_module.modules_to_save[adapter_key].weight
            lm_head_module.weight = trained_param

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


def build_eval_task_name(cli_task: str, lang: str) -> str:
    if cli_task == "multiblimp":
        return f"multiblimp_{lang}"
    if cli_task == "belebele_eng":
        return "belebele_eng_Latn"
    raise ValueError(f"Unhandled --tasks entry {cli_task!r}")


def build_eng_retention_lm(arm: str, kw: dict, tok_file, batch_size: int):
    """English-retention routing (plan item 3, belebele_eng). Explicit rule:
    each arm is evaluated on English through whatever tokenizer it actually
    trained English THROUGH -- not through the arm's target-language
    tokenizer.

    - original / langspec / placebo / base: finetune.py routes 'eng' through
      the model's own tokenizer for all four las_modes
      (`tokenizer_for_lang["eng"] = model_tokenizer.backend_tokenizer`), so
      belebele_eng for these arms is a plain HFLM.
    - vocab_extend: the extension protocol does NOT special-case eng -- it
      keeps every original BPE piece and only appends new per-language
      pieces (F15 plan A4 rationale: "keeps English handling ... comparable"
      vs a from-scratch replacement), so eng was trained through the SAME
      extended-vocab tokenizer as the target language. Reuse the arm's own
      ExtendedVocabHFLM/tok_file rather than the plain tokenizer.
    """
    if arm == "vocab_extend":
        if tok_file is None:
            raise ValueError("vocab_extend: tok_file must be set for eng routing")
        from las_experiment.downstream_eval import ExtendedVocabHFLM  # lazy: A4 lands this
        return ExtendedVocabHFLM(
            **kw, extended_tokenizer_path=str(tok_file), batch_size=batch_size
        )
    return HFLM(**kw, batch_size=batch_size)


def write_items_jsonl(out_dir: Path, tag: str, head_wiring: str, eval_task_name: str,
                       res: dict, metric: str = "acc", ckpt_suffix: str = "") -> None:
    """Optional sibling `<tag>__head<wiring>__items.jsonl` for
    scripts/compile_f15_stats.py's item-level paired bootstrap. Filename
    matches the results JSON's own keying (`<tag>__head<wiring>[__task].json`)
    -- pre-launch review fix: the previous bare `<tag>__items.jsonl` name
    dropped head_wiring, so a training- and a reshared-wiring eval of the
    SAME cell would silently clobber each other's items file. One JSON line
    per item: {"doc_id": ..., "<metric>": <float>}. lm_eval.simple_evaluate
    defaults log_samples=True already (unchanged here), so
    res["samples"][eval_task_name] is populated for free; this only adds the
    file write. Only called for multiblimp (<=2570 items/lang, so the file
    stays small); belebele_eng does not get one.

    `ckpt_suffix` (e.g. "__ckpt357") mirrors the results JSON's own naming so
    an intermediate-checkpoint items file never collides with (or is
    silently mixed into) the final-checkpoint items file for the same cell.
    """
    samples = (res.get("samples") or {}).get(eval_task_name)
    if not samples:
        return
    lines = []
    for ex in samples:
        if metric not in ex:
            raise KeyError(
                f"{tag}: per-sample record for {eval_task_name} missing "
                f"metric {metric!r}; available: {sorted(ex.keys())}"
            )
        lines.append(json.dumps({"doc_id": ex["doc_id"], metric: float(ex[metric])}))
    (out_dir / f"{tag}__head{head_wiring}{ckpt_suffix}__items.jsonl").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--family", choices=["f14", "f14b", "f15", "all"], default="f14",
        help="f14 = no _reinitemb/_truetie tags; f14b = only those (both "
             "unchanged from the original F14/F14b driver, independent of "
             "model/arm); f15 = arm in {base, vocab_extend} OR the "
             "llama3_2-1b model prefix -- note an f15 cell with neither "
             "reinit nor truetie suffix also passes the f14 filter, so "
             "combine with --model when that overlap matters. This f14/f14b/"
             "f15 split assumes ONE --cells-root mixing all three families "
             "(the original F14/F14b/F15 root). It does NOT fit F15 wave-1 "
             "cells (scripts/launch_f15_wave1.sh): those are true_tie "
             "throughout, so every wave-1 tag carries _truetie regardless of "
             "arm, which would require --family f14b for original/langspec/"
             "placebo arms but --family f15 for base/vocab_extend -- an "
             "arm-dependent split that defeats the point of --family. Since "
             "wave-1 cells live under their OWN --cells-root (disjoint from "
             "the F14/F14b/F15 root, no other family's cells to exclude), "
             "use --family all --cells-root <f15-wave1-root> instead.",
    )
    ap.add_argument(
        "--model", choices=[*MODEL_MAP.keys(), "all"], default="all",
        help="filter by model tag; 'all' = no filter",
    )
    ap.add_argument(
        "--expect-cells", type=int, default=None,
        help="abort if the post family/model-filter cell count under "
             "--cells-root differs from this (pre-shard: the same total is "
             "checked by every shard). The wrapper (run_f14_eval_login.sh) "
             "passes this when the expected count is known; omit to skip "
             "the check.",
    )
    ap.add_argument(
        "--tasks", default="multiblimp",
        help=f"comma list from {sorted(TASKS_AVAILABLE)}. belebele_eng is "
             f"5-shot English retention, evaluated with each arm's OWN "
             f"training-time English routing -- see build_eng_retention_lm.",
    )
    ap.add_argument(
        "--head-wiring", choices=["training", "reshared"], required=True,
        help="'training' = frozen-original output head (matches what the "
             "optimizer saw); 'reshared' = tied trained matrix re-share "
             "(F12/P4d-lineage wiring, a config training never saw for "
             "non-true-tie runs). SHORT-CIRCUIT: F15 arms are true_tie "
             "throughout (plan Design section), so for F15 cells both "
             "wirings load the SAME shared trained Parameter and are "
             "numerically identical -- this flag only distinguishes "
             "pre-true-tie F14/F14b cells.",
    )
    ap.add_argument("--shard", default="0/1", help="K/N by sha1(tag) mod N")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="Evaluate the intermediate checkpoint-N/ subdir inside the "
             "derived run dir (e.g. 357/1781/3561 for F15 wave1/wave2 "
             "cells) instead of the final trained state. Default: absent, "
             "current final-checkpoint behavior unchanged (byte-identical "
             "output). Hard-fails naming the missing path if checkpoint-N/ "
             "does not exist under the cell's derived dir. Output filenames "
             "and the sibling __items.jsonl get a __ckptN suffix so they "
             "never collide with the final-checkpoint files.",
    )
    args = ap.parse_args()

    requested_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad_tasks = set(requested_tasks) - TASKS_AVAILABLE
    if bad_tasks:
        raise ValueError(
            f"--tasks: unknown task(s) {sorted(bad_tasks)}; choose from "
            f"{sorted(TASKS_AVAILABLE)}"
        )
    if not requested_tasks:
        raise ValueError("--tasks: empty task list")

    k, n = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    gstate = git_state(repo)
    ckpt_suffix = f"__ckpt{args.checkpoint_step}" if args.checkpoint_step is not None else ""

    # Pass 1: family/model filter only (no shard split, no metadata reads) --
    # this is the population --expect-cells checks against, since shard is a
    # per-invocation runtime split of the SAME pool across parallel shards,
    # not a property of the pool itself.
    matched = []
    for cell in sorted(args.cells_root.iterdir()):
        m = TAG_RE.match(cell.name) if cell.is_dir() else None
        if not m:
            continue
        model_tag, arm = m["model"], m["arm"]
        is_f14b = bool(m["reinit"] or m["truetie"])
        is_f15 = arm in ("base", "vocab_extend") or model_tag == "llama3_2-1b"
        if args.family == "f14" and is_f14b:
            continue
        if args.family == "f14b" and not is_f14b:
            continue
        if args.family == "f15" and not is_f15:
            continue
        if args.model != "all" and model_tag != args.model:
            continue
        matched.append((cell, m))

    print(
        f"discovered {len(matched)} cells (family={args.family} "
        f"model={args.model}) under {args.cells_root}"
    )
    if args.expect_cells is not None and len(matched) != args.expect_cells:
        raise RuntimeError(
            f"--expect-cells={args.expect_cells} but discovered {len(matched)} "
            f"cells (family={args.family} model={args.model}) under "
            f"{args.cells_root} -- investigate before evaluating a wrong or "
            f"partial cell set."
        )

    for cell, m in matched:
        tag = cell.name
        model_tag, lang, arm = m["model"], m["lang"], m["arm"]
        base_model_id = MODEL_MAP[model_tag]

        if int(hashlib.sha1(tag.encode()).hexdigest(), 16) % n != k:
            continue

        out_paths = {
            t: args.out / (
                f"{tag}__head{args.head_wiring}{TASK_FILENAME_SUFFIX[t]}"
                f"{ckpt_suffix}.json"
            )
            for t in requested_tasks
        }
        pending_tasks = [t for t in requested_tasks if args.force or not out_paths[t].exists()]
        if not pending_tasks:
            print(f"[skip] {tag} (exists: {[p.name for p in out_paths.values()]})")
            continue

        derived = find_derived(cell)
        if args.checkpoint_step is not None:
            ckpt_load_dir = derived / f"checkpoint-{args.checkpoint_step}"
            if not ckpt_load_dir.exists():
                raise FileNotFoundError(
                    f"{tag}: --checkpoint-step={args.checkpoint_step} requested "
                    f"but {ckpt_load_dir} does not exist."
                )
        else:
            ckpt_load_dir = derived
        meta = json.loads((derived / "run_metadata.json").read_text())
        if meta.get("status") != "completed":
            print(f"[SKIP-INCOMPLETE] {tag}: status={meta.get('status')!r} — not evaluating")
            continue
        snap_reinit = bool(meta.get("cfg_snapshot", {}).get("reinit_embeddings", False))
        if snap_reinit != bool(m["reinit"]):
            raise ValueError(
                f"{tag}: cfg_snapshot.reinit_embeddings={snap_reinit} does not "
                f"match tag reinit={bool(m['reinit'])} — mislabeled cell."
            )
        meta_true_tie = meta.get("model", {}).get("true_tie")
        if meta_true_tie is not None and bool(meta_true_tie) != bool(m["truetie"]):
            raise ValueError(
                f"{tag}: run_metadata model.true_tie={meta_true_tie} does not "
                f"match tag truetie={bool(m['truetie'])} — mislabeled cell."
            )
        las_dir = meta.get("tokenizer", {}).get("las_tokenizer_path") or None
        if arm != "original":
            if las_dir is None:
                raise ValueError(f"{tag}: no las_tokenizer_path in metadata")
            # base-Unigram is language-AGNOSTIC (F15 plan A5) -- its overlay
            # dir is shared across a family's languages, so only the
            # genuinely per-language artifacts (langspec/placebo/
            # vocab_extend) get the wrong-language-overlay guard.
            if arm != "base" and Path(las_dir).name != lang:
                raise ValueError(
                    f"{tag}: overlay dir {las_dir!r} does not end in {lang!r} — "
                    f"wrong-language overlay. Refusing."
                )

        # Provenance (las-dir half is self-referential by construction; the
        # las_mode + eos-bug halves are the live checks). Checked against
        # ckpt_load_dir (== derived unless --checkpoint-step) so a
        # checkpoint-N/ dir missing/mismatched run_metadata is caught the
        # same way as the final state (_find_run_metadata in
        # downstream_eval.py walks up to the parent ft* dir either way).
        _assert_checkpoint_provenance(str(ckpt_load_dir), base_model_id, arm, las_dir)

        if arm == "vocab_extend":
            model_obj = load_model_vocab_extend(
                base_model_id, str(ckpt_load_dir), args.device, args.head_wiring, meta
            )
            kw = {"pretrained": model_obj, "tokenizer": base_model_id}
        elif args.head_wiring == "reshared":
            kw = _build_eval_lm_kwargs(
                base_pretrained=base_model_id, ckpt_path=str(ckpt_load_dir),
                device=args.device,
            )
        else:
            kw = {
                "pretrained": load_model_training_wiring(base_model_id, str(ckpt_load_dir), args.device),
                "tokenizer": base_model_id,
            }

        tok_file = None
        if arm == "original":
            lm_arm = HFLM(**kw, batch_size=args.batch_size)
        elif arm in ("langspec", "placebo"):
            suffix = LANG_SUFFIX[lang]
            tok_file = Path(las_dir) / f"{arm}_{lang}{suffix}.json"
            if not tok_file.exists():
                raise FileNotFoundError(f"{tag}: overlay {tok_file} missing.")
            lm_arm = BaseUnigramHFLM(
                **kw, base_tokenizer_path=str(tok_file), batch_size=args.batch_size
            )
        elif arm == "base":
            tok_file = Path(las_dir) / "base_tokenizer.json"
            if not tok_file.exists():
                raise FileNotFoundError(f"{tag}: base tokenizer {tok_file} missing.")
            lm_arm = BaseUnigramHFLM(
                **kw, base_tokenizer_path=str(tok_file), batch_size=args.batch_size
            )
        elif arm == "vocab_extend":
            suffix = LANG_SUFFIX[lang]
            tok_file = Path(las_dir) / f"vocabext_{lang}{suffix}.json"
            if not tok_file.exists():
                raise FileNotFoundError(f"{tag}: extended-vocab tokenizer {tok_file} missing.")
            from las_experiment.downstream_eval import ExtendedVocabHFLM  # lazy: A4 lands this
            lm_arm = ExtendedVocabHFLM(
                **kw, extended_tokenizer_path=str(tok_file), batch_size=args.batch_size
            )
        else:
            raise ValueError(f"{tag}: unhandled arm {arm!r}")

        for cli_task in pending_tasks:
            eval_task_name = build_eval_task_name(cli_task, lang)
            num_fewshot = TASK_NUM_FEWSHOT[cli_task]
            if cli_task == "multiblimp":
                lm = lm_arm
                # Only true for multiblimp: this is the tokenizer the arm
                # actually evaluates with for THIS task.
                record_tok_file = tok_file
            else:
                lm = build_eng_retention_lm(arm, kw, tok_file, args.batch_size)
                # belebele_eng: record_tok_file per the SAME routing rule as
                # build_eng_retention_lm (None unless vocab_extend, which
                # reuses tok_file for eng too).
                record_tok_file = tok_file if arm == "vocab_extend" else None

            print(f"[eval] {tag} ({args.head_wiring}) -> {eval_task_name}")
            res = lm_eval.simple_evaluate(
                model=lm, tasks=[eval_task_name], num_fewshot=num_fewshot
            )

            record = {
                "tag": tag,
                "lang": lang,
                "arm": arm,
                "model": base_model_id,
                "reinit_embeddings": bool(m["reinit"]),
                "truetie": bool(m["truetie"]),
                "seed": int(m["seed"]),
                "head_wiring": args.head_wiring,
                "task": eval_task_name,
                "num_fewshot": num_fewshot,
                "results": res["results"],
                "n_samples": res.get("n-samples"),
                "checkpoint": str(ckpt_load_dir),
                "train_git_commit": meta.get("git_commit"),
                "eval_git": gstate,
                "overlay_tokenizer": str(record_tok_file) if record_tok_file else None,
                "overlay_sha256": sha256(record_tok_file) if record_tok_file else None,
                "lm_eval_version": lm_eval.__version__,
                "batch_size": args.batch_size,
                "timestamp": datetime.now().isoformat(),
            }
            if args.checkpoint_step is not None:
                record["checkpoint_step"] = args.checkpoint_step
            out_paths[cli_task].write_text(
                json.dumps(record, indent=2, default=handle_non_serializable)
            )
            acc = next(iter(res["results"].values()))
            print(f"[done] {tag} ({args.head_wiring}) {cli_task}: {acc}")

            if cli_task == "multiblimp":
                write_items_jsonl(
                    args.out, tag, args.head_wiring, eval_task_name, res,
                    ckpt_suffix=ckpt_suffix,
                )

        # GPU cleanup (pre-launch review S11): a prior version of this loop
        # only deleted `lm` when it was a SEPARATE belebele_eng wrapper
        # (guarded by `if lm is not lm_arm`). For the common multiblimp-only
        # case `lm` is an ALIAS of `lm_arm` (`lm = lm_arm` above), so that
        # guard never fired and `lm` kept a live reference to the model after
        # `del lm_arm` -- the object never reached refcount 0 and
        # gc.collect()/empty_cache() below freed nothing. `lm` is always
        # bound here regardless of which branch ran (pending_tasks is
        # non-empty), so an unconditional `del lm` is always safe.
        # `model_obj` (vocab_extend arm only, ~line 496) is a THIRD name
        # bound to the same model object, independent of `kw` and `lm_arm`;
        # delete it too when it exists so no arm leaves a stray reference
        # alive across cells.
        del lm, lm_arm, kw
        if arm == "vocab_extend":
            del model_obj
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
