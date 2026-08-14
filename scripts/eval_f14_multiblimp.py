#!/usr/bin/env python3
"""MultiBLiMP eval for F14/F14b cells (PR#21 tab:extrinsic replication).

For each completed cell under --cells-root (family-filtered):
  - provenance checks: run_metadata status/las_mode/eos-guard, overlay dir
    matches the cell language, cfg_snapshot.reinit_embeddings matches the tag
  - model loading per --head-wiring:
      training  : PEFT adapter loaded, NO lm_head re-share — the output head
                  stays the frozen original the model actually trained
                  against (modules_to_save does not preserve the tie; see
                  memory modules-to-save-tie-not-preserved)
      reshared  : downstream_eval._build_eval_lm_kwargs re-share — the
                  F12/P4d-lineage wiring (tied trained matrix; a
                  configuration training never saw)
    Both avoid lm-eval's resize-before-adapter-load by pre-building the model.
  - encoding with the arm's own tokenizer: original -> model tokenizer
    (HFLM); langspec/placebo -> single overlay file (BaseUnigramHFLM),
    matching the single-overlay training protocol (no routing pool)
  - lm_eval multiblimp_<lang>, num_fewshot=0; one JSON per (cell, wiring)
    with full provenance.

Sharding is by sha1(tag) % N, so shard membership is independent of what
other cell dirs exist (no drift when new families land in the same root).

Usage:
  python scripts/eval_f14_multiblimp.py --cells-root <dir> --out <dir> \
      --family f14 --head-wiring training [--shard K/N] [--batch-size 16]
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

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
TAG_RE = re.compile(
    r"^qwen2_5-0_5b_(?P<lang>tur|fin|hun)_(?P<arm>original|langspec|placebo)"
    r"(?P<reinit>_reinitemb)?(?P<truetie>_truetie)?_s(?P<seed>\d+)$"
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
    cands = [d for d in cell_dir.iterdir()
             if d.is_dir() and d.name.startswith("ft") and
             (d / "run_metadata.json").exists()]
    if len(cands) != 1:
        raise RuntimeError(
            f"{cell_dir}: expected exactly 1 derived run dir with "
            f"run_metadata.json, found {[d.name for d in cands]}"
        )
    return cands[0]


def load_model_training_wiring(ckpt_path: str, device: str):
    """PEFT load WITHOUT the lm_head re-share: input lookup uses the trained
    modules_to_save copy (wrapper forward), output head stays the frozen
    original — exactly the training-time wiring of modules_to_save-only
    checkpoints. tie_word_embeddings is still disabled on the configs to
    dodge HFLM's tie_weights() crash (PEFT issue #2777); with the re-share
    skipped this does NOT change any weight, it only prevents a re-tie from
    clobbering the wiring."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", choices=["f14", "f14b", "all"], default="f14",
                    help="f14 = no _reinitemb/_truetie tags; f14b = only those")
    ap.add_argument("--head-wiring", choices=["training", "reshared"],
                    required=True)
    ap.add_argument("--shard", default="0/1", help="K/N by sha1(tag) mod N")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    k, n = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    gstate = git_state(repo)

    for cell in sorted(args.cells_root.iterdir()):
        m = TAG_RE.match(cell.name) if cell.is_dir() else None
        if not m:
            continue
        tag = cell.name
        is_f14b = bool(m["reinit"] or m["truetie"])
        if args.family == "f14" and is_f14b:
            continue
        if args.family == "f14b" and not is_f14b:
            continue
        if int(hashlib.sha1(tag.encode()).hexdigest(), 16) % n != k:
            continue
        out_path = args.out / f"{tag}__head{args.head_wiring}.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {tag} (exists)")
            continue
        lang, arm = m["lang"], m["arm"]

        derived = find_derived(cell)
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
        las_dir = meta.get("tokenizer", {}).get("las_tokenizer_path") or None
        if arm != "original":
            if las_dir is None:
                raise ValueError(f"{tag}: no las_tokenizer_path in metadata")
            if Path(las_dir).name != lang:
                raise ValueError(
                    f"{tag}: overlay dir {las_dir!r} does not end in {lang!r} — "
                    f"wrong-language overlay. Refusing."
                )

        # Provenance (las-dir half is self-referential by construction; the
        # las_mode + eos-bug halves are the live checks).
        _assert_checkpoint_provenance(str(derived), BASE_MODEL, arm, las_dir)

        if args.head_wiring == "reshared":
            kw = _build_eval_lm_kwargs(
                base_pretrained=BASE_MODEL, ckpt_path=str(derived),
                device=args.device,
            )
        else:
            kw = {
                "pretrained": load_model_training_wiring(str(derived), args.device),
                "tokenizer": BASE_MODEL,
            }

        tok_file = None
        if arm == "original":
            lm = HFLM(**kw, batch_size=args.batch_size)
        else:
            tok_file = Path(las_dir) / f"{arm}_{lang}_Latn.json"
            if not tok_file.exists():
                raise FileNotFoundError(f"{tag}: overlay {tok_file} missing.")
            lm = BaseUnigramHFLM(
                **kw, base_tokenizer_path=str(tok_file), batch_size=args.batch_size
            )

        task = f"multiblimp_{lang}"
        print(f"[eval] {tag} ({args.head_wiring}) -> {task}")
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0)

        record = {
            "tag": tag,
            "lang": lang,
            "arm": arm,
            "reinit_embeddings": bool(m["reinit"]),
            "truetie": bool(m["truetie"]),
            "seed": int(m["seed"]),
            "head_wiring": args.head_wiring,
            "task": task,
            "results": res["results"],
            "n_samples": res.get("n-samples"),
            "checkpoint": str(derived),
            "train_git_commit": meta.get("git_commit"),
            "eval_git": gstate,
            "overlay_tokenizer": str(tok_file) if tok_file else None,
            "overlay_sha256": sha256(tok_file) if tok_file else None,
            "lm_eval_version": lm_eval.__version__,
            "batch_size": args.batch_size,
            "timestamp": datetime.now().isoformat(),
        }
        out_path.write_text(
            json.dumps(record, indent=2, default=handle_non_serializable)
        )
        acc = next(iter(res["results"].values()))
        print(f"[done] {tag} ({args.head_wiring}): {acc}")
        del lm, kw
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
