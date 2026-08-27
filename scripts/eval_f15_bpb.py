#!/usr/bin/env python3
"""FLORES+ BpB eval for F15 cells (plan: `the-paper-has-not-stateful-bear.md`,
"Evaluation suite" item 5, BpB half). MultiBLiMP/Belebele-eng is
`scripts/eval_f14_multiblimp.py`; this script covers the secondary BpB metric
(plan C1/C2/C3) via FLORES+ (held-out, Wikipedia/news domain, disjoint from
the FineWeb2 training corpora).

For each completed cell under --cells-root (same discovery/filters/sharding
as eval_f14_multiblimp.py: TAG_RE dir-name match, --family/--model/--shard
filters, status=="completed" gate, cfg_snapshot.reinit_embeddings check,
overlay-dir-basename==lang guard, `_assert_checkpoint_provenance`):
  - loads the checkpoint per --head-wiring, mirroring
    eval_f14_multiblimp.py's three load paths verbatim (copied, not
    imported, so this script has no runtime dependency on a sibling script):
      training     -> `load_model_training_wiring` (frozen-original head)
      reshared     -> `downstream_eval._build_eval_lm_kwargs` (tied re-share)
      vocab_extend -> `load_model_vocab_extend` (resize-before-adapter-load;
                       head_wiring is accepted for interface parity but F15
                       arms are true_tie throughout, so both wirings load the
                       same shared trained Parameter for these cells --
                       see eval_f14_multiblimp.py's SHORT-CIRCUIT note)
  - resolves the arm's own tokenizer for the TARGET language and, separately,
    for English (build_eng_retention_lm's routing rule, mirrored here):
      original/langspec/placebo/base -> model's own tokenizer for English
      vocab_extend                   -> the SAME extended tokenizer for
                                         English too (the extension never
                                         special-cases eng; finetune.py keeps
                                         every original BPE piece)
    Target-language tokenizer per arm:
      original     -> model's own tokenizer
      langspec     -> `{las_dir}/langspec_{lang}{suffix}.json`
      placebo      -> `{las_dir}/placebo_{lang}{suffix}.json`
      base         -> `{las_dir}/base_tokenizer.json` (language-agnostic)
      vocab_extend -> `{las_dir}/vocabext_{lang}{suffix}.json`, loaded via
                       `las_experiment.vocab_extend.ExtendedTokenizer` for
                       BOTH encode and decode. A bare `tokenizers.Tokenizer`
                       (what `bpb_core.tokenize_texts` calls `.decode` on)
                       mis-decodes ~8.7% of appended pieces (see
                       vocab_extend.py's DECODING CAVEAT), which corrupts the
                       byte denominator of BpB for these cells specifically
                       -- so this script reimplements the 6-line
                       `tokenize_texts` loop for vocab_extend
                       (`tokenize_texts_extended` below) rather than call
                       `bpb_core.tokenize_texts` on it. `bpb_core.compute_bpb`
                       is reused unmodified for every arm (the NLL/byte
                       numerics don't touch decode).
  - computes BpB via `bpb_core.compute_bpb` (max_seq_len 1024, batch 8) on
    FLORES+ `{flores_root}/{lang}{suffix}.txt` for the target language and
    `{flores_root}/eng_Latn.txt` for English.
  - writes one `<tag>__bpb.json` per cell to --out, with a schema DELIBERATELY
    disjoint from `scripts/compile_f15_stats.py`'s REQUIRED_KEYS (that loader
    requires a `results` key shaped like an lm_eval results dict; this
    script's numbers live under `flores_bpb_target`/`flores_bpb_eng` instead,
    no `results` key at all) -- confirmed empirically (see the F15-BpB-driver
    build session) that `compile_f15_stats.py`'s `load_cells` SKIPS these
    files (reports them in `skipped`, does not crash) rather than erroring.
    --out defaults to `eval_results/f15_bpb/`, a directory separate from
    `eval_results/f14_multiblimp/`, so the two families of per-cell JSON never
    mix in the same directory regardless of the schema-tolerance behavior.

--single-checkpoint escape hatch: some checkpoints (e.g. ad hoc smoke runs)
live in a directory whose name does not match TAG_RE and whose overlay dir is
not named after the target language (both required by the standard discovery
path). `--single-checkpoint DIR --single-arm ARM --single-lang LANG` bypasses
--cells-root discovery/sharding/family/model filtering AND the
overlay-dir-basename==lang guard, but keeps every other check (status gate,
`_assert_checkpoint_provenance`, tokenizer-file existence). The output tag is
built from the SAME `<model>_<lang>_<arm>[_reinitemb][_truetie]_s<seed>`
convention TAG_RE encodes, using model/reinit/truetie/seed read from the
checkpoint's own run_metadata.json (never guessed) unless overridden with
--single-tag.

Usage:
  python scripts/eval_f15_bpb.py --cells-root <dir> --out eval_results/f15_bpb \
      --family f15 --head-wiring training [--model qwen2_5-0_5b] \
      [--shard K/N] [--batch-size 8]

  python scripts/eval_f15_bpb.py --single-checkpoint /path/to/f15_a4_smoke \
      --single-arm vocab_extend --single-lang tur --head-wiring training \
      --out eval_results/f15_bpb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from las_experiment.bpb_core import compute_bpb, pick_tokenizer, tokenize_texts
from las_experiment.downstream_eval import (
    _assert_checkpoint_provenance,
    _build_eval_lm_kwargs,
)
from las_experiment.vocab_extend import ExtendedTokenizer

# ---------------------------------------------------------------------------
# Tag schema -- copied verbatim from scripts/eval_f14_multiblimp.py so the two
# drivers can never silently drift on which directory names are valid F15
# cells. (Deliberately copied, not imported: this script owns no dependency
# on eval_f14_multiblimp.py, which is a script, not a library module.)
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "qwen2_5-0_5b": "Qwen/Qwen2.5-0.5B",
    "llama3_2-1b": "meta-llama/Llama-3.2-1B",
}
LANG_SUFFIX = {
    "tur": "_Latn",
    "fin": "_Latn",
    "hun": "_Latn",
    "arb": "_Arab",
    "hin": "_Deva",
}
ENG_FLORES_CODE = "eng_Latn"
_MODEL_ALT = "|".join(re.escape(k) for k in MODEL_MAP)
_LANG_ALT = "|".join(re.escape(k) for k in LANG_SUFFIX)
TAG_RE = re.compile(
    rf"^(?P<model>{_MODEL_ALT})_(?P<lang>{_LANG_ALT})_"
    r"(?P<arm>original|langspec|placebo|base|vocab_extend)"
    r"(?P<reinit>_reinitemb)?(?P<truetie>_truetie)?_s(?P<seed>\d+)$"
)

DEFAULT_FLORES_ROOT = Path(
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data"
)
# Magic number: matches the F15 plan's BpB protocol ("Evaluation suite" item
# 5) and every other project BpB driver (eval_flores_bpb.py, eval_heldout_bpb.py).
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 8


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


def _verify_skip_provenance(out_path: Path, derived: Path) -> None:
    """Before trusting an existing `<tag>__bpb[__ckptN].json` as satisfying a
    cell (skip-if-exists), verify it actually belongs to THIS cell rather
    than being a stale file left over from a different run under the same
    tag -- the 2026-08-18 incident (`eval_results/f15_bpb/
    qwen2_5-0_5b_tur_langspec_truetie_s42__bpb.json` was a stale F14c-era
    file whose `checkpoint` pointed at `pr21_rerun/`, silently accepted as
    satisfying the f15_wave1 cell of the same tag; see EXPERIMENTS_RESULTS.md
    provenance caveat). Reads the file's `checkpoint` field and checks it is
    `derived` itself or a path under it -- the cell's own derived run
    directory (`<cell_dir>/ft*/`, or its `checkpoint-N/` subdir when
    --checkpoint-step is used). The exact checkpoint-N is NOT separately
    re-verified here: the output filename already encodes the step (the
    `__ckptN` suffix), so a file for the wrong step lives at a different
    path and would never reach this skip check -- the cell-dir prefix is the
    only check that can actually fire.

    Malformed/unreadable JSON or a missing `checkpoint` key is ALSO a hard
    failure, not a silent re-evaluate: an existing file whose provenance
    cannot be verified is exactly as untrustworthy as one proven wrong.
    """
    try:
        data = json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"{out_path}: could not read/parse existing output while "
            f"checking skip-if-exists provenance ({e!r}). Refusing to "
            f"silently treat this file as satisfying the cell -- fix or "
            f"remove it before re-running."
        ) from e
    recorded = data.get("checkpoint")
    if not recorded:
        raise RuntimeError(
            f"{out_path}: existing output has no 'checkpoint' field -- "
            f"cannot verify it belongs to this cell before skipping."
        )
    recorded_path = Path(recorded).resolve()
    derived_resolved = derived.resolve()
    if recorded_path != derived_resolved and derived_resolved not in recorded_path.parents:
        raise RuntimeError(
            f"{out_path}: recorded checkpoint {recorded!r} is not under this "
            f"cell's own derived directory {derived_resolved} -- looks like "
            f"a stale/mismatched output from a different run standing in "
            f"for this cell. Refusing to skip; investigate before deleting "
            f"or forcing re-evaluation."
        )


# ---------------------------------------------------------------------------
# Model loading -- copied from scripts/eval_f14_multiblimp.py (same rationale:
# bpb_core.load_eval_model does an unconditional lm_head re-share and has no
# vocab_extend resize-before-adapter-load step, so it cannot express the
# --head-wiring=training path or load A4 checkpoints at all).
# ---------------------------------------------------------------------------

def load_model_training_wiring(base_model_id: str, ckpt_path: str, device: str):
    """PEFT load WITHOUT the lm_head re-share: input lookup uses the trained
    modules_to_save copy (wrapper forward), output head stays the frozen
    original -- exactly the training-time wiring of modules_to_save-only
    checkpoints. tie_word_embeddings is still disabled on the configs to
    dodge HFLM's tie_weights() crash (PEFT issue #2777); with the re-share
    skipped this does NOT change any weight, it only prevents a re-tie from
    clobbering the wiring."""
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
    `run_metadata.json`'s `tokenizer.vocab_extend.n_total_vocab`. Deliberately
    the ONLY source used -- no fallback to a value reverse-engineered from the
    saved adapter tensor shape."""
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
    size first shape-mismatches against the saved adapter. head_wiring
    'training' vs 'reshared' kept for interface parity with
    eval_f14_multiblimp.py: F15 arms are true_tie throughout, so 'reshared'
    re-ties lm_head onto the SAME trained Parameter 'training' already uses
    -- for F15 cells the two wirings are numerically identical."""
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


# ---------------------------------------------------------------------------
# FLORES+ text loading
# ---------------------------------------------------------------------------

def load_flores_texts(flores_root: Path, lang_code: str) -> list[str]:
    """Read FLORES+ plaintext for one script-suffixed language code (e.g.
    `tur_Latn`, `eng_Latn`). Empty lines are dropped. Raises loudly if the
    file is missing -- no silent fallback."""
    path = flores_root / f"{lang_code}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES+ data for {lang_code!r} not found at {path}. "
            f"Check the language code and --flores-root."
        )
    with open(path, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]
    return texts


def verify_flores_files(flores_root: Path) -> None:
    """Fail fast (before any GPU work) if a needed FLORES+ file is missing
    for any of the F15 target languages or English."""
    missing = []
    for lang, suffix in LANG_SUFFIX.items():
        p = flores_root / f"{lang}{suffix}.txt"
        if not p.exists():
            missing.append(str(p))
    eng_p = flores_root / f"{ENG_FLORES_CODE}.txt"
    if not eng_p.exists():
        missing.append(str(eng_p))
    if missing:
        raise FileNotFoundError(
            f"Missing FLORES+ file(s) under {flores_root}: {missing}"
        )


def tokenize_texts_extended(texts, ext_tok: ExtendedTokenizer, max_seq_len: int = 1024):
    """Reimplementation of `bpb_core.tokenize_texts` for `ExtendedTokenizer`
    (vocab_extend arm). `bpb_core.tokenize_texts` calls `tokenizer.decode`
    directly on a bare `tokenizers.Tokenizer`, which mis-decodes appended
    added_tokens whose surface characters all sit in the GPT-2 byte<->char
    table (see `las_experiment/vocab_extend.py` DECODING CAVEAT, measured
    8.7%-ish of appended pieces on Qwen2.5-0.5B) -- silently corrupting the
    BpB byte denominator for these cells. `ExtendedTokenizer.decode` is the
    byte-exact fix; this loop is otherwise identical to `tokenize_texts`."""
    token_ids_list: list[list[int]] = []
    num_bytes_list: list[int] = []
    for text in texts:
        enc = ext_tok.encode(text, add_special_tokens=False)
        ids = enc.ids[:max_seq_len]
        decoded = ext_tok.decode(ids, skip_special_tokens=True)
        n_bytes = len(decoded.encode("utf-8"))
        if ids and n_bytes:
            token_ids_list.append(ids)
            num_bytes_list.append(n_bytes)
    return token_ids_list, num_bytes_list


# ---------------------------------------------------------------------------
# Tokenizer selection per arm -- mirrors eval_f14_multiblimp.py's inline arm
# branching (its tok_file construction, not bpb_core.pick_tokenizer's
# filename_map, which only knows "langspec" among the overlay arms and would
# load the WRONG file for placebo).
# ---------------------------------------------------------------------------

def _assert_overlay_matches_model(tok, tok_file: Path, model_tokenizer,
                                   n_probe: int = 1000) -> None:
    """Pre-launch review S12: this driver resolves langspec/placebo/base
    tokenizers directly (bpb_core.pick_tokenizer / Tokenizer.from_file),
    never through downstream_eval.BaseUnigramHFLM, so that class's __init__
    vocab-size safety check never runs here. Reproduce it, plus a
    piece-order check over the first `n_probe` ids: langspec/placebo/base
    overlays are supposed to be the model's own tokenizer with Unigram
    RESCORING only (same vocabulary, same id assignment, different
    merge/score data) -- any id in this range that decodes to a different
    surface piece than the model tokenizer means the overlay silently
    reordered the vocabulary, and every id fed to the model afterward would
    be scored against the wrong piece."""
    overlay_vocab = tok.get_vocab_size()
    model_vocab = len(model_tokenizer)
    if overlay_vocab != model_vocab:
        raise AssertionError(
            f"Overlay tokenizer {tok_file} vocab {overlay_vocab} != model "
            f"tokenizer vocab {model_vocab} -- wrong tokenizer for this "
            f"model. Refusing (no silent id misalignment)."
        )
    n_check = min(n_probe, overlay_vocab)
    for tid in range(n_check):
        overlay_piece = tok.id_to_token(tid)
        model_piece = model_tokenizer.convert_ids_to_tokens(tid)
        if overlay_piece != model_piece:
            raise AssertionError(
                f"Overlay tokenizer {tok_file} piece order diverges from the "
                f"model tokenizer at id={tid}: overlay={overlay_piece!r} "
                f"model={model_piece!r} (checked ids 0..{n_check - 1}). "
                f"Refusing to score with a reordered vocabulary."
            )


def resolve_target_tokenizer(arm: str, lang: str, las_dir: str | None,
                              model_tokenizer):
    """Returns (kind, tok_file_path_or_None, tok_obj, is_extended)."""
    if arm == "original":
        tok = pick_tokenizer("original", None, lang, model_tokenizer=model_tokenizer)
        return "original", None, tok, False
    if las_dir is None:
        raise ValueError(f"arm={arm!r} requires a las_dir (none in metadata).")
    suffix = LANG_SUFFIX[lang]
    las_dir = Path(las_dir)
    if arm == "langspec":
        tok = pick_tokenizer("langspec", las_dir, lang, lang_suffix=suffix)
        tok_file = las_dir / f"langspec_{lang}{suffix}.json"
        _assert_overlay_matches_model(tok, tok_file, model_tokenizer)
        return "langspec", tok_file, tok, False
    if arm == "placebo":
        tok_file = las_dir / f"placebo_{lang}{suffix}.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"placebo overlay {tok_file} missing.")
        tok = Tokenizer.from_file(str(tok_file))
        _assert_overlay_matches_model(tok, tok_file, model_tokenizer)
        return "placebo", tok_file, tok, False
    if arm == "base":
        tok = pick_tokenizer("base", las_dir, lang)
        tok_file = las_dir / "base_tokenizer.json"
        _assert_overlay_matches_model(tok, tok_file, model_tokenizer)
        return "base", tok_file, tok, False
    if arm == "vocab_extend":
        tok_file = las_dir / f"vocabext_{lang}{suffix}.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"extended-vocab tokenizer {tok_file} missing.")
        return "vocab_extend", tok_file, ExtendedTokenizer.from_file(tok_file), True
    raise ValueError(f"Unhandled arm {arm!r}")


def resolve_eng_tokenizer(arm: str, target_tok_file, target_tok_obj, target_is_ext: bool,
                           model_tokenizer):
    """English-routing rule mirrored from eval_f14_multiblimp.py's
    build_eng_retention_lm: finetune.py routes 'eng' through the model's own
    tokenizer for original/langspec/placebo/base (all four non-A4 las_modes);
    vocab_extend does not special-case eng, so eng was trained through the
    SAME extended-vocab tokenizer as the target language -- reuse it."""
    if arm == "vocab_extend":
        return "vocab_extend", target_tok_file, target_tok_obj, target_is_ext
    tok = pick_tokenizer("original", None, "eng", model_tokenizer=model_tokenizer)
    return "original", None, tok, False


# ---------------------------------------------------------------------------
# Cell discovery
# ---------------------------------------------------------------------------

def discover_cells(args) -> list[dict]:
    k, n = (int(x) for x in args.shard.split("/"))

    # Pass 1: family/model filter only (no shard split, no metadata reads) --
    # this is the population --expect-cells checks against, since shard is a
    # per-invocation runtime split of the SAME pool across parallel shards,
    # not a property of the pool itself.
    matched = []
    for cell_dir in sorted(args.cells_root.iterdir()):
        m = TAG_RE.match(cell_dir.name) if cell_dir.is_dir() else None
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
        matched.append((cell_dir, m))

    print(
        f"discovered {len(matched)} cells (family={args.family} "
        f"model={args.model}) under {args.cells_root}"
    )
    if not matched:
        raise RuntimeError(
            f"discovered 0 cells under --cells-root={args.cells_root} with "
            f"--family={args.family} --model={args.model} -- no cell "
            f"directory names matched TAG_RE and these filters. This has "
            f"previously been caused by a wrong --family default; check "
            f"--cells-root/--family/--model, or retry with --family all to "
            f"see the full population before narrowing."
        )
    if args.expect_cells is not None and len(matched) != args.expect_cells:
        raise RuntimeError(
            f"--expect-cells={args.expect_cells} but discovered {len(matched)} "
            f"cells (family={args.family} model={args.model}) under "
            f"{args.cells_root} -- investigate before evaluating a wrong or "
            f"partial cell set."
        )

    cells = []
    for cell_dir, m in matched:
        tag = cell_dir.name
        model_tag, lang, arm = m["model"], m["lang"], m["arm"]
        base_model_id = MODEL_MAP[model_tag]

        if int(hashlib.sha1(tag.encode()).hexdigest(), 16) % n != k:
            continue

        out_path = args.out / f"{tag}__bpb{args.checkpoint_step_suffix}.json"
        derived = find_derived(cell_dir)
        if out_path.exists() and not args.force:
            _verify_skip_provenance(out_path, derived)
            print(f"[skip] {tag} (exists: {out_path.name})")
            continue

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
            if arm != "base" and Path(las_dir).name != lang:
                raise ValueError(
                    f"{tag}: overlay dir {las_dir!r} does not end in {lang!r} — "
                    f"wrong-language overlay. Refusing."
                )

        cells.append(dict(
            tag=tag, lang=lang, arm=arm, model_tag=model_tag,
            base_model_id=base_model_id, reinit=bool(m["reinit"]),
            truetie=bool(m["truetie"]), seed=int(m["seed"]),
            derived=derived, ckpt_load_dir=ckpt_load_dir, meta=meta,
            las_dir=las_dir, out_path=out_path,
            single_checkpoint=False,
        ))
    return cells


def resolve_single_checkpoint(args) -> dict:
    cell_dir = args.single_checkpoint
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"--single-checkpoint {cell_dir} is not a directory.")
    derived = find_derived(cell_dir)
    if args.checkpoint_step is not None:
        ckpt_load_dir = derived / f"checkpoint-{args.checkpoint_step}"
        if not ckpt_load_dir.exists():
            raise FileNotFoundError(
                f"--single-checkpoint {cell_dir}: --checkpoint-step="
                f"{args.checkpoint_step} requested but {ckpt_load_dir} does "
                f"not exist."
            )
    else:
        ckpt_load_dir = derived
    meta = json.loads((derived / "run_metadata.json").read_text())
    if meta.get("status") != "completed":
        raise ValueError(
            f"--single-checkpoint {cell_dir}: status={meta.get('status')!r}, "
            f"not 'completed'. Refusing to evaluate an incomplete run."
        )

    arm = args.single_arm
    lang = args.single_lang
    t = meta.get("tokenizer", {})
    if t.get("las_mode") != arm:
        raise ValueError(
            f"--single-arm={arm!r} but run_metadata.json tokenizer.las_mode="
            f"{t.get('las_mode')!r} for {derived} — mismatched arm."
        )
    base_model_id = meta.get("model", {}).get("name")
    if base_model_id is None:
        raise ValueError(f"{derived}/run_metadata.json missing model.name")
    model_tag = next((k for k, v in MODEL_MAP.items() if v == base_model_id), None)
    if model_tag is None:
        raise ValueError(
            f"run_metadata.json model.name={base_model_id!r} not in MODEL_MAP "
            f"{MODEL_MAP} — unrecognized model for --single-checkpoint."
        )
    reinit = bool(meta.get("model", {}).get("reinit_embeddings", False))
    truetie = bool(meta.get("model", {}).get("true_tie", False))
    seed = meta.get("training", {}).get("seed")
    if seed is None:
        raise ValueError(f"{derived}/run_metadata.json missing training.seed")

    las_dir = t.get("las_tokenizer_path") if arm != "original" else None
    if arm != "original" and las_dir is None:
        raise ValueError(f"{derived}: no las_tokenizer_path in metadata for arm={arm!r}")

    # NOTE: unlike discover_cells, the overlay-dir-basename==lang guard is
    # deliberately NOT applied here — this escape hatch exists precisely for
    # checkpoints whose overlay dir is not named after the target language
    # (e.g. ad hoc smoke-test tokenizer dirs). Every other check
    # (status/las_mode/provenance/tokenizer-file-existence) still applies.

    default_tag = f"{model_tag}_{lang}_{arm}"
    if reinit:
        default_tag += "_reinitemb"
    if truetie:
        default_tag += "_truetie"
    default_tag += f"_s{seed}"
    tag = args.single_tag or default_tag
    out_path = args.out / f"{tag}__bpb{args.checkpoint_step_suffix}.json"

    return dict(
        tag=tag, lang=lang, arm=arm, model_tag=model_tag,
        base_model_id=base_model_id, reinit=reinit, truetie=truetie,
        seed=int(seed), derived=derived, ckpt_load_dir=ckpt_load_dir, meta=meta,
        las_dir=las_dir, out_path=out_path, single_checkpoint=True,
    )


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------

def process_cell(cell: dict, args, gstate: dict) -> None:
    tag = cell["tag"]
    out_path = cell["out_path"]
    if out_path.exists() and not args.force:
        _verify_skip_provenance(out_path, cell["derived"])
        print(f"[skip] {tag} (exists: {out_path.name})")
        return

    base_model_id = cell["base_model_id"]
    arm = cell["arm"]
    lang = cell["lang"]
    derived = cell["derived"]
    ckpt_load_dir = cell["ckpt_load_dir"]
    meta = cell["meta"]
    las_dir = cell["las_dir"]

    _assert_checkpoint_provenance(str(ckpt_load_dir), base_model_id, arm, las_dir)

    print(f"[eval] {tag} ({args.head_wiring})")
    if arm == "vocab_extend":
        model = load_model_vocab_extend(
            base_model_id, str(ckpt_load_dir), args.device, args.head_wiring, meta
        )
    elif args.head_wiring == "reshared":
        kw = _build_eval_lm_kwargs(
            base_pretrained=base_model_id, ckpt_path=str(ckpt_load_dir), device=args.device,
        )
        model = kw["pretrained"]
    else:
        model = load_model_training_wiring(base_model_id, str(ckpt_load_dir), args.device)

    model_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    pad_id = model_tokenizer.pad_token_id or 0

    target_kind, target_tok_file, target_tok_obj, target_is_ext = resolve_target_tokenizer(
        arm, lang, las_dir, model_tokenizer
    )
    eng_kind, eng_tok_file, eng_tok_obj, eng_is_ext = resolve_eng_tokenizer(
        arm, target_tok_file, target_tok_obj, target_is_ext, model_tokenizer
    )

    target_flores_code = f"{lang}{LANG_SUFFIX[lang]}"
    target_texts = load_flores_texts(args.flores_root, target_flores_code)
    eng_texts = load_flores_texts(args.flores_root, ENG_FLORES_CODE)

    def _tok_and_bpb(texts, tok_obj, is_ext):
        if is_ext:
            ids_list, bytes_list = tokenize_texts_extended(texts, tok_obj, args.max_seq_len)
        else:
            ids_list, bytes_list = tokenize_texts(texts, tok_obj, args.max_seq_len)
        res = compute_bpb(
            model, ids_list, bytes_list, args.device,
            batch_size=args.batch_size, max_seq_len=args.max_seq_len, pad_id=pad_id,
        )
        res["n_sentences"] = len(texts)
        res["n_kept"] = len(ids_list)
        return res

    target_res = _tok_and_bpb(target_texts, target_tok_obj, target_is_ext)
    eng_res = _tok_and_bpb(eng_texts, eng_tok_obj, eng_is_ext)

    def _tok_provenance(kind, tok_file):
        if tok_file is None:
            return {
                "kind": kind, "path": None, "sha256": None,
                "note": f"model's own HF tokenizer ({base_model_id})",
            }
        return {"kind": kind, "path": str(tok_file), "sha256": sha256(tok_file)}

    record = {
        "tag": tag,
        "lang": lang,
        "arm": arm,
        "model": base_model_id,
        "reinit_embeddings": cell["reinit"],
        "truetie": cell["truetie"],
        "seed": cell["seed"],
        "head_wiring": args.head_wiring,
        "flores_bpb_target": target_res["bpb"],
        "flores_bpb_eng": eng_res["bpb"],
        "target_flores_code": target_flores_code,
        "eng_flores_code": ENG_FLORES_CODE,
        "n_tokens": {"target": target_res["n_tokens"], "eng": eng_res["n_tokens"]},
        "n_bytes": {"target": target_res["n_bytes"], "eng": eng_res["n_bytes"]},
        "n_sentences": {"target": target_res["n_sentences"], "eng": eng_res["n_sentences"]},
        "n_kept": {"target": target_res["n_kept"], "eng": eng_res["n_kept"]},
        "nll": {"target": target_res["nll"], "eng": eng_res["nll"]},
        "tokenizer_target": _tok_provenance(target_kind, target_tok_file),
        "tokenizer_eng": _tok_provenance(eng_kind, eng_tok_file),
        "checkpoint": str(ckpt_load_dir),
        "train_git_commit": meta.get("git_commit"),
        "eval_git": gstate,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "flores_root": str(args.flores_root),
        "single_checkpoint_mode": cell["single_checkpoint"],
        "timestamp": datetime.now().isoformat(),
    }
    if args.checkpoint_step is not None:
        record["checkpoint_step"] = args.checkpoint_step
    out_path.write_text(json.dumps(record, indent=2))
    print(
        f"[done] {tag}: flores_bpb_target={target_res['bpb']:.4f} "
        f"flores_bpb_eng={eng_res['bpb']:.4f}"
    )

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, default=None,
                    help="required unless --single-checkpoint is given")
    ap.add_argument("--out", type=Path, default=Path("eval_results/f15_bpb"),
                    help="separate from eval_results/f14_multiblimp/ -- see "
                         "module docstring for why (compile_f15_stats.py "
                         "schema tolerance).")
    ap.add_argument(
        "--family", choices=["f14", "f14b", "f15", "all"], default="f14",
        help="same semantics as eval_f14_multiblimp.py --family -- including "
             "that it assumes ONE --cells-root mixing F14/F14b/F15 cells. F15 "
             "wave-1 cells (scripts/launch_f15_wave1.sh) are true_tie "
             "throughout and live under their OWN --cells-root, so use "
             "--family all --cells-root <f15-wave1-root> for those rather "
             "than this arm-dependent split.",
    )
    ap.add_argument("--model", choices=[*MODEL_MAP.keys(), "all"], default="all")
    ap.add_argument(
        "--expect-cells", type=int, default=None,
        help="abort if the post family/model-filter cell count under "
             "--cells-root differs from this (pre-shard). The wrapper "
             "(run_f14_eval_login.sh) passes this when the expected count "
             "is known; omit to skip the check.",
    )
    ap.add_argument("--shard", default="0/1", help="K/N by sha1(tag) mod N")
    ap.add_argument(
        "--head-wiring", choices=["training", "reshared"], required=True,
        help="'training' = frozen-original output head; 'reshared' = tied "
             "trained matrix re-share. F15 arms are true_tie throughout, so "
             "for F15 cells both wirings are numerically identical.",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    ap.add_argument("--flores-root", type=Path, default=DEFAULT_FLORES_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--single-checkpoint", type=Path, default=None,
        help="Escape hatch for a checkpoint whose dir name / overlay-dir "
             "naming does not match the standard cell layout (see module "
             "docstring). Mutually exclusive with --cells-root.",
    )
    ap.add_argument("--single-arm", choices=["original", "langspec", "placebo",
                                              "base", "vocab_extend"], default=None)
    ap.add_argument("--single-lang", choices=list(LANG_SUFFIX.keys()), default=None)
    ap.add_argument("--single-tag", default=None,
                    help="override the auto-built output tag for --single-checkpoint")
    ap.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="Evaluate the intermediate checkpoint-N/ subdir inside the "
             "derived run dir (e.g. 357/1781/3561 for F15 wave1/wave2 "
             "cells) instead of the final trained state. Default: absent, "
             "current final-checkpoint behavior unchanged (byte-identical "
             "output). Hard-fails naming the missing path if checkpoint-N/ "
             "does not exist under the cell's derived dir. Output filenames "
             "get a __ckptN suffix so they never collide with the "
             "final-checkpoint files.",
    )
    args = ap.parse_args()
    args.checkpoint_step_suffix = (
        f"__ckpt{args.checkpoint_step}" if args.checkpoint_step is not None else ""
    )

    if args.single_checkpoint is not None:
        if args.cells_root is not None:
            raise ValueError("--single-checkpoint and --cells-root are mutually exclusive.")
        if args.single_arm is None or args.single_lang is None:
            raise ValueError("--single-checkpoint requires --single-arm and --single-lang.")
    elif args.cells_root is None:
        raise ValueError("--cells-root is required unless --single-checkpoint is given.")

    verify_flores_files(args.flores_root)

    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    gstate = git_state(repo)

    if args.single_checkpoint is not None:
        cells = [resolve_single_checkpoint(args)]
    else:
        cells = discover_cells(args)

    if not cells:
        print("No cells to evaluate.")
        return

    for cell in cells:
        process_cell(cell, args, gstate)


if __name__ == "__main__":
    main()
