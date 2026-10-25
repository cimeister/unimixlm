#!/usr/bin/env python3
"""Prepare tokenizers/overlays for the F15 campaign.

Deliverables, all under ``las_tokenizers/f15/<family>/`` where family is one
of ``qwen2_5-0_5b`` (Qwen/Qwen2.5-0.5B) or ``llama3.2-1b``
(meta-llama/Llama-3.2-1B):

  1. LangMAP overlays ``<lang>/langspec_<lang>_<Script>.json``
     (qwen x {arb, hin}; llama x {tur, arb, hin}). Fit via the existing LAS
     pipeline (``las_experiment/tokenizer_training/training.py``
     ``train_base_tokenizer`` + ``code.language_specific_unigramlm
     .LanguageSpecificUnigramLMTokenizer`` per-language re-estimation).
  2. Base-Unigram tokenizer ``base_tokenizer.json`` per family: language-
     agnostic rescoring of the model vocab, fit on the combined 5-language
     (tur/fin/hun/arb/hin) corpus. Doubles as the A5-arm tokenizer and the
     shared seed for every langspec overlay in (1) -- both must descend from
     the identical base vocab for the A2-vs-A5 comparison to be meaningful.
  3. Placebos ``<lang>/placebo_<lang>_<Script>.json`` (qwen x {arb, hin})
     via ``scripts/make_placebo.py --mode bucketed --bucket-size 500``.
  4. Monolingual 32k BPE fits ``<lang>/mono32k_<lang>_<Script>.json``
     (qwen x all 5 langs) via ``train_bpe_tokenizer`` -- fit-only, raw
     material for the separate A4 vocab-extension builder (not built here).

Corpora: 50k lines/language, deterministically the first N non-empty lines
(after the project's standard script filter -- see NOTE below) read in
file order from ``MultilingualUnigramLM/FineWeb2-{lang}-100M`` on the HF
Hub, pinned to the exact revision resolved at prep time. Corpus text is
family-independent (the model-specific pre-tokenizer is applied at
*training* time inside the LAS pipeline, not at extraction time) so it is
built once and shared across both families.

NOTE on script filtering: the FineWeb2-<lang>-100M repos are already
single-language splits, but this script still applies
``filter_line_by_script`` (the same quality gate ``prepare_language_corpora``
applies) to reject stray non-target-script lines while walking the corpus
in file order to collect the first 50k lines that pass. Devanagari
(hin_Deva) had no entry in ``SCRIPT_RANGES`` in
``las_experiment/tokenizer_training/training.py`` -- added per this task's
explicit instruction (ranges 0x0900-0x097F + 0xA8E0-0xA8FF), noted in the
manifest.

IMPORTANT DEVIATION FROM THE LIBRARY DEFAULT, REQUIRED BY THE F15 PROTOCOL:
``code/language_specific_unigramlm.py`` seeds
the per-language SentencePiece EM UNIFORMLY. The F15 protocol requires seeding
with the shared base tokenizer's already-trained log-scores instead --
the behavior of the earlier code-tokenizer pipeline that produced the
paper's tokenizers -- and treats this as a critical setting. This
script's ownership is scoped to itself and ``las_tokenizers/f15/`` only
(not ``code/language_specific_unigramlm.py``), so the change is made via a
local subclass, ``BaseScoreSeededLASTokenizer`` below, which overrides only
the seed-vocab-construction step of ``train_with_sentencepiece_direct``
(one line changed; everything else copied verbatim from the parent). See
that class's docstring for the exact diff. This is a significant,
consequential choice.

Do NOT commit any artifact under las_tokenizers/f15/. No HF Hub push.
"""

import os

# MUST run before any `tokenizers`/`transformers` import: the HF `tokenizers`
# Rust extension's in-process BPE trainer (train_bpe_tokenizer, deliverable 4)
# uses a global Rayon thread pool sized to the detected CPU count (288 on
# this login node) unless capped here first -- Rayon reads these env vars
# only once, at first use, so setting them later (even before the specific
# call) is too late once any other tokenizers/Rust call has already
# initialized the pool. Empirically: two full runs of this script were
# SIGKILLed (exit 137) at exactly the mono32k BPE-training step while the
# SentencePiece EM steps (external `spm_train` binary, explicitly capped at
# --num_threads=20) completed fine -- consistent with an unbounded
# per-thread-allocation blowup in the in-process trainer specifically.
os.environ.setdefault("RAYON_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import logging
import multiprocessing
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.constants import (  # noqa: E402
    NUM_FINETUNE_EM_ITERATIONS,
    SP_DEFAULT_ARGS,
    SPECIAL_TOKENS,
)
from code.language_specific_unigramlm import (  # noqa: E402
    LanguageSpecificUnigramLMTokenizer,
)
from code.tokenizer_formatting_utils import (  # noqa: E402
    _load_vocab_from_hf_tokenizer,
    _write_sp_seed_vocab_file,
    write_hf_bytelevel_corpus,
)
from las_experiment.tokenizer_training.training import (  # noqa: E402
    filter_line_by_script,
    train_base_tokenizer,
    train_bpe_tokenizer,
)
from las_experiment.tokenizer_training.validation import validate_tokenizers  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prep_f15_tokenizers")

# ---------------------------------------------------------------------------
# Constants (named per project convention)
# ---------------------------------------------------------------------------
LINES_PER_LANG = 50_000          # 50k lines per language
PLACEBO_SEED = 1234              # matches scripts/prep_qwen25_las_rerun_tokenizers.py
PLACEBO_MODE = "bucketed"        # bucketed placebo control
PLACEBO_BUCKET_SIZE = 500        # --bucket-size 500
MONO_BPE_VOCAB_SIZE = 32_000     # A4 builder input
ENCODE_PROBE_LINES = 5           # encode probe on 5 real lines

LANG_CODES = {
    "tur": "tur_Latn",
    "fin": "fin_Latn",
    "hun": "hun_Latn",
    "arb": "arb_Arab",
    "hin": "hin_Deva",
}
BASE_LANGS = ["tur", "fin", "hun", "arb", "hin"]  # deliverable 2: combined 5-lang corpus
HUB_REPO_TMPL = "MultilingualUnigramLM/FineWeb2-{lang_code}-100M"

FAMILIES = {
    "qwen2_5-0_5b": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        # tur/fin/hun overlays are fit from THIS family's F15
        # base_tokenizer.json (overlays from the earlier code-tokenizer
        # pipeline come from a different pipeline, which would make the
        # A5-vs-A2 control unclean for those three languages). No placebos
        # for tur/fin/hun, see placebo_langs below.
        "overlay_langs": ["arb", "hin", "tur", "fin", "hun"],
        "placebo_langs": ["arb", "hin"],
        "mono_langs": ["tur", "fin", "hun", "arb", "hin"],
    },
    "llama3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B",
        # fin/hun are included so all 5 langs
        # have a langspec overlay, as the 6 Llama joint-multilingual
        # cells need (scripts/launch_f15_wave2.sh INCLUDE_LLAMA_JOINT). No
        # placebos/mono32k for llama.
        "overlay_langs": ["tur", "arb", "hin", "fin", "hun"],
        "placebo_langs": [],
        "mono_langs": [],
    },
}

OUT_ROOT = REPO_ROOT / "las_tokenizers" / "f15"
_LAS_TOK_CORPORA_ROOT = os.environ.get("LAS_TOK_CORPORA_ROOT")
if not _LAS_TOK_CORPORA_ROOT:
    raise RuntimeError(
        "LAS_TOK_CORPORA_ROOT is not set. Point it at the scratch directory "
        "holding the LAS tokenizer-training corpora (parent of f15_corpora/)."
    )
CORPORA_ROOT = Path(_LAS_TOK_CORPORA_ROOT) / "f15_corpora"
MAKE_PLACEBO = REPO_ROOT / "scripts" / "make_placebo.py"

assert NUM_FINETUNE_EM_ITERATIONS == 10, (
    f"code/constants.py NUM_FINETUNE_EM_ITERATIONS={NUM_FINETUNE_EM_ITERATIONS}, "
    f"expected 10 (the F15 protocol value). Aborting rather than "
    f"silently training with a different EM sub-iteration count."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Base-log-score-seeded per-language tokenizer trainer
# ---------------------------------------------------------------------------
class BaseScoreSeededLASTokenizer(LanguageSpecificUnigramLMTokenizer):
    """Per-language LAS tokenizer trainer, seeded with BASE log-scores.

    ``code/language_specific_unigramlm.py`` on this branch (log_space) seeds
    the per-language SentencePiece EM uniformly: inside
    ``train_with_sentencepiece_direct``, the seed vocab file written for
    ``spm_train`` assigns every piece the same
    ``initial_log_score = log(1/num_pieces)``, discarding the actual
    ``vocab_dict`` (the base tokenizer's already-trained log-scores) that is
    passed into the method and used only for special-token scores and the
    post-hoc missing-token fallback.

    The earlier code-tokenizer pipeline, which produced the
    paper's tokenizers, seeds the SAME seed-vocab file with
    ``vocab_dict.get(t, initial_log_score)`` instead -- i.e. each piece's EM
    starting point is the base tokenizer's own trained log-score, falling
    back to the uniform value only for pieces the base tokenizer has no
    score for (there should be none, since every piece here comes from
    ``_load_vocab_from_hf_tokenizer(self.base_tokenizer)``).

    The F15 protocol requires that behavior (seed the
    per-language EM with base log-scores). This script's ownership is
    scoped to itself and ``las_tokenizers/f15/`` only, so rather than edit
    the shared library file, this subclass overrides
    ``train_with_sentencepiece_direct`` with the SP->HF handoff logic
    copied verbatim from the parent -- the only change is the seed-vocab
    list comprehension (see "F15 override" comment below).
    ``_write_sp_seed_vocab_file`` treats the incoming (piece, score) pairs
    as opaque, so this substitution is safe.
    """

    def train_with_sentencepiece_direct(
        self, corpus_file: str, vocab_dict: dict, num_iterations: int = 10
    ) -> dict:
        import shutil

        if not shutil.which("spm_train"):
            raise RuntimeError(
                "spm_train executable not found. Make sure sentencepiece is "
                "installed and in your PATH."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "model")
            vocab_path = os.path.join(tmpdir, "vocab.tsv")

            base_vocab = _load_vocab_from_hf_tokenizer(self.base_tokenizer)
            excluded = set(self.special_tokens) | {self.unk_token}
            num_pieces = sum(1 for t in base_vocab if t not in excluded)
            initial_log_score = np.log(1.0 / num_pieces) if num_pieces > 0 else -10.0

            # --- F15 override (see class docstring): seed each piece with
            # the base tokenizer's own trained log-score instead of a
            # constant uniform value. ---
            token_count = _write_sp_seed_vocab_file(
                [(t, vocab_dict.get(t, initial_log_score)) for t in base_vocab],
                vocab_path,
                self.unk_token,
            )

            hf_processed_text_path = os.path.join(tmpdir, "hf_text.txt")
            normalizer = self.base_tokenizer.normalizer if self.base_tokenizer else None
            write_hf_bytelevel_corpus(
                corpus_file, hf_processed_text_path, self._get_pretokenizer(),
                normalizer=normalizer,
            )

            expected_vocab_size = token_count + 3

            command = [
                "spm_train",
                f"--input={hf_processed_text_path}",
                f"--model_prefix={out_prefix}",
                f"--seed_sentencepieces_file={vocab_path}",
                f"--vocab_size={expected_vocab_size}",
                f"--num_sub_iterations={max(1, min(int(num_iterations), 20))}",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count() - 1)}",
            ] + SP_DEFAULT_ARGS

            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            output = subprocess.run(
                command, check=False, capture_output=True, text=True, encoding="utf-8"
            )
            if output.returncode != 0:
                raise RuntimeError(
                    f"spm_train failed with exit code {output.returncode}.\n"
                    f"Command: {' '.join(command)}\n"
                    f"--- stderr ---\n{output.stderr}\n"
                    f"--- stdout ---\n{output.stdout}"
                )
            logger.info("SentencePiece training successful.")

            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            token_logps = {}
            additional_tokens = set()
            for i in range(sp.get_piece_size()):
                p = sp.id_to_piece(i)
                p_hf = p
                if p_hf not in vocab_dict:
                    additional_tokens.add(p_hf)
                if p_hf in set(SPECIAL_TOKENS.values()) | {self.unk_token}:
                    try:
                        logp = float(vocab_dict[p_hf])
                    except KeyError:
                        logger.warning(
                            f"Special token {p_hf} not found in vocab_dict; skipping"
                        )
                        continue
                else:
                    logp = float(sp.get_score(i))
                token_logps[p_hf] = logp
            if additional_tokens:
                logger.warning(f"unknown tokens: {additional_tokens}")

            missing_tokens = set()
            for t in vocab_dict.keys():
                if t not in token_logps:
                    missing_tokens.add(t)
                    token_logps[t] = float(vocab_dict[t])
            if missing_tokens:
                logger.warning(
                    f"Tokens {missing_tokens} not in sentencepiece model; defaulting "
                    f"to base tokenizer log-prob {vocab_dict[t]:.5f}"
                )

            sp_meta = set(SPECIAL_TOKENS.values()) | {self.unk_token}
            truly_unknown = additional_tokens - sp_meta
            n_truly_unknown = len(truly_unknown)
            n_sp_pieces = sp.get_piece_size()
            n_base_vocab = len(vocab_dict)
            n_fallback = len(missing_tokens)
            pct_unknown = n_truly_unknown / max(n_sp_pieces, 1)
            pct_fallback = n_fallback / max(n_base_vocab, 1)
            logger.info(
                f"SP->HF handoff: sp_pieces={n_sp_pieces} base_vocab={n_base_vocab} "
                f"matched={n_base_vocab - n_fallback} fallback={n_fallback} "
                f"unknown_from_sp={n_truly_unknown} "
                f"(pct_unknown={pct_unknown:.4f}, pct_fallback={pct_fallback:.4f})"
            )
            if pct_unknown > 0.01:
                raise RuntimeError(
                    f"SP produced {pct_unknown:.1%} pieces outside HF base vocab "
                    f"({n_truly_unknown}/{n_sp_pieces}). Likely cause: corpus "
                    f"format mismatch between write_hf_bytelevel_corpus and what "
                    f"spm_train read. First 20 unknown: {list(truly_unknown)[:20]}"
                )
            if pct_fallback > 0.50:
                raise RuntimeError(
                    f"{pct_fallback:.1%} of base vocab fell back to base score "
                    f"({n_fallback}/{n_base_vocab}). SP training didn't see enough "
                    f"of the language, or the SP<->HF piece format diverged."
                )

            token_logps = self._log_normalize(token_logps)
            return token_logps


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------
def build_corpus(short_lang: str, api: HfApi) -> dict:
    """Deterministically extract the first LINES_PER_LANG non-empty,
    script-filtered lines (file order) from the F15 FineWeb2-100M hub repo
    for `short_lang`, pinned to the resolved revision. Idempotent: reuses
    an existing corpus file if one is already present with the same sha256
    recorded (verified by re-hash), else rebuilds.
    """
    lang_code = LANG_CODES[short_lang]
    repo = HUB_REPO_TMPL.format(lang_code=lang_code)
    info = api.dataset_info(repo)
    revision = info.sha

    CORPORA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CORPORA_ROOT / f"{lang_code}.txt"
    prov_path = CORPORA_ROOT / f"{lang_code}.provenance.json"

    if out_path.exists() and prov_path.exists():
        prov = json.loads(prov_path.read_text())
        if (
            prov.get("repo") == repo
            and prov.get("revision") == revision
            and prov.get("lines_written") == LINES_PER_LANG
            and prov.get("sha256") == sha256(out_path)
        ):
            logger.info(f"[{lang_code}] Reusing existing corpus at {out_path} (verified)")
            return prov
        logger.warning(
            f"[{lang_code}] Existing corpus/provenance stale or mismatched "
            f"(repo/revision/hash changed) -- rebuilding."
        )

    from datasets import load_dataset

    logger.info(f"[{lang_code}] Loading {repo}@{revision} ...")
    ds = load_dataset(repo, split="train", revision=revision)

    kept = []
    scanned = 0
    filtered_empty = 0
    filtered_script = 0
    for ex in ds:
        scanned += 1
        text = (ex.get("text") or "").strip()
        if not text:
            filtered_empty += 1
            continue
        if not filter_line_by_script(text, lang_code):
            filtered_script += 1
            continue
        kept.append(text)
        if len(kept) >= LINES_PER_LANG:
            break

    if len(kept) < LINES_PER_LANG:
        raise RuntimeError(
            f"[{lang_code}] Only collected {len(kept)}/{LINES_PER_LANG} lines "
            f"from {repo}@{revision} after scanning {scanned} rows "
            f"(filtered_empty={filtered_empty}, filtered_script={filtered_script}). "
            f"Corpus is too small or the script filter is rejecting too much -- "
            f"aborting rather than shipping a short corpus."
        )

    out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    prov = {
        "repo": repo,
        "revision": revision,
        "lines_written": len(kept),
        "lines_scanned": scanned,
        "lines_filtered_empty": filtered_empty,
        "lines_filtered_script": filtered_script,
        "sha256": sha256(out_path),
        "path": str(out_path),
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    logger.info(
        f"[{lang_code}] Wrote {len(kept)} lines to {out_path} "
        f"(scanned={scanned}, filtered_script={filtered_script})"
    )
    return prov


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def encode_probe(tok: Tokenizer, corpus_path: Path, embed_rows: Optional[int]) -> int:
    lines = []
    with open(corpus_path, "r", encoding="utf-8", errors="strict") as f:
        for line in f:
            if len(lines) >= ENCODE_PROBE_LINES:
                break
            line = line.rstrip("\n")
            if line:
                lines.append(line)
    if not lines:
        raise RuntimeError(f"No probe lines available from {corpus_path}")
    for line in lines:
        enc = tok.encode(line, add_special_tokens=False)
        if not enc.ids:
            raise RuntimeError(
                f"Empty encode for probe line from {corpus_path}: {line[:60]!r}"
            )
        if embed_rows is not None and max(enc.ids) >= embed_rows:
            raise RuntimeError(
                f"Probe encode out-of-range id (max={max(enc.ids)} >= "
                f"embed_rows={embed_rows}) for {corpus_path}"
            )
    return len(lines)


def verify_and_patch_specials(
    dest_path: Path,
    model_tok,
    model_tok_json: dict,
    embed_rows: int,
) -> dict:
    """Verify eos/pad resolve on the tokenizer at dest_path; patch in any
    missing added_tokens from the model's own tokenizer.json (mirrors
    scripts/prep_qwen25_las_rerun_tokenizers.py). Since dest_path is built
    from the SAME model's own tokenizer (not a cross-model transplant like
    prep_qwen25's code-tokenizer rerun), no patch is expected in practice -- this is
    a verification gate, and the patch path exists as a safety net.
    """
    tok = Tokenizer.from_file(str(dest_path))
    patched = 0
    if tok.token_to_id(model_tok.eos_token) is None:
        ls = json.loads(dest_path.read_text())
        vlen = len(ls["model"]["vocab"])
        have_id = {a["id"] for a in ls.get("added_tokens", []) or []}
        have_c = {a["content"] for a in ls.get("added_tokens", []) or []}
        add = [
            a for a in (model_tok_json.get("added_tokens", []) or [])
            if a["id"] not in have_id and a["content"] not in have_c
        ]
        for a in add:
            assert a["id"] >= vlen, (
                f"{dest_path}: special id {a['id']} collides with vocab (len {vlen})"
            )
        ls.setdefault("added_tokens", []).extend(add)
        dest_path.write_text(json.dumps(ls, ensure_ascii=False))
        patched = len(add)
        tok = Tokenizer.from_file(str(dest_path))
        logger.warning(f"{dest_path}: patched +{patched} added_tokens for eos resolution")

    eos_id = tok.token_to_id(model_tok.eos_token)
    pad_id = tok.token_to_id(model_tok.pad_token)
    if eos_id is None or pad_id is None:
        raise RuntimeError(
            f"{dest_path}: eos/pad do not resolve after patch "
            f"(eos_id={eos_id}, pad_id={pad_id})"
        )
    vocab_size_total = tok.get_vocab_size()
    if vocab_size_total > embed_rows:
        raise RuntimeError(
            f"{dest_path}: tokenizer can emit ids up to {vocab_size_total - 1} "
            f">= model embedding rows {embed_rows}"
        )
    return {
        "eos_id": eos_id,
        "pad_id": pad_id,
        "vocab_size_total": vocab_size_total,
        "patched_added_tokens": patched,
    }


# ---------------------------------------------------------------------------
# Per-family pipeline
# ---------------------------------------------------------------------------
def process_family(family_name: str, cfg: dict, corpora: dict, git_commit: str) -> dict:
    model_name = cfg["model_name"]
    family_dir = OUT_ROOT / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Family {family_name} ({model_name}) ===")
    model_tok = AutoTokenizer.from_pretrained(model_name)
    model_cfg = AutoConfig.from_pretrained(model_name)
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token
    embed_rows = int(model_cfg.vocab_size)

    seed_tok_dir = family_dir / "model_tokenizer"
    seed_tok_dir.mkdir(parents=True, exist_ok=True)
    model_tok.save_pretrained(seed_tok_dir)
    model_tokenizer_path = seed_tok_dir / "tokenizer.json"
    model_tok_json = json.loads(model_tokenizer_path.read_text())

    unk_token = model_tok.unk_token if model_tok.unk_token else model_tok.eos_token
    logger.info(
        f"[{family_name}] eos={model_tok.eos_token!r} pad={model_tok.pad_token!r} "
        f"unk_token(internal)={unk_token!r} embed_rows={embed_rows}"
    )

    manifest = {
        "script": "scripts/prep_f15_tokenizers.py",
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "family": family_name,
        "model_name": model_name,
        "embed_rows": embed_rows,
        "eos_token": model_tok.eos_token,
        "pad_token": model_tok.pad_token,
        "unk_token_internal": unk_token,
        "seeding": {
            "per_lang_em_seed": "base_log_scores",
            "note": (
                "Deviates from this branch's (log_space) uniform-seed default "
                "via a local subclass in this script; "
                "code/language_specific_unigramlm.py was not modified. See "
                "BaseScoreSeededLASTokenizer docstring."
            ),
            "num_finetune_em_iterations": NUM_FINETUNE_EM_ITERATIONS,
        },
        "corpora": corpora,
        "overlays": {},
        "placebos": {},
        "mono32k": {},
    }

    # ---- Deliverable 2: base-Unigram tokenizer (combined 5-lang corpus) ----
    base_tokenizer_path = family_dir / "base_tokenizer.json"
    base_corpus_info = {LANG_CODES[l]: corpora[LANG_CODES[l]]["path"] for l in BASE_LANGS}
    if base_tokenizer_path.exists():
        logger.info(f"[{family_name}] base_tokenizer.json already exists, reusing")
    else:
        train_base_tokenizer(
            corpus_info=base_corpus_info,
            model_tokenizer_path=str(model_tokenizer_path),
            base_output_path=str(base_tokenizer_path),
            em_mode="soft",
            unk_token=unk_token,
        )
    validate_tokenizers(model_tokenizer_path, base_tokenizer_path)
    base_specials = verify_and_patch_specials(
        base_tokenizer_path, model_tok, model_tok_json, embed_rows
    )
    base_tok = Tokenizer.from_file(str(base_tokenizer_path))
    for l in BASE_LANGS:
        encode_probe(base_tok, Path(corpora[LANG_CODES[l]]["path"]), embed_rows)
    manifest["base_tokenizer"] = {
        "path": str(base_tokenizer_path),
        "sha256": sha256(base_tokenizer_path),
        "vocab_size_trainable": len(_load_vocab_from_hf_tokenizer(base_tok)),
        **base_specials,
        "probe_langs": BASE_LANGS,
    }
    logger.info(f"[{family_name}] base_tokenizer OK: {manifest['base_tokenizer']}")

    # ---- Deliverable 1: LangMAP overlays ----
    overlay_lang_paths = {}
    for l in cfg["overlay_langs"]:
        lang_code = LANG_CODES[l]
        lang_dir = family_dir / l
        lang_dir.mkdir(parents=True, exist_ok=True)
        langspec_path = lang_dir / f"langspec_{lang_code}.json"
        if not langspec_path.exists():
            trainer = BaseScoreSeededLASTokenizer(
                reestimation_em_mode="soft",
                unk_token=unk_token,
            )
            assert trainer.num_reestimation_iterations == NUM_FINETUNE_EM_ITERATIONS
            trainer.train(
                corpus_info={lang_code: corpora[lang_code]["path"]},
                base_tokenizer_path=str(base_tokenizer_path),
                tokenizer_path_format=str(lang_dir / "langspec_{lang_code}.json"),
                load_base_if_exists=True,
                load_tokenizers_if_exists=False,
                use_sentencepiece=True,
                max_workers=1,
            )
        overlay_lang_paths[lang_code] = langspec_path

    if overlay_lang_paths:
        validate_tokenizers(model_tokenizer_path, base_tokenizer_path, overlay_lang_paths)

    for l in cfg["overlay_langs"]:
        lang_code = LANG_CODES[l]
        lang_dir = family_dir / l  # re-derive: do NOT reuse loop-1's stale value
        langspec_path = overlay_lang_paths[lang_code]
        specials = verify_and_patch_specials(
            langspec_path, model_tok, model_tok_json, embed_rows
        )
        tok = Tokenizer.from_file(str(langspec_path))
        encode_probe(tok, Path(corpora[lang_code]["path"]), embed_rows)
        manifest["overlays"][lang_code] = {
            "path": str(langspec_path),
            "sha256": sha256(langspec_path),
            **specials,
        }
        logger.info(f"[{family_name}] overlay {lang_code} OK: {manifest['overlays'][lang_code]}")

        # ---- Deliverable 3: placebo (qwen x {arb, hin} only) ----
        if l in cfg["placebo_langs"]:
            placebo_path = lang_dir / f"placebo_{lang_code}.json"
            if not placebo_path.exists():
                subprocess.run(
                    [
                        sys.executable, str(MAKE_PLACEBO),
                        "--langspec", str(langspec_path),
                        "--out", str(placebo_path),
                        "--seed", str(PLACEBO_SEED),
                        "--mode", PLACEBO_MODE,
                        "--bucket-size", str(PLACEBO_BUCKET_SIZE),
                    ],
                    check=True,
                )
            pl = json.loads(placebo_path.read_text())
            lsj = json.loads(langspec_path.read_text())
            assert [p for p, _ in pl["model"]["vocab"]] == \
                   [p for p, _ in lsj["model"]["vocab"]], "placebo piece-list drift"
            assert sorted(s for _, s in pl["model"]["vocab"]) == \
                   sorted(s for _, s in lsj["model"]["vocab"]), "placebo score-multiset drift"
            ptok = Tokenizer.from_file(str(placebo_path))
            eos_id = manifest["overlays"][lang_code]["eos_id"]
            assert ptok.token_to_id(model_tok.eos_token) == eos_id, "placebo eos drift"
            encode_probe(ptok, Path(corpora[lang_code]["path"]), embed_rows)
            manifest["placebos"][lang_code] = {
                "path": str(placebo_path),
                "sha256": sha256(placebo_path),
                "mode": PLACEBO_MODE,
                "bucket_size": PLACEBO_BUCKET_SIZE,
                "seed": PLACEBO_SEED,
                "piece_list_identical": True,
                "eos_id_match": True,
            }
            logger.info(f"[{family_name}] placebo {lang_code} OK")

    # ---- Deliverable 4: monolingual 32k BPE fits (qwen x all 5 langs) ----
    for l in cfg["mono_langs"]:
        lang_code = LANG_CODES[l]
        lang_dir = family_dir / l
        lang_dir.mkdir(parents=True, exist_ok=True)
        mono_path = lang_dir / f"mono32k_{lang_code}.json"
        if not mono_path.exists():
            train_bpe_tokenizer(
                corpus_files=[corpora[lang_code]["path"]],
                output_path=str(mono_path),
                vocab_size=MONO_BPE_VOCAB_SIZE,
            )
        mtok = Tokenizer.from_file(str(mono_path))
        for special in ["<unk>", "<s>", "</s>", "<pad>"]:
            if mtok.token_to_id(special) is None:
                raise RuntimeError(f"{mono_path}: special token {special!r} missing")
        encode_probe(mtok, Path(corpora[lang_code]["path"]), embed_rows=None)
        manifest["mono32k"][lang_code] = {
            "path": str(mono_path),
            "sha256": sha256(mono_path),
            "vocab_size": mtok.get_vocab_size(),
            "target_vocab_size": MONO_BPE_VOCAB_SIZE,
        }
        logger.info(f"[{family_name}] mono32k {lang_code} OK: vocab={mtok.get_vocab_size()}")

    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"[{family_name}] manifest written: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Routing pool (F15 joint-multilingual arm)
# ---------------------------------------------------------------------------
def build_pool(family_name: str, git_commit: str) -> dict:
    """Build las_tokenizers/f15/<family>/pool/: base_tokenizer.json plus
    every language's langspec overlay, as RELATIVE symlinks into the
    per-language dirs (no file copies, so the pool can never drift from the
    fitted overlays) -- what `las_experiment.downstream_eval.LasTokenizerHFLM`
    globs for routed eval in the joint-multilingual arm. Requires an overlay
    to already exist for every language in BASE_LANGS.

    Mirrors the convention already established by
    las_tokenizers/f15/qwen2_5-0_5b/pool/ (built manually for wave 2 prep,
    inspected on disk before writing this function so the schema matches
    field-for-field): pool_manifest.json with purpose/built_by/date/
    git_commit/family/langs/entries/note, entries keyed by the pool
    filename with symlink_target (relative), resolved (absolute), sha256.
    """
    family_dir = OUT_ROOT / family_name
    pool_dir = family_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    base_tokenizer_path = family_dir / "base_tokenizer.json"
    if not base_tokenizer_path.exists():
        raise RuntimeError(f"{family_name}: base_tokenizer.json missing, cannot build pool")

    def _link(pool_name: str, target_rel: str, resolved: Path) -> dict:
        if not resolved.exists():
            raise RuntimeError(
                f"{family_name}: pool source missing for {pool_name}: {resolved} "
                f"(every language in BASE_LANGS must have a langspec overlay "
                f"before building the pool)"
            )
        link_path = pool_dir / pool_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(target_rel, link_path)
        return {
            "symlink_target": target_rel,
            "resolved": str(resolved),
            "sha256": sha256(resolved),
        }

    entries = {}
    entries["base_tokenizer.json"] = _link(
        "base_tokenizer.json", "../base_tokenizer.json", base_tokenizer_path
    )
    langs = {}
    for l in BASE_LANGS:
        lang_code = LANG_CODES[l]
        script_suffix = "_" + lang_code.split("_", 1)[1]
        langs[l] = script_suffix
        pool_name = f"langspec_{lang_code}.json"
        target_rel = f"../{l}/{pool_name}"
        resolved = family_dir / l / pool_name
        entries[pool_name] = _link(pool_name, target_rel, resolved)

    pool_manifest = {
        "purpose": (
            "Routing pool for the F15 joint-multilingual arm: one directory "
            "holding base_tokenizer.json plus every langspec overlay, which "
            "is what las_experiment.downstream_eval.LasTokenizerHFLM globs "
            "for routed eval and what las_mode=langspec reads when a run "
            "trains several languages at once."
        ),
        "built_by": "scripts/prep_f15_tokenizers.py build_pool(), F15 wave 2 prep",
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "family": family_name,
        "langs": langs,
        "entries": entries,
        "note": (
            "Symlinks (relative) into the per-language dirs of this same "
            "family; no file is copied, so the pool can never drift from "
            "the fitted overlays."
        ),
    }
    pool_manifest_path = pool_dir / "pool_manifest.json"
    pool_manifest_path.write_text(json.dumps(pool_manifest, indent=2))
    logger.info(f"[{family_name}] pool built: {pool_dir} ({len(entries)} entries)")
    return pool_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES.keys()), default=None,
                     help="Restrict to one family (default: both, sequentially)")
    ap.add_argument("--build-pool", choices=list(FAMILIES.keys()), default=None,
                     help="Build las_tokenizers/f15/<family>/pool/ from "
                          "already-fitted overlays (no training) and exit.")
    args = ap.parse_args()

    if args.build_pool:
        git_commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        build_pool(args.build_pool, git_commit)
        return

    git_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    api = HfApi()
    logger.info(f"=== Building shared corpora ({LINES_PER_LANG} lines/lang, 5 langs) ===")
    corpora = {}
    for l in BASE_LANGS:
        prov = build_corpus(l, api)
        corpora[LANG_CODES[l]] = prov

    families = [args.family] if args.family else list(FAMILIES.keys())
    for fam in families:
        process_family(fam, FAMILIES[fam], corpora, git_commit)

    logger.info("=== F15 tokenizer prep complete ===")


if __name__ == "__main__":
    main()
