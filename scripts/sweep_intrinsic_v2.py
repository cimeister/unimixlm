#!/usr/bin/env python3
"""Driver for the 7-language x 11-tokenizer-family MorphScore sweep (v2).

WHAT THIS IS
------------
`eval_results/f15_intrinsic/f15_final_v2.py` measured MorphScore for 2 families
(Qwen2.5-0.5B, Llama-3.2-1B) x 5 languages (tur/fin/hun/arb/hin) with the
offsets-patched harness and the repaired Arabic gold. This driver generalises
that to every tokenizer family in the paper's tokenizer table
x the 7 sweep languages, doing the tokenizer fitting
(base-Unigram conversion + per-language LangMAP overlay) where it is missing.

One invocation = one (family, language) CELL. A cell is:

  1. ensure the family's model tokenizer is on disk (HF Hub, tokenizer files only);
  2. ensure the family's BASE Unigram tokenizer exists -- a language-agnostic
     re-scoring of the model's own vocabulary, fit with soft EM on the combined
     `--base-langs` corpus (the conversion established by
     scripts/prep_f15_tokenizers.py deliverable 2, which this driver calls
     through to; no algorithm is reimplemented here);
  3. ensure the (family, language) LangMAP overlay exists -- per-language
     re-estimation seeded with the BASE tokenizer's own trained log-scores
     (`prep_f15_tokenizers.BaseScoreSeededLASTokenizer`), 10 EM sub-iterations,
     on that language's 50k-line FineWeb2-100M corpus slice;
  4. score all three arms -- Original(BPE) / base_tokenizer.json /
     LangMAP(langspec) -- with the offsets-patched MorphScore harness at
     `eval_results/f15_intrinsic/harness_patched/morphscore_patched.py`;
  5. write one JSON per cell carrying the metrics AND the sha256 of every input
     that determined them (each tokenizer file, the gold CSV, the harness, the
     corpus files), so a later reader can tell whether a number is still valid.

Every gate in steps 1-3 is the repo's own: `validate_tokenizers` (pre/post-
tokenizer identity + token-order identity against the model tokenizer),
`verify_and_patch_specials` (eos/pad resolve, no id above the embedding table),
and a real-corpus encode probe.

NO SILENT FALLBACKS. A missing corpus, gold file, tokenizer, or manifest field
aborts and names the artifact. A family whose conversion fails a gate is
excluded and the failure recorded; it is never worked around.

ARTIFACT LOCATIONS (path-encoded)
---------------------------------
* Families already fitted by the F15 campaign (qwen2_5-0_5b, llama3.2-1b) keep
  their existing artifacts under `las_tokenizers/f15/<family>/`. Their protocol
  is not in the path (that tree predates this driver), so instead the driver
  READS `las_tokenizers/f15/<family>/manifest.json` and hard-asserts, field by
  field, that it matches this sweep: model id, base-log-score seeding, 10 EM
  iterations, the per-corpus repo/revision/sha256/lines_written, eos/pad/unk,
  and an unchanged base_tokenizer.json sha256. Reuse is what makes those cells
  comparable with `eval_results/f15_intrinsic/f15_morphscore_final_v2.json`;
  that reproduction was verified by hand on qwen2_5-0_5b x tur (3 arms x 7
  metrics + num_samples, all bit-identical) and is not re-checked automatically
  here. On the reuse path this driver never writes to those files.
  New-language overlays for these families are written into the same family
  directory, because an overlay is only meaningful next to the base it was
  seeded from.
* Newly converted families go under
  `las_tokenizers/intrinsic_sweep_v2/<PROTOCOL_TAG>/<family>/`, where
  PROTOCOL_TAG spells out every setting that changes the artifact.

USAGE
-----
  # one cell (fit-if-missing, then score)
  python scripts/sweep_intrinsic_v2.py cell --family qwen2_5-0_5b --lang tur \
      --base-langs tur,fin,hun,arb,hin

  # conversion only (base-Unigram + gates), no scoring
  python scripts/sweep_intrinsic_v2.py convert --family bloom-3b \
      --base-langs tur,fin,hun,arb,hin

  # corpus only
  python scripts/sweep_intrinsic_v2.py corpus --lang deu

  # aggregate every finished cell into one manifest
  python scripts/sweep_intrinsic_v2.py manifest

`--base-langs` has NO default: the base corpus composition changes every number
in the sweep, so it must be stated at every invocation and it is encoded in
PROTOCOL_TAG.
"""

import os

# Must precede any tokenizers/transformers import -- see the identical note in
# scripts/prep_f15_tokenizers.py (Rayon reads these once, at first use).
os.environ.setdefault("RAYON_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
# Hoisted to module scope so `from morphscore_patched import MorphScore`
# resolves from any entry point, not only after score_arms has run.
_HARNESS_DIR = REPO_ROOT / "reproduction" / "harness_patched"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import prep_f15_tokenizers as prep  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from code.constants import NUM_FINETUNE_EM_ITERATIONS  # noqa: E402
from code.tokenizer_formatting_utils import _load_vocab_from_hf_tokenizer  # noqa: E402
from las_experiment.tokenizer_training.training import (  # noqa: E402
    filter_line_by_script,
    train_base_tokenizer,
)
from las_experiment.tokenizer_training.validation import validate_tokenizers  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sweep_intrinsic_v2")

# ---------------------------------------------------------------------------
# Constants. Every one of these changes the numbers; none is buried.
# ---------------------------------------------------------------------------
EM_MODE = "soft"                      # base-conversion EM mode (F15 protocol)
PER_LANG_EM_SEED = "base_log_scores"  # F15 protocol (NOT this branch's uniform default)
LINES_PER_LANG = prep.LINES_PER_LANG  # 50_000, inherited so it cannot drift
SCRIPT_FILTER_THRESHOLD = 0.10        # filter_line_by_script default, restated for the record

import inspect  # noqa: E402

assert (inspect.signature(filter_line_by_script).parameters["threshold"].default
        == SCRIPT_FILTER_THRESHOLD), (
    f"SCRIPT_FILTER_THRESHOLD={SCRIPT_FILTER_THRESHOLD} but "
    f"filter_line_by_script's default is "
    f"{inspect.signature(filter_line_by_script).parameters['threshold'].default}; "
    f"prep.build_corpus calls it with the library default, so the value recorded "
    f"in every cell JSON would be a lie.")

# Resolved lazily in _morphscore_data_dir() so importing this module (or
# running --help) works without the environment variable set.
_REPRO = REPO_ROOT / "reproduction"
HARNESS_DIR = _REPRO / "harness_patched"
HARNESS_PY = HARNESS_DIR / "morphscore_patched.py"
ARB_REPAIRED_CSV = _REPRO / "morphscore" / "arabic_data_repaired.csv"

F15_ROOT = REPO_ROOT / "las_tokenizers" / "f15"
SWEEP_TOK_ROOT = REPO_ROOT / "las_tokenizers" / "intrinsic_sweep_v2"
OUT_ROOT = REPO_ROOT / "eval_results" / "intrinsic_sweep_v2"
CELL_DIR = OUT_ROOT / "cells"
CONV_DIR = OUT_ROOT / "conversions"

# The 7 sweep languages (ben/tha are dropped from the original 9).
SWEEP_LANGS = {
    "deu": "deu_Latn",
    "fin": "fin_Latn",
    "hun": "hun_Latn",
    "tur": "tur_Latn",
    "hin": "hin_Deva",
    "arb": "arb_Arab",
    "tam": "tam_Taml",
}

# The 11 tokenizer families of the paper's tokenizer table, in table order.
#
# `paper_V` is that table's |V| column ("full vocabulary size, added and special
# tokens included"); it is checked against the tokenizer actually downloaded and
# any mismatch is RECORDED, never silently accepted.
#
# Every model_id below was CONFIRMED against the paper's own measurements, not
# guessed: the paper's Devanagari/English subword table reports a Devanagari-Complete count per
# row, and recomputing that count under the paper's stated rule reproduces the
# published number exactly for each id here (Bloom 16189, Gemma 13684, Granite 16,
# Llama 958, Mistral-7b 44, Mistral-Nemo 1530, Phi-3-mini 39, Phi-4 23, Qwen 67,
# XGLM 5667, Yi 3).
#
# TWO ERRORS IN THE PAPER'S GEMMA ROW, found by that check:
#   * the row is Gemma 3, not Gemma 2. google/gemma-3-1b-pt reproduces 13,684
#     Devanagari-Complete pieces; google/gemma-2-2b gives 1,514. They are
#     different tokenizers, so the tokenizer table's label "Gemma 2 & 3" (one row for
#     both) is wrong.
#   * its |V| is 262,144 (262,145 counting the added token at id 262144), not the
#     256,000 printed in the paper's tokenizer and Devanagari/English subword tables.
# `gemma2-2b` is registered below so it can be measured on request, but it is NOT
# the paper's row and is excluded from the 11-family grid.
FAMILIES = {
    "bloom-3b":         {"model_id": "bigscience/bloom-3b",              "paper_row": "Bloom-3b",         "paper_V": 250880, "provenance": "new"},
    "gemma3-1b":        {"model_id": "google/gemma-3-1b-pt",             "paper_row": "Gemma 2 & 3",      "paper_V": 256000, "provenance": "new",
                         "paper_V_note": "The paper's tokenizer table prints 256,000; the tokenizer this row was measured on has 262,144 entries. See module docstring.",
                         # Sweep-scoped, family-scoped bypass.
                         # google/gemma-3-1b-pt places <image_soft_token> at id
                         # 262144 == config.vocab_size, so the embedding-row gate
                         # ("tokenizer can emit ids >= embedding rows") fires. That
                         # gate exists to stop an LM training run indexing past its
                         # embedding table; this sweep only measures MorphScore and
                         # never builds a model, so it does not apply here. The
                         # bypass is declared per family in this file, is recorded
                         # in every cell/conversion record it affects, and does NOT
                         # touch scripts/prep_f15_tokenizers.py -- the training
                         # pipeline's gate is unchanged.
                         "embed_rows_gate": "measurement_only_bypass",
                         "embed_rows_gate_reason": (
                             "gemma-3-1b-pt's <image_soft_token> sits at id 262144 == "
                             "config.vocab_size (262144), so vocab_size_total (262145) "
                             "exceeds the embedding rows by exactly that one token. The "
                             "gate guards LM training against out-of-range embedding "
                             "lookups; this sweep computes MorphScore from tokenizer "
                             "offsets alone and instantiates no model.")},
    "granite3.0-8b":    {"model_id": "ibm-granite/granite-3.0-8b-base",  "paper_row": "Granite 3.0-8b",   "paper_V": 49152,  "provenance": "new"},
    "llama3.2-1b":      {"model_id": "meta-llama/Llama-3.2-1B",          "paper_row": "Llama 3",          "paper_V": 128256, "provenance": "f15"},
    "mistral-7b-v0.3":  {"model_id": "mistralai/Mistral-7B-v0.3",        "paper_row": "Mistral-7b-v0.3",  "paper_V": 32768,  "provenance": "new"},
    "mistral-nemo-12b": {"model_id": "mistralai/Mistral-Nemo-Base-2407", "paper_row": "Mistral-Nemo-12b", "paper_V": 131072, "provenance": "new"},
    "olmo3-7b":         {"model_id": "allenai/Olmo-3-1025-7B",           "paper_row": None,               "paper_V": 100278, "provenance": "new",
                         "paper_V_note": (
                             "Not one of the paper's 11 tokenizer-table families (paper_row=None; "
                             "excluded from PAPER_GRID). OLMo-3-7B is the paper's own "
                             "base model and appears separately in the paper's OLMo appendix, "
                             "an unreproducible 19-"
                             "language, 10k-line-corpus, single-seed sweep this family "
                             "entry exists to replace with an artifact-backed cell grid "
                             "under THIS driver's 50k-line protocol. paper_V=100278 is "
                             "config.vocab_size, confirmed via AutoConfig.from_pretrained "
                             "and identical to the fitted tokenizer's vocab_size_total "
                             "(100278 entries + 22 added tokens, all within the same "
                             "100278 ids), not a tokenizer-table citation."),
                         "old_overlays_not_reused": (
                             "las_tokenizers/olmo3-7b_v100k_fineweb2sample_10klines_* "
                             "were fit under a different (10k-line) protocol and are "
                             "NOT read or reused by this driver; this family's base and "
                             "overlays are refit at the sweep's 50k-line protocol under "
                             "las_tokenizers/intrinsic_sweep_v2/<protocol_tag>/olmo3-7b/.")},
    "phi-3-mini":       {"model_id": "microsoft/Phi-3-mini-4k-instruct", "paper_row": "Phi-3-mini",       "paper_V": 32064,  "provenance": "new",
                         "excluded": (
                             "microsoft/Phi-3-mini-4k-instruct's tokenizer.json has "
                             "pre_tokenizer: null (its space handling lives entirely in "
                             "a Prepend+Replace normalizer). The base-Unigram conversion "
                             "pre-tokenizes the corpus with the pre-tokenizer and "
                             "validate_tokenizers requires it to be non-null, so this "
                             "family cannot be converted by this pipeline. Confirmed by "
                             "a failed conversion attempt; excluded. "
                             "No substitute was used.")},
    "phi-4":            {"model_id": "microsoft/phi-4",                  "paper_row": "Phi-4",            "paper_V": 100352, "provenance": "new"},
    "qwen2_5-0_5b":     {"model_id": "Qwen/Qwen2.5-0.5B",                "paper_row": "Qwen 2.5 & 3",     "paper_V": 151936, "provenance": "f15"},
    "xglm-2.9b":        {"model_id": "facebook/xglm-2.9B",               "paper_row": "XGLM-2.9b",        "paper_V": 256008, "provenance": "new"},
    "yi-1.5-6b":        {"model_id": "01-ai/Yi-1.5-6B",                  "paper_row": "Yi-1.5-6b",        "paper_V": 64000,  "provenance": "new",
                         # KEEP, measured as transformers loads
                         # it. The Hub tokenizer.json has pre_tokenizer: null and
                         # transformers 5.3.0's save_pretrained synthesises a
                         # Metaspace pre-tokenizer, moves the space->metaspace
                         # Replace out of the normalizer, adds a TemplateProcessing
                         # post_processor and drops one added token. That rewritten
                         # object is what any user of this checkpoint gets, so it is
                         # what is measured. Every cell records the Hub sha256 and
                         # the differing sections under
                         # model_tokenizer.hub_tokenizer_json.
                         "tokenizer_rewritten_by_transformers": True},
}
# The grid actually measured: paper rows that are not excluded.
PAPER_GRID = [f for f in FAMILIES
              if FAMILIES[f]["paper_row"] is not None and not FAMILIES[f].get("excluded")]

# Corrections to the paper's own tables, established by recomputing
# the paper's Devanagari/English subword table's Devanagari-Complete column under the paper's stated
# rule. Carried into the sweep manifest so the finding travels with the numbers.
PAPER_TABLE_CORRECTIONS = {
    "tab:tokenizers Gemma row |V|": {
        "published": 256000,
        "actual": 262144,
        "evidence": (
            "The row's Devanagari-Complete count (13,684 in the paper's Devanagari subword table) "
            "is reproduced exactly by google/gemma-3-1b-pt and not by "
            "google/gemma-2-2b (1,514). gemma-3-1b-pt's tokenizer has 262,144 vocab "
            "entries (262,145 counting the added token at id 262144); the published "
            "256,000 is Gemma 2's size."),
    },
    "tab:tokenizers Gemma row label": {
        "published": "Gemma 2 & 3",
        "problem": (
            "Gemma 2 and Gemma 3 do not share a tokenizer: 256,000 vs 262,144 "
            "entries, 1,514 vs 13,684 Devanagari-Complete pieces. One row cannot "
            "cover both. The measurements in the paper are Gemma 3's."),
    },
}

# Which arms are scored, and where each one's tokenizer file lives relative to
# the family's artifact root.
ARMS = ("Original(BPE)", "base_tokenizer.json", "LangMAP(langspec)")

# Metric keys, identical to eval_results/f15_intrinsic/f15_final_v2.py's `M`,
# so a cell JSON can be diffed against f15_morphscore_final_v2.json directly.
METRICS = ["micro_f1", "macro_f1", "morphscore_recall", "morphscore_precision",
           "micro_precision", "micro_recall", "mean_token_char_ratio"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def code_fingerprint() -> dict:
    """The driver is not committed, so the git HEAD alone does not identify the
    code that produced a record. Hash the two files that decide the numbers."""
    return {
        "sweep_intrinsic_v2.py": sha256(Path(__file__).resolve()),
        "prep_f15_tokenizers.py": sha256(REPO_ROOT / "scripts" / "prep_f15_tokenizers.py"),
        "standard_unigramlm.py": sha256(REPO_ROOT / "code" / "standard_unigramlm.py"),
        "language_specific_unigramlm.py":
            sha256(REPO_ROOT / "code" / "language_specific_unigramlm.py"),
    }


def parse_base_langs(spec: str) -> list:
    """Validate --base-langs. Order is significant: it is written into
    PROTOCOL_TAG verbatim, so two different orderings of the same set produce
    two different artifact directories (duplicated work, never a silent mix)."""
    langs = [x.strip() for x in spec.split(",") if x.strip()]
    if not langs:
        raise RuntimeError("--base-langs is empty")
    unknown = [l for l in langs if l not in SWEEP_LANGS]
    if unknown:
        raise RuntimeError(f"--base-langs contains unknown languages {unknown}; "
                           f"known: {sorted(SWEEP_LANGS)}")
    dupes = sorted({l for l in langs if langs.count(l) > 1})
    if dupes:
        raise RuntimeError(f"--base-langs contains duplicates {dupes}; the base "
                           f"corpus would silently be smaller than the tag claims")
    return langs


def protocol_tag(base_langs) -> str:
    """Directory-name-safe spelling of every setting that changes an artifact."""
    return (f"fw2-100m_{LINES_PER_LANG // 1000}klines_scriptfilt"
            f"_base-{'-'.join(base_langs)}"
            f"_{EM_MODE}em_seed-{PER_LANG_EM_SEED.replace('_', '')}"
            f"_em{NUM_FINETUNE_EM_ITERATIONS}")


def corpus_fingerprint(base_langs, corpora: dict) -> dict:
    """The identity of the corpora an artifact was fit on -- what a later reuse
    must still agree with."""
    return {SWEEP_LANGS[l]: {
        "repo": corpora[SWEEP_LANGS[l]]["repo"],
        "revision": corpora[SWEEP_LANGS[l]]["revision"],
        "sha256": corpora[SWEEP_LANGS[l]]["sha256"],
        "lines_written": corpora[SWEEP_LANGS[l]]["lines_written"],
    } for l in base_langs}


def write_provenance(artifact: Path, record: dict) -> None:
    Path(str(artifact) + ".provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False))


def check_provenance(artifact: Path, expected: dict, what: str) -> dict:
    """On REUSE of an already-fitted artifact, verify that everything that
    determined it is still what this invocation is claiming. prep's
    build_corpus silently REBUILDS a corpus when the Hub revision moves (it only
    logs a warning), so without this check a cell could record a new corpus
    revision beside a tokenizer fit on the old one."""
    p = Path(str(artifact) + ".provenance.json")
    if not p.is_file():
        raise RuntimeError(
            f"{what}: {artifact} exists but {p} does not, so the corpora and "
            f"settings behind it cannot be verified. Refusing to reuse an "
            f"artifact of unknown provenance -- delete it to refit, or restore "
            f"the provenance file."
        )
    have = json.loads(p.read_text())
    diffs = []
    for k, want in expected.items():
        if have.get(k) != want:
            diffs.append(f"{k}: recorded={have.get(k)!r} now={want!r}")
    if diffs:
        raise RuntimeError(
            f"{what}: {artifact} was fit under different inputs than this "
            f"invocation declares; refusing to reuse it.\n  - " + "\n  - ".join(diffs))
    return have


class _Lock:
    """O_EXCL lock so two concurrent invocations can never write the same
    tokenizer artifact. Not advisory: acquisition failure is an abort, not a
    wait, because the other writer's settings are unknown to this process."""

    def __init__(self, path: Path):
        self.path = Path(str(path) + ".lock")
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise RuntimeError(
                    f"Lock {self.path} already held -- another process is fitting "
                    f"this artifact. Aborting rather than racing it. If you are "
                    f"certain no such process exists, remove the lock file by hand."
                ) from None
            raise
        os.write(self.fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


def env_versions() -> dict:
    import numpy
    import pandas
    import sentencepiece
    import tokenizers
    import transformers
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "sentencepiece": sentencepiece.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
    }


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def register_sweep_langs() -> None:
    """Make prep_f15_tokenizers.build_corpus reachable for the two languages the
    F15 campaign never needed. Explicit, checked, and reported -- prep's own
    entries are never overwritten."""
    for short, code in SWEEP_LANGS.items():
        existing = prep.LANG_CODES.get(short)
        if existing is None:
            prep.LANG_CODES[short] = code
            logger.info(f"registered sweep language {short} -> {code} "
                        f"(absent from prep_f15_tokenizers.LANG_CODES)")
        elif existing != code:
            raise RuntimeError(
                f"Language code conflict for {short!r}: this sweep expects "
                f"{code!r} but scripts/prep_f15_tokenizers.py declares "
                f"{existing!r}. Refusing to fit against an ambiguous corpus."
            )


def ensure_corpus(short_lang: str, api: HfApi) -> dict:
    """Build (or verify) the 50k-line script-filtered corpus for one language.
    Delegates to prep_f15_tokenizers.build_corpus so the extraction rule cannot
    drift from the one that produced the existing F15 corpora."""
    if short_lang not in SWEEP_LANGS:
        raise RuntimeError(f"{short_lang!r} is not one of the sweep languages "
                           f"{sorted(SWEEP_LANGS)}")
    register_sweep_langs()
    lang_code = SWEEP_LANGS[short_lang]
    # Fail here rather than deep inside the corpus loop if the script has no
    # SCRIPT_RANGES entry.
    filter_line_by_script("x", lang_code)
    t0 = time.time()
    prov = prep.build_corpus(short_lang, api)
    prov["build_seconds"] = round(time.time() - t0, 2)
    return prov


# ---------------------------------------------------------------------------
# Family artifact resolution
# ---------------------------------------------------------------------------
def family_root(family: str, base_langs) -> Path:
    cfg = FAMILIES[family]
    if cfg["provenance"] == "f15":
        return F15_ROOT / family
    return SWEEP_TOK_ROOT / protocol_tag(base_langs) / family


def check_f15_protocol(family: str, base_langs, corpora: dict) -> dict:
    """Hard-verify that a reused F15 family's artifacts were produced under this
    sweep's protocol AND on the same corpora this invocation just resolved.
    Any disagreement aborts."""
    root = F15_ROOT / family
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise RuntimeError(
            f"{family}: reuse of F15 artifacts requires {mpath}, which is missing. "
            f"Without it the protocol behind {root} cannot be verified."
        )
    m = json.loads(mpath.read_text())
    exp_model = FAMILIES[family]["model_id"]
    problems = []
    if m.get("model_name") != exp_model:
        problems.append(f"model_name={m.get('model_name')!r} != {exp_model!r}")
    seeding = m.get("seeding") or {}
    if seeding.get("per_lang_em_seed") != PER_LANG_EM_SEED:
        problems.append(f"per_lang_em_seed={seeding.get('per_lang_em_seed')!r} "
                        f"!= {PER_LANG_EM_SEED!r}")
    if seeding.get("num_finetune_em_iterations") != NUM_FINETUNE_EM_ITERATIONS:
        problems.append(f"num_finetune_em_iterations="
                        f"{seeding.get('num_finetune_em_iterations')} "
                        f"!= {NUM_FINETUNE_EM_ITERATIONS}")

    # Corpus identity, not just the key set: the F15 manifests carry repo,
    # revision, sha256 and lines_written per language, and prep.build_corpus
    # silently rebuilds a corpus when the Hub revision moves.
    man_corp = m.get("corpora") or {}
    have, want = set(man_corp), {SWEEP_LANGS[l] for l in base_langs}
    if have != want:
        problems.append(f"base corpora {sorted(have)} != requested --base-langs "
                        f"{sorted(want)}")
    for lc in sorted(have & want):
        for field in ("repo", "revision", "sha256", "lines_written"):
            a, b = man_corp[lc].get(field), corpora[lc].get(field)
            if a != b:
                problems.append(f"corpus {lc} {field}: F15 manifest={a!r} "
                                f"now={b!r}")

    # Token identity: new deu/tam overlays are written into this same tree with
    # an unk/eos re-derived from HF today; they must match what the August
    # artifacts were fit with.
    model_tok_json = root / "model_tokenizer" / "tokenizer.json"
    if not model_tok_json.is_file():
        problems.append(f"{model_tok_json} missing")
    else:
        try:
            mt = AutoTokenizer.from_pretrained(model_tok_json.parent)
            if mt.pad_token is None:
                mt.pad_token = mt.eos_token
            now_unk = mt.unk_token or mt.eos_token
            for field, now in (("eos_token", mt.eos_token),
                               ("pad_token", mt.pad_token),
                               ("unk_token_internal", now_unk)):
                if m.get(field) != now:
                    problems.append(f"{field}: F15 manifest={m.get(field)!r} "
                                    f"now={now!r}")
        except Exception as e:
            problems.append(f"could not reload {model_tok_json.parent}: "
                            f"{type(e).__name__}: {e}")

    bt = root / "base_tokenizer.json"
    if not bt.is_file():
        problems.append(f"{bt} missing")
    else:
        rec = (m.get("base_tokenizer") or {}).get("sha256")
        cur = sha256(bt)
        if rec != cur:
            problems.append(f"base_tokenizer.json sha256 {cur} != manifest {rec} "
                            f"(the file changed after the manifest was written)")
    if problems:
        raise RuntimeError(
            f"{family}: F15 artifacts at {root} do NOT match this sweep's "
            f"protocol; refusing to mix protocols.\n  - " + "\n  - ".join(problems)
        )
    return {"f15_manifest": str(mpath), "f15_manifest_checked": True,
            "f15_embed_rows": m.get("embed_rows"),
            "corpora": man_corp}


def ensure_model_tokenizer(family: str, root: Path) -> dict:
    """Download the family's own tokenizer files (tokenizer.json +
    tokenizer_config + special_tokens_map) into <root>/model_tokenizer/.
    Weights are never fetched."""
    model_id = FAMILIES[family]["model_id"]
    dest = root / "model_tokenizer"
    tj = dest / "tokenizer.json"
    if not tj.is_file():
        with _Lock(tj):
            dest.mkdir(parents=True, exist_ok=True)
            tok = AutoTokenizer.from_pretrained(model_id)
            # pad BEFORE saving, so the saved dir carries pad=eos exactly as
            # scripts/prep_f15_tokenizers.py's model_tokenizer/ dirs do.
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.save_pretrained(dest)
        if not tj.is_file():
            raise RuntimeError(
                f"{family}: AutoTokenizer.save_pretrained({dest}) produced no "
                f"tokenizer.json -- the family has no fast-tokenizer JSON, so "
                f"the conversion pipeline (which reads tokenizer.json directly) "
                f"cannot run on it."
            )
    tok = AutoTokenizer.from_pretrained(dest)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if (tok.unk_token or tok.eos_token) is None:
        raise RuntimeError(
            f"{family} ({model_id}): both unk_token and eos_token are None, so "
            f"the internal unk piece handed to spm_train would be the literal "
            f"string 'None'. Aborting rather than fitting a wrong artifact.")
    cfg = AutoConfig.from_pretrained(model_id)
    embed_rows = getattr(cfg, "vocab_size", None)
    if embed_rows is None and hasattr(cfg, "text_config"):
        embed_rows = getattr(cfg.text_config, "vocab_size", None)
    if embed_rows is None:
        raise RuntimeError(
            f"{family}: could not read vocab_size from {model_id}'s config; "
            f"the embedding-row bound in verify_and_patch_specials would be "
            f"unchecked. Aborting rather than skipping that gate."
        )
    hf_tok = Tokenizer.from_file(str(tj))
    # The conversion pipeline drives the corpus through this pre-tokenizer and
    # `validate_tokenizers` requires a non-null one. A family whose saved
    # tokenizer.json has neither would otherwise die with a bare
    # "'NoneType' object has no attribute 'pre_tokenize_str'" inside a
    # ThreadPoolExecutor worker, ~10 minutes in, with no family name anywhere in
    # the traceback. (The Hub files for Yi-1.5-6B and Phi-3-mini both carry
    # "pre_tokenizer": null; transformers' save_pretrained happens to
    # reconstruct a Metaspace pre-tokenizer for them, but that is its choice,
    # not a guarantee.)
    if hf_tok.pre_tokenizer is None:
        raise RuntimeError(
            f"{family} ({model_id}): the tokenizer at {tj} has no pre_tokenizer. "
            f"The base-Unigram conversion pre-tokenizes the corpus with it and "
            f"validate_tokenizers requires it to be non-null, so this family "
            f"cannot be converted by this pipeline. EXCLUDED.")
    if hf_tok.decoder is None:
        raise RuntimeError(
            f"{family} ({model_id}): the tokenizer at {tj} has no decoder; "
            f"validate_tokenizers requires a non-null decoder. EXCLUDED.")

    # transformers' save_pretrained can REWRITE a tokenizer on the way to disk
    # (observed on Yi-1.5-6B under transformers 5.3.0: the space->metaspace
    # Replace normalizer is moved into a Metaspace pre_tokenizer, a
    # TemplateProcessing post_processor appears, and one added token is
    # dropped). The conversion, the gates and the Original(BPE) arm all run on
    # THIS saved file, so record what it is relative to the Hub file and say so
    # out loud rather than letting the difference pass unnoticed.
    hub_diff = {"checked": False}
    try:
        from huggingface_hub import hf_hub_download
        hub_tj = hf_hub_download(model_id, "tokenizer.json")
        saved_json, hub_json = json.loads(tj.read_text()), json.loads(Path(hub_tj).read_text())
        differing = [k for k in ("normalizer", "pre_tokenizer", "decoder",
                                 "post_processor", "added_tokens")
                     if saved_json.get(k) != hub_json.get(k)]
        hub_diff = {
            "checked": True,
            "hub_path": hub_tj,
            "hub_sha256": sha256(hub_tj),
            "identical_to_saved": sha256(hub_tj) == sha256(tj),
            "differing_sections": differing,
            "hub_vocab_entries": len(hub_json["model"]["vocab"]),
            "saved_vocab_entries": len(saved_json["model"]["vocab"]),
        }
        if differing:
            logger.warning(
                f"{family}: save_pretrained rewrote the tokenizer relative to the "
                f"Hub file; differing sections: {differing}. The conversion and "
                f"the Original(BPE) arm use the SAVED file at {tj}.")
    except Exception as e:
        hub_diff = {"checked": False,
                    "error": f"{type(e).__name__}: {e}"}
        logger.warning(f"{family}: could not compare the saved tokenizer against "
                       f"the Hub file ({type(e).__name__}: {e}); recorded as unchecked.")

    return {
        "model_id": model_id,
        "path": str(tj),
        "sha256": sha256(tj),
        "hub_tokenizer_json": hub_diff,
        "embed_rows": int(embed_rows),
        "vocab_size_total": hf_tok.get_vocab_size(),
        "trainable_vocab": len(_load_vocab_from_hf_tokenizer(hf_tok)),
        "paper_V": FAMILIES[family]["paper_V"],
        # The tokenizer table's |V| column tracks the model config's vocab_size (the
        # embedding rows), NOT the tokenizer's entry count -- the two differ for
        # every family that pads its embedding table. Compare against the right
        # quantity and say so when it disagrees.
        "paper_V_matches_embed_rows": int(embed_rows) == FAMILIES[family]["paper_V"],
        "eos_token": tok.eos_token,
        "pad_token": tok.pad_token,
        "unk_token": tok.unk_token,
        "_tok": tok,
        "_tok_json": json.loads(tj.read_text()),
    }


# Sentinel handed to prep.verify_and_patch_specials when a family carries an
# measurement-only bypass of the embedding-row gate. prep's function is
# left exactly as the training pipeline uses it; the driver performs the
# comparison itself and records the outcome, so the bypass is visible in the
# artifacts rather than hidden inside a shared helper.
_EMBED_ROWS_UNBOUNDED = float("inf")


def embed_rows_policy(family: str) -> tuple:
    """(effective_embed_rows_for_prep, bypass_record_or_None)."""
    cfg = FAMILIES[family]
    if cfg.get("embed_rows_gate") == "measurement_only_bypass":
        return _EMBED_ROWS_UNBOUNDED, {
            "embed_rows_gate": "measurement_only_bypass",
            "reason": cfg["embed_rows_gate_reason"],
            "scope": ("this sweep driver only; scripts/prep_f15_tokenizers.py and "
                      "the training pipeline's gate are unmodified"),
            "approved": "sweep filter policy",
        }
    return None, None


def apply_embed_rows_gate(family: str, path: Path, embed_rows: int,
                          specials: dict, what: str) -> dict:
    """Run the embedding-row check the driver owns, honouring a
    per-family bypass. Mutates nothing; returns the fields to record."""
    vocab_size_total = specials["vocab_size_total"]
    _, bypass = embed_rows_policy(family)
    over = vocab_size_total > embed_rows
    if over and bypass is None:
        raise RuntimeError(
            f"{what}: {path} can emit ids up to {vocab_size_total - 1} >= model "
            f"embedding rows {embed_rows}")
    rec = {"embed_rows": embed_rows,
           "vocab_size_total": vocab_size_total,
           "ids_exceed_embed_rows": bool(over)}
    if bypass is not None:
        rec["embed_rows_gate_bypass"] = bypass
        if over:
            logger.warning(
                f"{what}: {path} can emit ids up to {vocab_size_total - 1} >= "
                f"embedding rows {embed_rows}; allowed by the approved "
                f"measurement-only bypass declared for {family}. Recorded in the "
                f"cell/conversion record.")
    return rec


def verify_specials_readonly(family: str, path: Path, model_tok, embed_rows: int,
                             what: str) -> dict:
    """Read-only twin of prep.verify_and_patch_specials: same gates (eos and pad
    resolve; no id at or above the embedding rows, subject to a
    per-family bypass), but it never writes. Used on the REUSE path so an
    already-published artifact can never be rewritten and concurrent readers
    cannot tear."""
    tok = Tokenizer.from_file(str(path))
    eos_id = tok.token_to_id(model_tok.eos_token)
    pad_id = tok.token_to_id(model_tok.pad_token)
    if eos_id is None or pad_id is None:
        raise RuntimeError(
            f"{what}: {path} does not resolve eos={model_tok.eos_token!r} "
            f"(id={eos_id}) or pad={model_tok.pad_token!r} (id={pad_id}). The "
            f"artifact would need patching, which this read-only reuse path "
            f"refuses to do. Delete it and refit.")
    specials = {"eos_id": eos_id, "pad_id": pad_id,
                "vocab_size_total": tok.get_vocab_size(),
                "patched_added_tokens": 0, "specials_check": "read-only"}
    specials.update(apply_embed_rows_gate(family, path, embed_rows, specials, what))
    return specials


def _log_paper_V(family: str, mt: dict) -> None:
    if not mt["paper_V_matches_embed_rows"]:
        logger.warning(
            f"{family}: the paper's tokenizer table declares |V|={mt['paper_V']} but "
            f"{mt['model_id']}'s config.vocab_size is {mt['embed_rows']} "
            f"(tokenizer entries {mt['vocab_size_total']}). Recorded, not "
            f"corrected.")


def ensure_base_tokenizer(family: str, root: Path, base_langs, corpora: dict,
                          mt: dict) -> dict:
    """Ensure <root>/base_tokenizer.json: the family's own vocabulary re-scored
    as a Unigram model by soft EM on the combined `base_langs` corpus. Runs the
    repo's validation gates on the result."""
    base_path = root / "base_tokenizer.json"
    unk_token = mt["unk_token"] or mt["eos_token"]
    expected_prov = {
        "kind": "base_tokenizer",
        "family": family,
        "model_id": mt["model_id"],
        "model_tokenizer_sha256": mt["sha256"],
        "base_langs": list(base_langs),
        "corpora": corpus_fingerprint(base_langs, corpora),
        "em_mode": EM_MODE,
        "unk_token_internal": unk_token,
        "lines_per_lang": LINES_PER_LANG,
    }
    missing = [SWEEP_LANGS[l] for l in base_langs if SWEEP_LANGS[l] not in corpora]
    if missing:
        raise RuntimeError(f"{family}: base corpora missing for {missing}")

    fitted = False
    t0 = time.time()
    if not base_path.is_file():
        # Fit AND the patch step that follows it happen under one lock: the
        # patch WRITES base_path, and several language cells of the same family
        # may run concurrently against this one file.
        with _Lock(base_path):
            train_base_tokenizer(
                corpus_info={SWEEP_LANGS[l]: corpora[SWEEP_LANGS[l]]["path"]
                             for l in base_langs},
                model_tokenizer_path=str(root / "model_tokenizer" / "tokenizer.json"),
                base_output_path=str(base_path),
                em_mode=EM_MODE,
                unk_token=unk_token,
            )
            fitted = True
            validate_tokenizers(root / "model_tokenizer" / "tokenizer.json", base_path)
            override, _ = embed_rows_policy(family)
            specials = prep.verify_and_patch_specials(
                base_path, mt["_tok"], mt["_tok_json"],
                mt["embed_rows"] if override is None else override)
            specials.update(apply_embed_rows_gate(
                family, base_path, mt["embed_rows"], specials,
                f"{family} base_tokenizer"))
            write_provenance(base_path, expected_prov)
    else:
        # F15 bases predate this driver and carry no .provenance.json; for them
        # check_f15_protocol has already verified the manifest field by field
        # (model id, seeding, EM iterations, per-corpus repo/revision/sha256/
        # lines, eos/pad/unk, base sha256), which is the same guarantee.
        prov_file = Path(str(base_path) + ".provenance.json")
        if prov_file.is_file() or FAMILIES[family]["provenance"] != "f15":
            check_provenance(base_path, expected_prov, f"{family} base_tokenizer")
        else:
            logger.info(f"{family}: reusing pre-driver F15 base {base_path}; "
                        f"provenance verified via the F15 manifest.")
        validate_tokenizers(root / "model_tokenizer" / "tokenizer.json", base_path)
        # Reuse path is strictly READ-ONLY, so concurrent language cells of the
        # same family cannot collide, and a published F15 artifact can never be
        # rewritten underneath f15_morphscore_final_v2.json.
        specials = verify_specials_readonly(family, base_path, mt["_tok"],
                                            mt["embed_rows"],
                                            f"{family} base_tokenizer")
    fit_seconds = round(time.time() - t0, 2) if fitted else None

    tok = Tokenizer.from_file(str(base_path))
    for l in base_langs:
        prep.encode_probe(tok, Path(corpora[SWEEP_LANGS[l]]["path"]),
                          mt["embed_rows"])
    base_sha = sha256(base_path)
    return {
        "path": str(base_path),
        "sha256": base_sha,
        "fitted_now": fitted,
        "fit_seconds": fit_seconds,
        "base_langs": list(base_langs),
        "em_mode": EM_MODE,
        "unk_token_internal": unk_token,
        "vocab_size_trainable": len(_load_vocab_from_hf_tokenizer(tok)),
        "provenance_file": str(base_path) + ".provenance.json",
        **specials,
    }


def ensure_overlay(family: str, root: Path, short_lang: str, corpora: dict,
                   mt: dict) -> dict:
    """Ensure <root>/<lang>/langspec_<lang_code>.json, seeded with the base
    tokenizer's own trained log-scores (F15 protocol)."""
    lang_code = SWEEP_LANGS[short_lang]
    lang_dir = root / short_lang
    ls_path = lang_dir / f"langspec_{lang_code}.json"
    base_path = root / "base_tokenizer.json"
    if not base_path.is_file():
        raise RuntimeError(f"{family}: {base_path} missing; fit the base first")
    if lang_code not in corpora:
        raise RuntimeError(f"{family}/{short_lang}: corpus for {lang_code} missing")

    expected_prov = {
        "kind": "langspec_overlay",
        "family": family,
        "model_id": mt["model_id"],
        "model_tokenizer_sha256": mt["sha256"],
        "base_tokenizer_sha256": sha256(base_path),
        "lang_code": lang_code,
        "corpus": {
            "repo": corpora[lang_code]["repo"],
            "revision": corpora[lang_code]["revision"],
            "sha256": corpora[lang_code]["sha256"],
            "lines_written": corpora[lang_code]["lines_written"],
        },
        "em_mode": EM_MODE,
        "em_seed": PER_LANG_EM_SEED,
        "em_iterations": NUM_FINETUNE_EM_ITERATIONS,
        "unk_token_internal": mt["unk_token"] or mt["eos_token"],
    }

    fitted = False
    t0 = time.time()
    if not ls_path.is_file():
        with _Lock(ls_path):
            lang_dir.mkdir(parents=True, exist_ok=True)
            trainer = prep.BaseScoreSeededLASTokenizer(
                reestimation_em_mode=EM_MODE,
                unk_token=mt["unk_token"] or mt["eos_token"],
            )
            if trainer.num_reestimation_iterations != NUM_FINETUNE_EM_ITERATIONS:
                raise RuntimeError(
                    f"num_reestimation_iterations="
                    f"{trainer.num_reestimation_iterations}, expected "
                    f"{NUM_FINETUNE_EM_ITERATIONS}")
            trainer.train(
                corpus_info={lang_code: corpora[lang_code]["path"]},
                base_tokenizer_path=str(base_path),
                tokenizer_path_format=str(lang_dir / "langspec_{lang_code}.json"),
                load_base_if_exists=True,
                load_tokenizers_if_exists=False,
                use_sentencepiece=True,
                max_workers=1,
            )
            fitted = True
            validate_tokenizers(root / "model_tokenizer" / "tokenizer.json",
                                base_path, {lang_code: ls_path})
            override, _ = embed_rows_policy(family)
            specials = prep.verify_and_patch_specials(
                ls_path, mt["_tok"], mt["_tok_json"],
                mt["embed_rows"] if override is None else override)
            specials.update(apply_embed_rows_gate(
                family, ls_path, mt["embed_rows"], specials,
                f"{family}/{short_lang} overlay"))
            write_provenance(ls_path, expected_prov)
    else:
        # Existing F15 overlays predate this driver and carry no provenance
        # file; for them the F15 manifest is the record and check_f15_protocol
        # has already verified it. Everything this driver fits gets one, and it
        # is enforced.
        prov_file = Path(str(ls_path) + ".provenance.json")
        if prov_file.is_file() or FAMILIES[family]["provenance"] != "f15":
            check_provenance(ls_path, expected_prov, f"{family}/{short_lang} overlay")
        else:
            logger.info(f"{family}/{short_lang}: reusing pre-driver F15 overlay "
                        f"{ls_path}; provenance verified via the F15 manifest.")
        validate_tokenizers(root / "model_tokenizer" / "tokenizer.json", base_path,
                            {lang_code: ls_path})
        specials = verify_specials_readonly(family, ls_path, mt["_tok"],
                                            mt["embed_rows"],
                                            f"{family}/{short_lang} overlay")
    fit_seconds = round(time.time() - t0, 2) if fitted else None

    prep.encode_probe(Tokenizer.from_file(str(ls_path)),
                      Path(corpora[lang_code]["path"]), mt["embed_rows"])
    return {
        "path": str(ls_path),
        "sha256": sha256(ls_path),
        "fitted_now": fitted,
        "fit_seconds": fit_seconds,
        "lang_code": lang_code,
        "em_seed": PER_LANG_EM_SEED,
        "em_iterations": NUM_FINETUNE_EM_ITERATIONS,
        "provenance_file": (str(ls_path) + ".provenance.json"
                            if Path(str(ls_path) + ".provenance.json").is_file()
                            else None),
        **specials,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def gold_for(short_lang: str) -> dict:
    """Per-language gold + filter policy: default filter config
    everywhere; Arabic measured on the repaired gold.

    The path returned here is the file whose sha256 goes into the cell JSON, so
    it must be the file the harness will actually open. MorphScore._load_dataset
    tries `<data_dir>/<language.lower()>_data.csv` FIRST and only then consults
    FLORES_TO_MS_FILES, so that precedence is mirrored rather than assumed away."""
    lang_code = SWEEP_LANGS[short_lang]
    if short_lang == "arb":
        if not ARB_REPAIRED_CSV.is_file():
            raise RuntimeError(
                f"Arabic repaired gold {ARB_REPAIRED_CSV} missing; rebuild with "
                f"reproduction/morphscore/repair_arabic_gold.py. Refusing to "
                f"fall back to the original (broken) arabic_data.csv.")
        return {"gold": "repaired", "data_dir": str(ARB_REPAIRED_CSV.parent),
                "dataset_key": ARB_REPAIRED_CSV.name, "file": str(ARB_REPAIRED_CSV)}
    data_dir = os.environ.get("MORPHSCORE_DATA_DIR")
    if data_dir is None:
        raise RuntimeError(
            "MORPHSCORE_DATA_DIR is not set. Point it at a local checkout of "
            "the MorphScore evaluation data (per-language *_data.csv files).")
    from morphscore_patched import MorphScore
    first_choice = Path(data_dir) / f"{lang_code.lower()}_data.csv"
    if first_choice.exists():
        fpath = first_choice
    else:
        fname = MorphScore.FLORES_TO_MS_FILES.get(lang_code)
        if fname is None:
            raise RuntimeError(
                f"{lang_code} has no entry in MorphScore.FLORES_TO_MS_FILES and "
                f"{first_choice} does not exist; the harness cannot locate a "
                f"gold file for it.")
        fpath = Path(data_dir) / fname
    if not fpath.is_file():
        raise RuntimeError(f"gold file {fpath} missing for {lang_code}")
    return {"gold": "default", "data_dir": data_dir,
            "dataset_key": lang_code, "file": str(fpath)}


def score_arms(arm_paths: dict, short_lang: str, stem_eq_lemma: bool) -> tuple:
    """Score every arm. The MorphScore construction and the result extraction
    are byte-for-byte the ones in eval_results/f15_intrinsic/f15_final_v2.py
    run_cell(), so a cell here is directly comparable with
    f15_morphscore_final_v2.json."""
    from morphscore_patched import MorphScore  # noqa: E402

    if tuple(arm_paths) != ARMS:
        raise RuntimeError(f"arm keys {tuple(arm_paths)} != {ARMS}; the cell JSON "
                           f"would not line up with f15_morphscore_final_v2.json")
    g = gold_for(short_lang)
    out = {}
    for arm, path in arm_paths.items():
        t0 = time.time()
        ms = MorphScore(data_dir=g["data_dir"], language_subset=[g["dataset_key"]],
                        by_split=False, freq_scale=True, exclude_single_tok=False,
                        stem_eq_lemma=stem_eq_lemma)
        res = ms.eval(Tokenizer.from_file(str(path)))
        if g["dataset_key"] not in res:
            raise KeyError(f"MorphScore returned no entry for {g['dataset_key']!r} "
                           f"({short_lang}/{arm}); got {sorted(res)}")
        r = res[g["dataset_key"]]
        if "error" in r:
            raise RuntimeError(f"{short_lang}/{arm}: {r['error']}")
        rec = {k: float(r[k]) for k in METRICS}
        rec["num_samples"] = int(r["num_samples"])
        rec["score_seconds"] = round(time.time() - t0, 2)
        out[arm] = rec
        logger.info(f"  {arm:<22} n={rec['num_samples']:>6} "
                    + " ".join(f"{k}={rec[k]:.4f}" for k in METRICS))
    return out, g


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_corpus(args) -> None:
    api = HfApi()
    for short in args.lang.split(","):
        prov = ensure_corpus(short.strip(), api)
        print(json.dumps(prov, indent=2))


def prepare_family(family: str, base_langs, langs_needed, api: HfApi) -> tuple:
    """Corpora + model tokenizer + base tokenizer for one family."""
    cfg = FAMILIES[family]
    if cfg.get("excluded"):
        raise RuntimeError(
            f"{family} is EXCLUDED from this sweep and will not be fit or "
            f"measured.\n  reason: {cfg['excluded']}")
    root = family_root(family, base_langs)
    root.mkdir(parents=True, exist_ok=True)

    corpora = {}
    for l in sorted(set(list(base_langs) + list(langs_needed))):
        corpora[SWEEP_LANGS[l]] = ensure_corpus(l, api)

    reuse = {}
    if cfg["provenance"] == "f15":
        reuse = check_f15_protocol(family, base_langs, corpora)

    mt = ensure_model_tokenizer(family, root)
    _log_paper_V(family, mt)
    base = ensure_base_tokenizer(family, root, base_langs, corpora, mt)
    return root, corpora, mt, base, reuse


def cmd_convert(args) -> None:
    api = HfApi()
    base_langs = parse_base_langs(args.base_langs)
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for family in args.family.split(","):
        family = family.strip()
        if family not in FAMILIES:
            raise RuntimeError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
        rec = {"family": family, **{k: v for k, v in FAMILIES[family].items()},
               "protocol_tag": protocol_tag(base_langs),
               "date": datetime.now(timezone.utc).isoformat(),
               "git_commit": git_commit(), "code_fingerprint": code_fingerprint(),
               "env": env_versions()}
        t0 = time.time()
        try:
            root, corpora, mt, base, reuse = prepare_family(
                family, base_langs, [], api)
            mt_public = {k: v for k, v in mt.items() if not k.startswith("_")}
            rec.update(status="pass", artifact_root=str(root),
                       model_tokenizer=mt_public, base_tokenizer=base,
                       corpora={k: {kk: vv for kk, vv in v.items()}
                                for k, v in corpora.items()},
                       f15_reuse=reuse or None)
        except Exception as e:
            import traceback
            rec.update(status="fail", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())
            logger.error(f"{family}: CONVERSION FAILED: {type(e).__name__}: {e}")
            failures.append(family)
        rec["wall_seconds"] = round(time.time() - t0, 2)
        p = CONV_DIR / f"{family}__{protocol_tag(base_langs)}.json"
        p.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"[{rec['status']}] {family} -> {p}")
        if rec["status"] == "fail":
            print(rec["error"])
    if failures:
        # Exit non-zero so a batch launcher checking $? does not read a failed
        # conversion as a success.
        sys.exit(f"conversion failed for: {', '.join(failures)}")


def _recheck_cell_inputs(cell: dict) -> list:
    """Re-verify the hashes a finished cell recorded. Returns a list of
    human-readable differences (empty == the cell is still valid)."""
    diffs = []
    cur_harness = sha256(HARNESS_PY)
    if cell["harness"]["sha256"] != cur_harness:
        diffs.append(f"harness {HARNESS_PY}: recorded={cell['harness']['sha256']} "
                     f"now={cur_harness}")
    gold_file = cell["gold"]["file"]
    if not Path(gold_file).is_file():
        diffs.append(f"gold {gold_file} no longer exists")
    else:
        cur_gold = sha256(gold_file)
        if cell["gold"]["sha256"] != cur_gold:
            diffs.append(f"gold {gold_file}: recorded={cell['gold']['sha256']} "
                         f"now={cur_gold}")
    for arm, v in cell["arms"].items():
        if not Path(v["path"]).is_file():
            diffs.append(f"arm {arm}: {v['path']} no longer exists")
            continue
        cur = sha256(v["path"])
        if v["sha256"] != cur:
            diffs.append(f"arm {arm} {v['path']}: recorded={v['sha256']} now={cur}")
    return diffs


def cmd_cell(args) -> None:
    api = HfApi()
    base_langs = parse_base_langs(args.base_langs)
    family, short_lang = args.family, args.lang
    if family not in FAMILIES:
        raise RuntimeError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    if short_lang not in SWEEP_LANGS:
        raise RuntimeError(f"unknown language {short_lang!r}; known: {sorted(SWEEP_LANGS)}")

    tag = protocol_tag(base_langs)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (CELL_DIR /
                f"{family}__{SWEEP_LANGS[short_lang]}__stem_eq_lemma-"
                f"{args.stem_eq_lemma}__{tag}.json")
    if out_path.is_file() and not args.force:
        # The filename encodes family/lang/stem_eq_lemma/protocol but NOT the
        # harness, the gold CSV, or the arm tokenizers. Skipping on the filename
        # alone would silently republish stale numbers after any of those change,
        # so re-verify the hashes the cell itself recorded.
        existing = json.loads(out_path.read_text())
        diffs = _recheck_cell_inputs(existing)
        if diffs:
            raise RuntimeError(
                f"{out_path} exists but its recorded inputs no longer match what "
                f"is on disk; the numbers in it are stale. Re-run with --force to "
                f"remeasure.\n  - " + "\n  - ".join(diffs))
        logger.info(f"cell already present and its inputs still verify: {out_path} "
                    f"(pass --force to remeasure)")
        print(out_path)
        return

    t_all = time.time()
    root, corpora, mt, base, reuse = prepare_family(
        family, base_langs, [short_lang], api)
    overlay = ensure_overlay(family, root, short_lang, corpora, mt)

    arm_paths = {
        "Original(BPE)": root / "model_tokenizer" / "tokenizer.json",
        "base_tokenizer.json": root / "base_tokenizer.json",
        "LangMAP(langspec)": Path(overlay["path"]),
    }
    for a, p in arm_paths.items():
        if not Path(p).is_file():
            raise RuntimeError(f"{family}/{short_lang}: arm {a!r} tokenizer {p} missing")

    logger.info(f"scoring {family} / {short_lang} "
                f"(stem_eq_lemma={args.stem_eq_lemma})")
    metrics, gold = score_arms(arm_paths, short_lang, args.stem_eq_lemma)

    cell = {
        "schema": "intrinsic_sweep_v2/cell/1",
        "family": family,
        "model_id": FAMILIES[family]["model_id"],
        "paper_row": FAMILIES[family]["paper_row"],
        "provenance": FAMILIES[family]["provenance"],
        "lang": short_lang,
        "lang_code": SWEEP_LANGS[short_lang],
        "protocol_tag": tag,
        "config": {
            "stem_eq_lemma": args.stem_eq_lemma,
            "morphscore": ("unique_only=True, exclude_numbers=True, "
                           "splits=train/dev/test, exclude_single_morpheme=True, "
                           "exclude_single_tok=False, freq_scale=True, "
                           "by_split=False"),
            "base_langs": base_langs,
            "em_mode": EM_MODE,
            "per_lang_em_seed": PER_LANG_EM_SEED,
            "em_iterations": NUM_FINETUNE_EM_ITERATIONS,
            "lines_per_lang": LINES_PER_LANG,
            "script_filter_threshold": SCRIPT_FILTER_THRESHOLD,
        },
        "harness": {
            "path": str(HARNESS_PY),
            "sha256": sha256(HARNESS_PY),
            "kind": "offsets-patched",
        },
        "gold": {**gold, "sha256": sha256(gold["file"])},
        "artifact_root": str(root),
        "model_tokenizer": {k: v for k, v in mt.items() if not k.startswith("_")},
        "base_tokenizer": base,
        "overlay": overlay,
        "family_notes": {k: v for k, v in FAMILIES[family].items()
                         if k in ("paper_V_note", "embed_rows_gate",
                                  "embed_rows_gate_reason",
                                  "tokenizer_rewritten_by_transformers")} or None,
        "f15_reuse": reuse or None,
        "corpora": corpora,
        "arms": {a: {"path": str(p), "sha256": sha256(p)} for a, p in arm_paths.items()},
        "metrics": metrics,
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "env": env_versions(),
        "wall_seconds": round(time.time() - t_all, 2),
    }
    out_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False))
    logger.info(f"wrote {out_path}")
    print(out_path)


def cmd_backfill_provenance(args) -> None:
    """Write the `<artifact>.provenance.json` sidecar for a base tokenizer that
    was fit by an earlier revision of this driver, BEFORE the sidecar existed.

    The record is not invented: it is reconstructed field-for-field from that
    family's conversion JSON, which already stored the corpus repo/revision/
    sha256/lines and the model-tokenizer sha256. The reconstruction is refused
    unless the conversion JSON's recorded base sha256 still matches the file on
    disk, so a backfilled sidecar can only ever describe the artifact it names.
    """
    base_langs = parse_base_langs(args.base_langs)
    tag = protocol_tag(base_langs)
    for family in args.family.split(","):
        family = family.strip()
        if family not in FAMILIES:
            raise RuntimeError(f"unknown family {family!r}")
        conv = CONV_DIR / f"{family}__{tag}.json"
        if not conv.is_file():
            raise RuntimeError(f"{family}: no conversion record at {conv}; nothing "
                               f"to reconstruct from. Refusing to invent provenance.")
        c = json.loads(conv.read_text())
        if c.get("status") != "pass":
            raise RuntimeError(f"{family}: conversion record status={c.get('status')!r}; "
                               f"refusing to backfill provenance for a failed conversion.")
        base_path = Path(c["base_tokenizer"]["path"])
        if not base_path.is_file():
            raise RuntimeError(f"{family}: {base_path} missing")
        cur = sha256(base_path)
        if cur != c["base_tokenizer"]["sha256"]:
            raise RuntimeError(
                f"{family}: {base_path} sha256 {cur} != the conversion record's "
                f"{c['base_tokenizer']['sha256']}; the file changed after that "
                f"record was written, so the record does not describe it.")
        sidecar = Path(str(base_path) + ".provenance.json")
        if sidecar.is_file() and not args.force:
            print(f"[skip] {sidecar} already exists")
            continue
        rec = {
            "kind": "base_tokenizer",
            "family": family,
            "model_id": c["model_tokenizer"]["model_id"],
            "model_tokenizer_sha256": c["model_tokenizer"]["sha256"],
            "base_langs": c["base_tokenizer"]["base_langs"],
            "corpora": {SWEEP_LANGS[l]: {
                "repo": c["corpora"][SWEEP_LANGS[l]]["repo"],
                "revision": c["corpora"][SWEEP_LANGS[l]]["revision"],
                "sha256": c["corpora"][SWEEP_LANGS[l]]["sha256"],
                "lines_written": c["corpora"][SWEEP_LANGS[l]]["lines_written"],
            } for l in base_langs},
            "em_mode": c["base_tokenizer"]["em_mode"],
            "unk_token_internal": c["base_tokenizer"]["unk_token_internal"],
            "lines_per_lang": LINES_PER_LANG,
            "backfilled_from": str(conv),
            "backfilled_at": datetime.now(timezone.utc).isoformat(),
        }
        write_provenance(base_path, rec)
        print(f"[ok] wrote {sidecar}")


def cmd_manifest(args) -> None:
    cells = sorted(CELL_DIR.glob("*.json")) if CELL_DIR.is_dir() else []
    convs = sorted(CONV_DIR.glob("*.json")) if CONV_DIR.is_dir() else []
    man = {
        "schema": "intrinsic_sweep_v2/manifest/1",
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "driver": "scripts/sweep_intrinsic_v2.py",
        "families": FAMILIES,
        "langs": SWEEP_LANGS,
        "paper_table_corrections": PAPER_TABLE_CORRECTIONS,
        "excluded_families": {f: FAMILIES[f]["excluded"] for f in FAMILIES
                              if FAMILIES[f].get("excluded")},
        "not_paper_rows": [f for f in FAMILIES if FAMILIES[f]["paper_row"] is None],
        "conversions": {},
        "cells": {},
    }
    for p in convs:
        c = json.loads(p.read_text())
        man["conversions"][p.name] = {
            "family": c["family"], "status": c["status"],
            "error": c.get("error"),
            "base_sha256": (c.get("base_tokenizer") or {}).get("sha256"),
            "wall_seconds": c.get("wall_seconds"),
            # A record can outlive its family: gemma2-2b was converted while the
            # paper's Gemma row was still ambiguous, then dropped once the
            # fingerprint match identified the row as Gemma 3. The artifact and
            # its record are kept; the family is no longer part of the sweep.
            "family_no_longer_in_sweep": c["family"] not in FAMILIES,
        }
    for p in cells:
        c = json.loads(p.read_text())
        man["cells"][p.name] = {
            "family": c["family"], "lang": c["lang"],
            "stem_eq_lemma": c["config"]["stem_eq_lemma"],
            "protocol_tag": c["protocol_tag"],
            "harness_sha256": c["harness"]["sha256"],
            "gold_sha256": c["gold"]["sha256"],
            "arm_sha256": {a: v["sha256"] for a, v in c["arms"].items()},
            "metrics": c["metrics"],
        }
    # Coverage is keyed by (protocol_tag, stem_eq_lemma) as well as family: a
    # cell fit under different base-langs or a different gold filter is NOT
    # coverage of the same grid square.
    grid = {}
    for v in man["cells"].values():
        key = f"{v['protocol_tag']}|stem_eq_lemma={v['stem_eq_lemma']}"
        grid.setdefault(key, {}).setdefault(v["family"], set()).add(v["lang"])
    man["coverage"] = {k: {f: sorted(ls) for f, ls in fam.items()}
                       for k, fam in grid.items()}
    man["missing"] = {
        k: {f: sorted(set(SWEEP_LANGS) - set(fam.get(f, [])))
            for f in PAPER_GRID
            if sorted(set(SWEEP_LANGS) - set(fam.get(f, [])))}
        for k, fam in grid.items()
    }
    man["paper_grid"] = {"families": PAPER_GRID, "langs": sorted(SWEEP_LANGS),
                         "n_cells": len(PAPER_GRID) * len(SWEEP_LANGS)}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    p = OUT_ROOT / "manifest.json"
    p.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"wrote {p}: {len(man['cells'])} cells, {len(man['conversions'])} conversions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_base_langs(p):
        p.add_argument("--base-langs", required=True,
                       help="comma-separated sweep languages whose combined corpus "
                            "the BASE Unigram conversion is fit on. No default: it "
                            "changes every number and is encoded in the artifact path.")

    p = sub.add_parser("corpus", help="build/verify a language corpus only")
    p.add_argument("--lang", required=True, help="comma-separated sweep languages")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("convert", help="base-Unigram conversion + gates, no scoring")
    p.add_argument("--family", required=True, help="comma-separated family keys")
    add_base_langs(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("cell", help="fit-if-missing then score one (family, lang)")
    p.add_argument("--family", required=True)
    p.add_argument("--lang", required=True)
    add_base_langs(p)
    p.add_argument("--stem-eq-lemma", dest="stem_eq_lemma", action="store_true",
                   default=False,
                   help="gold filter: keep only rows whose stem == lemma. "
                        "Default False.")
    p.add_argument("--force", action="store_true",
                   help="re-score even if the cell JSON already exists")
    p.set_defaults(func=cmd_cell)

    p = sub.add_parser("backfill-provenance",
                       help="reconstruct a base tokenizer's .provenance.json "
                            "from its conversion record (for artifacts fit "
                            "before the sidecar existed)")
    p.add_argument("--family", required=True, help="comma-separated family keys")
    add_base_langs(p)
    p.add_argument("--force", action="store_true", help="overwrite an existing sidecar")
    p.set_defaults(func=cmd_backfill_provenance)

    p = sub.add_parser("manifest", help="aggregate finished cells + conversions")
    p.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
