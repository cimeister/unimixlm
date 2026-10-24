"""Paired bootstrap CIs on the v2 tab:morphscore deltas (LangMAP(langspec) minus
Original(BPE)), stem_eq_lemma=False config, for all 10 (family, language) pairs.

v2 differences from f15_morphscore_bootstrap.py:
  * harness = harness_patched/morphscore_patched.py (boundaries and per-token
    strings from tokenizers.Encoding.offsets, not per-id decoding);
  * Arabic uses arabic_data_repaired.csv (undiacritized lemma re-located inside
    the wordform), aligned subset only;
  * self-check reference is f15_morphscore_final_v2.json.

Everything else is unchanged from v1, deliberately: same MorphScore config, same
per-word arrays, same aggregate() formulas lifted from
MorphScore.get_morphscore, same resample unit (word type, paired across arms),
N_BOOT=2000, SEED=1234, 95% percentile interval, and the same RNG consumption
order (a fresh set of 2000 index draws per metric, in METRICS order) so the CIs
are directly comparable to v1's.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness_patched"))
from morphscore_patched import MorphScore  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import os

def _env_dir(var, what):
    v = os.environ.get(var)
    if v is None:
        raise RuntimeError(f"{var} is not set; point it at {what}.")
    return v


REPO = HERE.parents[1]
DATA_DIR = _env_dir("MORPHSCORE_DATA_DIR", "the MorphScore per-language *_data.csv files")
F15 = REPO / "las_tokenizers" / "f15"
FINAL_JSON = HERE / "f15_morphscore_final_v2.json"
OUT = HERE / "f15_morphscore_bootstrap_v2.json"
PARTS = Path(_env_dir("BOOTSTRAP_PARTS_DIR", "a scratch directory for per-language bootstrap part files"))

LANGS = {"tur": "tur_Latn", "fin": "fin_Latn", "hun": "hun_Latn",
         "arb": "arb_Arab", "hin": "hin_Deva"}
FAMS = ["qwen2_5-0_5b", "llama3.2-1b"]
METRICS = ["micro_f1", "macro_f1", "morphscore_recall", "morphscore_precision"]

N_BOOT = 2000
SEED = 1234
CI_LO_PCT, CI_HI_PCT = 2.5, 97.5  # 95% percentile interval
SELF_CHECK_TOL = 1e-12

MS_CONFIG = dict(by_split=False, freq_scale=True, exclude_single_tok=False,
                 stem_eq_lemma=False)

# per-language (data_dir, dataset key passed to MorphScore)
DATASET = {lang: (DATA_DIR, flores) for lang, flores in LANGS.items()}
DATASET["arb"] = (str(HERE), "arabic_data_repaired.csv")


def harm(a, b):
    """Verbatim guard from morphscore.py get_morphscore (micro_f1/macro_f1
    branches)."""
    return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


def aggregate(pr, pp, m, g, p, w):
    """Recompute the four tab:morphscore aggregates from per-word arrays,
    using the exact aggregate formulas in MorphScore.get_morphscore."""
    mean_recall = float(np.sum(pr * w) / np.sum(w))
    mean_precision = float(np.sum(pp * w) / np.sum(w))

    n_matched, n_gold, n_pred = float(np.sum(m)), float(np.sum(g)), float(np.sum(p))
    micro_precision = n_matched / n_pred
    micro_recall = n_matched / n_gold
    micro_f1 = harm(micro_precision, micro_recall)

    macro_precision = float(np.mean(m / p))
    macro_recall = float(np.mean(m / g))
    macro_f1 = harm(macro_precision, macro_recall)

    return {
        "morphscore_recall": mean_recall,
        "morphscore_precision": mean_precision,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def load_tok(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(f"missing tokenizer file: {path}")
    return Tokenizer.from_file(str(path))


def extract_arrays(df):
    return dict(
        pr=df["morphscore_recall"].to_numpy(dtype=np.float64),
        pp=df["morphscore_precision"].to_numpy(dtype=np.float64),
        m=df["matched_subwords"].to_numpy(dtype=np.float64),
        g=df["gold_subwords"].to_numpy(dtype=np.float64),
        p=df["pred_subwords"].to_numpy(dtype=np.float64),
        w=df["word_freq_norm"].to_numpy(dtype=np.float64),
        wordform=df["wordform"].to_numpy(),
        index=df.index.to_numpy(),
    )


def run_cell(fam, lang, final_ref):
    flores = LANGS[lang]
    ddir, dsname = DATASET[lang]
    fam_dir = F15 / fam
    tok_orig = load_tok(fam_dir / "model_tokenizer" / "tokenizer.json")
    tok_lang = load_tok(fam_dir / lang / f"langspec_{flores}.json")

    ms_o = MorphScore(data_dir=ddir, language_subset=[dsname], **MS_CONFIG)
    res_o, df_o = ms_o.eval(tok_orig, return_df=True)
    ms_l = MorphScore(data_dir=ddir, language_subset=[dsname], **MS_CONFIG)
    res_l, df_l = ms_l.eval(tok_lang, return_df=True)

    arr_o = extract_arrays(df_o)
    arr_l = extract_arrays(df_l)

    key_o = f"stem_eq_lemma=False|{fam}|{flores}|Original(BPE)"
    key_l = f"stem_eq_lemma=False|{fam}|{flores}|LangMAP(langspec)"
    if key_o not in final_ref or key_l not in final_ref:
        raise KeyError(f"missing reference keys {key_o!r} / {key_l!r} in {FINAL_JSON}")

    # ---- self-check A: each arm's own un-resampled aggregates must bit-match
    # f15_morphscore_final_v2.json. This holds whether or not the two arms
    # retain the same rows, so it is the check that always applies.
    agg_o_full = aggregate(arr_o["pr"], arr_o["pp"], arr_o["m"], arr_o["g"], arr_o["p"], arr_o["w"])
    agg_l_full = aggregate(arr_l["pr"], arr_l["pp"], arr_l["m"], arr_l["g"], arr_l["p"], arr_l["w"])
    for label, key, agg, arr in (("Original(BPE)", key_o, agg_o_full, arr_o),
                                 ("LangMAP(langspec)", key_l, agg_l_full, arr_l)):
        if final_ref[key]["num_samples"] != len(arr["wordform"]):
            raise AssertionError(
                f"{fam}/{flores}/{label}: n mismatch vs {FINAL_JSON.name} "
                f"({final_ref[key]['num_samples']} vs {len(arr['wordform'])})")
        for metric in METRICS:
            d = abs(agg[metric] - final_ref[key][metric])
            if d > SELF_CHECK_TOL:
                raise AssertionError(
                    f"SELF-CHECK FAILED {fam}/{flores}/{label}/{metric}: "
                    f"recomputed={agg[metric]!r} vs reference={final_ref[key][metric]!r} "
                    f"(|diff|={d:.3e} > {SELF_CHECK_TOL:.0e})")

    # ---- resample set: the rows both arms retain.
    # For every cell except Arabic-on-repaired-gold the two arms retain exactly
    # the same rows and this is a no-op. Arabic-on-repaired-gold is the one gold
    # that contains single-morpheme rows (lemma == wordform, 35% of the aligned
    # subset); under exclude_single_tok=False such a row is KEPT when the
    # tokenizer emits one token for it and DROPPED (NaN) when it emits more, so
    # the retained row set is tokenizer-dependent there. The paired bootstrap
    # needs a common row set, so it runs on the intersection; the full-set delta
    # from the v2 table is carried alongside as delta_full_table.
    same_rows = (len(arr_o["index"]) == len(arr_l["index"])
                 and np.array_equal(arr_o["index"], arr_l["index"]))
    if same_rows:
        if not np.array_equal(arr_o["wordform"], arr_l["wordform"]):
            raise RuntimeError(f"{fam}/{flores}: wordform sequence differs between arms")
        if not np.allclose(arr_o["w"], arr_l["w"]):
            raise RuntimeError(f"{fam}/{flores}: per-word weights differ between arms")
        sel_o = sel_l = slice(None)
        subset = "full"
    else:
        common = np.intersect1d(arr_o["index"], arr_l["index"])
        sel_o = np.searchsorted(arr_o["index"], common)
        sel_l = np.searchsorted(arr_l["index"], common)
        if not (np.array_equal(arr_o["index"][sel_o], common)
                and np.array_equal(arr_l["index"][sel_l], common)):
            raise RuntimeError(f"{fam}/{flores}: index intersection alignment failed")
        if not np.array_equal(arr_o["wordform"][sel_o], arr_l["wordform"][sel_l]):
            raise RuntimeError(f"{fam}/{flores}: wordform sequence differs on the intersection")
        if not np.allclose(arr_o["w"][sel_o], arr_l["w"][sel_l]):
            raise RuntimeError(f"{fam}/{flores}: per-word weights differ on the intersection")
        subset = "intersection"

    bo = {k: arr_o[k][sel_o] for k in ("pr", "pp", "m", "g", "p", "w")}
    bl = {k: arr_l[k][sel_l] for k in ("pr", "pp", "m", "g", "p", "w")}
    n = len(bo["pr"])

    agg_o = aggregate(bo["pr"], bo["pp"], bo["m"], bo["g"], bo["p"], bo["w"])
    agg_l = aggregate(bl["pr"], bl["pp"], bl["m"], bl["g"], bl["p"], bl["w"])

    rng = np.random.default_rng(SEED)
    pair_out = {}
    rows = []
    for metric in METRICS:
        delta_here = agg_l[metric] - agg_o[metric]
        delta_full = final_ref[key_l][metric] - final_ref[key_o][metric]
        if subset == "full":
            # ---- self-check B: on a common row set the un-resampled delta must
            # bit-match the v2 table.
            diff = abs(delta_here - delta_full)
            if diff > SELF_CHECK_TOL:
                raise AssertionError(
                    f"SELF-CHECK FAILED {fam}/{flores}/{metric}: "
                    f"recomputed delta={delta_here!r} vs reference delta={delta_full!r} "
                    f"(|diff|={diff:.3e} > {SELF_CHECK_TOL:.0e})")

        boot_deltas = np.empty(N_BOOT, dtype=np.float64)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            ao = aggregate(bo["pr"][idx], bo["pp"][idx], bo["m"][idx],
                           bo["g"][idx], bo["p"][idx], bo["w"][idx])
            al = aggregate(bl["pr"][idx], bl["pp"][idx], bl["m"][idx],
                           bl["g"][idx], bl["p"][idx], bl["w"][idx])
            boot_deltas[b] = al[metric] - ao[metric]

        ci_low, ci_high = np.percentile(boot_deltas, [CI_LO_PCT, CI_HI_PCT])
        pair_out[metric] = {
            "delta": delta_here,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_boot": N_BOOT,
            "n_samples": n,
        }
        if subset != "full":
            pair_out[metric]["delta_full_table"] = delta_full
        rows.append((fam, flores, metric, delta_here, ci_low, ci_high,
                     (ci_low > 0) or (ci_high < 0)))
    pair_out["resample_subset"] = subset
    pair_out["n_rows_original_arm"] = int(len(arr_o["index"]))
    pair_out["n_rows_langmap_arm"] = int(len(arr_l["index"]))
    if lang == "arb":
        pair_out["gold"] = "repaired"
    return {f"{fam}|{flores}": pair_out}, rows


def metadata():
    return {
        "reference_arm": "Original(BPE)",
        "target_arm": "LangMAP(langspec)",
        "harness": "offsets-patched",
        "harness_path": str(HERE / "harness_patched" / "morphscore_patched.py"),
        "metric_config": "stem_eq_lemma=False, exclude_single_tok=False, freq_scale=True, by_split=False",
        "resample_unit": "word type (row), paired across arms, with replacement",
        "n_boot": N_BOOT,
        "seed": SEED,
        "rng_note": ("a fresh set of N_BOOT index draws is consumed per metric, in "
                     "METRICS order, per (family, language) cell -- identical to v1"),
        "ci_method": "percentile",
        "ci_level": 0.95,
        "self_check_tolerance": SELF_CHECK_TOL,
        "self_check_passed": True,
        "self_check_a": ("each arm's un-resampled aggregates bit-match "
                         "f15_morphscore_final_v2.json (all cells)"),
        "self_check_b": ("un-resampled delta bit-matches f15_morphscore_final_v2.json "
                         "on every cell whose two arms retain the same rows "
                         "(all cells except arb_Arab)"),
        "arb_gold": "repaired (arabic_data_repaired.csv, aligned subset)",
        "arb_note": ("arb_Arab is the only cell where the two arms retain different "
                     "row sets: the repaired gold contains single-morpheme rows and, "
                     "under exclude_single_tok=False, such a row is kept when the "
                     "tokenizer emits one token and dropped when it emits more. The "
                     "paired bootstrap therefore runs on the intersection of the two "
                     "arms' retained rows; 'delta' is the intersection delta (the one "
                     "the CI brackets) and 'delta_full_table' is the full-set delta "
                     "reported in f15_morphscore_final_v2.json"),
        "source_final_json": str(FINAL_JSON),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", help="fam:lang")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        out = {}
        rows = []
        parts = sorted(glob.glob(str(PARTS / "*.json")))
        expected = len(FAMS) * len(LANGS)
        if len(parts) != expected:
            raise RuntimeError(f"expected {expected} part files in {PARTS}, found {len(parts)}")
        for p in parts:
            out.update(json.loads(Path(p).read_text()))
        out["_metadata"] = metadata()
        OUT.write_text(json.dumps(out, indent=2))
        print("wrote", OUT)
        print(f"\n{'family':<14}{'lang':<10}{'metric':<22}{'delta':>10}  {'95% CI':>22}  excl.0")
        for fam in FAMS:
            for lang, flores in LANGS.items():
                cell = out[f"{fam}|{flores}"]
                for metric in METRICS:
                    v = cell[metric]
                    excl = (v["ci_low"] > 0) or (v["ci_high"] < 0)
                    print(f"{fam:<14}{flores:<10}{metric:<22}{v['delta']:>+10.4f}  "
                          f"[{v['ci_low']:>+8.4f}, {v['ci_high']:>+8.4f}]  "
                          f"{'yes' if excl else 'no'}")
        return

    if not FINAL_JSON.is_file():
        raise FileNotFoundError(f"missing reference file: {FINAL_JSON}")
    final_ref = json.loads(FINAL_JSON.read_text())

    t0 = time.time()
    if args.cell:
        fam, lang = args.cell.split(":")
        res, rows = run_cell(fam, lang, final_ref)
        PARTS.mkdir(parents=True, exist_ok=True)
        (PARTS / f"{fam}__{lang}.json").write_text(json.dumps(res, indent=2))
        for r in rows:
            print(f"{r[0]:<14}{r[1]:<10}{r[2]:<22}{r[3]:>+10.4f}  "
                  f"[{r[4]:>+8.4f}, {r[5]:>+8.4f}]  {'yes' if r[6] else 'no'}")
        print(f"wrote part {PARTS / f'{fam}__{lang}.json'} ({time.time()-t0:.1f}s)")
        return

    out = {}
    for fam in FAMS:
        for lang in LANGS:
            res, rows = run_cell(fam, lang, final_ref)
            out.update(res)
            print(f"done {fam} {lang} ({time.time()-t0:.1f}s)", flush=True)
    out["_metadata"] = metadata()
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
