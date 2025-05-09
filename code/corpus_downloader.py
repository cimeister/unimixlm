import os
import json 
import random
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# language selection & caps  (matches the proposal in the previous answer)
# ---------------------------------------------------------------------------
_DATASET_CODE = {
    "zh":  "zh-Hans",        # default to Simplified
    # add more aliases here if needed
}
_LANG2TIER: Dict[str, str] = {
    # high
    "en":"high","zh":"high","es":"high","ar":"high","ru":"high","de":"high",
    # medium
    "hi":"med","tr":"med","id":"med","vi":"med","pl":"med","sw":"med","el":"med",
    # low
    "cy":"low","ga":"low","ku":"low","mg":"low","yo":"low","gn":"low",
}

_TIER_CAP_GB = {"high": 15, "med": 5, "low": None}   # None = keep all
_VAL_FRAC  = 0.001
_TEST_FRAC = 0.001
_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
def _ds_code(lang: str) -> str:
    """Return the code understood by the cc100 dataset loader."""
    return _DATASET_CODE.get(lang, lang)

def _cap_bytes(lang: str) -> Optional[int]:
    tier = _LANG2TIER[lang]
    gb = _TIER_CAP_GB[tier]
    return None if gb is None else int(gb * 1024**3)

def _write_splits(lang: str, out_dir: Path, cap: Optional[int]) -> None:
    ds_code = _ds_code(lang)
    ds = load_dataset("cc100", ds_code, split="train", streaming=True)
    random.seed(_RANDOM_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = os.path.join(out_dir, "train.txt")
    val_file = os.path.join(out_dir, "val.txt")
    test_file = os.path.join(out_dir, "test.txt")
    f_train = open(train_file, "w", encoding="utf-8")
    f_val = open(val_file, "w", encoding="utf-8")
    f_test = open(test_file, "w", encoding="utf-8")

    written_bytes = 0
    for ex in ds:
        line = ex["text"].strip() + "\n"
        if cap is not None and written_bytes + len(line.encode("utf-8")) > cap:
            break

        r = random.random()
        if r < _VAL_FRAC:
            f_val.write(line)
        elif r < _VAL_FRAC + _TEST_FRAC:
            f_test.write(line)
        else:
            f_train.write(line)

        written_bytes += len(line.encode("utf-8"))

    for fh in (f_train, f_val, f_test):
        fh.close()

    meta = {
        "bytes_written": written_bytes,
        "val_frac": _VAL_FRAC,
        "test_frac": _TEST_FRAC,
    }
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    print(f"{lang}: {written_bytes/1e9:.2f} GB written to {out_dir}")
    return {'train': train_file,
            'val': val_file,
            'test': test_file
            }

# ---------------------------------------------------------------------------
def download_corpus(config: Dict) -> None:
    """Entry‑point used by run_experiment.py."""
    langs: List[str] = config["corpus"].get("languages", [])
    if not langs:
        langs = list(_LANG2TIER.keys())

    data_dir = Path(config["corpus"]["processed_dir"].format(data_dir=config["experiment"]["data_dir"]))
    data_dir.mkdir(parents=True, exist_ok=True)

    missing = [lg for lg in langs if lg not in _LANG2TIER]
    if missing:
        raise ValueError(f"Languages not configured for download: {missing}")

    corpus_info = {}
    for lg in langs:
        out_dir = data_dir / lg
        if (out_dir / "train.txt").exists():
            print(f"{lg}: already downloaded, skipping")
            corpus_info[lg] = {'splits': 
                               {'train': os.path.join(out_dir, "train.txt"),
                                'val': os.path.join(out_dir, "val.txt"),
                                'test': os.path.join(out_dir, "test.txt")
                                }
                               }
            continue
        cap = _cap_bytes(lg)
        corpus_info[lg]['splits'] = _write_splits(lg, out_dir, cap)
