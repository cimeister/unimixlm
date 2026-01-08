"""
Downloading Corpora for Aya-Expanse 8B Languages
"""

#!/usr/bin/env python3
import subprocess
import os
from collections import defaultdict

# Path to corpus_downloader.py in the same directory as this script
SCRIPT_DIR = os.path.expanduser("~")  # home directory
CORPUS_DOWNLOADER = os.path.join(SCRIPT_DIR, "corpus_downloader.py")

# === Define HR and LR languages per script ===
aya_languages = {
    "ar": {"script": "Arabic", "resource": "HR"},
    "fa": {"script": "Arabic", "resource": "LR"},
    "zh-CN": {"script": "Han", "resource": "HR"},
    "zh-TW": {"script": "Han", "resource": "HR"},
    "cs": {"script": "Latin", "resource": "HR"},
    "ro": {"script": "Latin", "resource": "LR"},
    "tr": {"script": "Latin", "resource": "LR"},
    "vi": {"script": "Latin", "resource": "LR"},
    "nl": {"script": "Latin", "resource": "HR"},
    "en": {"script": "Latin", "resource": "HR"},
    "fr": {"script": "Latin", "resource": "HR"},
    "de": {"script": "Latin", "resource": "HR"},
    "id": {"script": "Latin", "resource": "HR"},
    "it": {"script": "Latin", "resource": "HR"},
    "pl": {"script": "Latin", "resource": "HR"},
    "pt": {"script": "Latin", "resource": "HR"},
    "es": {"script": "Latin", "resource": "HR"},
    "ru": {"script": "Cyrillic", "resource": "HR"},
    "uk": {"script": "Cyrillic", "resource": "LR"},
    "el": {"script": "Greek", "resource": "LR"},
    "he": {"script": "Hebrew", "resource": "LR"},
    "hi": {"script": "Devanagari", "resource": "LR"},
    "ja": {"script": "Kanji/Kana", "resource": "HR"},
    "ko": {"script": "Hangul", "resource": "HR"},
}

# === Group languages by script and resource ===
script_groups = defaultdict(lambda: {"HR": [], "LR": []})
for lang, info in aya_languages.items():
    script_groups[info["script"]][info["resource"]].append(lang)

# === Create HR-LR pairs ===
nested_pairs = []
for groups in script_groups.values():
    for hr in groups["HR"]:
        for lr in groups["LR"]:
            nested_pairs.append((hr, lr))

# === Loop over each pair and run corpus_downloader.py ===
for hr, lr in nested_pairs:
    print(f"Downloading corpus for pair: {hr} (HR) and {lr} (LR)")

    subprocess.run([
        "python3",
        CORPUS_DOWNLOADER,
        "--dataset", "cc100",
        "--lang_set", f"{hr},{lr}",
        "--output_dir", os.path.join(SCRIPT_DIR, "data"),
        "--total_size_gb", "0.1",
        "--val_frac", "0.1",
        "--test_frac", "0.1",
        "--seed", "42",
        "--main_sampling_temp", "0.7",
        "--min_lines_floor", "50000"
    ], check=True)

    print(f"Finished downloading {hr},{lr}\n")
