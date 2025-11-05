#!/usr/bin/env python3
"""
Refactored script to download and prepare multilingual corpora (CC100, FineWeb-2)
and parallel data (FLORES+) for language model and tokenizer training.

This script implements:
1.  Diverse language sets (20, 60, 100 languages) present in all corpora.
2.  Support for 'cc100' and 'HuggingFaceFW/fineweb-2' datasets.
    - Special case: Loads English from 'HuggingFaceFW/fineweb' (v1)
3.  Temperature-based sampling for the main corpus (--main_sampling_temp).
4.  A 50% cap for any single language.
5.  A minimum line floor (--min_lines_floor) for train/tokenizer data.
6.  Temperature-based sampling for the tokenizer corpus (--tokenizer_sampling_temp).
7.  Correct download and structuring of FLORES+ dev and devtest splits.
8.  A modular, class-based structure for maintainability.
"""

import os
import json
import random
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset
from datasets import logging as datasets_logging

# ---------------------------------------------------------------------------
# --- Configuration & Constants
# ---------------------------------------------------------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# --- Default Splitting and Capping ---
from code.constants import MAX_VAL_TEST_SIZE

DEFAULT_VAL_FRAC = 0.005  # 0.5% for validation
DEFAULT_TEST_FRAC = 0.005 # 0.5% for test
DEFAULT_SEED = 42
XLM_R_TEMP = 5.0 # Default temperature for main corpus sampling

# --- Language Metadata ---
# This is the central source of truth for all languages.
# Sizes are estimates in GB based on search results.
# FineWeb-2: Total ~20,200 GB / ~5B docs = ~0.00000404 GB/row
FINEWEB_GB_PER_ROW = 0.00000404

# ---
# UPDATED:
# - 'zh' flores_code changed from 'cmn_Hans' to 'zho_Hans'.
# - 'ko' flores_code changed from 'kor_Kore' to 'kor_Hang'.
# These now match the 'facebook/flores' 'all' config.
# ---
LANGUAGE_METADATA = {
    # Code: { name, script, family, flores_code, cc100_code, cc100_est_gb, fineweb_code (for fineweb-2), fineweb_est_gb }
    # High-Resource, Indo-European
    'en': {'name': 'English', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'eng_Latn', 'cc100_code': 'en', 'cc100_est_gb': 82.0, 
           'fineweb_code': 'N/A (uses fineweb v1)', 'fineweb_est_gb': 10000.0, 'fineweb_v1_est_gb': 10000.0},
    'de': {'name': 'German', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'deu_Latn', 'cc100_code': 'de', 'cc100_est_gb': 18.0, 'fineweb_code': 'deu_Latn', 'fineweb_est_gb': 264.0},
    'fr': {'name': 'French', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'fra_Latn', 'cc100_code': 'fr', 'cc100_est_gb': 14.0, 'fineweb_code': 'fra_Latn', 'fineweb_est_gb': 188.0},
    'es': {'name': 'Spanish', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'spa_Latn', 'cc100_code': 'es', 'cc100_est_gb': 14.0, 'fineweb_code': 'spa_Latn', 'fineweb_est_gb': 190.0},
    'ru': {'name': 'Russian', 'script': 'Cyrl', 'family': 'Indo-European', 'flores_code': 'rus_Cyrl', 'cc100_code': 'ru', 'cc100_est_gb': 46.0, 'fineweb_code': 'rus_Cyrl', 'fineweb_est_gb': 404.0},
    'pt': {'name': 'Portuguese', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'por_Latn', 'cc100_code': 'pt', 'cc100_est_gb': 13.0, 'fineweb_code': 'por_Latn', 'fineweb_est_gb': 117.0},
    'it': {'name': 'Italian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'ita_Latn', 'cc100_code': 'it', 'cc100_est_gb': 7.8, 'fineweb_code': 'ita_Latn', 'fineweb_est_gb': 80.0},
    'pl': {'name': 'Polish', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'pol_Latn', 'cc100_code': 'pl', 'cc100_est_gb': 12.0, 'fineweb_code': 'pol_Latn', 'fineweb_est_gb': 80.0},
    'nl': {'name': 'Dutch', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'nld_Latn', 'cc100_code': 'nl', 'cc100_est_gb': 7.9, 'fineweb_code': 'nld_Latn', 'fineweb_est_gb': 60.0},
    
    # High-Resource, Other Families
    'zh': {'name': 'Chinese (Hans/Hani)', 'script': 'Hans', 'family': 'Sino-Tibetan', 'flores_code': 'cmn_Hans', 'cc100_code': 'zh-Hans', 'cc100_est_gb': 14.0, 'fineweb_code': 'cmn_Hani', 'fineweb_est_gb': 161.0},
    'ja': {'name': 'Japanese', 'script': 'Jpan', 'family': 'Japonic', 'flores_code': 'jpn_Jpan', 'cc100_code': 'ja', 'cc100_est_gb': 15.0, 'fineweb_code': 'jpn_Jpan', 'fineweb_est_gb': 121.0},
    'ar': {'name': 'Arabic', 'script': 'Arab', 'family': 'Afro-Asiatic', 'flores_code': 'arb_Arab', 'cc100_code': 'ar', 'cc100_est_gb': 5.4, 'fineweb_code': 'arb_Arab', 'fineweb_est_gb': 250.0},
    'ko': {'name': 'Korean', 'script': 'Kore', 'family': 'Koreanic', 'flores_code': 'kor_Hang', 'cc100_code': 'ko', 'cc100_est_gb': 14.0, 'fineweb_code': 'kor_Hang', 'fineweb_est_gb': 30.0},

    # Medium-Resource, Diverse Families
    'id': {'name': 'Indonesian', 'script': 'Latn', 'family': 'Austronesian', 'flores_code': 'ind_Latn', 'cc100_code': 'id', 'cc100_est_gb': 3.6, 'fineweb_code': 'ind_Latn', 'fineweb_est_gb': 20.0},
    'vi': {'name': 'Vietnamese', 'script': 'Latn', 'family': 'Austroasiatic', 'flores_code': 'vie_Latn', 'cc100_code': 'vi', 'cc100_est_gb': 2.8, 'fineweb_code': 'vie_Latn', 'fineweb_est_gb': 28.0},
    'tr': {'name': 'Turkish', 'script': 'Latn', 'family': 'Turkic', 'flores_code': 'tur_Latn', 'cc100_code': 'tr', 'cc100_est_gb': 5.4, 'fineweb_code': 'tur_Latn', 'fineweb_est_gb': 32.0},
    'hi': {'name': 'Hindi', 'script': 'Deva', 'family': 'Indo-European', 'flores_code': 'hin_Deva', 'cc100_code': 'hi', 'cc100_est_gb': 2.5, 'fineweb_code': 'hin_Deva', 'fineweb_est_gb': 16.0},
    'fa': {'name': 'Persian', 'script': 'Arab', 'family': 'Indo-European', 'flores_code': 'fas_Arab', 'cc100_code': 'fa', 'cc100_est_gb': 2.0, 'fineweb_code': 'fas_Arab', 'fineweb_est_gb': 16.0},
    'sw': {'name': 'Swahili', 'script': 'Latn', 'family': 'Niger-Congo', 'flores_code': 'swh_Latn', 'cc100_code': 'sw', 'cc100_est_gb': 0.33, 'fineweb_code': 'swh_Latn', 'fineweb_est_gb': 1.6},
    'el': {'name': 'Greek', 'script': 'Grek', 'family': 'Indo-European', 'flores_code': 'ell_Grek', 'cc100_code': 'el', 'cc100_est_gb': 7.4, 'fineweb_code': 'ell_Grek', 'fineweb_est_gb': 28.0},
    'bn': {'name': 'Bengali', 'script': 'Beng', 'family': 'Indo-European', 'flores_code': 'ben_Beng', 'cc100_code': 'bn', 'cc100_est_gb': 0.86, 'fineweb_code': 'ben_Beng', 'fineweb_est_gb': 61.0},
    'th': {'name': 'Thai', 'script': 'Thai', 'family': 'Kra-Dai', 'flores_code': 'tha_Thai', 'cc100_code': 'th', 'cc100_est_gb': 8.7, 'fineweb_code': 'tha_Thai', 'fineweb_est_gb': 20.0},
    'he': {'name': 'Hebrew', 'script': 'Hebr', 'family': 'Afro-Asiatic', 'flores_code': 'heb_Hebr', 'cc100_code': 'he', 'cc100_est_gb': 6.1, 'fineweb_code': 'heb_Hebr', 'fineweb_est_gb': 13.7},

    # Low-Resource, Diverse Families
    'yo': {'name': 'Yoruba', 'script': 'Latn', 'family': 'Niger-Congo', 'flores_code': 'yor_Latn', 'cc100_code': 'yo', 'cc100_est_gb': 0.011, 'fineweb_code': 'yor_Latn', 'fineweb_est_gb': 0.04},
    'cy': {'name': 'Welsh', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'cym_Latn', 'cc100_code': 'cy', 'cc100_est_gb': 0.179, 'fineweb_code': 'cym_Latn', 'fineweb_est_gb': 0.2},
    'ga': {'name': 'Irish', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'gle_Latn', 'cc100_code': 'ga', 'cc100_est_gb': 0.108, 'fineweb_code': 'gle_Latn', 'fineweb_est_gb': 0.08},
    'ku': {'name': 'Kurdish (Kurmanji)', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'kmr_Latn', 'cc100_code': 'ku', 'cc100_est_gb': 0.09, 'fineweb_code': 'kmr_Latn', 'fineweb_est_gb': 0.08},
    'mg': {'name': 'Malagasy', 'script': 'Latn', 'family': 'Austronesian', 'flores_code': 'plt_Latn', 'cc100_code': 'mg', 'cc100_est_gb': 0.029, 'fineweb_code': 'plt_Latn', 'fineweb_est_gb': 0.012},
    'gn': {'name': 'Guarani', 'script': 'Latn', 'family': 'Tupian', 'flores_code': 'gug_Latn', 'cc100_code': 'gn', 'cc100_est_gb': 0.0015, 'fineweb_code': 'gug_Latn', 'fineweb_est_gb': 0.004},
    
    # --- Additions for L60/L100 (Corrected fineweb_code) ---
    'uk': {'name': 'Ukrainian', 'script': 'Cyrl', 'family': 'Indo-European', 'flores_code': 'ukr_Cyrl', 'cc100_code': 'uk', 'cc100_est_gb': 1.4, 'fineweb_code': 'ukr_Cyrl', 'fineweb_est_gb': 28.0},
    'ro': {'name': 'Romanian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'ron_Latn', 'cc100_code': 'ro', 'cc100_est_gb': 1.6, 'fineweb_code': 'ron_Latn', 'fineweb_est_gb': 32.0},
    'cs': {'name': 'Czech', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'ces_Latn', 'cc100_code': 'cs', 'cc100_est_gb': 4.4, 'fineweb_code': 'ces_Latn', 'fineweb_est_gb': 40.0},
    'hu': {'name': 'Hungarian', 'script': 'Latn', 'family': 'Uralic', 'flores_code': 'hun_Latn', 'cc100_code': 'hu', 'cc100_est_gb': 1.5, 'fineweb_code': 'hun_Latn', 'fineweb_est_gb': 24.0},
    'sv': {'name': 'Swedish', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'swe_Latn', 'cc100_code': 'sv', 'cc100_est_gb': 2.1, 'fineweb_code': 'swe_Latn', 'fineweb_est_gb': 40.0},
    'fi': {'name': 'Finnish', 'script': 'Latn', 'family': 'Uralic', 'flores_code': 'fin_Latn', 'cc100_code': 'fi', 'cc100_est_gb': 1.5, 'fineweb_code': 'fin_Latn', 'fineweb_est_gb': 24.0},
    'da': {'name': 'Danish', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'dan_Latn', 'cc100_code': 'da', 'cc100_est_gb': 1.2, 'fineweb_code': 'dan_Latn', 'fineweb_est_gb': 20.0},
    'no': {'name': 'Norwegian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'nob_Latn', 'cc100_code': 'no', 'cc100_est_gb': 1.3, 'fineweb_code': 'nob_Latn', 'fineweb_est_gb': 24.0},
    'bg': {'name': 'Bulgarian', 'script': 'Cyrl', 'family': 'Indo-European', 'flores_code': 'bul_Cyrl', 'cc100_code': 'bg', 'cc100_est_gb': 0.93, 'fineweb_code': 'bul_Cyrl', 'fineweb_est_gb': 105.0},
    'hr': {'name': 'Croatian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'hrv_Latn', 'cc100_code': 'hr', 'cc100_est_gb': 0.57, 'fineweb_code': 'hrv_Latn', 'fineweb_est_gb': 14.9},
    'sr': {'name': 'Serbian', 'script': 'Cyrl', 'family': 'Indo-European', 'flores_code': 'srp_Cyrl', 'cc100_code': 'sr', 'cc100_est_gb': 0.15, 'fineweb_code': 'srp_Cyrl', 'fineweb_est_gb': 16.1},
    'sk': {'name': 'Slovak', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'slk_Latn', 'cc100_code': 'sk', 'cc100_est_gb': 0.61, 'fineweb_code': 'slk_Latn', 'fineweb_est_gb': 17.8},
    'sl': {'name': 'Slovenian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'slv_Latn', 'cc100_code': 'sl', 'cc100_est_gb': 0.28, 'fineweb_code': 'slv_Latn', 'fineweb_est_gb': 10.9},
    'lt': {'name': 'Lithuanian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'lit_Latn', 'cc100_code': 'lt', 'cc100_est_gb': 0.34, 'fineweb_code': 'lit_Latn', 'fineweb_est_gb': 8.5},
    'lv': {'name': 'Latvian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'lvs_Latn', 'cc100_code': 'lv', 'cc100_est_gb': 0.21, 'fineweb_code': 'lvs_Latn', 'fineweb_est_gb': 4.8},
    'et': {'name': 'Estonian', 'script': 'Latn', 'family': 'Uralic', 'flores_code': 'est_Latn', 'cc100_code': 'et', 'cc100_est_gb': 0.17, 'fineweb_code': 'ekk_Latn', 'fineweb_est_gb': 4.8},
    'ur': {'name': 'Urdu', 'script': 'Arab', 'family': 'Indo-European', 'flores_code': 'urd_Arab', 'cc100_code': 'ur', 'cc100_est_gb': 0.088, 'fineweb_code': 'urd_Arab', 'fineweb_est_gb': 1.8},
    'ms': {'name': 'Malay', 'script': 'Latn', 'family': 'Austronesian', 'flores_code': 'zsm_Latn', 'cc100_code': 'ms', 'cc100_est_gb': 0.21, 'fineweb_code': 'zsm_Latn', 'fineweb_est_gb': 14.1},
    'tl': {'name': 'Tagalog', 'script': 'Latn', 'family': 'Austronesian', 'flores_code': 'fil_Latn', 'cc100_code': 'tl', 'cc100_est_gb': 0.07, 'fineweb_code': 'fil_Latn', 'fineweb_est_gb': 0.8},
    'ta': {'name': 'Tamil', 'script': 'Taml', 'family': 'Dravidian', 'flores_code': 'tam_Taml', 'cc100_code': 'ta', 'cc100_est_gb': 0.13, 'fineweb_code': 'tam_Taml', 'fineweb_est_gb': 2.2},
    'te': {'name': 'Telugu', 'script': 'Telu', 'family': 'Dravidian', 'flores_code': 'tel_Telu', 'cc100_code': 'te', 'cc100_est_gb': 0.05, 'fineweb_code': 'tel_Telu', 'fineweb_est_gb': 0.9},
    'kn': {'name': 'Kannada', 'script': 'Knda', 'family': 'Dravidian', 'flores_code': 'kan_Knda', 'cc100_code': 'kn', 'cc100_est_gb': 0.036, 'fineweb_code': 'kan_Knda', 'fineweb_est_gb': 0.4},
    'ml': {'name': 'Malayalam', 'script': 'Mlym', 'family': 'Dravidian', 'flores_code': 'mal_Mlym', 'cc100_code': 'ml', 'cc100_est_gb': 0.083, 'fineweb_code': 'mal_Mlym', 'fineweb_est_gb': 1.3},
    'mr': {'name': 'Marathi', 'script': 'Deva', 'family': 'Indo-European', 'flores_code': 'mar_Deva', 'cc100_code': 'mr', 'cc100_est_gb': 0.033, 'fineweb_code': 'mar_Deva', 'fineweb_est_gb': 0.8},
    'gu': {'name': 'Gujarati', 'script': 'Gujr', 'family': 'Indo-European', 'flores_code': 'guj_Gujr', 'cc100_code': 'gu', 'cc100_est_gb': 0.024, 'fineweb_code': 'guj_Gujr', 'fineweb_est_gb': 0.6},
    'pa': {'name': 'Punjabi', 'script': 'Guru', 'family': 'Indo-European', 'flores_code': 'pan_Guru', 'cc100_code': 'pa', 'cc100_est_gb': 0.009, 'fineweb_code': 'pan_Guru', 'fineweb_est_gb': 0.16},
    'ha': {'name': 'Hausa', 'script': 'Latn', 'family': 'Afro-Asiatic', 'flores_code': 'hau_Latn', 'cc100_code': 'ha', 'cc100_est_gb': 0.006, 'fineweb_code': 'hau_Latn', 'fineweb_est_gb': 0.08},
    'so': {'name': 'Somali', 'script': 'Latn', 'family': 'Afro-Asiatic', 'flores_code': 'som_Latn', 'cc100_code': 'so', 'cc100_est_gb': 0.0078, 'fineweb_code': 'som_Latn', 'fineweb_est_gb': 0.08},
    'zu': {'name': 'Zulu', 'script': 'Latn', 'family': 'Niger-Congo', 'flores_code': 'zul_Latn', 'cc100_code': 'zu', 'cc100_est_gb': 0.004, 'fineweb_code': 'zul_Latn', 'fineweb_est_gb': 0.02},
    'ig': {'name': 'Igbo', 'script': 'Latn', 'family': 'Niger-Congo', 'flores_code': 'ibo_Latn', 'cc100_code': 'ig', 'cc100_est_gb': 0.006, 'fineweb_code': 'ibo_Latn', 'fineweb_est_gb': 0.012},
    'am': {'name': 'Amharic', 'script': 'Ethi', 'family': 'Afro-Asiatic', 'flores_code': 'amh_Ethi', 'cc100_code': 'am', 'cc100_est_gb': 0.013, 'fineweb_code': 'amh_Ethi', 'fineweb_est_gb': 1.74},
    'hy': {'name': 'Armenian', 'script': 'Armn', 'family': 'Indo-European', 'flores_code': 'hye_Armn', 'cc100_code': 'hy', 'cc100_est_gb': 0.077, 'fineweb_code': 'hye_Armn', 'fineweb_est_gb': 1.6},
    'ka': {'name': 'Georgian', 'script': 'Geor', 'family': 'Kartvelian', 'flores_code': 'kat_Geor', 'cc100_code': 'ka', 'cc100_est_gb': 0.11, 'fineweb_code': 'kat_Geor', 'fineweb_est_gb': 2.0},
    'my': {'name': 'Burmese', 'script': 'Mymr', 'family': 'Sino-Tibetan', 'flores_code': 'mya_Mymr', 'cc100_code': 'my', 'cc100_est_gb': 0.046, 'fineweb_code': 'mya_Mymr', 'fineweb_est_gb': 0.4},
    'km': {'name': 'Khmer', 'script': 'Khmr', 'family': 'Austroasiatic', 'flores_code': 'khm_Khmr', 'cc100_code': 'km', 'cc100_est_gb': 0.015, 'fineweb_code': 'khm_Khmr', 'fineweb_est_gb': 0.28},
    'ne': {'name': 'Nepali', 'script': 'Deva', 'family': 'Indo-European', 'flores_code': 'npi_Deva', 'cc100_code': 'ne', 'cc100_est_gb': 0.039, 'fineweb_code': 'npi_Deva', 'fineweb_est_gb': 0.8},
    'si': {'name': 'Sinhala', 'script': 'Sinh', 'family': 'Indo-European', 'flores_code': 'sin_Sinh', 'cc100_code': 'si', 'cc100_est_gb': 0.045, 'fineweb_code': 'sin_Sinh', 'fineweb_est_gb': 0.6},
    'sq': {'name': 'Albanian', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'als_Latn', 'cc100_code': 'sq', 'cc100_est_gb': 0.13, 'fineweb_code': 'als_Latn', 'fineweb_est_gb': 35.0},
    'az': {'name': 'Azerbaijani', 'script': 'Latn', 'family': 'Turkic', 'flores_code': 'azj_Latn', 'cc100_code': 'az', 'cc100_est_gb': 0.13, 'fineweb_code': 'azj_Latn', 'fineweb_est_gb': 29.6},
    'kk': {'name': 'Kazakh', 'script': 'Cyrl', 'family': 'Turkic', 'flores_code': 'kaz_Cyrl', 'cc100_code': 'kk', 'cc100_est_gb': 0.088, 'fineweb_code': 'kaz_Cyrl', 'fineweb_est_gb': 1.6},
    'uz': {'name': 'Uzbek', 'script': 'Latn', 'family': 'Turkic', 'flores_code': 'uzn_Latn', 'cc100_code': 'uz', 'cc100_est_gb': 0.015, 'fineweb_code': 'uzn_Latn', 'fineweb_est_gb': 2.0},
    'af': {'name': 'Afrikaans', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'afr_Latn', 'cc100_code': 'af', 'cc100_est_gb': 0.03, 'fineweb_code': 'afr_Latn', 'fineweb_est_gb': 8.1},
    'mn': {'name': 'Mongolian', 'script': 'Cyrl', 'family': 'Mongolic', 'flores_code': 'khk_Cyrl', 'cc100_code': 'mn', 'cc100_est_gb': 0.039, 'fineweb_code': 'khk_Cyrl', 'fineweb_est_gb': 0.8},
    'gl': {'name': 'Galician', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'glg_Latn', 'cc100_code': 'gl', 'cc100_est_gb': 0.07, 'fineweb_code': 'glg_Latn', 'fineweb_est_gb': 2.0},
    'eu': {'name': 'Basque', 'script': 'Latn', 'family': 'Language isolate', 'flores_code': 'eus_Latn', 'cc100_code': 'eu', 'cc100_est_gb': 0.048, 'fineweb_code': 'eus_Latn', 'fineweb_est_gb': 1.0},
    'is': {'name': 'Icelandic', 'script': 'Latn', 'family': 'Indo-European', 'flores_code': 'isl_Latn', 'cc100_code': 'is', 'cc100_est_gb': 0.077, 'fineweb_code': 'isl_Latn', 'fineweb_est_gb': 2.0},
}

# --- Language Sets ---
LANG_SET_20 = [
    'en', 'zh', 'es', 'ar', 'ru', 'de', 'fr', 'ja', 'hi', 'pt', # High-resource
    'id', 'vi', 'tr', 'pl', 'el', 'sw', 'fa', 'bn', 'th', 'ko'  # Med/Low-resource, diverse scripts/families
]

LANG_SET_60 = LANG_SET_20 + [
    'it', 'nl', 'uk', 'ro', 'cs', 'hu', 'sv', 'fi', 'da', 'no', # More European
    'he', 'ms', 'tl', 'ta', 'te', 'mr', 'ur', 'gu', 'pa', 'kn', # More Asian (Indo-Aryan, Dravidian, Austronesian)
    'ml', 'ne', 'si', 'my', 'km', 'gn', 'ga',
    'af', 'ha', 'so', 'yo', 'zu', 'ig', 'am', # More African (Afro-Asiatic, Niger-Congo)
    'az', 'kk', 'uz', 'hy', 'ka', 'mn', # More Turkic, Caucasian, Mongolic
]

LANG_SET_100 = LANG_SET_60 + [
    'bg', 'hr', 'sr', 'sk', 'sl', 'lt', 'lv', 'et', 'sq', 'gl', # Rest of Europe
    'eu', 'is', 'cy', 'ku', 'mg',  # Low-resource / Isolates
    # Note: L100 is < 100, constrained by data in all 3 sources.
]

# ---------------------------------------------------------------------------
# --- Helper Functions
# ---------------------------------------------------------------------------

def _get_flores_col(example: Dict, code: str) -> Optional[str]:
    """Return the correct column name for a FLORES language code."""
    if code in example:
        return code
    pref = f"sentence_{code}"
    if pref in example:
        return pref
    # This was the source of the warning. The codes were wrong in LANGUAGE_METADATA.
    log.warning(f"Could not find column for FLORES code {code} in example. Keys: {list(example.keys())}")
    return None

def resolve_lang_set(lang_set_str: str) -> List[str]:
    """Resolves the language set argument into a list of language codes."""
    if lang_set_str == 'L20':
        return LANG_SET_20
    elif lang_set_str == 'L60':
        return LANG_SET_60
    elif lang_set_str == 'L100':
        return LANG_SET_100
    else:
        langs = [lg.strip() for lg in lang_set_str.split(',') if lg.strip()]
        log.info(f"Using custom language list: {langs}")
        return langs

def _calculate_proportions(
    original_sizes: Dict[str, float], 
    temperature: float, 
    cap: float = 0.5
) -> Dict[str, float]:
    """Calculates language proportions with temperature and capping."""
    
    # --- 1. Apply Temperature ---
    if temperature == 1.0:
        powered_sizes = original_sizes
    else:
        log.info(f"Applying temperature {temperature} to sampling.")
        powered_sizes = {lg: size**(1.0/temperature) for lg, size in original_sizes.items()}
    
    total_powered_size = sum(powered_sizes.values())
    if total_powered_size == 0:
        return {}
        
    lang_proportions = {lg: p_size / total_powered_size for lg, p_size in powered_sizes.items()}
    
    # --- 2. Apply Capping ---
    capping_needed = False
    for lg, prop in lang_proportions.items():
        if prop > cap:
            log.warning(f"[{lg}] exceeds cap (at {prop*100:.1f}%). Capping at {cap*100}%.")
            lang_proportions[lg] = cap
            capping_needed = True
    
    if capping_needed:
        # Re-normalize the rest
        capped_total_prop = sum(prop for lg, prop in lang_proportions.items() if prop == cap)
        uncapped_total_original_size = sum(
            powered_sizes[lg] for lg in powered_sizes if lang_proportions[lg] < cap
        )
        remaining_prop_to_distribute = 1.0 - capped_total_prop
        
        if uncapped_total_original_size > 0:
            for lg in original_sizes:
                if lang_proportions[lg] < cap:
                    lang_proportions[lg] = (powered_sizes[lg] / uncapped_total_original_size) * remaining_prop_to_distribute
        elif remaining_prop_to_distribute > 0:
            log.warning("Only capped languages present. Final distribution may not be 100%.")

    # --- 3. Final Normalization ---
    final_total_prop = sum(lang_proportions.values())
    if not math.isclose(final_total_prop, 1.0, rel_tol=1e-5):
        log.info(f"Final proportions sum to {final_total_prop}. Re-normalizing.")
        for lg in lang_proportions:
            lang_proportions[lg] = lang_proportions[lg] / final_total_prop
            
    return lang_proportions

# ---------------------------------------------------------------------------
# --- Data Processing Classes
# ---------------------------------------------------------------------------

class BaseCorpusDownloader:
    """Base class for streaming and splitting a main corpus."""
    def __init__(self, lg: str, lang_dir: Path, cap_bytes: int, args: argparse.Namespace):
        self.lg = lg
        self.lang_dir = lang_dir
        self.cap_bytes = cap_bytes
        self.args = args
        self.meta_file = self.lang_dir / "meta.json"

    def _load_dataset(self) -> Optional[Dataset]:
        """Abstract method to load the streaming dataset."""
        raise NotImplementedError
        
    def run(self):
        """Streams, splits, and writes the data for one language."""
        if self.meta_file.exists():
            log.info(f"[{self.lg}] Already processed. Skipping download.")
            return

        if self.cap_bytes == 0 and self.args.min_lines_floor == 0:
            log.warning(f"[{self.lg}] Calculated target size is 0 bytes and floor is 0. Skipping.")
            return
            
        log.info(f"[{self.lg}] Processing... Target size: {self.cap_bytes / 1e9:.3f} GB (Floor: {self.args.min_lines_floor} lines)")

        try:
            ds = self._load_dataset()
            if ds is None:
                return
        except Exception as e:
            log.error(f"[{self.lg}] Failed to load dataset: {e}")
            return

        # --- Write Splits ---
        f_train, f_val, f_test = None, None, None
        pbar = tqdm(desc=f"Writing {self.lg}", unit="B", unit_scale=True, smoothing=0.1)
        written_bytes = 0
        counts = {"train": 0, "val": 0, "test": 0}
        
        try:
            self.lang_dir.mkdir(parents=True, exist_ok=True)
            f_train = open(self.lang_dir / "train.txt", "w", encoding="utf-8")
            f_val = open(self.lang_dir / "val.txt", "w", encoding="utf-8")
            f_test = open(self.lang_dir / "test.txt", "w", encoding="utf-8")
            
            for ex in ds:
                line = ex.get("text")
                if line is None:
                    continue
                # do not strip or normalize; just ensure newline termination
                if not line.endswith("\n"):
                    line_with_newline = line + "\n"
                else:
                    line_with_newline = line

                line_bytes = len(line_with_newline.encode("utf-8"))
                            
                # Check floor logic FIRST
                # If we're over byte budget BUT under train line floor, force-write to train.
                is_over_budget = (written_bytes + line_bytes > self.cap_bytes) and self.cap_bytes > 0
                is_under_floor = counts["train"] < self.args.min_lines_floor

                if is_over_budget and is_under_floor:
                    f_train.write(line_with_newline)
                    counts["train"] += 1
                    written_bytes += line_bytes
                    pbar.update(line_bytes)
                    continue # Skip normal assignment and break check
                
                # Check cap logic SECOND
                if is_over_budget:
                    break # We're over budget and meet the floor
                    
                # Normal assignment
                r = random.random()
                if r < self.args.val_frac and counts["val"] < MAX_VAL_TEST_SIZE:
                    f_val.write(line_with_newline)
                    counts["val"] += 1
                elif r < (self.args.val_frac + self.args.test_frac) and counts["test"] < MAX_VAL_TEST_SIZE:
                    f_test.write(line_with_newline)
                    counts["test"] += 1
                else:
                    f_train.write(line_with_newline)
                    counts["train"] += 1
                    
                written_bytes += line_bytes
                pbar.update(line_bytes)
                
        except Exception as e:
            log.error(f"[{self.lg}] Error during writing: {e}")
        finally:
            if f_train: f_train.close()
            if f_val: f_val.close()
            if f_test: f_test.close()
            pbar.close()
            
            meta = {
                "language": self.lg,
                "dataset": self.args.dataset,
                "target_bytes": self.cap_bytes,
                "bytes_written": written_bytes,
                "lines": counts,
            }
            with open(self.meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            log.info(f"[{self.lg}] Finished. Wrote {written_bytes / 1e9:.3f} GB.")
            if is_under_floor and counts["train"] < self.args.min_lines_floor:
                log.warning(f"[{self.lg}] Could not meet line floor. Dataset exhausted. Total lines: {counts['train']}")


class Cc100Downloader(BaseCorpusDownloader):
    def _load_dataset(self) -> Optional[Dataset]:
        ds_code = LANGUAGE_METADATA[self.lg]['cc100_code']
        log.info(f"[{self.lg}] Loading from 'cc100' config: {ds_code}")
        return load_dataset("cc100", ds_code, split="train", streaming=True)


class FineWebDownloader(BaseCorpusDownloader):
    def _load_dataset(self) -> Optional[Dataset]:
        if self.lg == 'en':
            log.info(f"[{self.lg}] Loading from 'HuggingFaceFW/fineweb' (v1)")
            return load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        else:
            ds_code = LANGUAGE_METADATA[self.lg]['fineweb_code']
            log.info(f"[{self.lg}] Loading from 'HuggingFaceFW/fineweb-2' config: {ds_code}")
            return load_dataset("HuggingFaceFW/fineweb-2", ds_code, split="train", streaming=True)


class TokenizerSampler:
    """
    Creates the tokenizer set by downsampling in a single pass.
    REFACTOR: This class no longer uses _sample_lines_from_file.
    It performs a clean, single-pass probabilistic sample over the train.txt files.
    REFACTOR: All oversampling logic is REMOVED.
    """
    def __init__(self, args: argparse.Namespace, main_corpus_dir: Path, tokenizer_corpus_dir: Path):
        self.args = args
        self.main_corpus_dir = main_corpus_dir
        self.tokenizer_corpus_dir = tokenizer_corpus_dir
        self.tokenizer_corpus_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)

    def run(self, lang_list: List[str]):
        log.info(f"--- Creating Tokenizer Corpus (Target: {self.args.tokenizer_size_gb} GB) ---")
        if self.args.tokenizer_size_gb <= 0:
            log.info("Tokenizer size is 0, skipping.")
            return

        # --- 1. Find all train.txt files and their sizes ---
        train_files: Dict[str, Tuple[Path, int]] = {} # {lg: (path, size)}
        for lg in lang_list:
            if lg not in LANGUAGE_METADATA: continue
            train_file = self.main_corpus_dir / lg / "train.txt"
            if train_file.exists():
                try:
                    f_size = train_file.stat().st_size
                    if f_size > 0:
                        train_files[lg] = (train_file, f_size)
                    else:
                        log.warning(f"[{lg}] train.txt is empty. Skipping for tokenizer set.")
                except OSError as e:
                    log.warning(f"[{lg}] Could not stat {train_file}: {e}")
            else:
                log.warning(f"[{lg}] No train.txt found. Skipping for tokenizer set.")

        if not train_files:
            log.error("No training files found. Cannot create tokenizer set.")
            return

        # --- 2. Calculate Tokenizer Proportions ---
        original_sizes = {lg: size for lg, (path, size) in train_files.items()}
        lang_proportions = _calculate_proportions(
            original_sizes,
            self.args.tokenizer_sampling_temp,
            cap=1.0 # No cap for tokenizer, just temp sampling
        )
        
        log.info(f"Tokenizer Language Proportions (relative to train set): {json.dumps({k: f'{v*100:.2f}%' for k, v in lang_proportions.items()}, indent=2)}")

        # --- 3. Sample from each train file ---
        for lg, (train_file, f_size) in train_files.items():
            tok_lang_dir = self.tokenizer_corpus_dir / lg
            tok_lang_dir.mkdir(parents=True, exist_ok=True)
            tok_file_path = tok_lang_dir / "train.txt"
            
            if tok_file_path.exists():
                log.info(f"[{lg}] Tokenizer file already exists. Skipping.")
                continue
            
            lang_prop = lang_proportions.get(lg, 0)
            tok_target_bytes = int(self.args.tokenizer_size_gb * lang_prop * (1024**3))
            sample_prop = min(1.0, tok_target_bytes / f_size if f_size > 0 else 0)
            
            log.info(f"[{lg}] Sampling for tokenizer: {sample_prop*100:.4f}% of {f_size/1e6:.2f}MB -> target {tok_target_bytes/1e6:.2f}MB")
            
            if sample_prop == 0:
                log.warning(f"[{lg}] Sample proportion is 0. Skipping.")
                continue

            # --- 4. Cleaned-up Sampling Pass ---
            # This new logic streams the train file once and applies probabilistic sampling.
            # It no longer oversamples or re-reads the file.
            lines_written = 0
            bytes_written = 0
            try:
                with open(train_file, 'r', encoding='utf-8') as f_in, \
                     open(tok_file_path, 'w', encoding='utf-8') as f_out:
                    
                    pbar = tqdm(desc=f"Sampling {lg} for tok", total=f_size, unit="B", unit_scale=True, smoothing=0.1)
                    
                    # Read line by line
                    for line in f_in:
                        line_bytes = len(line.encode('utf-8'))
                        
                        if random.random() < sample_prop:
                            f_out.write(line) # line already includes newline
                            lines_written += 1
                            bytes_written += line_bytes
                        
                        pbar.update(line_bytes)
                    pbar.close()

                log.info(f"[{lg}] Wrote {lines_written} lines ({bytes_written/1e6:.2f} MB) to tokenizer file.")
                
                # Check floor without oversampling
                if lines_written < self.args.min_lines_floor:
                    log.warning(f"[{lg}] Tokenizer set has {lines_written} lines (below floor {self.args.min_lines_floor}). Not oversampling.")

            except IOError as e:
                log.error(f"[{lg}] Failed to read/write tokenizer file: {e}")

        log.info("--- Tokenizer Corpus Creation Finished ---")


class FloresDownloader:
    """Downloads and structures the FLORES+ parallel dataset."""
    def __init__(self, parallel_dir: Path):
        self.parallel_dir = parallel_dir
        self.parallel_dir.mkdir(parents=True, exist_ok=True)
        # maps our short language code -> (iso_639_3, iso_15924)
        self.flores_codes: Dict[str, Tuple[str, Optional[str]]] = {}

    def _write_split(self, split_name: str, dataset: Dataset):
        """Writes a single FLORES+ split (dev or devtest) to per-language files."""
        split_dir = self.parallel_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if not self.flores_codes:
            log.warning(f"No FLORES+ languages configured. Skipping {split_name} split.")
            return

        # skip if we've already written this split
        first_lang = list(self.flores_codes.keys())[0]
        first_lang_file = split_dir / f"{first_lang}.txt"
        if first_lang_file.exists():
            log.info(f"FLORES+ {split_name} split already exists. Skipping.")
            return

        log.info(f"Writing FLORES+ {split_name} split...")
        file_handlers = {}

        try:
            # open all target files
            for lg in self.flores_codes.keys():
                file_handlers[lg] = open(split_dir / f"{lg}.txt", "w", encoding="utf-8")

            # FLORES+ is row-wise; we filter by iso_639_3 + iso_15924
            for ex in tqdm(dataset, desc=f"Writing {split_name}"):
                ex_iso = ex.get("iso_639_3")
                ex_script = ex.get("iso_15924")
                text = ex.get("text")
                if text is None:
                    continue
                # don’t change the text, just make sure it’s newline-terminated
                if not text.endswith("\n"):
                    text = text + "\n"

                for lg, (iso_code, script_code) in self.flores_codes.items():
                    if ex_iso == iso_code and (script_code is None or ex_script == script_code):
                        file_handlers[lg].write(text)

        except Exception as e:
            log.error(f"Error writing FLORES+ split {split_name}: {e}")
        finally:
            for fh in file_handlers.values():
                fh.close()

    def run(self, lang_list: List[str]):
        log.info("--- Downloading FLORES+ Parallel Data ---")

        # build the mapping from our short codes to FLORES+ iso/script
        for lg in lang_list:
            md = LANGUAGE_METADATA.get(lg)
            if not md:
                log.warning(f"[{lg}] not found in LANGUAGE_METADATA. Skipping for parallel data.")
                continue

            flores_code = md.get("flores_code")
            if not flores_code:
                log.warning(f"[{lg}] has no flores_code in LANGUAGE_METADATA. Skipping for parallel data.")
                continue

            # FLORES+ uses "<iso_639_3>_<iso_15924>"
            if "_" in flores_code:
                iso_part, script_part = flores_code.split("_", 1)
            else:
                iso_part, script_part = flores_code, None

            self.flores_codes[lg] = (iso_part, script_part)

        if not self.flores_codes:
            log.error("No valid languages for FLORES+. Skipping parallel download.")
            return

        try:
            # FLORES+ exposes dev and devtest as separate splits
            ds_dev = load_dataset("openlanguagedata/flores_plus", split="dev", verification_mode="no_checks",)
            ds_devtest = load_dataset("openlanguagedata/flores_plus", split="devtest", verification_mode="no_checks",)

            self._write_split("dev", ds_dev)
            self._write_split("devtest", ds_devtest)

            log.info("--- FLORES+ Download Finished ---")
        except Exception as e:
            log.error(f"Failed to download FLORES+ dataset: {e}")



def clean_hf_cache():
    log.info("--- Cleaning Hugging Face datasets cache ---")
    try:
        log.info("Loading 'openlanguagedata/flores_plus' to access cache...")
        ds = load_dataset("openlanguagedata/flores_plus", split="dev")
        cleaned_bytes_struct = ds.cleanup_cache_files()
        
        total_cleaned = sum(cleaned_bytes_struct.values())
        log.info(f"Cleaned {total_cleaned / 1e9:.3f} GB from cache.")
        log.info(f"Cache cleanup details: {cleaned_bytes_struct}")
        
    except Exception as e:
        log.error(f"Could not clean Hugging Face cache: {e}")
        log.warning("Cache cleaning failed. You may need to manually clear it.")

# ---------------------------------------------------------------------------
# --- Main Execution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and prepare multilingual corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['cc100', 'fineweb'],
        help="The main dataset to download. 'fineweb' uses FineWeb-2 for most languages but FineWeb-v1 for English."
    )
    parser.add_argument(
        "--lang_set",
        type=str,
        required=True,
        help="Language set to use. Choices: 'L20', 'L60', 'L100', or a comma-separated list (e.g., 'en,fr,de')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The root directory to save all processed data."
    )
    parser.add_argument(
        "--total_size_gb",
        type=float,
        required=True,
        help="Total desired size (in GB) for the main corpus (train+val+test)."
    )
    parser.add_argument(
        "--main_sampling_temp",
        type=float,
        default=XLM_R_TEMP,
        help="Temperature for sampling main corpus. T=1.0 is proportional, T>1.0 upsamples LRLs."
    )
    parser.add_argument(
        "--tokenizer_size_gb",
        type=float,
        default=10.0,
        help="Desired size (in GB) for the downsampled tokenizer corpus."
    )
    parser.add_argument(
        "--tokenizer_sampling_temp",
        type=float,
        default=1.0,
        help="Temperature for sampling tokenizer data from the main train data. T>1 (e.g., 7.0) will create a "
             "more uniform set. Note: this compounds with --main_sampling_temp."
    )
    parser.add_argument(
        "--min_lines_floor",
        type=int,
        default=1000,
        help="Minimum number of lines for train.txt (main) and tokenizer files. Will NOT oversample to meet."
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of data to use for validation."
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of data to use for test."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for splitting and sampling."
    )
    parser.add_argument(
        "--skip_cache_clean",
        action='store_true',
        help="Skip the final cache cleaning step."
    )
    
    args = parser.parse_args()
    
    if (args.val_frac + args.test_frac) > 0.1:
        log.warning(f"Validation ({args.val_frac}) + Test ({args.test_frac}) > 10%. This is high.")
        
    return args

def main():
    """Main function to orchestrate the data preparation pipeline."""
    args = parse_args()
    random.seed(args.seed) 
    
    # Suppress excessive logging from datasets library
    datasets_logging.set_verbosity_error()
    
    log.info(f"Starting data preparation with args: {json.dumps(vars(args), indent=2)}")
    
    # 1. Resolve language list
    lang_list = resolve_lang_set(args.lang_set)
    if not lang_list:
        log.error("Language set is empty. Exiting.")
        return
        
    # --- Define output paths ---
    output_dir = Path(args.output_dir)
    main_corpus_dir = output_dir / "main_corpus"
    tokenizer_corpus_dir = output_dir / "tokenizer_corpus"
    parallel_dir = output_dir / "parallel"
    
    # --- Run pipeline ---
    try:
        # 2. Download and split main corpus
        log.info(f"--- 1. Starting Main Corpus Download ({args.dataset}) ---")
        dataset_key = f"{args.dataset}_est_gb"
        original_sizes = {}
        for lg in lang_list:
            if lg not in LANGUAGE_METADATA:
                log.warning(f"[{lg}] not in LANGUAGE_METADATA. Skipping.")
                continue
            
            if args.dataset == 'fineweb' and lg == 'en':
                size = LANGUAGE_METADATA['en'].get('fineweb_v1_est_gb', 10000.0) # Use v1 size
                original_sizes[lg] = size
            elif dataset_key in LANGUAGE_METADATA[lg]:
                original_sizes[lg] = LANGUAGE_METADATA[lg][dataset_key]
            else:
                log.warning(f"[{lg}] missing size key '{dataset_key}'. Skipping.")
        
        if not original_sizes:
            raise RuntimeError(f"No valid languages with size info found for dataset {args.dataset}.")
        
        # Calculate final proportions
        lang_proportions = _calculate_proportions(
            original_sizes,
            args.main_sampling_temp,
            cap=0.5
        )
        log.info(f"Final Language Proportions: {json.dumps({k: f'{v*100:.2f}%' for k, v in lang_proportions.items()}, indent=2)}")

        # Create and run downloaders for each language
        for lg, proportion in lang_proportions.items():
            lang_dir = main_corpus_dir / lg
            cap_bytes = int(args.total_size_gb * proportion * (1024**3))
            
            downloader: Optional[BaseCorpusDownloader] = None
            if args.dataset == 'cc100':
                downloader = Cc100Downloader(lg, lang_dir, cap_bytes, args)
            elif args.dataset == 'fineweb':
                downloader = FineWebDownloader(lg, lang_dir, cap_bytes, args)
            
            if downloader:
                downloader.run()
            else:
                log.error(f"No valid downloader found for dataset {args.dataset}")
        
        # 3. Create tokenizer set from the 'train' files
        log.info(f"--- 2. Starting Tokenizer Corpus Creation ---")
        tokenizer_sampler = TokenizerSampler(args, main_corpus_dir, tokenizer_corpus_dir)
        tokenizer_sampler.run(lang_list)
        
        # 4. Download parallel data
        log.info(f"--- 3. Starting FLORES+ Parallel Data Download ---")
        flores_downloader = FloresDownloader(parallel_dir)
        flores_downloader.run(lang_list)
        
        log.info("--- Pipeline Finished Successfully ---")
        
    except Exception as e:
        log.critical(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        # 5. Clean up cache
        if not args.skip_cache_clean:
            clean_hf_cache()
            
    log.info(f"All data saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()