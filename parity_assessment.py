# parallel_corpus_analysis.py

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.multigramlm import MultigramLMTokenizer  # Adjust path if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_parallel_corpus(
    parallel_files,
    standard_tokenizer_path,
    langspec_tokenizer_paths,
    multigram_tokenizer_paths,
    max_lines=2000,
    output_prefix="token_analysis",
    show_plots=False
):
    """
    Analyze token lengths across three tokenization methods (standard, language-specific, multigram)
    for a parallel corpus. Each language's file is read line by line. For each line:
      - We measure how many tokens are produced by each of the three tokenizers.
    Then we produce:
      1) A side-by-side boxplot per language with (standard, langspec, multigram).
      2) A bar chart showing for each language how often each method yields the "shortest" segmentation.

    Args:
        parallel_files: dict { lang_code: "/path/to/corpus.txt", ... } 
                       Each file is the monolingual corpus for that language. If it’s a true “parallel” set 
                       in the sense of translations, we’re still just measuring lengths from each file.
        standard_tokenizer_path: path to a trained standard UnigramLM tokenizer.
        langspec_tokenizer_paths: { 'base':..., 'languages':{...} } for LanguageSpecificUnigramLMTokenizer.
        multigram_tokenizer_paths: { 'base':..., 'global_probs':...} for MultigramLMTokenizer.
        max_lines: how many lines to read from each file (to limit runtime).
        output_prefix: base filename for saving plots (like "analysis_boxplot.png" etc.).
        show_plots: if True, we call plt.show() instead of saving.

    Produces files:
      - <output_prefix>_boxplot.png
      - <output_prefix>_shortest_bar.png
    (and optionally shows the plots if show_plots=True).
    """
    logger.info("Loading tokenizers...")

    # 1) Standard
    std_tok = StandardUnigramLMTokenizer()
    std_tok.load(standard_tokenizer_path)

    # 2) Language-specific
    lang_tok = LanguageSpecificUnigramLMTokenizer()
    lang_tok.load(langspec_tokenizer_paths["base"], langspec_tokenizer_paths["languages"])

    # 3) Multigram
    multi_tok = MultigramLMTokenizer()
    multi_tok.load(
        base_path=multigram_tokenizer_paths["base"],
        global_probs_path=multigram_tokenizer_paths["global_probs"]
    )
    # If multi_tok.token_probs is None, we might skip analyzing multi-gram.

    # We'll store lengths in a dict: 
    #   lengths["standard"][lang_code] = list of segment lengths
    #   lengths["langspec"][lang_code] = ...
    #   lengths["multigram"][lang_code] = ...
    lengths = {
        "standard": {},
        "langspec": {},
        "multigram": {}
    }

    # We'll also track how often each method yields the "shortest" segmentation for that line.
    # shortest_count[lang_code][method] = how many lines that method is the shortest
    # We'll do a tie-break if multiple are equally short, we can either give them each 1 
    # or do something else. We'll do "if there's a tie, each gets +1".
    shortest_count = {}
    total_count = {}

    # Initialize data structures
    for lang_code in parallel_files.keys():
        lengths["standard"][lang_code] = []
        lengths["langspec"][lang_code] = []
        lengths["multigram"][lang_code] = []
        shortest_count[lang_code] = {"standard": 0, "langspec": 0, "multigram": 0}
        total_count[lang_code] = 0

    # 4) Read each language file, measure lengths
    logger.info("Measuring token lengths for each language file...")
    for lang_code, path in parallel_files.items():
        if not os.path.isfile(path):
            logger.warning(f"No file found at {path} for {lang_code}, skipping.")
            continue

        logger.info(f"Reading lines from {lang_code} corpus at {path}")
        line_idx = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                line_idx += 1
                if max_lines and line_idx > max_lines:
                    break

                # A) standard
                std_ids = std_tok.encode(text)
                std_len = len(std_ids)

                # B) language-specific
                # We'll do the "native approach" for that language if we want, or we do 
                # custom_viterbi_best_language. 
                # But presumably, the user wants the "native approach" for that language.
                # If you want the best among all languages, do custom_viterbi_best_language.
                # We'll do the "native" approach for that language:
                lang_probs = lang_tok.get_token_probs(lang_code)
                if lang_probs is not None:
                    # We'll do a Viterbi for just that language's distribution:
                    #   from code.utils import segment_text_viterbi
                    from code.utils import segment_text_viterbi
                    tokens_lang = segment_text_viterbi(text, lang_probs, unk_token=lang_tok.unk_token)
                    lang_len = len(tokens_lang)
                else:
                    # fallback
                    lang_len = std_len  # fallback
                # Another approach if you want best_lang approach:
                # best_lang, best_tokens = lang_tok.custom_viterbi_best_language(text)
                # lang_len = len(best_tokens)

                # C) multigram
                if multi_tok.token_probs:
                    from code.utils import segment_text_viterbi
                    tokens_multi = segment_text_viterbi(text, multi_tok.token_probs, unk_token=multi_tok.unk_token)
                    multi_len = len(tokens_multi)
                else:
                    # fallback
                    multi_len = std_len

                lengths["standard"][lang_code].append(std_len)
                lengths["langspec"][lang_code].append(lang_len)
                lengths["multigram"][lang_code].append(multi_len)

                # Which is shortest? 
                lens_list = [("standard", std_len),
                             ("langspec", lang_len),
                             ("multigram", multi_len)]
                lens_list.sort(key=lambda x: x[1])
                # the smallest length
                smallest_len = lens_list[0][1]
                # count ties
                # any that match smallest_len get +1
                for (method, ln) in lens_list:
                    if ln == smallest_len:
                        shortest_count[lang_code][method] += 1

                total_count[lang_code] += 1

        logger.info(f"Read {min(line_idx, max_lines)} lines for {lang_code}.")

    # 5) produce side-by-side boxplot for each language
    logger.info("Producing side-by-side boxplots with (standard, langspec, multigram).")
    languages = sorted(list(parallel_files.keys()))
    data_for_boxplot = []
    group_labels = []

    # We'll gather data in the order: [ (std_data), (lang_data), (multi_data) ] per language
    # so we get 3 boxes per language
    for lang in languages:
        # skip if we have no data
        if not lengths["standard"][lang]:
            continue
        data_for_boxplot.append(lengths["standard"][lang])
        data_for_boxplot.append(lengths["langspec"][lang])
        data_for_boxplot.append(lengths["multigram"][lang])
        group_labels.append(f"{lang}-std")
        group_labels.append(f"{lang}-lspec")
        group_labels.append(f"{lang}-multi")

    plt.figure(figsize=(max(8, len(group_labels)*0.5), 6))
    plt.boxplot(data_for_boxplot, labels=group_labels)
    plt.xticks(rotation=45)
    plt.title("Token Length Comparison: standard vs. lang-spec vs. multigram (by language)")
    plt.xlabel("Language-Method")
    plt.ylabel("Tokens per line")

    boxplot_path = f"{output_prefix}_boxplot.png"
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.savefig(boxplot_path, dpi=150)
        logger.info(f"Saved boxplot to {boxplot_path}")
    plt.close()

    # 6) produce bar chart: for each language, how often each method is the shortest
    # We'll do a grouped bar chart with 3 bars per language
    method_names = ["standard", "langspec", "multigram"]
    bar_data = {}
    for lang in languages:
        sum_count = total_count[lang]
        if sum_count == 0:
            bar_data[lang] = [0, 0, 0]
            continue
        # fraction or percentage
        bar_data[lang] = [
            (shortest_count[lang]["standard"] / sum_count)*100.0,
            (shortest_count[lang]["langspec"] / sum_count)*100.0,
            (shortest_count[lang]["multigram"] / sum_count)*100.0
        ]

    # We'll produce a bar chart with each language on x-axis groups, and 3 bars per group
    x_langs = np.arange(len(languages))
    width = 0.25  # bar width
    plt.figure(figsize=(max(8, len(languages)*2.5), 5))

    # standard
    standard_vals = [bar_data[l][0] for l in languages]
    langspec_vals = [bar_data[l][1] for l in languages]
    multi_vals = [bar_data[l][2] for l in languages]

    plt.bar(x_langs - width, standard_vals, width, label="Standard")
    plt.bar(x_langs,         langspec_vals, width, label="LangSpec")
    plt.bar(x_langs + width, multi_vals,    width, label="MultiGram")

    plt.xticks(x_langs, languages)
    plt.ylim(0, 105)
    plt.title("Percentage of lines for which each method yields the shortest segmentation")
    plt.ylabel("Percentage (%)")
    plt.legend()

    barplot_path = f"{output_prefix}_shortest_bar.png"
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.savefig(barplot_path, dpi=150)
        logger.info(f"Saved bar chart to {barplot_path}")
    plt.close()

    logger.info("Done analyzing token counts with standard, lang-spec, and multigram.")


if __name__ == "__main__":
    # Example usage
    logger.info("Running a quick test of the parallel_corpus_analysis with multi-gram included.")
    parallel_files = {
        "en": "data/parallel_text_dataset/final/parallel_all_langs/en.txt",
        "es": "data/parallel_text_dataset/final/parallel_all_langs/es.txt",
        "ar": "data/parallel_text_dataset/final/parallel_all_langs/ar.txt",
        "fr": "data/parallel_text_dataset/final/parallel_all_langs/fr.txt",
        "de": "data/parallel_text_dataset/final/parallel_all_langs/de.txt",
        "zh": "data/parallel_text_dataset/final/parallel_all_langs/zh.txt",
        "ru": "data/parallel_text_dataset/final/parallel_all_langs/ru.txt",
        "hi": "data/parallel_text_dataset/final/parallel_all_langs/hi.txt",
        "swh_Latn": "data/parallel_text_dataset/final/parallel_all_langs/swh_Latn.txt",
        "yor_Latn": "data/parallel_text_dataset/final/parallel_all_langs/yor_Latn.txt",
        "hau_Latn": "data/parallel_text_dataset/final/parallel_all_langs/hau_Latn.txt",
        "zul_Latn": "data/parallel_text_dataset/final/parallel_all_langs/zul_Latn.txt"
    }
    standard_path = "data/tokenizers/standard_unigramlm.json"
    langspec_paths = {
        "base": "data/tokenizers/langspec_base_tokenizer.json",
        "languages": {
            "en": "data/tokenizers/langspec_en_em_probs.npy",
            "fr": "data/tokenizers/langspec_fr_em_probs.npy",
            "ar": "data/tokenizers/langspec_ar_em_probs.npy",
            "de": "data/tokenizers/langspec_de_em_probs.npy",
            "es": "data/tokenizers/langspec_es_em_probs.npy",
            "hau_Latn": "data/tokenizers/langspec_hau_Latn_em_probs.npy",
            "hi": "data/tokenizers/langspec_hi_em_probs.npy",
            "ru": "data/tokenizers/langspec_ru_em_probs.npy",
            "zh": "data/tokenizers/langspec_zh_em_probs.npy",
            "zul_Latn": "data/tokenizers/langspec_zul_Latn_em_probs.npy",
            "swh_Latn": "data/tokenizers/langspec_swh_Latn_em_probs.npy",
            "yor_Latn": "data/tokenizers/langspec_yor_Latn_em_probs.npy",
        }
    }
    multigram_tokenizer_paths = { "base": "data/tokenizers/multi_base_tokenizer.json", "global_probs": "data/tokenizers/multigram_global_probs.npy" }
    # Suppose parallel_files = {
    #   "en": "data/english.txt",
    #   "fr": "data/french.txt",
    #   ...
    # }
    # standard_tokenizer_path = "..."
    # langspec_tokenizer_paths = { "base": "...", "languages": {...} }
    # multigram_tokenizer_paths = { "base": "...", "global_probs": "..." }
    #
    analyze_parallel_corpus(
        parallel_files,
        standard_path,
        langspec_paths,
        multigram_tokenizer_paths,
        max_lines=1000,
        output_prefix="token_analysis",
        show_plots=False
    )

    pass

