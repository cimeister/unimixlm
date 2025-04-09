# compare_three_tokenizers.py

import os
import logging
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm 

from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.multigramlm import MultigramLMTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_tokenizers_on_parallel_corpus(
    parallel_files,
    standard_tokenizer,
    langspec_tokenizer,
    multigram_tokenizer,
    multi_langspec_tokenizer,
    multi_soft_tokenizer,
    max_lines=2000,
    output_prefix="tokenizers_analysis",
    show_plots=False,
    save_tokenized_lines=False,
    tokenized_lines_output_dir="tokenized_analysis_output"
):
    """
    Compare three tokenizers (Standard, Language-Specific, Multigram) on a parallel corpus.

    1) We parse each line from each language. 
    2) For Standard => standard_tokenizer.encode(text).
    3) For Language-Specific => best-language approach to see how often the 'native' language is actually the shortest.
    4) For Multigram => e.g. mixture_viterbi_tokenize => encode or tokens -> IDs.

    We'll gather:
      - token length distribution
      - subword fertility (# tokens / # whitespace words)
      - best-language correctness (native is shortest)
      - compression ratio (tokens/char) per lang & overall
      - ratio of distinct subwords to total subwords (type-token ratio)
      - how many unique subwords are used in each language (vocab utilization)

    We produce:
      - boxplots (token lengths) including a "flush" version with each method grouped
      - bar charts for compression, TTR, etc.
      - final stats printed once at the end

    If save_tokenized_lines=True, we store tokenization results in JSON for debugging:
      { "lang":..., "line_idx":..., "text":..., 
        "standard_tokens":..., "standard_ids":..., 
        "langspec_tokens":..., "langspec_ids":...,
        "multigram_tokens":..., "multigram_ids":... }
      in a subdirectory under tokenized_lines_output_dir.
    """

    methods = ["standard", "langspec", "multigram", "multi-langspec", "multi-soft"]
    # We'll store token lengths in lengths[method][lang] = list of counts
    lengths = {m: defaultdict(list) for m in methods}

    # usage_counts => how many times each subword ID is used (across the entire corpus, all languages).
    usage_counts = {m: Counter() for m in methods}

    # We'll also track usage per language, to measure per-language vocab usage.
    usage_counts_per_lang = {m: defaultdict(Counter) for m in methods}
    # e.g. usage_counts_per_lang[method][lang][token_id]++

    # total chars, total tokens
    total_chars_by_lang = defaultdict(int)
    total_tokens_by_method_lang = {m: defaultdict(int) for m in methods}

    # For subword fertility => we track total "whitespace words" by language, method's total subwords
    # Actually subword fertility is # subwords / # whitespace-words.
    whitespace_word_counts = defaultdict(int)
    total_subwords_for_fertility = {m: defaultdict(int) for m in methods}

    # For best-language analysis => how often the 'native' language is the shortest
    # We'll define best_lang_count[lang] = how many lines the "native" lang is the shortest in the language-specific approach
    best_lang_count = defaultdict(int)
    best_lang_count_mm = defaultdict(int)
    total_lines_lang = defaultdict(int)

    # If user wants to save tokenized lines:
    if save_tokenized_lines:
        os.makedirs(tokenized_lines_output_dir, exist_ok=True)

    def encode_standard(text):
        return standard_tokenizer.encode(text)

    def encode_langspec(tokenizer, text, lang):
        # We'll do a best-language approach for the segmentation
        best_lang, best_tokens = tokenizer.custom_viterbi_best_language(text)
        # Convert tokens -> IDs
        vocab = tokenizer.get_vocab()
        unk = tokenizer.unk_token
        ids = [vocab.get(t, vocab.get(unk,0)) for t in best_tokens]
        return ids, best_lang, best_tokens

    def encode_multigram(tokenizer, text):
        # tokens => mixture_viterbi_tokenize
        tokens = tokenizer.mixture_viterbi_tokenize(text)
        vocab = tokenizer.get_vocab()
        unk = tokenizer.unk_token
        ids = [vocab.get(t, vocab.get(unk, 0)) for t in tokens]
        return ids, tokens


    logger.info("Beginning analysis over parallel corpus.")
    for lang, path in parallel_files.items():
        logger.info(f"Processing language {lang} from {path}")
        line_idx=0

        out_path = os.path.join(tokenized_lines_output_dir, f"analysis_{lang}.jsonl")
        cur_results = {}
        if os.path.isfile(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    cur_results[tmp['text']] = tmp
        if save_tokenized_lines:
            records = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Analyzing {lang}"):
                text = line.strip()
                if not text:
                    continue
                if text in cur_results:
                    std_ids = cur_results[text].get('standard_ids', None)
                    ls_ids = cur_results[text].get('langspec_ids', None)
                    ls_tokens = cur_results[text].get('langspec_tokens', None)
                    best_lang = cur_results[text].get('best_lang', None)
                    mm_ids = cur_results[text].get('multigram_ids', None)
                    mm_tokens = cur_results[text].get('multigram_tokens', None)
                    mm_soft_ids = cur_results[text].get('multi_soft_ids', None)
                    mm_soft_tokens = cur_results[text].get('multi_soft_tokens', None)
                    ls_mm_ids = cur_results[text].get('ls_multigram_ids', None)
                    ls_mm_tokens = cur_results[text].get('ls_multigram_tokens', None)
                    best_lang_mm = cur_results[text].get('best_lang_mm', None)

                line_idx+=1
                if max_lines and line_idx>max_lines:
                    break

                # Count chars
                total_chars_by_lang[lang]+= len(text)

                # Count whitespace words
                words = text.split()
                whitespace_word_counts[lang]+= len(words)

                # 1) standard
                if std_ids is None:
                    std_ids = encode_standard(text)
                len_std = len(std_ids)
                lengths["standard"][lang].append(len_std)
                total_tokens_by_method_lang["standard"][lang]+= len_std
                usage_counts["standard"].update(std_ids)
                usage_counts_per_lang["standard"][lang].update(std_ids)
                total_subwords_for_fertility["standard"][lang]+= len_std

                # 2) langspec
                if not ls_ids or not best_lang or not ls_tokens:
                    ls_ids, best_lang, ls_tokens = encode_langspec(langspec_tokenizer, text, lang)
                len_ls = len(ls_ids)
                lengths["langspec"][lang].append(len_ls)
                total_tokens_by_method_lang["langspec"][lang]+= len_ls
                usage_counts["langspec"].update(ls_ids)
                usage_counts_per_lang["langspec"][lang].update(ls_ids)
                total_subwords_for_fertility["langspec"][lang]+= len_ls

                # If the line is "native" to lang, check if best_lang==lang => i.e. "native is best"
                # Then check if it's indeed the shortest among all distributions => we need the length of that distribution vs. standard/multigram??? 
                # Actually we interpret "shortest segmentation" only among the language-specific approach? 
                # We'll interpret "best_lang" as the chosen distribution. If best_lang == lang => "native is best".
                if best_lang == lang:
                    best_lang_count[lang]+= 1

                # 3) multigram
                if not mm_ids or not mm_tokens:
                    mm_ids, mm_tokens = encode_multigram(multigram_tokenizer, text)
                len_mm = len(mm_ids)
                lengths["multigram"][lang].append(len_mm)
                total_tokens_by_method_lang["multigram"][lang]+= len_mm
                usage_counts["multigram"].update(mm_ids)
                usage_counts_per_lang["multigram"][lang].update(mm_ids)
                total_subwords_for_fertility["multigram"][lang]+= len_mm
                
                 # 4) multi-langspec
                if not ls_mm_ids or not best_lang_mm or not ls_mm_tokens:
                    ls_mm_ids, best_lang_mm, ls_mm_tokens = encode_langspec(multi_langspec_tokenizer, text, lang)
                len_ls = len(ls_mm_ids)
                lengths["multi-langspec"][lang].append(len_ls)
                total_tokens_by_method_lang["multi-langspec"][lang]+= len_ls
                usage_counts["multi-langspec"].update(ls_mm_ids)
                usage_counts_per_lang["multi-langspec"][lang].update(ls_mm_ids)
                total_subwords_for_fertility["multi-langspec"][lang]+= len_ls

                if best_lang_mm == lang:
                    best_lang_count_mm[lang]+= 1

                # 5) multigram-soft
                if not mm_soft_ids or not mm_soft_tokens:
                    mm_soft_ids, mm_soft_tokens = encode_multigram(multi_soft_tokenizer, text)
                len_soft_mm = len(mm_soft_ids)
                lengths["multi-soft"][lang].append(len_soft_mm)
                total_tokens_by_method_lang["multi-soft"][lang]+= len_soft_mm
                usage_counts["multi-soft"].update(mm_soft_ids)
                usage_counts_per_lang["multi-soft"][lang].update(mm_soft_ids)
                total_subwords_for_fertility["multi-soft"][lang]+= len_soft_mm

                total_lines_lang[lang]+= 1

                # optionally write out JSON record
                if save_tokenized_lines:
                    record = {
                        "lang": lang,
                        "line_idx": line_idx,
                        "text": text,
                        "standard_ids": std_ids,
                        "standard_tokens": [standard_tokenizer.base_tokenizer.id_to_token(i) for i in std_ids],
                        "langspec_ids": ls_ids,
                        "langspec_tokens": ls_tokens,
                        "best_lang": best_lang,
                        "multigram_ids": mm_ids,
                        "multigram_tokens": mm_tokens,
                        "ls_multigram_ids": ls_mm_ids,
                        "ls_multigram_tokens": ls_mm_tokens,
                        "best_lang_mm": best_lang_mm,
                        "multi_soft_ids": mm_soft_ids,
                        "multi_soft_tokens": mm_soft_tokens,
                    }
                    records.append(record)
                    
        # We'll store lines if user wants:
        if save_tokenized_lines:

            with open(out_path, 'w', encoding='utf-8') as fout:
                for record in records:
                    fout.write(json.dumps(record) + "\n")

       
        logger.info(f"Processed {min(line_idx, max_lines)} lines for {lang}.")


    # All lines processed, now we compute stats.

    # 1) "Flush" token parity plot: 
    #    we'll produce a boxplot with each method, each language, but the box plots are grouped by method 
    #    or grouped by language. Let's do grouped by language. 
    # We'll do both variations:

    languages = sorted(parallel_files.keys())
    methods_label = ["std", "ls", "mm"]

    # Variation A: group by language => for each language, we have 3 boxplots next to each other
    data_for_box = []
    group_labels = []
    positions = []  # we'll store the x-axis positions here

    for i, lang in enumerate(languages):
        # For each language, we add 5 boxes with consecutive positions.
        base = i * 5  # 5 boxes per language group
        data_for_box.append(lengths["standard"][lang])
        data_for_box.append(lengths["langspec"][lang])
        data_for_box.append(lengths["multigram"][lang])
        data_for_box.append(lengths["multi-soft"][lang])
        data_for_box.append(lengths["multi-langspec"][lang])
        
        group_labels.append(f"{lang}-std")
        group_labels.append(f"{lang}-ls")
        group_labels.append(f"{lang}-mm")
        group_labels.append(f"{lang}-ms")
        group_labels.append(f"{lang}-ls-mm")
        
        # Create positions with no gap between boxes within the same language.
        positions.extend([base + j for j in range(5)])

    plt.figure(figsize=(max(10, len(group_labels) * 0.5), 6))
    plt.boxplot(data_for_box, positions=positions, labels=group_labels)
    plt.title("Token parity plot (group by language)")
    plt.xticks(rotation=45)
    plt.ylabel("Tokens per line")
    plt.tight_layout()
    flush_box_out = f"{output_prefix}_parity_boxplot_group_by_lang.png"
    plt.savefig(flush_box_out, dpi=150)
    plt.close()

    import pandas as pd
    from plotnine import ggplot, aes, geom_boxplot, facet_wrap, labs, theme, element_text, scale_x_discrete

    # Create a long DataFrame from your data.
    rows = []
    for lang in languages:
        for method, label_suffix in [("standard", "std"),
                                     ("langspec", "ls"),
                                     ("multigram", "mm"),
                                     ("multi-langspec", "ls-mm"),
                                     ("multi-soft", "ms")]:
            # Assuming lengths[method][lang] is a list of token values.
            for token_value in lengths[method][lang]:
                rows.append({
                    "language": lang,
                    "method": label_suffix,
                    "tokens": token_value,
                })

    df = pd.DataFrame(rows)

    # Create the Plotnine plot using facet_wrap to split by language.
    p = (ggplot(df, aes(x='method', y='tokens'))
         + geom_boxplot(width=0.8)
         + scale_x_discrete(limits=["std", "ls", "mm", "ls-mm", "ms"])
         + facet_wrap('~language', nrow=1)
         + labs(title="Token parity plot (group by language)",
                x="Method", y="Tokens per line")
         + theme(axis_text_x=element_text(rotation=45, ha='right'))
    )

    # Save the plot.
    p.save(f"{output_prefix}_parity_boxplot_group_by_lang_plotnine.png", dpi=300, width=15, height=6)

    # Create the Plotnine plot using facet_wrap to split by language.
    p = (ggplot(df, aes(x='language', y='tokens'))
         + geom_boxplot(width=0.8)
         #+ scale_x_discrete(limits=["std", "ls", "mm", "ls-mm", "ms"])
         + facet_wrap('~method', nrow=1)
         + labs(title="Token parity plot (group by language)",
                x="Method", y="Tokens per line")
         + theme(axis_text_x=element_text(rotation=45, ha='right'))
    )

    # Save the plot.
    p.save(f"{output_prefix}_parity_boxplot_group_by_method_plotnine.png", dpi=300, width=15, height=6)


    # Variation B: group by method => for each method, we have len(languages) boxplots. 
    # This might be less common, but we show the idea:
    data_method = []
    labels_method = []
    for m in methods:
        for lang in languages:
            data_method.append(lengths[m][lang])
            labels_method.append(f"{m[:2]}-{lang}")
    plt.figure(figsize=(max(10, len(labels_method)*0.5),6))
    plt.boxplot(data_method, labels=labels_method)
    plt.title("Token parity plot grouped by method.")
    plt.xticks(rotation=45)
    plt.ylabel("Tokens per line")
    meth_box_out = f"{output_prefix}_parity_boxplot_group_by_method.png"
    plt.tight_layout()
    plt.savefig(meth_box_out, dpi=150)
    plt.close()

    # 2) subword fertility => # tokens / # whitespace words
    # We'll do a dictionary fertility[method][lang]
    fertility = {m: {} for m in methods}
    for m in methods:
        for lang in languages:
            wcount = whitespace_word_counts[lang]
            if wcount>0:
                fertility[m][lang] = total_subwords_for_fertility[m][lang]/ wcount
            else:
                fertility[m][lang] = 0.0

    # 3) average tokens per line => just the average of lengths
    # average_tokens_per_line[method][lang]
    avg_tokens_per_line = {m: {} for m in methods}
    for m in methods:
        for lang in languages:
            lines_len = lengths[m][lang]
            if lines_len:
                avg_tokens_per_line[m][lang] = sum(lines_len)/ len(lines_len)
            else:
                avg_tokens_per_line[m][lang] = 0.0

    # 4) best-language correctness => proportion of lines for which the "native" language is best 
    # best_lang_count[lang] / total_lines_lang[lang]
    best_lang_proportion = {}
    for lang in languages:
        if total_lines_lang[lang]>0:
            best_lang_proportion[lang] = best_lang_count[lang]/ total_lines_lang[lang]
        else:
            best_lang_proportion[lang] = 0.0

    # 5) compression => tokens/char, per language, and aggregated across all
    # aggregated => sum of tokens for all lines in that method / sum of chars for all lines
    compression = {m: {} for m in methods}
    sum_chars_total = 0
    sum_tokens_total = {m:0 for m in methods}
    for lang in languages:
        sum_chars = total_chars_by_lang[lang]
        sum_chars_total += sum_chars
        for m in methods:
            sum_tokens_m_lang = total_tokens_by_method_lang[m][lang]
            if sum_chars>0:
                c_rate = sum_tokens_m_lang/ sum_chars
            else:
                c_rate = 0
            compression[m][lang] = c_rate
            sum_tokens_total[m]+= sum_tokens_m_lang
    compression_all = {}
    for m in methods:
        if sum_chars_total>0:
            compression_all[m] = sum_tokens_total[m]/ sum_chars_total
        else:
            compression_all[m] = 0

    # 6) ratio of distinct subwords to total subwords => TTR (type-token ratio)
    # distinct subwords = size of usage_counts[m].keys(), total subwords = sum(usage_counts[m].values())
    ttr = {}
    for m in methods:
        distinct_count = len(usage_counts[m])
        total_occ = sum(usage_counts[m].values())
        if total_occ>0:
            ttr[m] = distinct_count / total_occ
        else:
            ttr[m] = 0.0

    # 7) vocabulary utilization per language => how many unique subwords are used for that language
    # usage_counts_per_lang[m][lang] is a Counter => len(...) is the distinct subwords
    vocab_utilization = {m: {} for m in methods}
    for m in methods:
        for lang in languages:
            vocab_utilization[m][lang] = len(usage_counts_per_lang[m][lang])

    # Now let's produce a bar chart for some of these stats, and print them once at the end.

    # We'll produce a bar chart for best-lang correctness:
    plt.figure()
    x = np.arange(len(languages))
    rates = [best_lang_proportion[lang]*100.0 for lang in languages]
    plt.bar(x, rates)
    plt.xticks(x, languages)
    plt.ylabel("Percentage (%)")
    plt.title("Proportion of lines for which native language is best (lang-spec).")
    bestlang_path = f"{output_prefix}_best_lang_native.png"
    plt.tight_layout()
    plt.savefig(bestlang_path, dpi=150)
    plt.close()

    # We'll produce a bar chart for TTR across methods
    plt.figure()
    mvals = [ttr[m]*100.0 for m in methods]
    x2 = np.arange(len(methods))
    plt.bar(x2, mvals)
    plt.xticks(x2, methods)
    plt.ylabel("TTR (%) = (distinct subwords / total subwords)*100")
    plt.title("Type-Token Ratio across entire corpus")
    ttr_path = f"{output_prefix}_ttr_bar.png"
    plt.tight_layout()
    plt.savefig(ttr_path, dpi=150)
    plt.close()

    # We'll produce a bar chart for compression_all => tokens/char across entire corpus
    plt.figure()
    cvals = [compression_all[m] for m in methods]
    x3 = np.arange(len(methods))
    plt.bar(x3, cvals)
    plt.xticks(x3, methods)
    plt.ylabel("Tokens per char (lower => more compressed) [all languages]")
    plt.title("Overall compression across entire parallel corpus")
    compall_path = f"{output_prefix}_overall_compression_bar.png"
    plt.tight_layout()
    plt.savefig(compall_path, dpi=150)
    plt.close()

    # Print stats once at the end:
    logger.info("\n===== Final Analysis Summary =====")
    # best-language correctness
    logger.info("Best-language correctness (lang-spec) => proportion of lines where native is best:")
    for lang in languages:
        logger.info(f"  {lang}: {best_lang_proportion[lang]*100:.2f}%  ( {best_lang_count[lang]}/{total_lines_lang[lang]} )")

    # subword fertility
    logger.info("\nSubword Fertility = #subwords / #whitespace_words:")
    for m in methods:
        logger.info(f"{m}:")
        for lang in languages:
            logger.info(f"  {lang}: {fertility[m][lang]:.3f}")

    # average tokens per line
    logger.info("\nAverage tokens per line:")
    for m in methods:
        logger.info(f"{m}:")
        for lang in languages:
            logger.info(f"  {lang}: {avg_tokens_per_line[m][lang]:.3f}")

    # compression per lang + overall
    logger.info("\nCompression (tokens/char) - smaller => more compressed:")
    for m in methods:
        comp_str = ", ".join(f"{lang}:{compression[m][lang]:.3f}" for lang in languages)
        logger.info(f"{m}: {comp_str}")
    logger.info(f"Overall compression => tokens/char across all languages:")
    for m in methods:
        logger.info(f"  {m}: {compression_all[m]:.3f}")

    # TTR
    logger.info("\nType-Token Ratio (distinct subwords / total subwords), entire corpus:")
    for m in methods:
        logger.info(f"  {m}: {ttr[m]:.4f}")

    # vocab utilization per language
    logger.info("\nVocabulary utilization (# unique subwords used in each language):")
    for m in methods:
        logger.info(f" {m}:")
        for lang in languages:
            logger.info(f"   {lang}: {vocab_utilization[m][lang]}")

    logger.info("Done.")


if __name__=="__main__":
    
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
    multi_soft_tokenizer_paths = { "base": "data/tokenizers/multi_base_tokenizer_soft.json", "global_probs": "data/tokenizers/multigram_global_probs_soft.npy" }

    multi_langspec_paths = {
        "base": multigram_tokenizer_paths['base'], 
        "languages": {
            "en": "data/tokenizers/multi_langspec_en_em_probs.npy",
            "fr": "data/tokenizers/multi_langspec_fr_em_probs.npy",
            "ar": "data/tokenizers/multi_langspec_ar_em_probs.npy",
            "de": "data/tokenizers/multi_langspec_de_em_probs.npy",
            "es": "data/tokenizers/multi_langspec_es_em_probs.npy",
            "hau_Latn": "data/tokenizers/multi_langspec_hau_Latn_em_probs.npy",
            "hi": "data/tokenizers/multi_langspec_hi_em_probs.npy",
            "ru": "data/tokenizers/multi_langspec_ru_em_probs.npy",
            "zh": "data/tokenizers/multi_langspec_zh_em_probs.npy",
            "zul_Latn": "data/tokenizers/multi_langspec_zul_Latn_em_probs.npy",
            "swh_Latn": "data/tokenizers/multi_langspec_swh_Latn_em_probs.npy",
            "yor_Latn": "data/tokenizers/multi_langspec_yor_Latn_em_probs.npy",
        }
    }

    standard_tok = StandardUnigramLMTokenizer()
    standard_tok.load(standard_path) # re-indexed
    
    langspec_tok = LanguageSpecificUnigramLMTokenizer()
    langspec_tok.load(langspec_paths['base'], langspec_paths['languages']) # re-indexed
    
    multi_tok    = MultigramLMTokenizer()
    multi_tok.load(multigram_tokenizer_paths['base'], multigram_tokenizer_paths['global_probs'])

    multi_langspec_tok = LanguageSpecificUnigramLMTokenizer()
    multi_langspec_tok.load(multi_langspec_paths['base'], multi_langspec_paths['languages'])

    multi_soft_tok    = MultigramLMTokenizer()
    multi_soft_tok.load(multi_soft_tokenizer_paths['base'], multi_soft_tokenizer_paths['global_probs'])
    
    analyze_tokenizers_on_parallel_corpus(
      parallel_files,
      standard_tok,
      langspec_tok,
      multi_tok,
      multi_langspec_tok,
      multi_soft_tok,
      max_lines=2000,
      output_prefix="data/results/analysis",
      show_plots=False,
      save_tokenized_lines=True,
      tokenized_lines_output_dir="data/parallel_text_dataset/tokenized"
    )

