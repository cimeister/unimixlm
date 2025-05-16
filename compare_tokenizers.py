import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
from code.utils import encode_text, load_tokenizer_from_config
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

# Import and setup environment variables
from code.environment_setup import setup_environment
setup_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerAnalyzer:
    """
    A class for analyzing and comparing multiple tokenizers on parallel corpora.
    
    This analyzer measures metrics like:
    - Token length distribution
    - Subword fertility (# tokens / # whitespace words)
    - Compression ratio (tokens/char) per language & overall
    - Type-token ratio (distinct subwords to total subwords)
    - Vocabulary utilization per language
    """
    
    def __init__(self, 
                 parallel_files: Dict[str, str], 
                 tokenizers: Dict[str, Any],
                 output_prefix: str = "tokenizers_analysis",
                 max_lines: Optional[int] = None,
                 save_tokenized_lines: bool = False,
                 tokenized_lines_output_dir: str = "tokenized_analysis_output"):
        """
        Initialize the TokenizerAnalyzer.
        
        Args:
            parallel_files: Dictionary mapping language codes to corpus files
            tokenizers: Dictionary mapping tokenizer names to tokenizer objects
            output_prefix: Prefix for output files
            max_lines: Maximum number of lines to process per language (None for all)
            save_tokenized_lines: Whether to save tokenized outputs to files
            tokenized_lines_output_dir: Directory to save tokenized outputs
        """
        self.parallel_files = parallel_files
        self.tokenizers = tokenizers
        self.output_prefix = output_prefix
        self.max_lines = max_lines
        self.save_tokenized_lines = save_tokenized_lines
        self.tokenized_lines_output_dir = tokenized_lines_output_dir
        
        # Create output directory if needed
        if self.save_tokenized_lines:
            os.makedirs(self.tokenized_lines_output_dir, exist_ok=True)
        
        # Create output directory for plots
        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)
        
        # Data structures to store analysis results
        self.languages = sorted(self.parallel_files.keys())
        self.tokenizer_names = list(self.tokenizers.keys())
        
        # Initialize data structures
        self._initialize_data_structures()
    
    def _initialize_data_structures(self) -> None:
        """Initialize all data structures needed for the analysis."""
        # Token lengths for each tokenizer and language
        self.lengths = {t_name: defaultdict(list) for t_name in self.tokenizer_names}
        
        # Token usage counts across all languages
        self.usage_counts = {t_name: Counter() for t_name in self.tokenizer_names}
        
        # Token usage counts per language
        self.usage_counts_per_lang = {t_name: defaultdict(Counter) for t_name in self.tokenizer_names}
        
        # Character and token counts
        self.total_chars_by_lang = defaultdict(int)
        self.total_tokens_by_method_lang = {t_name: defaultdict(int) for t_name in self.tokenizer_names}
        
        # Whitespace word counts (for fertility calculation)
        self.whitespace_word_counts = defaultdict(int)
        self.total_subwords_for_fertility = {t_name: defaultdict(int) for t_name in self.tokenizer_names}
        
        # Counts for language-specific tokenizers
        self.best_lang_counts = defaultdict(lambda: defaultdict(int))
        self.total_lines_lang = defaultdict(int)
        
        # Results
        self.fertility = None
        self.avg_tokens_per_line = None
        self.best_lang_proportion = None
        self.compression = None
        self.compression_all = None
        self.ttr = None
        self.vocab_utilization = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze all tokenizers on the parallel corpus.
        
        Returns:
            Dictionary containing all computed metrics
        """
        # Process each language
        for lang, path in self.parallel_files.items():
            logger.info(f"Processing language {lang} from {path}")
            self._process_language(lang, path)
        
        # Compute metrics
        self._compute_metrics()
        
        # Generate plots
        self._generate_plots()
        
        # Print summary
        self._print_summary()
        
        # Return all metrics
        return {
            'fertility': self.fertility,
            'avg_tokens_per_line': self.avg_tokens_per_line,
            'best_lang_proportion': self.best_lang_proportion,
            'compression': self.compression,
            'compression_all': self.compression_all,
            'ttr': self.ttr,
            'vocab_utilization': self.vocab_utilization
        }
    
    def _process_language(self, lang: str, path: str) -> None:
        """
        Analyse one language file and store per-tokenizer statistics.
        """
        out_path = os.path.join(self.tokenized_lines_output_dir,
                                f"analysis_{lang}.jsonl")

        # ---------------- load any existing cache -------------------------
        existing = {}
        if os.path.isfile(out_path):
            with open(out_path, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    rec = json.loads(line)
                    existing[rec["text"]] = rec

        records, line_idx = [], 0

        with open(path, "r", encoding="utf-8", errors="replace") as fin:
            for raw in tqdm(fin, desc=f"Analyzing {lang}"):
                text = raw.strip()
                if not text:
                    continue
                line_idx += 1
                if self.max_lines and line_idx > self.max_lines:
                    break

                # basic char / whitespace stats
                self.total_chars_by_lang[lang] += len(text)
                self.whitespace_word_counts[lang] += len(text.split())

                # --------- initialise output record ----------------------
                rec = {"lang": lang, "line_idx": line_idx, "text": text}
                prev = existing.get(text, {})

                # --------- per-tokenizer processing ----------------------
                for name, tok in self.tokenizers.items():
                    k_ids     = f"{name}_ids"
                    k_tokens  = f"{name}_tokens"
                    k_bestlg  = f"{name}_best_lang"

                    if k_ids in prev and k_tokens in prev:
                        ids    = prev[k_ids]
                        tokens = prev[k_tokens]
                        bestlg = prev.get(k_bestlg)
                    else:
                        # decide whether to invoke language detection
                        use_best = isinstance(tok, LanguageSpecificUnigramLMTokenizer)
                        enc = encode_text(
                            tok,
                            text,
                            use_best_language=use_best,
                            language=lang if not use_best else None,
                        )
                        ids    = enc["input_ids"]
                        tokens = enc.get("tokens", [])
                        bestlg = enc.get("best_lang")

                        rec[k_ids]    = ids
                        rec[k_tokens] = tokens
                        if bestlg is not None:
                            rec[k_bestlg] = bestlg

                    # ----- update global stats --------------------------
                    n_tok = len(ids)
                    self.lengths[name][lang].append(n_tok)
                    self.total_tokens_by_method_lang[name][lang] += n_tok
                    self.usage_counts[name].update(ids)
                    self.usage_counts_per_lang[name][lang].update(ids)
                    self.total_subwords_for_fertility[name][lang] += n_tok
                    if bestlg and bestlg == lang:
                        self.best_lang_counts[name][lang] += 1

                self.total_lines_lang[lang] += 1
                if self.save_tokenized_lines and rec:
                    records.append(rec)

        logger.info(f"Processed {min(line_idx, self.max_lines or float('inf'))} lines for {lang}.")

        if self.save_tokenized_lines and records:
            with open(out_path, "w", encoding="utf-8") as fout:
                for rec in records:
                    fout.write(json.dumps(rec) + "\n")

    
    def _compute_metrics(self) -> None:
        """Compute all metrics from the collected data."""
        # 1. Subword fertility (tokens per whitespace word)
        self.fertility = {t_name: {} for t_name in self.tokenizer_names}
        for t_name in self.tokenizer_names:
            for lang in self.languages:
                wcount = self.whitespace_word_counts[lang]
                if wcount > 0:
                    self.fertility[t_name][lang] = self.total_subwords_for_fertility[t_name][lang] / wcount
                else:
                    self.fertility[t_name][lang] = 0.0
        
        # 2. Average tokens per line
        self.avg_tokens_per_line = {t_name: {} for t_name in self.tokenizer_names}
        for t_name in self.tokenizer_names:
            for lang in self.languages:
                lines_len = self.lengths[t_name][lang]
                if lines_len:
                    self.avg_tokens_per_line[t_name][lang] = sum(lines_len) / len(lines_len)
                else:
                    self.avg_tokens_per_line[t_name][lang] = 0.0
        
        # 3. Best-language correctness (for language-specific tokenizers)
        self.best_lang_proportion = {t_name: {} for t_name in self.tokenizer_names}
        for t_name in self.tokenizer_names:
            for lang in self.languages:
                if lang in self.best_lang_counts[t_name] and self.total_lines_lang[lang] > 0:
                    self.best_lang_proportion[t_name][lang] = self.best_lang_counts[t_name][lang] / self.total_lines_lang[lang]
                else:
                    self.best_lang_proportion[t_name][lang] = 0.0
        
        # 4. Compression (tokens per character)
        self.compression = {t_name: {} for t_name in self.tokenizer_names}
        sum_chars_total = 0
        sum_tokens_total = {t_name: 0 for t_name in self.tokenizer_names}
        
        for lang in self.languages:
            sum_chars = self.total_chars_by_lang[lang]
            sum_chars_total += sum_chars
            
            for t_name in self.tokenizer_names:
                sum_tokens_m_lang = self.total_tokens_by_method_lang[t_name][lang]
                if sum_chars > 0:
                    c_rate = sum_tokens_m_lang / sum_chars
                else:
                    c_rate = 0
                self.compression[t_name][lang] = c_rate
                sum_tokens_total[t_name] += sum_tokens_m_lang
        
        # Overall compression
        self.compression_all = {}
        for t_name in self.tokenizer_names:
            if sum_chars_total > 0:
                self.compression_all[t_name] = sum_tokens_total[t_name] / sum_chars_total
            else:
                self.compression_all[t_name] = 0
        
        # 5. Type-Token Ratio (TTR)
        self.ttr = {}
        for t_name in self.tokenizer_names:
            distinct_count = len(self.usage_counts[t_name])
            total_occ = sum(self.usage_counts[t_name].values())
            if total_occ > 0:
                self.ttr[t_name] = distinct_count / total_occ
            else:
                self.ttr[t_name] = 0.0
        
        # 6. Vocabulary utilization
        self.vocab_utilization = {t_name: {} for t_name in self.tokenizer_names}
        for t_name in self.tokenizer_names:
            for lang in self.languages:
                self.vocab_utilization[t_name][lang] = len(self.usage_counts_per_lang[t_name][lang])
    
    def _generate_plots(self) -> None:
        """Generate all visualization plots."""
        self._plot_boxplots()
        self._plot_best_lang_proportions()
        self._plot_ttr()
        self._plot_compression()
    
    def _plot_boxplots(self) -> None:
        """Generate boxplots for token lengths."""
        # Create a DataFrame for plotnine
        rows = []
        for lang in self.languages:
            for t_name in self.tokenizer_names:
                for token_value in self.lengths[t_name][lang]:
                    rows.append({
                        "language": lang,
                        "method": t_name,
                        "tokens": token_value,
                    })
        
        df = pd.DataFrame(rows)
        
        try:
            from plotnine import ggplot, aes, geom_boxplot, facet_wrap, labs, theme, element_text, scale_x_discrete
            
            # Plot 1: Group by language
            p = (ggplot(df, aes(x='method', y='tokens'))
                + geom_boxplot(width=0.8)
                + facet_wrap('~language', nrow=1)
                + labs(title="Token parity plot (group by language)",
                        x="Method", y="Tokens per line")
                + theme(axis_text_x=element_text(rotation=45, ha='right'))
            )
            
            p.save(f"{self.output_prefix}_parity_boxplot_group_by_lang.png", dpi=300, 
                  width=min(20, max(15, len(self.languages) * 1.5)), height=6)
            
            # Plot 2: Group by method
            p = (ggplot(df, aes(x='language', y='tokens'))
                + geom_boxplot(width=0.8)
                + facet_wrap('~method', nrow=1)
                + labs(title="Token parity plot (group by method)",
                        x="Language", y="Tokens per line")
                + theme(axis_text_x=element_text(rotation=45, ha='right'))
            )
            
            p.save(f"{self.output_prefix}_parity_boxplot_group_by_method.png", dpi=300, 
                  width=min(20, max(15, len(self.tokenizer_names) * 2)), height=6)
        
        except ImportError:
            logger.warning("plotnine not available. Using matplotlib for boxplots instead.")
            
            # Fallback to matplotlib (simpler plots)
            # Plot 1: Group by language
            data_for_box = []
            group_labels = []
            
            for lang in self.languages:
                for t_name in self.tokenizer_names:
                    data_for_box.append(self.lengths[t_name][lang])
                    group_labels.append(f"{lang}-{t_name}")
            
            plt.figure(figsize=(max(10, len(group_labels) * 0.5), 6))
            plt.boxplot(data_for_box, labels=group_labels)
            plt.title("Token parity plot (group by language)")
            plt.xticks(rotation=45)
            plt.ylabel("Tokens per line")
            plt.tight_layout()
            plt.savefig(f"{self.output_prefix}_parity_boxplot.png", dpi=150)
            plt.close()
    
    def _plot_best_lang_proportions(self) -> None:
        """Generate bar chart for best language proportions."""
        # Only plot for tokenizers that have language-specific data
        valid_tokenizers = [t_name for t_name in self.tokenizer_names 
                           if any(self.best_lang_proportion[t_name].values())]
        
        if not valid_tokenizers:
            logger.info("No valid tokenizers for lang specific decoding")
            return
        
        for t_name in valid_tokenizers:
            plt.figure(figsize=(max(8, len(self.languages) * 0.8), 5))
            x = np.arange(len(self.languages))
            rates = [self.best_lang_proportion[t_name].get(lang, 0) * 100.0 for lang in self.languages]
            plt.bar(x, rates)
            plt.xticks(x, self.languages, rotation=45)
            plt.ylabel("Percentage (%)")
            plt.title(f"Proportion of lines where native language is best ({t_name})")
            plt.tight_layout()
            plt.savefig(f"{self.output_prefix}_{t_name}_best_lang_native.png", dpi=150)
            plt.close()
    
    def _plot_ttr(self) -> None:
        """Generate bar chart for Type-Token Ratio."""
        plt.figure(figsize=(max(8, len(self.tokenizer_names) * 0.8), 5))
        mvals = [self.ttr[t_name] * 100.0 for t_name in self.tokenizer_names]
        x = np.arange(len(self.tokenizer_names))
        plt.bar(x, mvals)
        plt.xticks(x, self.tokenizer_names, rotation=45)
        plt.ylabel("TTR (%) = (distinct subwords / total subwords)*100")
        plt.title("Type-Token Ratio across entire corpus")
        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_ttr_bar.png", dpi=150)
        plt.close()
    
    def _plot_compression(self) -> None:
        """Generate bar chart for overall compression."""
        plt.figure(figsize=(max(8, len(self.tokenizer_names) * 0.8), 5))
        cvals = [self.compression_all[t_name] for t_name in self.tokenizer_names]
        x = np.arange(len(self.tokenizer_names))
        plt.bar(x, cvals)
        plt.xticks(x, self.tokenizer_names, rotation=45)
        plt.ylabel("Tokens per char (lower => more compressed)")
        plt.title("Overall compression across entire parallel corpus")
        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_overall_compression_bar.png", dpi=150)
        plt.close()
    
    def _print_summary(self) -> None:
        """Print a summary of the analysis results."""
        logger.info("\n===== Final Analysis Summary =====")
        
        # Best-language correctness
        has_best_lang_data = any(any(self.best_lang_proportion[t_name].values()) 
                                for t_name in self.tokenizer_names)
        
        if has_best_lang_data:
            logger.info("Best-language correctness => proportion of lines where native is best:")
            for t_name in self.tokenizer_names:
                if any(self.best_lang_proportion[t_name].values()):
                    logger.info(f"\n{t_name}:")
                    for lang in self.languages:
                        count = self.best_lang_counts[t_name].get(lang, 0)
                        total = self.total_lines_lang[lang]
                        if total > 0:
                            logger.info(f"  {lang}: {self.best_lang_proportion[t_name].get(lang, 0)*100:.2f}% ({count}/{total})")
        
        # Subword fertility
        logger.info("\nSubword Fertility = #subwords / #whitespace_words:")
        for t_name in self.tokenizer_names:
            logger.info(f"\n{t_name}:")
            for lang in self.languages:
                logger.info(f"  {lang}: {self.fertility[t_name][lang]:.3f}")
        
        # Average tokens per line
        logger.info("\nAverage tokens per line:")
        for t_name in self.tokenizer_names:
            logger.info(f"\n{t_name}:")
            for lang in self.languages:
                logger.info(f"  {lang}: {self.avg_tokens_per_line[t_name][lang]:.3f}")
        
        # Compression
        logger.info("\nCompression (tokens/char) - smaller => more compressed:")
        for t_name in self.tokenizer_names:
            comp_str = ", ".join(f"{lang}:{self.compression[t_name][lang]:.3f}" for lang in self.languages)
            logger.info(f"{t_name}: {comp_str}")
        
        logger.info("\nOverall compression => tokens/char across all languages:")
        for t_name in self.tokenizer_names:
            logger.info(f"  {t_name}: {self.compression_all[t_name]:.3f}")
        
        # TTR
        logger.info("\nType-Token Ratio (distinct subwords / total subwords), entire corpus:")
        for t_name in self.tokenizer_names:
            logger.info(f"  {t_name}: {self.ttr[t_name]:.4f}")
        
        # Vocab utilization
        logger.info("\nVocabulary utilization (# unique subwords used in each language):")
        for t_name in self.tokenizer_names:
            logger.info(f"\n{t_name}:")
            for lang in self.languages:
                logger.info(f"  {lang}: {self.vocab_utilization[t_name][lang]}")
        
        logger.info("\nAnalysis complete.")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare tokenizers on parallel corpora')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--max_lines', type=int, default=2000, help='Maximum lines to process per language')
    parser.add_argument('--output_prefix', type=str, default='data/results/analysis', help='Prefix for output files')
    parser.add_argument('--save_tokenized', action='store_true', help='Save tokenized outputs')
    parser.add_argument('--tokenized_dir', type=str, default='tokenized', help='Directory for tokenized outputs')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8', errors="replace") as f:
        config = json.load(f)
    
    # Load parallel files
    parallel_files = config['parallel_files']
    
    # Load tokenizers
    tokenizers = {}
    for name, tok_config in config['tokenizers'].items():
        tokenizer = load_tokenizer_from_config(tok_config)
        tokenizers[name] = tokenizer
        logger.info(f"Loaded tokenizer '{name}'")
    
    # Create analyzer
    analyzer = TokenizerAnalyzer(
        parallel_files=parallel_files,
        tokenizers=tokenizers,
        output_prefix=args.output_prefix,
        max_lines=args.max_lines,
        save_tokenized_lines=args.save_tokenized,
        tokenized_lines_output_dir=args.tokenized_dir
    )
    
    # Run analysis
    analyzer.analyze()


if __name__ == "__main__":
    main()
