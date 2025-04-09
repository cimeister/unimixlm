"""
Results comparison and report generation module for multilingual tokenization experiments.
This module generates a comprehensive report comparing the standard and language-specific
UnigramLM tokenization methods.
"""

import os
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.language_selection import SELECTED_LANGUAGES, LANGUAGES

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
REPORT_DIR = os.path.join(DATA_DIR, "report")

# Create directories if they don't exist
os.makedirs(REPORT_DIR, exist_ok=True)

def load_perplexity_results():
    """
    Load perplexity results from file.
    
    Returns:
        Dictionary with perplexity results
    """
    results_path = os.path.join(RESULTS_DIR, "perplexity_results.json")
    
    if not os.path.exists(results_path):
        logger.error(f"Perplexity results file not found: {results_path}")
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        logger.error(f"Error loading perplexity results: {str(e)}")
        return None

def load_perplexity_comparison():
    """
    Load perplexity comparison from file.
    
    Returns:
        Dictionary with perplexity comparison
    """
    comparison_path = os.path.join(RESULTS_DIR, "perplexity_comparison.json")
    
    if not os.path.exists(comparison_path):
        logger.error(f"Perplexity comparison file not found: {comparison_path}")
        return None
    
    try:
        with open(comparison_path, 'r', encoding='utf-8') as f:
            comparison = json.load(f)
        
        return comparison
    
    except Exception as e:
        logger.error(f"Error loading perplexity comparison: {str(e)}")
        return None

def generate_perplexity_chart(results, output_path):
    """
    Generate a chart comparing perplexity scores.
    
    Args:
        results: Dictionary with perplexity results
        output_path: Path to save the chart
    
    Returns:
        Path to the generated chart
    """
    if not results:
        logger.error("No perplexity results to generate chart")
        return None
    
    try:
        # Extract perplexity scores for each language
        languages = []
        standard_scores = []
        langspec_scores = []
        
        for lang_code in SELECTED_LANGUAGES:
            if lang_code in results['standard'] and lang_code in results['langspec']:
                languages.append(LANGUAGES[lang_code]['name'])
                standard_scores.append(results['standard'][lang_code])
                langspec_scores.append(results['langspec'][lang_code])
        
        # Create a bar chart
        fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
        
        x = np.arange(len(languages))
        width = 0.35
        
        ax.bar(x - width/2, standard_scores, width, label='Standard UnigramLM')
        ax.bar(x + width/2, langspec_scores, width, label='Language-Specific UnigramLM')
        
        ax.set_title('Perplexity Comparison by Language')
        ax.set_xlabel('Language')
        ax.set_ylabel('Perplexity (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Generated perplexity chart: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating perplexity chart: {str(e)}")
        return None

def generate_improvement_chart(comparison, output_path):
    """
    Generate a chart showing improvement percentages.
    
    Args:
        comparison: Dictionary with perplexity comparison
        output_path: Path to save the chart
    
    Returns:
        Path to the generated chart
    """
    if not comparison or 'improvement_percentage' not in comparison:
        logger.error("No improvement percentages to generate chart")
        return None
    
    try:
        # Extract improvement percentages for each language
        languages = []
        improvements = []
        colors = []
        
        for lang_code, percentage in comparison['improvement_percentage'].items():
            languages.append(LANGUAGES[lang_code]['name'])
            improvements.append(percentage)
            
            # Color based on which model is better
            if comparison['better_model_per_language'][lang_code] == 'standard':
                colors.append('blue')
            else:
                colors.append('orange')
        
        # Create a bar chart
        fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
        
        ax.bar(languages, improvements, color=colors)
        
        ax.set_title('Improvement Percentage by Language')
        ax.set_xlabel('Language')
        ax.set_ylabel('Improvement Percentage')
        plt.xticks(rotation=45, ha='right')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Standard UnigramLM better'),
            Patch(facecolor='orange', label='Language-Specific UnigramLM better')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Generated improvement chart: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating improvement chart: {str(e)}")
        return None

def generate_resource_level_analysis(results, comparison):
    """
    Analyze perplexity results by language resource level.
    
    Args:
        results: Dictionary with perplexity results
        comparison: Dictionary with perplexity comparison
    
    Returns:
        Dictionary with resource level analysis
    """
    if not results or not comparison:
        logger.error("Missing results or comparison for resource level analysis")
        return None
    
    try:
        # Group languages by resource level
        resource_levels = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for lang_code in SELECTED_LANGUAGES:
            if lang_code in results['standard'] and lang_code in results['langspec']:
                resource_level = LANGUAGES[lang_code]['resource_level']
                resource_levels[resource_level].append(lang_code)
        
        # Calculate average perplexity and improvement by resource level
        analysis = {
            'high': {
                'standard_avg_perplexity': 0,
                'langspec_avg_perplexity': 0,
                'avg_improvement': 0,
                'better_model': None,
                'language_count': len(resource_levels['high'])
            },
            'medium': {
                'standard_avg_perplexity': 0,
                'langspec_avg_perplexity': 0,
                'avg_improvement': 0,
                'better_model': None,
                'language_count': len(resource_levels['medium'])
            },
            'low': {
                'standard_avg_perplexity': 0,
                'langspec_avg_perplexity': 0,
                'avg_improvement': 0,
                'better_model': None,
                'language_count': len(resource_levels['low'])
            }
        }
        
        # Calculate averages for each resource level
        for level, lang_codes in resource_levels.items():
            if not lang_codes:
                continue
            
            standard_sum = sum(results['standard'][lang] for lang in lang_codes)
            langspec_sum = sum(results['langspec'][lang] for lang in lang_codes)
            
            analysis[level]['standard_avg_perplexity'] = standard_sum / len(lang_codes)
            analysis[level]['langspec_avg_perplexity'] = langspec_sum / len(lang_codes)
            
            # Determine better model and calculate improvement
            if analysis[level]['standard_avg_perplexity'] < analysis[level]['langspec_avg_perplexity']:
                analysis[level]['better_model'] = 'standard'
                improvement = (analysis[level]['langspec_avg_perplexity'] - analysis[level]['standard_avg_perplexity']) / analysis[level]['langspec_avg_perplexity'] * 100
            else:
                analysis[level]['better_model'] = 'langspec'
                improvement = (analysis[level]['standard_avg_perplexity'] - analysis[level]['langspec_avg_perplexity']) / analysis[level]['standard_avg_perplexity'] * 100
            
            analysis[level]['avg_improvement'] = improvement
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error performing resource level analysis: {str(e)}")
        return None

def generate_report():
    """
    Generate a comprehensive report comparing the tokenization methods.
    
    Returns:
        Path to the generated report
    """
    # Load results and comparison
    results = load_perplexity_results()
    comparison = load_perplexity_comparison()
    
    if not results or not comparison:
        logger.error("Missing results or comparison for report generation")
        return None
    
    try:
        # Generate charts
        charts_dir = os.path.join(REPORT_DIR, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        perplexity_chart = generate_perplexity_chart(
            results,
            os.path.join(charts_dir, "perplexity_comparison.png")
        )
        
        improvement_chart = generate_improvement_chart(
            comparison,
            os.path.join(charts_dir, "improvement_percentage.png")
        )
        
        # Perform resource level analysis
        resource_analysis = generate_resource_level_analysis(results, comparison)
        
        # Save resource analysis
        if resource_analysis:
            resource_analysis_path = os.path.join(RESULTS_DIR, "resource_level_analysis.json")
            with open(resource_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(resource_analysis, f, indent=4)
        
        # Generate the report
        report_path = os.path.join(REPORT_DIR, "multilingual_tokenization_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Title and introduction
            f.write("# Multilingual Tokenization Experiment Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Introduction\n\n")
            f.write("This report presents the results of an experiment comparing two multilingual tokenization methods:\n\n")
            f.write("1. **Standard UnigramLM Tokenization**: The vocabulary is chosen via the standard UnigramLM tokenization algorithm, and inference is done using the same algorithm.\n\n")
            f.write("2. **Language-Specific UnigramLM Tokenization**: The vocabulary is chosen via the standard UnigramLM tokenization algorithm, but inference is done differently. Per-language unigram probabilities are computed, where each language has its own token probabilities derived solely from the portion of the corpus in that language. At inference time, tokenization probabilities are computed according to the standard UnigramLM algorithm (via Viterbi) for each language separately, and the tokenization with the maximum probability across all languages is selected.\n\n")
            
            # Experiment setup
            f.write("## Experiment Setup\n\n")
            f.write("### Languages\n\n")
            f.write("The experiment included the following languages:\n\n")
            
            # Create a table of languages by resource level
            f.write("| Resource Level | Languages |\n")
            f.write("|---------------|----------|\n")
            
            high_resource = [f"{LANGUAGES[code]['name']} ({code})" for code in SELECTED_LANGUAGES if LANGUAGES[code]['resource_level'] == 'high']
            medium_resource = [f"{LANGUAGES[code]['name']} ({code})" for code in SELECTED_LANGUAGES if LANGUAGES[code]['resource_level'] == 'medium']
            low_resource = [f"{LANGUAGES[code]['name']} ({code})" for code in SELECTED_LANGUAGES if LANGUAGES[code]['resource_level'] == 'low']
            
            f.write(f"| High | {', '.join(high_resource)} |\n")
            f.write(f"| Medium | {', '.join(medium_resource)} |\n")
            f.write(f"| Low | {', '.join(low_resource)} |\n\n")
            
            # Methodology
            f.write("### Methodology\n\n")
            f.write("1. **Data Collection**: Multilingual corpora were downloaded and processed incrementally to manage disk space constraints.\n\n")
            f.write("2. **Tokenizer Implementation**: Both the standard UnigramLM tokenizer and the language-specific variant were implemented.\n\n")
            f.write("3. **Corpus Tokenization**: The corpora were tokenized using both methods, processing languages incrementally.\n\n")
            f.write("4. **Language Model Training**: Language models were trained on the tokenized corpora using HuggingFace's transformers library.\n\n")
            f.write("5. **Perplexity Evaluation**: The perplexity of both models was evaluated on test data for each language.\n\n")
            
            # Results
            f.write("## Results\n\n")
            
            # Overall results
            f.write("### Overall Results\n\n")
            f.write(f"**Overall Winner**: {comparison['overall_winner'].capitalize()} UnigramLM Tokenization\n\n")
            f.write(f"**Overall Improvement**: {comparison['overall_improvement_percentage']:.2f}%\n\n")
            f.write(f"**Standard UnigramLM Wins**: {comparison['win_counts']['standard']} languages\n\n")
            f.write(f"**Language-Specific UnigramLM Wins**: {comparison['win_counts']['langspec']} languages\n\n")
            
            # Perplexity by language
            f.write("### Perplexity by Language\n\n")
            f.write("| Language | Standard UnigramLM | Language-Specific UnigramLM | Better Model | Improvement |\n")
            f.write("|----------|-------------------|----------------------------|-------------|------------|\n")
            
            for lang_code in SELECTED_LANGUAGES:
                if lang_code in results['standard'] and lang_code in results['langspec']:
                    std_ppl = results['standard'][lang_code]
                    lang_ppl = results['langspec'][lang_code]
                    better_model = comparison['better_model_per_language'][lang_code]
                    improvement = comparison['improvement_percentage'][lang_code]
                    
                    f.write(f"| {LANGUAGES[lang_code]['name']} ({lang_code}) | {std_ppl:.4f} | {lang_ppl:.4f} | {better_model.capitalize()} | {improvement:.2f}% |\n")
            
            f.write("\n")
            
            # Resource level analysis
            if resource_analysis:
                f.write("### Analysis by Resource Level\n\n")
                f.write("| Resource Level | Standard UnigramLM | Language-Specific UnigramLM | Better Model | Improvement |\n")
                f.write("|---------------|-------------------|----------------------------|-------------|------------|\n")
                
                for level, data in resource_analysis.items():
                    if data['language_count'] > 0:
                        f.write(f"| {level.capitalize()} | {data['standard_avg_perplexity']:.4f} | {data['langspec_avg_perplexity']:.4f} | {data['better_model'].capitalize()} | {data['avg_improvement']:.2f}% |\n")
                
                f.write("\n")
            
            # Charts
            if perplexity_chart:
                f.write("### Perplexity Comparison Chart\n\n")
                f.write("![Perplexity Comparison](charts/perplexity_comparison.png)\n\n")
            
            if improvement_chart:
                f.write("### Improvement Percentage Chart\n\n")
                f.write("![Improvement Percentage](charts/improvement_percentage.png)\n\n")
            
            # Discussion
            f.write("## Discussion\n\n")
            
            # Analyze the results
            if comparison['overall_winner'] == 'standard':
                f.write("The standard UnigramLM tokenization method outperformed the language-specific variant overall, achieving lower perplexity across more languages. This suggests that the shared probability model of the standard approach provides better generalization across languages.\n\n")
            else:
                f.write("The language-specific UnigramLM tokenization method outperformed the standard approach overall, achieving lower perplexity across more languages. This suggests that using language-specific probability models and selecting the maximum probability tokenization provides better results for multilingual text.\n\n")
            
            # Resource level insights
            if resource_analysis:
                f.write("### Resource Level Insights\n\n")
                
                for level, data in resource_analysis.items():
                    if data['language_count'] > 0:
                        f.write(f"**{level.capitalize()} Resource Languages**: ")
                        
                        if data['better_model'] == 'standard':
                            f.write(f"The standard UnigramLM approach performed better by {data['avg_improvement']:.2f}%. ")
                        else:
                            f.write(f"The language-specific UnigramLM approach performed better by {data['avg_improvement']:.2f}%. ")
                        
                        f.write("\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            
            if comparison['overall_winner'] == 'standard':
                f.write("Based on the experimental results, the standard UnigramLM tokenization method is recommended for multilingual applications. While the language-specific approach showed advantages for certain languages, the standard method provided better overall performance and generalization.\n\n")
            else:
                f.write("Based on the experimental results, the language-specific UnigramLM tokenization method is recommended for multilingual applications. By computing per-language unigram probabilities and selecting the maximum probability tokenization, this approach achieved better overall performance compared to the standard method.\n\n")
            
            # Future work
            f.write("## Future Work\n\n")
            f.write("1. **Larger Scale Evaluation**: Expand the experiment to include more languages and larger corpora.\n\n")
            f.write("2. **Hybrid Approaches**: Explore combinations of standard and language-specific tokenization methods.\n\n")
            f.write("3. **Language Identification**: Integrate automatic language identification to select the appropriate tokenization model.\n\n")
            f.write("4. **Downstream Tasks**: Evaluate the impact of these tokenization methods on downstream NLP tasks such as translation, classification, and named entity recognition.\n\n")
        
        logger.info(f"Generated report: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None

def generate_experiment_runner():
    """
    Generate a script to run the complete experiment pipeline.
    
    Returns:
        Path to the generated script
    """
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_experiment.py")
    
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Multilingual Tokenization Experiment Runner
This script runs the complete experiment pipeline.
\"\"\"

import os
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run multilingual tokenization experiments")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples per language")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size for tokenizers")
    parser.add_argument("--skip-download", action="store_true", help="Skip corpus download if already available")
    parser.add_argument("--skip-tokenization", action="store_true", help="Skip tokenization if already done")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training if already done")
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting experiment at {start_time}")
    
    try:
        # Step 1: Download and process corpora
        if not args.skip_download:
            logger.info("Step 1: Downloading and processing corpora...")
            from code.corpus_downloader import process_all_corpora
            corpus_info = process_all_corpora(max_samples_per_language=args.max_samples)
        else:
            logger.info("Skipping corpus download (using existing data)")
            # Create a dummy corpus_info structure
            corpus_info = {}
            processed_dir = os.path.join("data", "processed")
            for lang_code in os.listdir(processed_dir):
                if os.path.isdir(os.path.join(processed_dir, lang_code)):
                    corpus_info[lang_code] = {
                        'splits': {
                            'train': os.path.join(processed_dir, lang_code, "train.txt"),
                            'val': os.path.join(processed_dir, lang_code, "val.txt"),
                            'test': os.path.join(processed_dir, lang_code, "test.txt")
                        }
                    }
        
        # Step 2: Train tokenizers
        logger.info("Step 2: Training tokenizers...")
        from code.standard_unigramlm import train_standard_tokenizer
        from code.language_specific_unigramlm import train_language_specific_tokenizer
        
        standard_tokenizer_path = train_standard_tokenizer(corpus_info, vocab_size=args.vocab_size)
        langspec_tokenizer_paths = train_language_specific_tokenizer(corpus_info, vocab_size=args.vocab_size)
        
        # Step 3: Tokenize corpora
        if not args.skip_tokenization:
            logger.info("Step 3: Tokenizing corpora...")
            from code.corpus_tokenizer import tokenize_all_corpora
            
            tokenized_info = tokenize_all_corpora(
                corpus_info,
                standard_tokenizer_path,
                langspec_tokenizer_paths,
                max_lines_per_file=args.max_samples
            )
        else:
            logger.info("Skipping tokenization (using existing tokenized data)")
            # Create a dummy tokenized_info structure
            tokenized_info = {'standard': {}, 'langspec': {}}
            standard_dir = os.path.join("data", "tokenized", "standard")
            langspec_dir = os.path.join("data", "tokenized", "langspec")
            
            for lang_code in os.listdir(standard_dir):
                if os.path.isdir(os.path.join(standard_dir, lang_code)):
                    tokenized_info['standard'][lang_code] = {
                        'train': os.path.join(standard_dir, lang_code, "train.jsonl"),
                        'val': os.path.join(standard_dir, lang_code, "val.jsonl"),
                        'test': os.path.join(standard_dir, lang_code, "test.jsonl")
                    }
            
            for lang_code in os.listdir(langspec_dir):
                if os.path.isdir(os.path.join(langspec_dir, lang_code)):
                    tokenized_info['langspec'][lang_code] = {
                        'train': os.path.join(langspec_dir, lang_code, "train.jsonl"),
                        'val': os.path.join(langspec_dir, lang_code, "val.jsonl"),
                        'test': os.path.join(langspec_dir, lang_code, "test.jsonl")
                    }
        
        # Step 4: Train language models
        if not args.skip_training:
            logger.info("Step 4: Training language models...")
            from code.model_trainer import train_models_on_both_tokenization_methods
            
            model_paths = train_models_on_both_tokenization_methods(
                tokenized_info['standard'],
                tokenized_info['langspec'],
                standard_tokenizer_path,
                langspec_tokenizer_paths
            )
        else:
            logger.info("Skipping model training (using existing models)")
            model_paths = {
                'standard': os.path.join("data", "models", "standard_model"),
                'langspec': os.path.join("data", "models", "langspec_model")
            }
        
        # Step 5: Evaluate perplexity
        logger.info("Step 5: Evaluating perplexity...")
        from code.perplexity_evaluator import evaluate_both_models, compare_perplexity_results
        
        perplexity_results = evaluate_both_models(
            model_paths['standard'],
            model_paths['langspec'],
            tokenized_info['standard'],
            tokenized_info['langspec'],
            standard_tokenizer_path,
            langspec_tokenizer_paths['base']
        )
        
        comparison = compare_perplexity_results(perplexity_results)
        
        # Step 6: Generate report
        logger.info("Step 6: Generating report...")
        from code.report_generator import generate_report
        
        report_path = generate_report()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Experiment completed at {end_time} (duration: {duration})")
        logger.info(f"Report generated at: {report_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
""")
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Generated experiment runner script: {script_path}")
        return script_path
    
    except Exception as e:
        logger.error(f"Error generating experiment runner script: {str(e)}")
        return None

if __name__ == "__main__":
    # Generate the report
    report_path = generate_report()
    
    # Generate the experiment runner script
    script_path = generate_experiment_runner()
    
    if report_path and script_path:
        print(f"Report generated: {report_path}")
        print(f"Experiment runner script generated: {script_path}")
        print("To run the complete experiment, execute: python3 run_experiment.py")
    else:
        print("Failed to generate report or experiment runner script")
