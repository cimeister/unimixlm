#!/usr/bin/env python3
"""
Multilingual Tokenization Experiment Runner
This script runs the complete experiment pipeline.
"""

import os
import logging
import argparse
from datetime import datetime

# Import and setup environment variables
from code.environment_setup import setup_environment
setup_environment()

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
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--max-samples", type=int, default=None, 
                      help="Maximum samples per language (default: None)")
    sampling_group.add_argument("--max-total-samples", type=int, 
                      help="Maximum total samples across all languages, distributed proportionally")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size for tokenizers")
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
            
            # Determine sampling mode
            proportional_sampling = args.max_total_samples is not None
            
            if proportional_sampling:
                logger.info(f"Using proportional sampling with max total samples: {args.max_total_samples}")
                corpus_info = process_all_corpora(
                    max_total_samples=args.max_total_samples,
                    proportional_sampling=True
                )
            else:
                logger.info(f"Using fixed sampling with max {args.max_samples} samples per language")
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
        from code.multigramlm import train_multigram_tokenizer
        
        #standard_tokenizer_path = train_standard_tokenizer(corpus_info, vocab_size=args.vocab_size)
        standard_tokenizer_path_soft = train_standard_tokenizer(corpus_info, vocab_size=args.vocab_size, em_mode='soft')
        #os.path.join("data", "tokenizers", "standard_unigramlm.json")#train_standard_tokenizer(corpus_info, vocab_size=args.vocab_size)
        exit(0)
        #langspec_tokenizer_paths = train_language_specific_tokenizer(corpus_info, vocab_size=args.vocab_size)
        multigram_paths = train_multigram_tokenizer(corpus_info, vocab_size=args.vocab_size, load_if_exists=False)
        
        #multigram_soft_paths = train_multigram_tokenizer(corpus_info, vocab_size=args.vocab_size, load_if_exists=False, soft_em=True)
        langspec_multi_tokenizer_paths = train_language_specific_tokenizer(corpus_info, 
            base_tokenizer_path=multigram_paths['base'], 
            probability_path_format='data/tokenizers/multi_langspec_{lang_code}_em_probs.npy', 
            vocab_size=args.vocab_size)
        
        # Step 3: Tokenize corpora
        if not args.skip_tokenization:
            logger.info("Step 3: Tokenizing corpora...")
            from code.corpus_tokenizer import tokenize_all_corpora
            
            tokenized_info = tokenize_all_corpora(
                corpus_info,
                standard_tokenizer_path,
                langspec_tokenizer_paths,
                multigram_soft_paths,
                langspec_multi_tokenizer_paths,
                max_lines_per_file=args.max_samples
            )
            exit(0)
        else:
            logger.info("Skipping tokenization (using existing tokenized data)")
            # Create a dummy tokenized_info structure
            
            tokenized_dirs = {
                'standard': os.path.join("data", "tokenized", "standard"), 
                'langspec': os.path.join("data", "tokenized", "langspec_viterbi_parallel"), 
                'multigram':os.path.join("data", "tokenized", "multigram_viterbi_parallel"), 
                "langspec-multigram":os.path.join("data", "tokenized", "langspec_multigram_viterbi_parallel")
            }
            tokenized_info = {k: {} for k in tokenized_dirs.keys()}

            for name, directory in tokenized_dirs.items():
                for lang_code in os.listdir(directory):
                    if os.path.isdir(os.path.join(directory, lang_code)):
                        tokenized_info[name][lang_code] = {
                            'train': os.path.join(directory, lang_code, "train.jsonl"),
                            'val': os.path.join(directory, lang_code, "val.jsonl"),
                            'test': os.path.join(directory, lang_code, "test.jsonl")
                    }
            
        # Step 4: Train language models
        if not args.skip_training:
            logger.info("Step 4: Training language models...")
            from code.model_trainer import train_models_on_all_tokenization_methods
            
            model_paths = train_models_on_all_tokenization_methods(
                tokenized_info['standard'],
                tokenized_info['langspec'],
                tokenized_info['multigram'],
                tokenized_info['langspec-multigram'],
                standard_tokenizer_path,
                langspec_tokenizer_paths,
                multigram_paths,
                langspec_multi_tokenizer_paths
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
