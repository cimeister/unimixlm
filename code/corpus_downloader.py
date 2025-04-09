"""
Corpus downloader module for multilingual tokenization experiments.
This module handles downloading and processing multilingual corpora incrementally.
"""

import os
import logging
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import language selection
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.language_selection import SELECTED_LANGUAGES, LANGUAGES

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Define dataset sources for different languages
DATASET_SOURCES = {
    # High-resource languages
    'en': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_en', 'split': 'train', 'streaming': True},
    'es': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_es', 'split': 'train', 'streaming': True},
    'fr': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_fr', 'split': 'train', 'streaming': True},
    'de': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_de', 'split': 'train', 'streaming': True},
    'zh': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_zh', 'split': 'train', 'streaming': True},
    'ru': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_ru', 'split': 'train', 'streaming': True},
    
    # Medium-resource languages
    'ar': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_ar', 'split': 'train', 'streaming': True},
    'hi': {'dataset': 'oscar', 'name': 'unshuffled_deduplicated_hi', 'split': 'train', 'streaming': True},
    
    # Low-resource languages - using OPUS Books for some low-resource languages
    'swh_Latn': {'dataset': 'facebook/flores', 'name': 'swh_Latn', 'split': 'dev', 'streaming': False},
    'yor_Latn': {'dataset': 'facebook/flores', 'name': 'yor_Latn', 'split': 'dev', 'streaming': False},
    'hau_Latn': {'dataset': 'facebook/flores', 'name': 'hau_Latn', 'split': 'dev', 'streaming': False},
    'zul_Latn': {'dataset': 'facebook/flores', 'name': 'zul_Latn', 'split': 'dev', 'streaming': False},
}

# Fallback to smaller datasets if the primary ones are too large
FALLBACK_DATASET_SOURCES = {
    # All languages - smaller dataset alternative
    'en': {'dataset': 'opus_books', 'name': 'en', 'split': 'train', 'streaming': False},
    'es': {'dataset': 'opus_books', 'name': 'es', 'split': 'train', 'streaming': False},
    'fr': {'dataset': 'opus_books', 'name': 'fr', 'split': 'train', 'streaming': False},
    'de': {'dataset': 'opus_books', 'name': 'de', 'split': 'train', 'streaming': False},
    'zh': {'dataset': 'opus_books', 'name': 'zh', 'split': 'train', 'streaming': False},
    'ru': {'dataset': 'opus_books', 'name': 'ru', 'split': 'train', 'streaming': False},
    'ar': {'dataset': 'opus_books', 'name': 'ar', 'split': 'train', 'streaming': False},
    'hi': {'dataset': 'opus_books', 'name': 'hi', 'split': 'train', 'streaming': False},
}

def count_lines(filename):
    with open(filename, 'rb') as f:
        return sum(buf.count(b'\n') for buf in iter(lambda: f.read(1024*1024), b''))


def clean_text(text):
    """Basic text cleaning function."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text

def download_and_process_language(lang_code, output_file, max_samples=10000, use_fallback=False):
    """
    Download and process corpus for a specific language.
    
    Args:
        lang_code: The language code to download
        max_samples: Maximum number of samples to process
        use_fallback: Whether to use fallback dataset source
    
    Returns:
        Path to the processed corpus file
    """
    logger.info(f"Processing {LANGUAGES[lang_code]['name']} ({lang_code}) corpus...")
    
    # Select dataset source
    if use_fallback and lang_code in FALLBACK_DATASET_SOURCES:
        source = FALLBACK_DATASET_SOURCES[lang_code]
        logger.info(f"Using fallback dataset for {lang_code}")
    else:
        source = DATASET_SOURCES[lang_code]
        
    try:
        # Load dataset with streaming if available
        dataset = load_dataset(
            source['dataset'],
            source['name'],
            split=source['split'],
            streaming=source['streaming']
        )
        
        # Process the dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            sample_count = 0
            
            # Use take() for streaming datasets to limit memory usage
            if source['streaming']:
                dataset = dataset.take(max_samples)
            
            # Process samples
            for sample in tqdm(dataset, desc=f"Processing {lang_code}", total=max_samples):
                # Handle different dataset structures
                if 'text' in sample:
                    text = clean_text(sample['text'])
                    if text:
                        f.write(text + '\n')
                        sample_count += 1
                # Special handling for facebook/flores dataset
                elif 'sentence' in sample:
                    text = clean_text(sample['sentence'])
                    if text:
                        f.write(text + '\n')
                        sample_count += 1
                        
                # Check if we've reached the maximum number of samples
                if sample_count >= max_samples:
                    break
            
            logger.info(f"Processed {sample_count} samples for {lang_code}")
        
        # Check file size and content
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
        logger.info(f"Corpus file for {lang_code} created: {output_file} ({file_size:.2f} MB)")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error processing {lang_code}: {str(e)}")
        
        # Try fallback if not already using it
        if not use_fallback and lang_code in FALLBACK_DATASET_SOURCES:
            logger.info(f"Attempting fallback dataset for {lang_code}")
            return download_and_process_language(lang_code, output_file, max_samples, use_fallback=True)
        
        # Create empty file if all attempts fail
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Failed to download {lang_code} corpus\n")
    
        return None

def download_all_corpora(max_samples_per_language=10000, max_total_samples=None, proportional_sampling=False):
    """
    Download and process all selected language corpora incrementally.
    
    Args:
        max_samples_per_language: Maximum number of samples per language (used if proportional_sampling is False)
        max_total_samples: Maximum total samples across all languages (used if proportional_sampling is True)
        proportional_sampling: Whether to distribute samples proportionally based on corpus sizes
    
    Returns:
        Dictionary mapping language codes to corpus file paths
    """
    corpus_files = {}
    
    # If using proportional sampling with a total sample limit
    if proportional_sampling and max_total_samples:
        logger.info(f"Using proportional sampling with max total samples: {max_total_samples}")
        
        # First, estimate the size of each corpus to determine proportions
        corpus_sizes = {}
        total_size = 0
        
        # Get an estimate of corpus sizes
        for lang_code in SELECTED_LANGUAGES:
            try:
                # Use dataset info to get an estimate of corpus size
                source = DATASET_SOURCES[lang_code]
                
                if source['streaming']:
                    # For streaming datasets, we'll use a fixed large number as an estimate
                    # This is a rough approximation for OSCAR datasets which are typically large
                    estimated_size = 1000000  # 1M samples for high-resource languages
                    if LANGUAGES[lang_code]['resource_level'] == 'medium':
                        estimated_size = 500000  # 500K for medium-resource
                    elif LANGUAGES[lang_code]['resource_level'] == 'low':
                        estimated_size = 100000  # 100K for low-resource
                else:
                    # For non-streaming datasets, we can load the dataset and get its size
                    try:
                        dataset = load_dataset(
                            source['dataset'],
                            source['name'],
                            split=source['split'],
                            streaming=False
                        )
                        estimated_size = len(dataset)
                    except Exception:
                        # If we can't get the size, use a reasonable default based on resource level
                        if LANGUAGES[lang_code]['resource_level'] == 'high':
                            estimated_size = 100000
                        elif LANGUAGES[lang_code]['resource_level'] == 'medium':
                            estimated_size = 50000
                        else:
                            estimated_size = 10000
                
                corpus_sizes[lang_code] = estimated_size
                total_size += estimated_size
                logger.info(f"Estimated size for {lang_code}: {estimated_size}")
            
            except Exception as e:
                logger.error(f"Error estimating size for {lang_code}: {str(e)}")
                # Use a default size if estimation fails
                estimated_size = 10000
                corpus_sizes[lang_code] = estimated_size
                total_size += estimated_size
        
        # Calculate samples per language proportionally
        samples_per_language = {}
        for lang_code, size in corpus_sizes.items():
            proportion = size / total_size
            lang_samples = int(max_total_samples * proportion)
            # Ensure at least 100 samples per language
            samples_per_language[lang_code] = max(lang_samples, min(size,1000))
            logger.info(f"Allocated {samples_per_language[lang_code]} samples for {lang_code} ({proportion:.2%} of total)")
        
        # Process one language at a time with proportional sample sizes
        for lang_code in SELECTED_LANGUAGES:
            file_name = os.path.join(RAW_DATA_DIR, f"{lang_code}_corpus.txt")
            if os.path.isfile(file_name):
                if count_lines(file_name) == samples_per_language[lang_code]:
                    corpus_files[lang_code] = file_name
                    logging.info(f"Corpus for {lang_code} already downloaded. Re-using...")
                    continue
                logging.info(f"Replacing downloaded corpus for {lang_code}.")
            
            try:
                # Check available disk space before downloading
                disk_usage = shutil.disk_usage(DATA_DIR)
                free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
                
                if free_space_gb < 0.5:  # Less than 500MB free
                    logger.warning(f"Low disk space: {free_space_gb:.2f}GB. Reducing sample size.")
                    max_samples = min(samples_per_language[lang_code], 1000)  # Reduce samples if low disk space
                else:
                    max_samples = samples_per_language[lang_code]
                
                # Download and process the language corpus
                corpus_file = download_and_process_language(lang_code, file_name, max_samples)
                if corpus_file:
                    corpus_files[lang_code] = corpus_file
            
            except Exception as e:
                logger.error(f"Failed to process {lang_code}: {str(e)}")
    
    # Traditional approach with fixed samples per language
    else:
        logger.info(f"Using fixed sampling with max {max_samples_per_language} samples per language")
        
        # Process one language at a time to manage disk space
        for lang_code in SELECTED_LANGUAGES:
            try:
                # Check available disk space before downloading
                disk_usage = shutil.disk_usage(DATA_DIR)
                free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
                
                if free_space_gb < 0.5:  # Less than 500MB free
                    logger.warning(f"Low disk space: {free_space_gb:.2f}GB. Reducing sample size.")
                    max_samples = min(max_samples_per_language, 5000)  # Reduce samples if low disk space
                else:
                    max_samples = max_samples_per_language
                
                # Download and process the language corpus
                corpus_file = download_and_process_language(lang_code, max_samples)
                
                if corpus_file:
                    corpus_files[lang_code] = corpus_file
            
            except Exception as e:
                logger.error(f"Failed to process {lang_code}: {str(e)}")
    
    return corpus_files

def split_corpus(corpus_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a corpus file into train, validation, and test sets.
    
    Args:
        corpus_file: Path to the corpus file
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    
    Returns:
        Dictionary with paths to the split files
    """
    if not corpus_file or not os.path.exists(corpus_file):
        return None
    
    base_name = os.path.basename(corpus_file)
    lang_code = base_name.split('_corpus.txt')[0]
    
    processed_lang_dir = os.path.join(PROCESSED_DATA_DIR, lang_code)
    os.makedirs(processed_lang_dir, exist_ok=True)
    train_file = os.path.join(processed_lang_dir, "train.txt")
    val_file = os.path.join(processed_lang_dir, "val.txt")
    test_file = os.path.join(processed_lang_dir, "test.txt")
    
    try:
        # Count total lines
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if total_lines == 0:
            logger.warning(f"Corpus file {corpus_file} is empty")
            return None
        
        # Calculate split indices
        train_end = int(total_lines * train_ratio)
        val_end = train_end + int(total_lines * val_ratio)
        
        # Write splits
        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[:train_end])
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[train_end:val_end])
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[val_end:])
        
        logger.info(f"Split {corpus_file} into train ({train_end} lines), "
                   f"val ({val_end - train_end} lines), and test ({total_lines - val_end} lines)")
        
        return {
            'train': train_file,
            'val': val_file,
            'test': test_file
        }
    
    except Exception as e:
        logger.error(f"Error splitting {corpus_file}: {str(e)}")
        return None

def process_all_corpora(max_samples_per_language=10000, max_total_samples=None, proportional_sampling=False):
    """
    Download and process all corpora, then split into train/val/test sets.
    
    Returns:
        Dictionary with corpus information for all languages
    """
    # Download all corpora
    corpus_files = download_all_corpora(max_samples_per_language, max_total_samples, proportional_sampling)
    
    # Split each corpus
    corpus_info = {}
    for lang_code, corpus_file in corpus_files.items():
        if corpus_file:
            split_files = split_corpus(corpus_file)
            if split_files:
                corpus_info[lang_code] = {
                    'full': corpus_file,
                    'splits': split_files
                }
    
    return corpus_info

if __name__ == "__main__":
    # Process all corpora
    corpus_info = process_all_corpora()
    
    # Print summary
    print("\nCorpus Processing Summary:")
    print("--------------------------")
    for lang_code, info in corpus_info.items():
        print(f"{LANGUAGES[lang_code]['name']} ({lang_code}):")
        print(f"  Full corpus: {info['full']}")
        print(f"  Train: {info['splits']['train']}")
        print(f"  Validation: {info['splits']['val']}")
        print(f"  Test: {info['splits']['test']}")
        print()
