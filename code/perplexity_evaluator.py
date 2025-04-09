"""
Perplexity evaluation module for multilingual tokenization experiments.
This module handles evaluating perplexity of language models trained on different tokenization methods.
"""

import os
import logging
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

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
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_perplexity(model, tokenizer, dataset):
    """
    Calculate perplexity of a model on a dataset.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset: Dataset with input_ids
    
    Returns:
        Perplexity score
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables for perplexity calculation
    total_loss = 0.0
    total_tokens = 0
    
    # Process the dataset in batches
    batch_size = 4
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating perplexity"):
            # Get batch
            batch = dataset[i:i+batch_size]
            
            # Get input_ids
            input_ids = torch.tensor(batch['input_ids'])
            
            # Calculate loss
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Update totals
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def evaluate_model_perplexity(model_path, tokenized_files, tokenizer_path):
    """
    Evaluate perplexity of a model on tokenized test data.
    
    Args:
        model_path: Path to the trained model
        tokenized_files: Dictionary mapping language codes to tokenized file paths
        tokenizer_path: Path to the tokenizer file
    
    Returns:
        Dictionary with perplexity scores for each language
    """
    logger.info(f"Evaluating perplexity of model at {model_path}...")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize results dictionary
    perplexity_scores = {}
    
    # Evaluate perplexity for each language
    for lang_code, files in tokenized_files.items():
        if 'test' not in files:
            logger.warning(f"No test file found for {lang_code}, skipping")
            continue
        
        test_file = files['test']
        
        try:
            # Load the tokenized test data
            data = []
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append({
                        'text': item['text'],
                        'input_ids': item['token_ids']
                    })
            
            # Create a HuggingFace Dataset
            test_dataset = Dataset.from_list(data)
            
            # Calculate perplexity
            perplexity = calculate_perplexity(model, tokenizer, test_dataset)
            
            logger.info(f"Perplexity for {lang_code}: {perplexity:.4f}")
            perplexity_scores[lang_code] = perplexity
        
        except Exception as e:
            logger.error(f"Error evaluating perplexity for {lang_code}: {str(e)}")
    
    # Calculate average perplexity across all languages
    if perplexity_scores:
        avg_perplexity = sum(perplexity_scores.values()) / len(perplexity_scores)
        perplexity_scores['average'] = avg_perplexity
        logger.info(f"Average perplexity across all languages: {avg_perplexity:.4f}")
    
    return perplexity_scores

def evaluate_both_models(standard_model_path, langspec_model_path, 
                        standard_tokenized_files, langspec_tokenized_files,
                        standard_tokenizer_path, langspec_tokenizer_path):
    """
    Evaluate perplexity of both models on their respective tokenized test data.
    
    Args:
        standard_model_path: Path to the standard model
        langspec_model_path: Path to the language-specific model
        standard_tokenized_files: Dictionary mapping language codes to standard tokenized file paths
        langspec_tokenized_files: Dictionary mapping language codes to language-specific tokenized file paths
        standard_tokenizer_path: Path to the standard tokenizer file
        langspec_tokenizer_path: Path to the language-specific base tokenizer file
    
    Returns:
        Dictionary with perplexity scores for both models
    """
    # Evaluate standard model
    logger.info("Evaluating standard model perplexity...")
    standard_perplexity = evaluate_model_perplexity(
        standard_model_path,
        standard_tokenized_files,
        standard_tokenizer_path
    )
    
    # Evaluate language-specific model
    logger.info("Evaluating language-specific model perplexity...")
    langspec_perplexity = evaluate_model_perplexity(
        langspec_model_path,
        langspec_tokenized_files,
        langspec_tokenizer_path
    )
    
    # Combine results
    results = {
        'standard': standard_perplexity,
        'langspec': langspec_perplexity
    }
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "perplexity_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved perplexity results to {results_path}")
    
    return results

def compare_perplexity_results(results):
    """
    Compare perplexity results between standard and language-specific models.
    
    Args:
        results: Dictionary with perplexity scores for both models
    
    Returns:
        Dictionary with comparison results
    """
    standard_perplexity = results['standard']
    langspec_perplexity = results['langspec']
    
    # Initialize comparison results
    comparison = {
        'better_model_per_language': {},
        'overall_winner': None,
        'improvement_percentage': {}
    }
    
    # Compare perplexity for each language
    for lang_code in set(standard_perplexity.keys()).intersection(langspec_perplexity.keys()):
        if lang_code == 'average':
            continue
        
        std_ppl = standard_perplexity[lang_code]
        lang_ppl = langspec_perplexity[lang_code]
        
        if std_ppl < lang_ppl:
            comparison['better_model_per_language'][lang_code] = 'standard'
            improvement = (lang_ppl - std_ppl) / lang_ppl * 100
        else:
            comparison['better_model_per_language'][lang_code] = 'langspec'
            improvement = (std_ppl - lang_ppl) / std_ppl * 100
        
        comparison['improvement_percentage'][lang_code] = improvement
    
    # Determine overall winner
    if standard_perplexity.get('average', float('inf')) < langspec_perplexity.get('average', float('inf')):
        comparison['overall_winner'] = 'standard'
        overall_improvement = (langspec_perplexity['average'] - standard_perplexity['average']) / langspec_perplexity['average'] * 100
    else:
        comparison['overall_winner'] = 'langspec'
        overall_improvement = (standard_perplexity['average'] - langspec_perplexity['average']) / standard_perplexity['average'] * 100
    
    comparison['overall_improvement_percentage'] = overall_improvement
    
    # Count wins per model
    standard_wins = sum(1 for model in comparison['better_model_per_language'].values() if model == 'standard')
    langspec_wins = sum(1 for model in comparison['better_model_per_language'].values() if model == 'langspec')
    
    comparison['win_counts'] = {
        'standard': standard_wins,
        'langspec': langspec_wins
    }
    
    # Save comparison results
    comparison_path = os.path.join(RESULTS_DIR, "perplexity_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Saved perplexity comparison to {comparison_path}")
    
    return comparison

if __name__ == "__main__":
    # For testing purposes
    from code.corpus_downloader import process_all_corpora
    from code.standard_unigramlm import train_standard_tokenizer
    from code.language_specific_unigramlm import train_language_specific_tokenizer
    from code.corpus_tokenizer import tokenize_all_corpora
    from code.model_trainer import train_models_on_both_tokenization_methods
    
    # Process corpora
    corpus_info = process_all_corpora()
    
    # Train tokenizers
    standard_tokenizer_path = train_standard_tokenizer(corpus_info)
    langspec_tokenizer_paths = train_language_specific_tokenizer(corpus_info)
    
    # Tokenize corpora
    tokenized_info = tokenize_all_corpora(
        corpus_info,
        standard_tokenizer_path,
        langspec_tokenizer_paths
    )
    
    # Train models
    model_paths = train_models_on_both_tokenization_methods(
        tokenized_info['standard'],
        tokenized_info['langspec'],
        standard_tokenizer_path,
        langspec_tokenizer_paths
    )
    
    # Evaluate perplexity
    perplexity_results = evaluate_both_models(
        model_paths['standard'],
        model_paths['langspec'],
        tokenized_info['standard'],
        tokenized_info['langspec'],
        standard_tokenizer_path,
        langspec_tokenizer_paths['base']
    )
    
    # Compare results
    comparison = compare_perplexity_results(perplexity_results)
    
    # Print summary
    print("\nPerplexity Evaluation Summary:")
    print("-----------------------------")
    print(f"Overall winner: {comparison['overall_winner']} model")
    print(f"Overall improvement: {comparison['overall_improvement_percentage']:.2f}%")
    print(f"Standard model wins: {comparison['win_counts']['standard']}")
    print(f"Language-specific model wins: {comparison['win_counts']['langspec']}")
