import os
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import and setup environment variables
from code.environment_setup import setup_environment
setup_environment()
from code.utils import SafeFormattingDict
from code.corpus_downloader import process_all_corpora
from code.standard_unigramlm import train_standard_tokenizer
from code.language_specific_unigramlm import train_language_specific_tokenizer
from code.corpus_tokenizer import CorpusTokenizer
from code.model_trainer import train_models_from_config

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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise

def process_corpora(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Download and process corpora."""
    logger.info("Step 1: Downloading and processing corpora...")
    
    # Override from args if specified
    if args.skip_download or not config["steps"]["download"]:
        logger.info("Skipping corpus download (using existing data)")
        # Create a corpus_info structure from processed directory
        corpus_info = {}
        processed_dir = config["corpus"]["processed_dir"].format(data_dir=config["experiment"]["data_dir"])

        if not os.path.exists(processed_dir):
            logger.error(f"Processed directory {processed_dir} does not exist. Cannot skip download step.")
            raise FileNotFoundError(f"Directory not found: {processed_dir}")
            
        for lang_code in os.listdir(processed_dir):
            lang_dir = os.path.join(processed_dir, lang_code)
            if os.path.isdir(lang_dir):
                corpus_info[lang_code] = {
                    'splits': {
                        'train': os.path.join(lang_dir, "train.txt"),
                        'val': os.path.join(lang_dir, "val.txt"),
                        'test': os.path.join(lang_dir, "test.txt")
                    }
                }
        
        return corpus_info
        
    # Determine sampling mode
    max_samples = args.max_samples if args.max_samples is not None else config["experiment"].get("max_samples_per_language")
    max_total = args.max_total_samples if args.max_total_samples is not None else config["experiment"].get("max_total_samples")
    proportional = max_total is not None or config["experiment"].get("proportional_sampling", False)
    
    if proportional:
        logger.info(f"Using proportional sampling with max total samples: {max_total}")
        corpus_info = process_all_corpora(
            max_total_samples=max_total,
            proportional_sampling=True,
            languages=config["corpus"].get("languages")
        )
    else:
        logger.info(f"Using fixed sampling with max {max_samples} samples per language")
        corpus_info = process_all_corpora(
            max_samples_per_language=max_samples,
            languages=config["corpus"].get("languages")
        )
    
    return corpus_info


def train_tokenizers(config: Dict[str, Any], corpus_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Train various tokenizers specified in the configuration."""
    logger.info("Step 2: Training tokenizers...")
    
    # Dictionary to store tokenizer paths and configurations
    tokenizer_paths = {}
    tokenizers_dir = config["experiment"]["tokenizers_dir"].format(data_dir=config["experiment"]["data_dir"])
    
    # Process each tokenizer configuration
    for tokenizer_config in config["tokenizers"]:
        tokenizer_name = tokenizer_config["name"]
        tokenizer_type = tokenizer_config["type"]
        em_mode = tokenizer_config.get("em_mode", "soft")
        
        logger.info(f"Processing tokenizer: {tokenizer_name} (type: {tokenizer_type})")
        
        # Check if we should use a pretrained tokenizer
        if tokenizer_config.get("skip_if_exists", False):
            logger.info(f"Using pretrained tokenizer for {tokenizer_name}")
            
            if tokenizer_type == "standard":
                pretrained_path = tokenizer_config["path"].format(tokenizers_dir=tokenizers_dir, em_mode=em_mode)
                if os.path.exists(pretrained_path):
                    tokenizer_paths[tokenizer_name] = {
                        "path": pretrained_path,
                        "type": "standard",
                        "multigram": tokenizer_config.get("multigram", False)
                    }
                    logger.info(f"Loaded pretrained tokenizer from {pretrained_path}")
                    continue
                else:
                    logger.warning(f"Pretrained tokenizer not found at {pretrained_path}, will train a new one")
            
            elif tokenizer_type == "langspec":
                if not tokenizer_config.get("skip_base_if_exists", False):
                    logger.error("skip_base_if_exists must also be set to true to use skip_if_exists for langspec tokenizer. Skipping tokenizer...")
                    continue

                pretrained_base_path = tokenizer_config.get("base_path", "").format(tokenizers_dir=tokenizers_dir, em_mode=em_mode)
                pretrained_prob_path_format = tokenizer_config.get("probability_dir", "").format_map(
                    SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
                pretrained_prob_dir = os.path.dirname(pretrained_prob_path_format)
                
                if pretrained_base_path and os.path.exists(pretrained_base_path) and pretrained_prob_dir and os.path.exists(pretrained_prob_dir):
                    # Load language-specific paths
                    language_paths = {}
                    for lang_code in corpus_info.keys():
                        prob_path = pretrained_prob_path_format.format(lang_code=lang_code)
                        if os.path.exists(prob_path):
                            language_paths[lang_code] = prob_path
                    
                    tokenizer_paths[tokenizer_name] = {
                        "base_path": pretrained_base_path,
                        "language_paths": language_paths,
                        "type": "langspec"
                    }
                    logger.info(f"Loaded pretrained language-specific tokenizer from {pretrained_base_path}")
                    continue

        
        if tokenizer_type == "standard":
            # Handle standard tokenizer
            em_mode = tokenizer_config.get("em_mode", "soft")
            multigram = tokenizer_config.get("multigram", False)
            output_file = tokenizer_config.get("path", f"standard_unigramlm_{em_mode}.json").format_map(
                SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
                        
            # Train new tokenizer
            vocab_size = config["experiment"].get("vocab_size", 8000)
            path = train_standard_tokenizer(
                corpus_info, 
                vocab_size=vocab_size, 
                em_mode=em_mode, 
                multigram=multigram,
                output_path=output_file
            )
            
            if path:
                tokenizer_paths[tokenizer_name] = {
                    "path": path,
                    "type": "standard",
                    "multigram": multigram
                }
                    
        elif tokenizer_type == "langspec":
            # Handle language-specific tokenizer            
            em_mode = tokenizer_config.get("em_mode", "soft")
            base_tokenizer_name = tokenizer_config.get("base_tokenizer", "").format_map(
                SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
            if not base_tokenizer_name:
                logger.info(f"Missing base_tokenizer for language-specific tokenizer: {tokenizer_name}. Will train a new one...")
                
            probability_path_format = tokenizer_config.get(
                "probability_path_format", ""
            ).format_map(SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
            
            # Train language-specific tokenizer
            vocab_size = config["experiment"].get("vocab_size", 8000)
            paths = train_language_specific_tokenizer(
                corpus_info,
                vocab_size=vocab_size,
                reestimation_em_mode=em_mode,
                output_dir=tokenizers_dir,
                base_tokenizer_path=base_tokenizer_name,
                probability_path_format=probability_path_format,
                load_base_if_exists=base_tokenizer_name and tokenizer_config.get("skip_base_if_exists", True),
                load_probs_if_exists=tokenizer_config.get("skip_if_exists", True)
            )
            
            if paths:
                tokenizer_paths[tokenizer_name] = {
                    "base_path": paths["base"],
                    "language_paths": paths.get("languages", {}),
                    "type": "langspec"
                }
        
        else:
            logger.warning(f"Unsupported tokenizer type: {tokenizer_type}")
    
    return tokenizer_paths


def tokenize_corpora(config: Dict[str, Any], corpus_info: Dict[str, Any], tokenizer_paths: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Tokenize corpora using the trained tokenizers."""
    
    # Check if we should use pre-tokenized data
    tokenized_data_dir = config["experiment"].get("tokenized_data_dir", "").format(data_dir=config["experiment"]["data_dir"])
    if config.get("tokenized_data", {}).get("use_pretokenized", False):
        logger.info("Using pre-tokenized data (skipping tokenization)")
        
        pretokenized_dirs = config["tokenized_data"].get("pretokenized_dirs", {})
        tokenized_info = {}
        
        for tokenizer_name, dir_path in pretokenized_dirs.items():
            dir_path = dir_path.format(tokenized_data_dir=tokenized_data_dir)
            if os.path.exists(dir_path):
                tokenized_info[tokenizer_name] = {}
                logger.info(f"Using files from {dir_path} (expecting one directory per language)...")
                for lang_code in os.listdir(dir_path):
                    lang_dir = os.path.join(dir_path, lang_code)
                    
                    if os.path.isdir(lang_dir):
                        tokenized_info[tokenizer_name][lang_code] = {
                            'train': os.path.join(lang_dir, "train.jsonl"),
                            'val': os.path.join(lang_dir, "val.jsonl"),
                            'test': os.path.join(lang_dir, "test.jsonl")
                        }
        return tokenized_info

    """Tokenize corpora using the trained tokenizers."""
    if not config["steps"].get("tokenize_corpora", True):
        logger.info("Skipping corpus tokenization (configured to skip)")
        # Create a tokenized_info structure from tokenized directory
        tokenized_info = {}
        for tokenizer_config in config["tokenizers"]:
            name = tokenizer_config["name"]
            output_dir = tokenizer_config.get("output_dir", name)
            dir_path = os.path.join(tokenized_data_dir, output_dir)
            
            if os.path.exists(dir_path):
                tokenized_info[name] = {}
                for lang_code in os.listdir(dir_path):
                    lang_dir = os.path.join(dir_path, lang_code)
                    if os.path.isdir(lang_dir):
                        tokenized_info[name][lang_code] = {
                            'train': os.path.join(lang_dir, "train.jsonl"),
                            'val': os.path.join(lang_dir, "val.jsonl"),
                            'test': os.path.join(lang_dir, "test.jsonl")
                        }
        
        return tokenized_info
    
    logger.info("Step 3: Tokenizing corpora...")
    # Prepare tokenizer configurations for the corpus tokenizer
    tokenizer_configs = {}
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        if name not in tokenizer_paths:
            logger.warning(f"No trained tokenizer found for {name}, skipping tokenization")
            continue
            
        tokenizer_info = tokenizer_paths[name]
        tokenizer_type = tokenizer_info.get("type", "standard")
        
        if tokenizer_type == "standard":
            tokenizer_configs[name] = {
                "type": "standard",
                "path": tokenizer_info["path"],
                "multigram": tokenizer_info.get("multigram", False),
                "use_best_lang": False
            }
        elif tokenizer_type == "langspec":
            tokenizer_configs[name] = {
                "type": "langspec",
                "base_path": tokenizer_info["base_path"],
                "language_paths": tokenizer_info.get("language_paths", {}),
                "use_best_lang": True
            }
    
    # Create corpus tokenizer
    max_lines = config["experiment"].get("max_samples_per_language")
    tokenizer = CorpusTokenizer(
        corpus_info=corpus_info,
        tokenizers=tokenizer_configs,
        output_dir=tokenized_data_dir,
        max_lines_per_file=max_lines,
        skip_if_exists=True
    )
    
    # Tokenize all corpora
    tokenized_info = tokenizer.tokenize_all()
    
    # Map the tokenized files to the corresponding output directories
    result = {}
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        if name in tokenized_info:
            result[name] = tokenized_info[name]
    
    return result


def train_models(config: Dict[str, Any], tokenized_info: Dict[str, Dict[str, Dict[str, str]]], tokenizer_paths: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Train language models on the tokenized corpora."""
    
    # Check for pretrained models first
    model_paths = {}
    models_dir = config["experiment"]["models_dir"].format(data_dir=config["experiment"]["data_dir"])
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        model_config = config["models"].get(name, {})
        
        if model_config.get("pretrained", {}).get("enabled", False):
            pretrained_path = model_config["pretrained"]["path"].format(models_dir=models_dir)
            if os.path.exists(pretrained_path):
                logger.info(f"Using pretrained model for {name} from {pretrained_path}")
                model_paths[name] = pretrained_path
    
    # If we have all models needed and train_models is False, return early
    if len(model_paths) == len(config["tokenizers"]) and not config["steps"].get("train_models", True):
        return model_paths
    
    # Otherwise, continue with training for any missing models
    if not config["steps"].get("train_models", True):
        logger.info("Skipping model training (configured to skip)")
        # Create a model_paths structure from models directory
        model_paths = {}
        for tokenizer_config in config["tokenizers"]:
            name = tokenizer_config["name"]
            model_dir = os.path.join(models_dir, f"{name}_model")
            if os.path.exists(model_dir):
                model_paths[name] = model_dir
        
        return model_paths
    
    logger.info("Step 4: Training language models...")
    # Prepare model training configuration
    model_config = {
        "tokenizers": [],
        "global_settings": {
            "model_output_dir": models_dir,
            "skip_existing": True,
            "seed": 42
        }
    }
    
    # Add each tokenizer to the configuration
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        if name not in tokenized_info or name not in tokenizer_paths:
            logger.warning(f"Missing tokenized data or tokenizer path for {name}, skipping model training")
            continue
        
        tokenizer_info = tokenizer_paths[name]
        tokenizer_type = tokenizer_info.get("type", "standard")
        
        # Get model configuration for this tokenizer
        model_params = config["models"].get(name, {})

        tokenizer_entry = {
            "name": name,
            "tokenized_files": tokenized_info[name],
            "model_config": model_params
        }
        
        # Add tokenizer path based on type
        if tokenizer_type == "standard":
            tokenizer_entry["tokenizer_path"] = tokenizer_info["path"]
        elif tokenizer_type == "langspec":
            tokenizer_entry["tokenizer_path"] = tokenizer_info["base_path"]
        
        model_config["tokenizers"].append(tokenizer_entry)
    
    # Train models
    model_paths = train_models_from_config(model_config)
    
    return model_paths


def evaluate_models(
    config: Dict[str, Any], 
    model_paths: Dict[str, str], 
    tokenized_info: Dict[str, Dict[str, Dict[str, str]]],
    tokenizer_paths: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate the trained models."""
    if not config["steps"].get("evaluate", True):
        logger.info("Skipping model evaluation (configured to skip)")
        return None
    
    logger.info("Step 5: Evaluating models...")
    try:
        from code.perplexity_evaluator import evaluate_models, compare_perplexity_results
        
        # Prepare evaluation configuration
        eval_config = []
        for tokenizer_config in config["tokenizers"]:
            name = tokenizer_config["name"]
            if name not in model_paths or name not in tokenized_info or name not in tokenizer_paths:
                logger.warning(f"Missing model, tokenized data, or tokenizer path for {name}, skipping evaluation")
                continue
            
            tokenizer_info = tokenizer_paths[name]
            tokenizer_type = tokenizer_info.get("type", "standard")
            
            eval_entry = {
                "name": name,
                "model_path": model_paths[name],
                "tokenized_files": tokenized_info[name]
            }
            
            # Add tokenizer path based on type
            if tokenizer_type == "standard":
                eval_entry["tokenizer_path"] = tokenizer_info["path"]
            elif tokenizer_type == "langspec":
                eval_entry["tokenizer_path"] = tokenizer_info["base_path"]
            
            eval_config.append(eval_entry)
        
        # Evaluate models
        perplexity_results = evaluate_models(eval_config)
        
        # Compare results
        comparison = compare_perplexity_results(perplexity_results)
        
        return {
            "perplexity": perplexity_results,
            "comparison": comparison
        }
    except ImportError:
        logger.warning("Perplexity evaluator module not found, skipping evaluation")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run multilingual tokenization experiments")
    
    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    config_group.add_argument("--output-config", type=str, default="experiment_config.json", 
                            help="Path to save generated config (with --create-config)")
    
    # Sampling options
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--max-samples", type=int, default=None, 
                      help="Maximum samples per language (default: None)")
    sampling_group.add_argument("--max-total-samples", type=int, 
                      help="Maximum total samples across all languages, distributed proportionally")
    
    # Vocabulary options
    parser.add_argument("--vocab-size", type=int, default=None, 
                      help="Vocabulary size for tokenizers (overrides config)")
    
    # Step control
    step_group = parser.add_argument_group("Step control")
    step_group.add_argument("--skip-download", action="store_true", 
                         help="Skip corpus download if already available")
    step_group.add_argument("--skip-tokenization", action="store_true", 
                         help="Skip tokenization if already done")
    step_group.add_argument("--skip-training", action="store_true", 
                         help="Skip model training if already done")
    step_group.add_argument("--skip-evaluation", action="store_true", 
                         help="Skip model evaluation")
    
    args = parser.parse_args()
        
    # Load configuration
    config = load_config(args.config)
    data_dir = config["experiment"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    for sub_dir_name in ["data_dir","tokenizers_dir","tokenized_data_dir","models_dir"]:
        sub_dir = config["experiment"].get(sub_dir_name, "").format(data_dir=data_dir)
        os.makedirs(sub_dir, exist_ok=True)
    
    # Override configuration with command-line arguments
    if args.vocab_size:
        config["experiment"]["vocab_size"] = args.vocab_size
    
    # Update step control based on arguments
    if args.skip_download:
        config["steps"]["download"] = False
    if args.skip_tokenization:
        config["steps"]["tokenize_corpora"] = False
    if args.skip_training:
        config["steps"]["train_models"] = False
    if args.skip_evaluation:
        config["steps"]["evaluate"] = False
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting experiment at {start_time}")
    
    try:
        # Step 1: Download and process corpora
        corpus_info = process_corpora(config, args)
        
        # Step 2: Train tokenizers
        tokenizer_paths = train_tokenizers(config, corpus_info)
        
        # Step 3: Tokenize corpora
        tokenized_info = tokenize_corpora(config, corpus_info, tokenizer_paths)
        
        # Step 4: Train language models
        model_paths = train_models(config, tokenized_info, tokenizer_paths)
        
        # Step 5: Evaluate models
       # evaluation_results = evaluate_models(config, model_paths, tokenized_info, tokenizer_paths)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Experiment completed at {end_time} (duration: {duration})")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
