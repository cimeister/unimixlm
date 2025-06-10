import os
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

# Import and setup environment variables
from code.environment_setup import setup_environment
setup_environment()
from code.utils import SafeFormattingDict, proportional_sample
from code.corpus_downloader import download_corpus
from code.standard_unigramlm import train_standard_tokenizer, train_lang_tokenizers, train_tokmix
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
            if lang_code == "sampled":
                continue
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
            
    corpus_info = download_corpus(config)
    return corpus_info


def train_tokenizers(config: Dict[str, Any], corpus_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Train various tokenizers specified in the configuration."""
    logger.info("Step 2: Training tokenizers...")
    
    # Dictionary to store tokenizer paths and configurations
    tokenizer_paths = {}
    tokenizers_dir = config["experiment"]["tokenizers_dir"].format(data_dir=config["experiment"]["data_dir"])

    sample_root = os.path.join(config["corpus"]["processed_dir"], "sampled")
    tokenizer_training_files = proportional_sample({lg: info['splits']["train"] for lg, info in corpus_info.items()}, sample_root)

    # Process each tokenizer configuration
    for tokenizer_config in config["tokenizers"]:
        tokenizer_name = tokenizer_config["name"]
        tokenizer_type = tokenizer_config["type"]
        byte_level = config["experiment"].get("byte_level", False)
        em_mode = tokenizer_config.get("em_mode", "soft")
        min_char_occurence_threshold = tokenizer_config.get("min_char_occurence_threshold")

        logger.info(f"Processing tokenizer: {tokenizer_name} (type: {tokenizer_type} at {'byte' if byte_level else 'char'}-level)")
        
        # Check if we should use a pretrained tokenizer
        
        if tokenizer_config.get("skip_if_exists", False):
            logger.info(f"Using pretrained tokenizer for {tokenizer_name}")
            
            if tokenizer_type == "langspec":
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
                    for lang_code in tokenizer_training_files.keys():
                        prob_path = pretrained_prob_path_format.format(lang_code=lang_code)
                        if os.path.exists(prob_path):
                            language_paths[lang_code] = prob_path
                    
                    tokenizer_paths[tokenizer_name] = {
                        "base_path": pretrained_base_path,
                        "language_paths": language_paths,
                        "class": "langspec"
                    }
                    logger.info(f"Loaded pretrained language-specific tokenizer from {pretrained_base_path}")
                    continue
            else:
                pretrained_path = tokenizer_config["path"].format(tokenizers_dir=tokenizers_dir, em_mode=em_mode)
                if os.path.exists(pretrained_path):
                    tokenizer_paths[tokenizer_name] = {
                        "path": pretrained_path,
                        "class": "standard",
                        "multigram": tokenizer_config.get("multigram", False)
                    }
                    logger.info(f"Loaded pretrained tokenizer from {pretrained_path}")
                    continue
                else:
                    logger.warning(f"Pretrained tokenizer not found at {pretrained_path}, will train a new one")

        vocab_size = config["experiment"].get("vocab_size", 8000)
        if tokenizer_type == "langspec":
            # Handle language-specific tokenizer            
            em_mode = tokenizer_config.get("em_mode", "soft")
            base_tokenizer_name = tokenizer_config.get("base_tokenizer", "").format_map(
                SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
            if not base_tokenizer_name:
                logger.info(f"Missing base_tokenizer for language-specific tokenizer: {tokenizer_name}. Will train a new one...")
                
            tokenizer_path_format = tokenizer_config.get(
                "tokenizer_path_format", ""
            ).format_map(SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
            
            # Train language-specific tokenizer
            paths = train_language_specific_tokenizer(
                tokenizer_training_files,
                vocab_size=vocab_size,
                reestimation_em_mode=em_mode,
                byte_level=byte_level,
                output_dir=tokenizers_dir,
                base_tokenizer_path=base_tokenizer_name,
                tokenizer_path_format=tokenizer_path_format,
                load_base_if_exists=base_tokenizer_name and tokenizer_config.get("skip_base_if_exists", True),
                load_tokenizers_if_exists=tokenizer_config.get("skip_if_exists", True)
            )
            
            if paths:
                paths["class"] ="langspec"
                tokenizer_paths[tokenizer_name] = paths


        elif tokenizer_type == "tokmix":
            output_file = tokenizer_config.get("path", f"tokmix.json").format_map(
                SafeFormattingDict(tokenizers_dir=tokenizers_dir))

            paths = train_lang_tokenizers(tokenizer_training_files, tokenizers_dir,
                                          vocab_size=vocab_size, 
                                          byte_level=byte_level)

            tokmix_path = train_tokmix(paths,
                                       k_final=vocab_size,
                                       byte_level=byte_level,
                                       output_path=output_file)

            tokenizer_paths[tokenizer_name] = {"path": tokmix_path,
                                               "class": "standard",
                                               "multigram": False}
        
        elif tokenizer_type in ["standard", "bpe"]:
            # Handle standard tokenizer
            em_mode = tokenizer_config.get("em_mode", "soft") if tokenizer_type == "standard" else "bpe"
            multigram = tokenizer_config.get("multigram", False) if tokenizer_type == "standard" else False
            output_file = tokenizer_config.get("path", f"standard_unigramlm_{em_mode}.json").format_map(
                SafeFormattingDict(tokenizers_dir=tokenizers_dir, em_mode=em_mode))
                        
            # Train new tokenizer
            
            path = train_standard_tokenizer(
                tokenizer_training_files, 
                vocab_size=vocab_size, 
                em_mode=em_mode, 
                multigram=multigram,
                byte_level=byte_level,
                output_path=output_file,
                min_char_occurence_threshold=min_char_occurence_threshold
            )
            
            if path:
                tokenizer_paths[tokenizer_name] = {
                    "path": path,
                    "class": "standard",
                    "multigram": multigram
                }

        else:
            logger.warning(f"Unsupported tokenizer type: {tokenizer_type}")
    
    return tokenizer_paths


def tokenize_corpora(config: Dict[str, Any], corpus_info: Dict[str, Any], tokenizer_paths: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, str]]]: 
    """Tokenize corpora using the trained tokenizers."""
    
    # Check if we should use pre-tokenized data
    tokenized_data_dir = config["experiment"].get("tokenized_data_dir", "").format(data_dir=config["experiment"]["data_dir"])
    
    """Tokenize corpora using the trained tokenizers."""
    if not config["steps"].get("tokenize_corpora", True):
        logger.info("Skipping corpus tokenization (configured to skip)") 
        tokenized_info = {}
        for tokenizer_config in config["tokenizers"]:
            tokenizer_name = tokenizer_config["name"]
            output_dir = tokenizer_config.get("tokenized_data", tokenizer_name)
            dir_path = os.path.join(tokenized_data_dir, output_dir)   
            for lang_code in os.listdir(dir_path):
                lang_dir = os.path.join(dir_path, lang_code)
                
                if os.path.isdir(lang_dir):
                    tokenized_info[tokenizer_name][lang_code] = {
                        'train': os.path.join(lang_dir, "train.jsonl"),
                        'val': os.path.join(lang_dir, "val.jsonl"),
                        'test': os.path.join(lang_dir, "test.jsonl")
                    }
        return tokenized_info
    
    logger.info("Step 3: Tokenizing corpora...")
    use_pretokenized_data_if_exists = config["tokenized_data"].get("use_pretokenized_data_if_exists", False)

    # Prepare tokenizer configurations for the corpus tokenizer
    tokenizer_configs = {}
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        if name not in tokenizer_paths:
            logger.warning(f"No trained tokenizer found for {name}, skipping tokenization")
            continue
        tokenizer_configs[name] = tokenizer_paths[name]
    
    # Create corpus tokenizer
    max_lines = config["experiment"].get("max_samples_per_language")
    tokenizer = CorpusTokenizer(
        corpus_info=corpus_info,
        tokenizers=tokenizer_configs,
        output_dir=tokenized_data_dir,
        max_lines_per_file=max_lines,
        skip_if_exists=use_pretokenized_data_if_exists
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
            "seed": config["experiment"]["seed"]
        }
    }
    
    # Add each tokenizer to the configuration
    for tokenizer_config in config["tokenizers"]:
        name = tokenizer_config["name"]
        if name not in tokenized_info or name not in tokenizer_paths:
            logger.warning(f"Missing tokenized data or tokenizer path for {name}, skipping model training")
            continue
        
        tokenizer_info = tokenizer_paths[name]        
        # Get model configuration for this tokenizer
        model_params = config["models"].get(name, {})

        tokenizer_entry = {
            "name": name,
            "tokenized_files": tokenized_info[name],
            "model_config": model_params
        }
        
        tokenizer_entry["tokenizer_path"] = tokenizer_info.get("path") or tokenizer_info["base_path"]
        model_config["tokenizers"].append(tokenizer_entry)
    
    # Train models
    model_paths = train_models_from_config(model_config)
    
    return model_paths


def finetune_models(
    config: Dict[str, Any],
    model_paths: Dict[str, str],
    tokenizer_paths: Dict[str, Dict[str, Any]],
    languages: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Fine‑tune each language model on the tasks listed in
    config['finetune_tasks'] and return
        { tokenizer_name -> { task -> metrics_path } }
    """

    # 0) quick exit ---------------------------------------------------------
    if not config["steps"].get("finetune_models", True):
        logger.info("Skipping fine‑tuning (configured to skip).")
        return {}

    if not config.get("finetune_tasks"):
        logger.info("No finetune_tasks found in config; nothing to do.")
        return {}

    # 1) check for existing results ----------------------------------------
    results_dir = config["experiment"]["finetune_dir"].format(
        data_dir=config["experiment"]["data_dir"]
    )
    ft_results: Dict[str, Dict[str, str]] = {}
    for tok_cfg in config["tokenizers"]:
        name = tok_cfg["name"]
        model_dir = model_paths.get(name)
        if not model_dir:
            logger.warning("No trained model for %s; skipping fine‑tune.", name)
            continue

        # pretrained_ft = config["finetune_tasks"].get(name, {}).get("pretrained", {})
        # if pretrained_ft.get("enabled", False):
        #     res_path = pretrained_ft["path"].format(finetune_dir=results_dir)
        #     if os.path.exists(res_path):
        #         logger.info("Using pretrained fine‑tune results for %s from %s", name, res_path)
        #         with open(res_path) as fh:
        #             ft_results[name] = json.load(fh)
        #         continue  # next tokenizer

        # 2) build per‑model entry -----------------------------------------
        if name not in tokenizer_paths:
            logger.warning("Tokenizer path missing for %s; skipping.", name)
            continue

        pretrained_model_dir_base = os.path.basename(model_dir)

        ft_entry = {
            "name":          name,
            "model_path":    model_dir,
            "tokenizer_info": tokenizer_paths[name],
            "tasks":         config["finetune_tasks"],
            "hyper":         config.get("finetune_hyper", {}),
            "output_dir":    os.path.join(results_dir, pretrained_model_dir_base),
            "languages":     languages
        }

        # 3) launch in‑process fine‑tune -----------------------------------
        from code.model_finetuner import finetune_from_config  # import here to avoid heavy deps at startup
        metrics = finetune_from_config(ft_entry)   # returns {task: json_path}
        ft_results[name] = metrics

    return ft_results


def evaluate_model_perpexity(
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
    parser.add_argument("--byte-level", action="store_true", 
                      help="Use byte level tokenizers (overrides config)")
    
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
    if args.byte_level:
        config["experiment"]["byte_level"] = True
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
        
        ft_results = finetune_models(config, model_paths, tokenizer_paths, languages=list(corpus_info.keys()))
        # Step 5: Evaluate models
        # TODO: update evaluate_models func  
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
