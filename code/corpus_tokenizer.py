import os
import logging
import json
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
# New faster helpers (module‑level – put them near the top of the file)
# ----------------------------------------------------------------------
import ujson                # 2‑3× faster than stdlib json
from tokenizers import AddedToken    # only used in one fast‑path check
from code.utils import encode_text,load_tokenizer_from_config

# -- multiprocessing initialisation ------------------------------------
# We keep one global copy of the tokenizer in each worker; that avoids
# pickling it for every task *and* gives copy‑on‑write sharing on Unix.
_WORKER_TOKENIZER = None
_WORKER_VOCAB = None
_WORKER_UNK_ID = 0
_WORKER_MODE = None      # "best" | "specific"
_WORKER_LANG = None

def _init_langspec_worker(tokenizer, mode, lang):
    global _WORKER_TOKENIZER, _WORKER_VOCAB, _WORKER_UNK_ID
    global _WORKER_MODE, _WORKER_LANG
    _WORKER_TOKENIZER = tokenizer
    _WORKER_VOCAB = tokenizer.get_vocab()
    _WORKER_UNK_ID = _WORKER_VOCAB[tokenizer.unk_token]
    _WORKER_MODE = mode
    _WORKER_LANG = lang
    # Deactivate HF’s own thread‑pool so we do not nest parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -- per‑line worker ----------------------------------------------------
def _process_line_line(line: str):
    """
    Tokenise one line and return a JSON‑serialisable dict.
    Called inside the process pool – *must* be top‑level picklable.
    """
    if not line:
        return None

    if _WORKER_MODE == "best":
        best_lang, tokens = _WORKER_TOKENIZER.custom_viterbi_best_language(line)
    else:
        tokens = _WORKER_TOKENIZER.custom_viterbi_tokenize(line, _WORKER_LANG)
        best_lang = None

    ids = [_WORKER_VOCAB.get(tok, _WORKER_UNK_ID) for tok in tokens]
    rec = {
        "text": line,
        "tokens": tokens,
        "token_ids": ids,
    }
    if best_lang:
        rec["best_lang"] = best_lang
    return rec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorpusTokenizer:
    """
    A class for tokenizing corpora with various tokenizers 
    
    """
    
    def __init__(self, 
                 corpus_info: Dict[str, Dict[str, Any]],
                 tokenizers: Dict[str, Dict[str, Any]],
                 output_dir: str = "",
                 max_lines_per_file: Optional[int] = 10000,
                 processes: Optional[int] = None,
                 skip_if_exists: bool = True):
        """
        Initialize the CorpusTokenizer.
        
        Args:
            corpus_info: Dictionary mapping language codes to corpus information
            tokenizers: Dictionary mapping tokenizer names to tokenizer configurations
            output_dir: Base output directory for tokenized files
            max_lines_per_file: Maximum number of lines to process per file
            processes: Number of processes for parallel processing
            skip_if_exists: Whether to skip processing if output file exists
        """
        self.corpus_info = corpus_info
        self.tokenizer_configs = tokenizers
        self.output_dir = output_dir
        self.max_lines_per_file = max_lines_per_file
        self.skip_if_exists = skip_if_exists
        
        # Set number of processes
        if processes is None:
            self.processes = max(1, min(20, cpu_count() - 1))
        else:
            self.processes = processes
            
        # Load tokenizers
        self.tokenizers = self._load_tokenizers()
        
        # Track tokenized files
        self.tokenized_files = {name: {} for name in self.tokenizers.keys()}
    
    def _load_tokenizers(self) -> Dict[str, Any]:
        """
        Load all tokenizers from configurations.
        
        Returns:
            Dictionary mapping tokenizer names to loaded tokenizer objects
        """
        loaded_tokenizers = {}
        
        for name, config in self.tokenizer_configs.items():
            try:
                tokenizer = load_tokenizer_from_config(config)
                loaded_tokenizers[name] = {
                    'tokenizer': tokenizer,
                    'config': config
                }
                logger.info(f"Loaded tokenizer '{name}'")
                
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{name}' with config {config}: {str(e)}")
        
        return loaded_tokenizers
    
    def tokenize_all(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Tokenize all corpora with all loaded tokenizers.
        
        Returns:
            Dictionary mapping tokenizer names to language codes to split names to output file paths
        """
        
        for tokenizer_name, tokenizer_data in self.tokenizers.items():
            tokenizer = tokenizer_data['tokenizer']
            config = tokenizer_data['config']
            
            # Create output directory for this tokenizer
            tokenizer_output_dir = os.path.join(self.output_dir, tokenizer_name)
            os.makedirs(tokenizer_output_dir, exist_ok=True)
            
            logger.info(f"Tokenizing corpora with '{tokenizer_name}'")
            
            # Initialize results tracking for this tokenizer
            self.tokenized_files[tokenizer_name] = {}
            
            for lang_code, info in self.corpus_info.items():
                if "splits" not in info:
                    continue
                    
                self.tokenized_files[tokenizer_name][lang_code] = {}
                lang_dir = os.path.join(tokenizer_output_dir, lang_code)
                os.makedirs(lang_dir, exist_ok=True)
                
                for split_name, split_file in info["splits"].items():
                    if not os.path.isfile(split_file):
                        logger.warning(f"File not found: {split_file}")
                        continue
                        
                    out_path = os.path.join(lang_dir, f"{split_name}.jsonl")
                    if self.skip_if_exists and os.path.isfile(out_path):
                        self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path
                        logger.info(f"Skipping existing file: {out_path}")
                        continue
                    
                    # Determine tokenization parameters based on tokenizer type
                    use_best_lang = config.get('use_best_lang', True)
                    if not use_best_lang:
                        raise NotImplementedError("Need to implement still")
                    
                    # Process the file in batches using a streaming approach
                    line_count = 0
                    unk_count = 0
                    batch_size = 1024  # Adjust batch size as needed
                    
                    with open(split_file, "r", encoding="utf-8", errors="replace") as fin, \
                         open(out_path, "w", encoding="utf-8") as fout:
                        
                        batch_lines = []
                        
                        for line in fin:
                            line = line.rstrip("\n")
                            if not line.strip():
                                continue
                                
                            batch_lines.append(line)
                            
                            # Process batch when it reaches the desired size
                            if len(batch_lines) >= batch_size:
                                self._process_batch(
                                    tokenizer, 
                                    batch_lines, 
                                    fout, 
                                    use_best_lang=use_best_lang
                                )
                                
                                line_count += len(batch_lines)
                                if hasattr(tokenizer, 'unk_token'):
                                    unk_token = tokenizer.unk_token
                                    for record in batch_lines:
                                        if 'tokens' in record:
                                            unk_count += record['tokens'].count(unk_token)
                                            
                                batch_lines = []
                        
                        # Process any remaining lines
                        if batch_lines:
                            self._process_batch(
                                tokenizer, 
                                batch_lines, 
                                fout, 
                                use_best_lang=use_best_lang
                            )
                            line_count += len(batch_lines)
                    
                    logger.info(f"[{tokenizer_name}] {lang_code}-{split_name} => "
                                f"{line_count} lines, {unk_count} UNK tokens.")
                    
                    if line_count:
                        self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path
        
        return self.tokenized_files

    def _process_batch(
        self,
        tokenizer,
        batch_lines,
        output_file,
        *,
        use_best_lang: bool = True,
        language: str | None = None,
    ):
        """
        Tokenise a batch of lines with the unified `encode_text` helper and
        write one JSON object per line to `output_file`.
        """
        # --- 1) encode ----------------------------------------------------
        enc = encode_text(
            tokenizer,
            batch_lines,
            use_best_language=use_best_lang,
            language=language,
            # we don't need extra tensors or masks here
        )

        # enc fields are always batch-first lists
        ids_batch   = enc["input_ids"]
        tok_batch   = enc.get("tokens", [[] for _ in batch_lines])
        best_lang_b = enc.get("best_lang")

        # --- 2) dump to disk ---------------------------------------------
        for i, text in enumerate(batch_lines):
            rec = {
                "text":      text,
                "token_ids": ids_batch[i],
                "tokens":    tok_batch[i],
            }
            if best_lang_b is not None:
                rec["best_lang"] = best_lang_b[i]
            output_file.write(ujson.dumps(rec) + "\n")


def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8', errors="replace") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Tokenize corpora with multiple tokenizers')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--output_dir', type=str, default="", help='Base output directory')
    parser.add_argument('--max_lines', type=int, default=10000, help='Maximum lines per file')
    parser.add_argument('--processes', type=int, help='Number of processes for parallel processing')
    parser.add_argument('--no_skip', action='store_true', help='Do not skip existing files')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create corpus tokenizer
    corpus_tokenizer = CorpusTokenizer(
        corpus_info=config['corpus_info'],
        tokenizers=config['tokenizers'],
        output_dir=args.output_dir,
        max_lines_per_file=args.max_lines,
        processes=args.processes,
        skip_if_exists=not args.no_skip
    )
    
    # Tokenize all corpora
    results = corpus_tokenizer.tokenize_all()
    
    # Print summary
    logger.info("\n===== Tokenization Summary =====")
    for tokenizer_name, lang_data in results.items():
        logger.info(f"Tokenizer: {tokenizer_name}")
        for lang, split_data in lang_data.items():
            logger.info(f"  Language: {lang}")
            for split, path in split_data.items():
                logger.info(f"    {split}: {path}")


if __name__ == "__main__":
    main()
