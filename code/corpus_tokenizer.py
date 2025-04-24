import os
import logging
import json
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path

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
                tokenizer_type = config.get('type', 'standard')
                
                if tokenizer_type == 'standard':
                    from code.standard_unigramlm import StandardUnigramLMTokenizer
                    tokenizer = StandardUnigramLMTokenizer(
                        multigram=False
                    )
                    tokenizer.load(config['path'])
                    
                elif tokenizer_type == 'langspec':
                    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
                    tokenizer = LanguageSpecificUnigramLMTokenizer()
                    tokenizer.load(config['base_path'], config.get('language_paths'))
                    
                elif tokenizer_type == 'multigram':
                    from code.standard_unigramlm import StandardUnigramLMTokenizer
                    tokenizer = StandardUnigramLMTokenizer(
                        multigram=True
                    )
                    tokenizer.load(config['base_path'])
                else:
                    # Try loading as a HuggingFace tokenizer
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(config['path'])
                    except:
                        logger.error(f"Failed to load tokenizer '{name}' of unknown type: {tokenizer_type}")
                        continue
                
                loaded_tokenizers[name] = {
                    'tokenizer': tokenizer,
                    'config': config
                }
                logger.info(f"Loaded tokenizer '{name}' of type '{tokenizer_type}'")
                
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{name}': {str(e)}")
        
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
            
            # Determine tokenization method
            if hasattr(tokenizer, 'custom_viterbi_best_language') and config.get('use_best_lang', True):
                # Language-specific tokenizer with best language selection
                self._tokenize_with_langspec(
                    tokenizer_name=tokenizer_name,
                    tokenizer=tokenizer, 
                    output_dir=tokenizer_output_dir,
                    use_specific_lang=False
                )
            elif hasattr(tokenizer, 'custom_viterbi_tokenize') and not config.get('use_best_lang', True):
                # Language-specific tokenizer with specific language
                self._tokenize_with_langspec(
                    tokenizer_name=tokenizer_name,
                    tokenizer=tokenizer, 
                    output_dir=tokenizer_output_dir,
                    use_specific_lang=True
                )
            else:
                # Standard tokenizer or other types
                self._tokenize_with_standard(
                    tokenizer_name=tokenizer_name,
                    tokenizer=tokenizer,
                    output_dir=tokenizer_output_dir
                )
        
        return self.tokenized_files
    
    def _tokenize_with_standard(self, tokenizer_name: str, tokenizer: Any, output_dir: str) -> None:
        """
        Tokenize corpora with a standard tokenizer.
        
        Args:
            tokenizer_name: Name of the tokenizer
            tokenizer: Tokenizer object
            output_dir: Output directory for tokenized files
        """
        self.tokenized_files[tokenizer_name] = {}

        for lang_code, info in self.corpus_info.items():
            if 'splits' not in info:
                continue
                
            self.tokenized_files[tokenizer_name][lang_code] = {}
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            
            for split_name, split_file in info['splits'].items():
                if not os.path.isfile(split_file):
                    logger.warning(f"File not found: {split_file}")
                    continue
                unk_count = 0
                out_path = os.path.join(lang_dir, f"{split_name}.jsonl")
                
                if self.skip_if_exists and os.path.isfile(out_path):
                    self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path
                    logger.info(f"Skipping existing file: {out_path}")
                    continue
                
                line_count = 0
                with open(split_file, 'r', encoding='utf-8') as fin, \
                     open(out_path, 'w', encoding='utf-8') as fout:
                    for i, line in enumerate(tqdm(fin, desc=f"{tokenizer_name} {lang_code}-{split_name}")):
                        if self.max_lines_per_file and i >= self.max_lines_per_file:
                            break
                        
                        text = line.strip()
                        if not text:
                            continue
                        
                        # Encode the text
                        try:
                            if hasattr(tokenizer, 'encode'):
                                ids = tokenizer.encode(text)
                                # Try to get tokens if possible
                                if hasattr(tokenizer, 'base_tokenizer') and hasattr(tokenizer.base_tokenizer, 'id_to_token'):
                                    tokens = [tokenizer.base_tokenizer.id_to_token(x) for x in ids]
                                else:
                                    # Use decode on individual tokens if id_to_token not available
                                    tokens = [tokenizer.decode([i]) for i in ids]
                            elif hasattr(tokenizer, 'encode_plus'):
                                # HuggingFace tokenizer
                                encoding = tokenizer.encode_plus(text, add_special_tokens=False)
                                ids = encoding.input_ids
                                tokens = tokenizer.convert_ids_to_tokens(ids)
                            else:
                                # Fallback
                                encoding = tokenizer(text)
                                ids = encoding.input_ids
                                tokens = tokenizer.convert_ids_to_tokens(ids)
                                
                            record = {
                                "text": text,
                                "token_ids": ids,
                                "tokens": tokens
                            }
                            unk_count += sum([1 for t in tokens if t == tokenizer.unk_token])
                            fout.write(json.dumps(record) + "\n")
                            line_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error tokenizing text with {tokenizer_name}: {str(e)}")
                    logger.info()
                
                logger.info(f"[{tokenizer_name}] {lang_code}-{split_name} => {line_count} lines with {unk_count} UNK tokens.")
                if line_count > 0:
                    self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path
    
    def _tokenize_with_langspec(self, 
                               tokenizer_name: str, 
                               tokenizer: Any, 
                               output_dir: str,
                               use_specific_lang: bool = False) -> None:
        """
        Tokenize corpora with a language-specific tokenizer.
        
        Args:
            tokenizer_name: Name of the tokenizer
            tokenizer: Language-specific tokenizer object
            output_dir: Output directory for tokenized files
            use_specific_lang: Whether to use the specific language for tokenization
        """
        self.tokenized_files[tokenizer_name] = {}
        
        for lang_code, info in self.corpus_info.items():
            if 'splits' not in info:
                continue
                
            self.tokenized_files[tokenizer_name][lang_code] = {}
            lang_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)
            
            for split_name, split_file in info['splits'].items():
                if not os.path.isfile(split_file):
                    logger.warning(f"File not found: {split_file}")
                    continue
                    
                unk_count = 0
                out_path = os.path.join(lang_dir, f"{split_name}.jsonl")
                
                if self.skip_if_exists and os.path.isfile(out_path):
                    self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path
                    logger.info(f"Skipping existing file: {out_path}")
                    continue
                
                # Read lines
                lines = []
                with open(split_file, 'r', encoding='utf-8') as fin:
                    for i, line in enumerate(fin):
                        if self.max_lines_per_file and i >= self.max_lines_per_file:
                            break
                        line = line.strip()
                        if line:
                            lines.append(line)
                
                if not lines:
                    logger.warning(f"No lines found in {split_file}")
                    continue
                
                # Process in chunks
                chunk_size = 1000
                chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
                
                # Parallel processing
                if use_specific_lang:
                    # Use specific language for tokenization
                    encode_fn = lambda text: self._encode_with_specific_lang(tokenizer, text, lang_code)
                    worker_args = [(chunk, tokenizer, encode_fn) for chunk in chunks]
                    worker_func = self._langspec_chunk_worker
                else:
                    # Use best language detection
                    worker_args = [(chunk, tokenizer) for chunk in chunks]
                    worker_func = self._langspec_best_lang_chunk_worker
                
                all_results = []
                with Pool(self.processes) as pool:
                    for partial_res in tqdm(
                        pool.imap_unordered(worker_func, worker_args),
                        total=len(chunks),
                        desc=f"{tokenizer_name} {lang_code}-{split_name}"
                    ):
                        all_results.extend(partial_res)
                
                # Write results
                with open(out_path, 'w', encoding='utf-8') as fout:
                    for rec in all_results:
                        unk_count += sum([1 for t in rec['tokens'] if t == tokenizer.unk_token])
                        fout.write(json.dumps(rec) + "\n")
                
                logger.info(f"[{tokenizer_name}] {lang_code}-{split_name} => {len(all_results)} lines with {unk_count} UNK tokens.")
                self.tokenized_files[tokenizer_name][lang_code][split_name] = out_path

    
    @staticmethod
    def _langspec_best_lang_chunk_worker(args):
        """
        Worker for processing a chunk of lines with language-specific tokenizer using best language.
        
        Args:
            args: Tuple of (chunk_lines, tokenizer)
            
        Returns:
            List of tokenized records
        """
        chunk_lines, tokenizer = args
        results = []
        base_vocab = tokenizer.get_vocab()
        unk_token = tokenizer.unk_token
        
        for line in chunk_lines:
            if not line:
                continue
                
            best_lang, tokens = tokenizer.custom_viterbi_best_language(line)
            ids = [base_vocab.get(t, base_vocab.get(unk_token, 0)) for t in tokens]
            
            record = {
                "text": line,
                "best_lang": best_lang,
                "tokens": tokens,
                "token_ids": ids
            }
            results.append(record)
            
        return results
    
    @staticmethod
    def _encode_with_specific_lang(tokenizer, text, lang_code):
        """
        Encode text with a specific language.
        
        Args:
            tokenizer: Language-specific tokenizer
            text: Text to encode
            lang_code: Language code to use
            
        Returns:
            Tuple of (tokens, None) - No best language since it's specified
        """
        tokens = tokenizer.custom_viterbi_tokenize(text, lang_code)
        return tokens, None
    
    
    @staticmethod
    def _langspec_chunk_worker(args):
        """
        Worker for processing a chunk of lines with language-specific tokenizer using a specified language.
        
        Args:
            args: Tuple of (chunk_lines, tokenizer, encode_fn)
            
        Returns:
            List of tokenized records
        """
        chunk_lines, tokenizer, encode_fn = args
        results = []
        base_vocab = tokenizer.get_vocab()
        unk_token = tokenizer.unk_token
        
        for line in chunk_lines:
            if not line:
                continue
                
            tokens, best_lang = encode_fn(tokenizer, line)
            ids = [base_vocab.get(t, base_vocab.get(unk_token, 0)) for t in tokens]
            
            record = {
                "text": line,
                "tokens": tokens,
                "token_ids": ids
            }
            if best_lang:
                record["best_lang"] = best_lang
                
            results.append(record)
            
        return results
    

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
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
