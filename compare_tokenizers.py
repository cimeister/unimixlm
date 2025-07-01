import logging
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from pathlib import Path
import unicodedata
import string
from code.utils import load_tokenizer_from_config, encode_text
from code.environment_setup import setup_environment
setup_environment()

MORPHOLEX_LANGCODE_MAPPING = {
            'en': 'eng',
            'es': 'spa',
            'de': 'deu',
            'ru': 'rus',
            'pl': 'pol',
            'it': 'ita',
            'fr': 'fra',
            'pt': 'por',
            'hu': 'hun',
            'fi': 'fin',
            'sv': 'swe',
            'cs': 'ces',
            'ca': 'cat',
            'mn': 'mon',
            'sh': 'hbs'  }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MorphologicalDataLoader:
    """
    Loader for various morphological datasets including LADEC, MorphoLex, MorphyNet, and DagoBert.
    Provides unified interface for accessing morphological segmentations.
    """
    
    def __init__(self, morphological_config: Optional[Dict[str, str]] = None):
        """
        Initialize morphological data loader.
        
        Args:
            morphological_config: Dict mapping dataset names to file paths
                Example: {
                    'ladec': 'path/to/ladec.txt',
                    'morpholex': 'path/to/morpholex.csv',
                    'morphynet': 'path/to/morphynet.tsv',
                    'dagobert': 'path/to/dagobert.json'
                }
        """
        self.config = morphological_config or {}
        self.morphological_data = {}  # lang -> dataset -> word -> morphemes
        self.available_languages = set()
    
    def load_all_datasets(self) -> None:
        """Load all configured morphological datasets."""
        for dataset_name, file_path in self.config.items():
            if not os.path.exists(file_path):
                logger.warning(f"Morphological dataset file not found: {file_path}")
                continue
                
            try:
                if dataset_name.lower() == 'ladec':
                    self._load_ladec(file_path)
                elif dataset_name.lower() == 'morpholex':
                    self._load_morpholex(file_path)
                elif dataset_name.lower() == 'morphynet':
                    self._load_morphynet(file_path)
                elif dataset_name.lower() == 'dagobert':
                    self._load_dagobert(file_path)
                else:
                    logger.warning(f"Unknown morphological dataset: {dataset_name}")
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {e}")
    
    def _load_ladec(self, file_path: str) -> None:
        """Load LADEC dataset (Gagné et al., 2019)."""
        logger.info(f"Loading LADEC from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0].lower()
                    morphemes = parts[1].split('-')  # Assuming morphemes separated by '-'
                    lang = 'en'  # Assume English for LADEC
                    if lang not in self.morphological_data:
                        self.morphological_data[lang] = {}
                    if 'ladec' not in self.morphological_data[lang]:
                        self.morphological_data[lang]['ladec'] = {}
                    self.morphological_data[lang]['ladec'][word] = morphemes
                    self.available_languages.add(lang)
    
    def _load_morpholex(self, file_path: str) -> None:
        """Load MorphoLex dataset (Sánchez-Gutiérrez et al., 2018)."""
        logger.info(f"Loading MorphoLex from {file_path}")
        try:
            df = pd.read_csv(file_path)
            word_col = 'Word' if 'Word' in df.columns else 'word'
            morph_col = 'MorphoLexSegm' if 'MorphoLexSegm' in df.columns else 'morphemes'
            
            for _, row in df.iterrows():
                word = str(row[word_col]).lower()
                morphemes_str = str(row[morph_col])
                if pd.isna(morphemes_str):
                    continue
                morphemes = re.split(r'[.\-+]', morphemes_str)
                morphemes = [m.strip() for m in morphemes if m.strip()]
                
                lang = 'en'  # Assume English for MorphoLex
                if lang not in self.morphological_data:
                    self.morphological_data[lang] = {}
                if 'morpholex' not in self.morphological_data[lang]:
                    self.morphological_data[lang]['morpholex'] = {}
                self.morphological_data[lang]['morpholex'][word] = morphemes
                self.available_languages.add(lang)
        except ImportError:
            logger.error("pandas required for MorphoLex loading")
    
    def _load_morphynet(self, file_path: str) -> None:
        """Load MorphyNet dataset (Batsuren et al., 2021)."""
        logger.info(f"Loading MorphyNet from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    lang = parts[0]
                    word = parts[1].lower()
                    derivation = parts[2]
                    
                    # Parse derivation: base_word→derived_word:POS
                    if '→' in derivation and ':' in derivation:
                        base_derived, pos = derivation.rsplit(':', 1)
                        if '→' in base_derived:
                            base_word, derived_word_check = base_derived.split('→', 1)
                            
                            # Infer morphemes by analyzing the transformation
                            morphemes = self._infer_morphemes_from_derivation(base_word, word)
                            
                            if lang not in self.morphological_data:
                                self.morphological_data[lang] = {}
                            if 'morphynet' not in self.morphological_data[lang]:
                                self.morphological_data[lang]['morphynet'] = {}
                            self.morphological_data[lang]['morphynet'][word] = morphemes
                            self.available_languages.add(lang)

    def _infer_morphemes_from_derivation(self, base: str, derived: str) -> List[str]:
        """
        Infer morpheme segmentation from base→derived transformation.
        This is a heuristic approach that handles common morphological patterns.
        """
        base = base.lower()
        derived = derived.lower()
        
        # Find common prefix
        common_prefix_len = 0
        for i in range(min(len(base), len(derived))):
            if base[i] == derived[i]:
                common_prefix_len += 1
            else:
                break
        
        # Find common suffix (from the end)
        common_suffix_len = 0
        for i in range(1, min(len(base), len(derived)) + 1):
            if base[-i] == derived[-i]:
                common_suffix_len += 1
            else:
                break
        
        # Handle different morphological patterns
        if derived.startswith(base):
            # Suffixation: base + suffix
            suffix = derived[len(base):]
            if suffix:
                return [base, suffix]
            else:
                return [base]
        elif base.startswith(derived):
            # Truncation: derived is shorter than base
            return [derived]
        elif derived.endswith(base):
            # Prefixation: prefix + base
            prefix = derived[:-len(base)]
            if prefix:
                return [prefix, base]
            else:
                return [base]
        elif common_prefix_len > 0:
            # Complex derivation with stem change
            # Try to identify the stem and affixes
            stem = base[:common_prefix_len]
            
            # Look for common patterns
            if len(derived) > len(base):
                # Possibly prefix + stem + suffix
                remaining_derived = derived[common_prefix_len:]
                remaining_base = base[common_prefix_len:]
                
                # Check if it's a simple prefix addition
                if remaining_derived.endswith(remaining_base):
                    prefix = remaining_derived[:-len(remaining_base)]
                    return [prefix, base] if prefix else [base]
                else:
                    # More complex - just use stem + remainder
                    return [stem, remaining_derived] if remaining_derived else [stem]
            else:
                # Use the common stem plus any suffix
                suffix = derived[common_prefix_len:]
                return [stem, suffix] if suffix else [stem]
        else:
            # No clear relationship - return as single morpheme
            return [derived]

    
    def _load_dagobert(self, file_path: str) -> None:
        """Load DagoBert dataset."""
        logger.info(f"Loading DagoBert from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for entry in data:
            if 'word' in entry and 'morphemes' in entry:
                word = entry['word'].lower()
                morphemes = entry['morphemes']
                lang = entry.get('language', 'de')  # Default to German
                
                if lang not in self.morphological_data:
                    self.morphological_data[lang] = {}
                if 'dagobert' not in self.morphological_data[lang]:
                    self.morphological_data[lang]['dagobert'] = {}
                self.morphological_data[lang]['dagobert'][word] = morphemes
                self.available_languages.add(lang)
    
    def get_morphemes(self, word: str, language: str, dataset: Optional[str] = None) -> Optional[List[str]]:
        """
        Get morphological segmentation for a word.
        
        Args:
            word: The word to segment
            language: Language code
            dataset: Specific dataset to use, or None for any available
            
        Returns:
            List of morphemes or None if not found
        """
        word = word.lower()
        if language not in self.morphological_data:
            if language in MORPHOLEX_LANGCODE_MAPPING and MORPHOLEX_LANGCODE_MAPPING[language] in self.morphological_data:
                language = MORPHOLEX_LANGCODE_MAPPING[language]
            else:
                return None
        
        if dataset:
            if dataset in self.morphological_data[language]:
                return self.morphological_data[language][dataset].get(word)
        else:
            # Try all available datasets for this language
            for dataset_name, words in self.morphological_data[language].items():
                if word in words:
                    return words[word]
        return None


class MultilingualTokenizerComparator:
    def __init__(self, tokenizer_configs: Dict[str, Dict], 
             morphological_config: Optional[Dict[str, str]] = None,
             renyi_alphas: List[float] = [0.5, 1.0, 2.0, 3.0]):
        """
        Initialize with multiple tokenizers.
        
        Args:
            tokenizer_configs: Dictionary mapping tokenizer names to their configs
            morphological_config: Configuration for morphological datasets
            renyi_alphas: List of alpha values for Rényi entropy computation
        """
        self.tokenizers = {}
        self.tokenizer_names = []
        self.vocabularies = {}
        self.renyi_alphas = renyi_alphas
        
        # Initialize morphological data loader
        self.morphological_loader = MorphologicalDataLoader(morphological_config)
        if morphological_config:
            self.morphological_loader.load_all_datasets()
        
        # Load all tokenizers
        for name, config in tokenizer_configs.items():
            self.tokenizers[name] = load_tokenizer_from_config(config)
            self.tokenizer_names.append(name)
            self.vocabularies[name] = self._extract_vocab(self.tokenizers[name])
        
        print(f"Loaded {len(self.tokenizers)} byte-level multilingual tokenizers:")
        for name, vocab in self.vocabularies.items():
            print(f"  {name}: {len(vocab)} tokens")
        
        # For backward compatibility
        if len(self.tokenizers) == 2:
            names = list(self.tokenizers.keys())
            self.old_tok = self.tokenizers[names[0]]
            self.new_tok = self.tokenizers[names[1]]
            self.old_vocab = self.vocabularies[names[0]]
            self.new_vocab = self.vocabularies[names[1]]
    
    def _extract_vocab(self, tokenizer) -> Dict[str, int]:
        """Extract vocabulary from tokenizer as {token: id} mapping."""
        vocab = {}
        for i in range(tokenizer.get_vocab_size()):
            token = tokenizer.id_to_token(i)
            if token is not None:
                vocab[token] = i
        return vocab
    
    def _load_language_texts(self, language_files: Dict[str, str], num_samples_per_lang: int = 1000) -> Dict[str, List[str]]:
        """Load and sample texts from language-specific files."""
        language_texts = {}
        
        for lang, file_path in language_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_lines = [line.strip() for line in f if line.strip()]
                
                # Sample lines for this language
                import random
                random.seed(42)
                sampled_lines = random.sample(all_lines, min(num_samples_per_lang, len(all_lines)))
                language_texts[lang] = sampled_lines
                
                print(f"Loaded {len(sampled_lines)} samples for {lang}")
                
            except FileNotFoundError:
                print(f"Warning: Could not find file for {lang}: {file_path}")
                language_texts[lang] = []
        
        return language_texts        

    
    def vocabulary_overlap_analysis(self) -> Dict:
        """Analyze vocabulary overlap between old and new tokenizers."""
        old_tokens = set(self.old_vocab.keys())
        new_tokens = set(self.new_vocab.keys())
        
        intersection = old_tokens & new_tokens
        old_only = old_tokens - new_tokens
        new_only = new_tokens - old_tokens
        
        # Calculate overlap statistics
        jaccard_similarity = len(intersection) / len(old_tokens | new_tokens)
        overlap_percentage = len(intersection) / max(len(old_tokens), len(new_tokens))
        
        results = {
            'total_old': len(old_tokens),
            'total_new': len(new_tokens),
            'intersection': len(intersection),
            'old_only': len(old_only),
            'new_only': len(new_only),
            'jaccard_similarity': jaccard_similarity,
            'overlap_percentage': overlap_percentage,
            'old_only_tokens': sorted(list(old_only))[:100],  # Sample for inspection
            'new_only_tokens': sorted(list(new_only))[:100],  # Sample for inspection
        }
        
        print(f"\n=== Vocabulary Overlap Analysis ===")
        print(f"Intersection: {results['intersection']} tokens ({overlap_percentage:.1%})")
        print(f"Old only: {results['old_only']} tokens")
        print(f"New only: {results['new_only']} tokens") 
        print(f"Jaccard similarity: {jaccard_similarity:.3f}")
        
        return results
    
    def token_length_analysis(self) -> Dict:
        """Analyze token length distributions for byte-level tokens."""
        old_lengths = [len(token) for token in self.old_vocab.keys()]
        new_lengths = [len(token) for token in self.new_vocab.keys()]
        
        results = {
            'old_stats': {
                'mean': np.mean(old_lengths),
                'median': np.median(old_lengths),
                'std': np.std(old_lengths),
                'min': np.min(old_lengths),
                'max': np.max(old_lengths)
            },
            'new_stats': {
                'mean': np.mean(new_lengths),
                'median': np.median(new_lengths),
                'std': np.std(new_lengths),
                'min': np.min(new_lengths),
                'max': np.max(new_lengths)
            },
            'old_lengths': old_lengths,
            'new_lengths': new_lengths
        }
        
        print(f"\n=== Token Length Analysis (Byte-Level) ===")
        print(f"Old tokenizer - Mean: {results['old_stats']['mean']:.2f}, "
              f"Median: {results['old_stats']['median']:.1f}, "
              f"Std: {results['old_stats']['std']:.2f}")
        print(f"New tokenizer - Mean: {results['new_stats']['mean']:.2f}, "
              f"Median: {results['new_stats']['median']:.1f}, "
              f"Std: {results['new_stats']['std']:.2f}")
        
        return results
    
    def segmentation_comparison_per_language(self, language_texts: Dict[str, List[str]],
        old_encodings: Dict[str, List[List[int]]], new_encodings: Dict[str, List[List[int]]]) -> Dict:
        """Compare segmentation per language."""
        per_language_results = {}
        global_results = {
            'all_old_lengths': [],
            'all_new_lengths': [],
            'total_old_tokens': 0,
            'total_new_tokens': 0
        }
        
        print(f"\n=== Per-Language Segmentation Comparison ===")
        
        for lang, texts in language_texts.items():
            if not texts:
                continue
                
            lang_results = {
                'texts': texts,
                'old_lengths': [],
                'new_lengths': [],
                'differences': []
            }
            
            lang_old_tokens = 0
            lang_new_tokens = 0
            old_encodings_per_lang = old_encodings[lang]
            new_encodings_per_lang = new_encodings[lang]

            for text, old_tokens, new_tokens in zip(texts, old_encodings_per_lang, new_encodings_per_lang):
            
                # old_tokens = self.old_tok.encode(text).tokens
                # new_tokens = self.new_tok.encode(text).tokens
                
                lang_results['old_lengths'].append(len(old_tokens))
                lang_results['new_lengths'].append(len(new_tokens))
                lang_results['differences'].append(len(old_tokens) - len(new_tokens))
                
                lang_old_tokens += len(old_tokens)
                lang_new_tokens += len(new_tokens)
                
                # Add to global stats
                global_results['all_old_lengths'].append(len(old_tokens))
                global_results['all_new_lengths'].append(len(new_tokens))
            
            # Calculate language-specific statistics
            avg_old_length = np.mean(lang_results['old_lengths'])
            avg_new_length = np.mean(lang_results['new_lengths'])
            compression_ratio = avg_new_length / avg_old_length if avg_old_length > 0 else 1.0
            
            lang_results.update({
                'avg_old_length': avg_old_length,
                'avg_new_length': avg_new_length,
                'compression_ratio': compression_ratio,
                'total_old_tokens': lang_old_tokens,
                'total_new_tokens': lang_new_tokens
            })
            
            per_language_results[lang] = lang_results
            global_results['total_old_tokens'] += lang_old_tokens
            global_results['total_new_tokens'] += lang_new_tokens
            
            print(f"{lang}: avg tokens Old={avg_old_length:.1f}, New={avg_new_length:.1f}, "
                  f"compression={compression_ratio:.3f}")
        
        # Calculate global statistics
        if global_results['all_old_lengths']:
            global_avg_old = np.mean(global_results['all_old_lengths'])
            global_avg_new = np.mean(global_results['all_new_lengths'])
            global_compression = global_avg_new / global_avg_old if global_avg_old > 0 else 1.0
            
            global_results.update({
                'avg_old_length': global_avg_old,
                'avg_new_length': global_avg_new,
                'compression_ratio': global_compression
            })
        
        results = {
            'per_language': per_language_results,
            'global': global_results
        }
        
        print(f"Global: avg tokens Old={global_results.get('avg_old_length', 0):.1f}, "
              f"New={global_results.get('avg_new_length', 0):.1f}, "
              f"compression={global_results.get('compression_ratio', 1):.3f}")
        
        return results
    
    def fertility_analysis_per_language(self, language_texts: Dict[str, List[str]], 
        old_encodings: Dict[str, List[List[int]]], new_encodings: Dict[str, List[List[int]]]) -> Dict:
        """Analyze fertility (tokens per character) per language."""
        per_language_results = {}
        global_fertilities = {'old': [], 'new': []}
        
        print(f"\n=== Per-Language Fertility Analysis ===")
        
        for lang, texts in language_texts.items():
            if not texts:
                continue
                
            old_fertilities = []
            new_fertilities = []
            old_encodings_per_lang = old_encodings[lang]
            new_encodings_per_lang = new_encodings[lang]

            for text, old_tokens, new_tokens in zip(texts, old_encodings_per_lang, new_encodings_per_lang):
                if not text.strip():
                    continue
                    
                char_count = len(text)
                old_tokens = len(old_tokens)#self.old_tok.encode(text).tokens)
                new_tokens = len(new_tokens) #self.new_tok.encode(text).tokens)
                
                old_fertility = old_tokens / char_count if char_count > 0 else 0
                new_fertility = new_tokens / char_count if char_count > 0 else 0
                
                old_fertilities.append(old_fertility)
                new_fertilities.append(new_fertility)
                
                global_fertilities['old'].append(old_fertility)
                global_fertilities['new'].append(new_fertility)
            
            if old_fertilities:
                lang_results = {
                    'old_fertility_mean': np.mean(old_fertilities),
                    'old_fertility_std': np.std(old_fertilities),
                    'new_fertility_mean': np.mean(new_fertilities),
                    'new_fertility_std': np.std(new_fertilities),
                    'fertility_change': np.mean(new_fertilities) - np.mean(old_fertilities),
                    'old_fertilities': old_fertilities,
                    'new_fertilities': new_fertilities
                }
                
                per_language_results[lang] = lang_results
                
                print(f"{lang}: Old={lang_results['old_fertility_mean']:.4f}±{lang_results['old_fertility_std']:.4f}, "
                      f"New={lang_results['new_fertility_mean']:.4f}±{lang_results['new_fertility_std']:.4f}, "
                      f"change={lang_results['fertility_change']:+.4f}")
        
        # Global statistics
        global_results = {}
        if global_fertilities['old']:
            global_results = {
                'old_fertility_mean': np.mean(global_fertilities['old']),
                'old_fertility_std': np.std(global_fertilities['old']),
                'new_fertility_mean': np.mean(global_fertilities['new']),
                'new_fertility_std': np.std(global_fertilities['new']),
                'fertility_change': np.mean(global_fertilities['new']) - np.mean(global_fertilities['old']),
                'old_fertilities': global_fertilities['old'],
                'new_fertilities': global_fertilities['new']
            }
            
            print(f"Global: Old={global_results['old_fertility_mean']:.4f}±{global_results['old_fertility_std']:.4f}, "
                  f"New={global_results['new_fertility_mean']:.4f}±{global_results['new_fertility_std']:.4f}, "
                  f"change={global_results['fertility_change']:+.4f}")
        
        return {
            'per_language': per_language_results,
            'global': global_results
        }
    
    def multilingual_analysis(self) -> Dict:
        """Analyze multilingual properties of the vocabularies using Unicode categories and scripts."""
        
        def analyze_vocab_unicode(vocab):
            unicode_categories = defaultdict(int)
            script_distribution = defaultdict(int)
            byte_patterns = defaultdict(int)
            special_tokens = set()
            
            for token in vocab:
                # Count Unicode categories for each character in token
                for char in token:
                    category = unicodedata.category(char)
                    unicode_categories[category] += 1
                    
                    # Detect script (simplified)
                    try:
                        script = unicodedata.name(char).split()[0] if unicodedata.name(char) else "UNKNOWN"
                        script_distribution[script] += 1
                    except ValueError:
                        script_distribution["CONTROL"] += 1
                
                # Analyze byte-level patterns
                if token.startswith('Ġ'):  # GPT-style space marker
                    byte_patterns['space_marker'] += 1
                elif token.startswith('▁'):  # SentencePiece style space marker  
                    byte_patterns['sentencepiece_space'] += 1
                elif len(token) == 1 and ord(token) < 128:
                    byte_patterns['ascii_single'] += 1
                elif len(token) == 1 and ord(token) >= 128:
                    byte_patterns['non_ascii_single'] += 1
                elif all(ord(c) < 128 for c in token):
                    byte_patterns['ascii_multi'] += 1
                else:
                    byte_patterns['mixed_multi'] += 1
                
                # Detect special tokens (heuristic)
                if token.startswith('<') and token.endswith('>'):
                    special_tokens.add(token)
                elif token in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']:
                    special_tokens.add(token)
            
            return {
                'unicode_categories': dict(unicode_categories),
                'script_distribution': dict(script_distribution), 
                'byte_patterns': dict(byte_patterns),
                'special_tokens': special_tokens,
                'total_tokens': len(vocab)
            }
        
        old_analysis = analyze_vocab_unicode(self.old_vocab.keys())
        new_analysis = analyze_vocab_unicode(self.new_vocab.keys())
        
        # Compare distributions
        results = {
            'old_analysis': old_analysis,
            'new_analysis': new_analysis,
            'unicode_category_diff': {},
            'script_diff': {},
            'byte_pattern_diff': {}
        }
        
        # Calculate differences in distributions
        all_categories = set(old_analysis['unicode_categories'].keys()) | set(new_analysis['unicode_categories'].keys())
        for cat in all_categories:
            old_count = old_analysis['unicode_categories'].get(cat, 0)
            new_count = new_analysis['unicode_categories'].get(cat, 0)
            old_ratio = old_count / old_analysis['total_tokens']
            new_ratio = new_count / new_analysis['total_tokens']
            results['unicode_category_diff'][cat] = new_ratio - old_ratio
        
        all_patterns = set(old_analysis['byte_patterns'].keys()) | set(new_analysis['byte_patterns'].keys())
        for pattern in all_patterns:
            old_count = old_analysis['byte_patterns'].get(pattern, 0)
            new_count = new_analysis['byte_patterns'].get(pattern, 0)
            old_ratio = old_count / old_analysis['total_tokens']
            new_ratio = new_count / new_analysis['total_tokens']
            results['byte_pattern_diff'][pattern] = new_ratio - old_ratio
        
        print(f"\n=== Multilingual Analysis ===")
        print(f"Special tokens: Old={len(old_analysis['special_tokens'])}, "
              f"New={len(new_analysis['special_tokens'])}")
        print(f"Unicode categories: Old={len(old_analysis['unicode_categories'])}, "
              f"New={len(new_analysis['unicode_categories'])}")
        print(f"Byte patterns:")
        for pattern in ['ascii_single', 'non_ascii_single', 'ascii_multi', 'mixed_multi']:
            old_count = old_analysis['byte_patterns'].get(pattern, 0)
            new_count = new_analysis['byte_patterns'].get(pattern, 0)
            print(f"  {pattern}: Old={old_count}, New={new_count}")
        
        return results
    
    def script_specific_analysis(self, language_texts: Dict[str, List[str]],
        old_encodings: Dict[str, List[List[int]]], new_encodings: Dict[str, List[List[int]]]) -> Dict:
        """Analyze performance on different writing systems, now using per-language data."""
        
        def detect_primary_script(text: str) -> str:
            """Detect the primary script in a text."""
            script_counts = defaultdict(int)
            
            for char in text:
                if char.isspace() or char in string.punctuation:
                    continue
                try:
                    # Simplified script detection
                    if '\u0600' <= char <= '\u06FF':  # Arabic
                        script_counts['Arabic'] += 1
                    elif '\u4e00' <= char <= '\u9fff':  # Chinese
                        script_counts['Chinese'] += 1
                    elif '\u0400' <= char <= '\u04FF':  # Cyrillic
                        script_counts['Cyrillic'] += 1
                    elif '\u0370' <= char <= '\u03FF':  # Greek
                        script_counts['Greek'] += 1
                    elif '\u0900' <= char <= '\u097F':  # Devanagari (Hindi)
                        script_counts['Devanagari'] += 1
                    elif 'a' <= char.lower() <= 'z':  # Latin
                        script_counts['Latin'] += 1
                    else:
                        script_counts['Other'] += 1
                except:
                    script_counts['Unknown'] += 1
            
            if not script_counts:
                return 'Empty'
            return max(script_counts.items(), key=lambda x: x[1])[0]
        
        # Combine script detection with language information
        script_language_results = {}
        
        print(f"\n=== Script-Specific Analysis by Language ===")
        
        for lang, texts in language_texts.items():
            if not texts:
                continue
                
            lang_script_results = defaultdict(lambda: {'old_tokens': [], 'new_tokens': [], 'char_counts': []})
            
            old_encodings_per_lang = old_encodings[lang]
            new_encodings_per_lang = new_encodings[lang]

            for text, old_tokens, new_tokens in zip(texts, old_encodings_per_lang, new_encodings_per_lang):
                if not text.strip():
                    continue
                    
                script = detect_primary_script(text)
                char_count = len(text)
                
                old_tokens = len(old_tokens)#self.old_tok.encode(text).tokens)
                new_tokens = len(new_tokens) #self.new_tok.encode(text).tokens)
                
                lang_script_results[script]['old_tokens'].append(old_tokens)
                lang_script_results[script]['new_tokens'].append(new_tokens)
                lang_script_results[script]['char_counts'].append(char_count)
            
            # Calculate statistics for this language
            lang_final_results = {}
            for script, data in lang_script_results.items():
                if not data['old_tokens']:
                    continue
                    
                old_fertility = np.mean([t/c for t, c in zip(data['old_tokens'], data['char_counts']) if c > 0])
                new_fertility = np.mean([t/c for t, c in zip(data['new_tokens'], data['char_counts']) if c > 0])
                
                lang_final_results[script] = {
                    'sample_count': len(data['old_tokens']),
                    'old_avg_fertility': old_fertility,
                    'new_avg_fertility': new_fertility,
                    'fertility_change': new_fertility - old_fertility,
                    'compression_ratio': new_fertility / old_fertility if old_fertility > 0 else 1.0
                }
            
            script_language_results[lang] = lang_final_results
            
            # Print language results
            print(f"{lang}:")
            for script, stats in lang_final_results.items():
                print(f"  {script}: {stats['sample_count']} samples, "
                      f"fertility change: {stats['fertility_change']:+.4f}, "
                      f"compression: {stats['compression_ratio']:.3f}")
        
        return script_language_results
    
    def consistency_analysis_per_language(self, language_texts: Dict[str, List[str]]) -> Dict:
        """Analyze segmentation consistency per language."""
        per_language_results = {}
        global_stats = {'total_substrings': 0, 'total_differences': 0}
        
        print(f"\n=== Per-Language Consistency Analysis ===")
        
        for lang, texts in language_texts.items():
            if not texts:
                continue
                
            old_segmentations = {}
            new_segmentations = {}
            
            # For byte-level, we'll look at character-level substrings
            substrings = []
            for text in texts:
                # Extract meaningful substrings (3+ characters)
                for i in range(len(text)):
                    for j in range(i + 3, min(i + 15, len(text) + 1)):  # Limit length
                        substring = text[i:j]
                        if len(substring.strip()) >= 3:  # Only meaningful substrings
                            substrings.append(substring)
            
            # Limit analysis to reasonable number of substrings per language
            if len(substrings) > 1000:
                import random
                random.seed(42)
                substrings = random.sample(substrings, 1000)

            old_tokens_list = encode_text(self.old_tok, substrings)["tokens"]
            new_tokens_list = encode_text(self.new_tok, substrings)["tokens"]
            
            for substring, old_tokens, new_tokens in zip(substrings, old_tokens_list, new_tokens_list):
                old_segmentations[substring] = old_tokens
                new_segmentations[substring] = new_tokens
            
            # Calculate consistency metrics for this language
            differences = []
            for substring in substrings:
                if old_segmentations[substring] != new_segmentations[substring]:
                    differences.append({
                        'substring': substring,
                        'old': old_segmentations[substring],
                        'new': new_segmentations[substring],
                        'old_length': len(old_segmentations[substring]),
                        'new_length': len(new_segmentations[substring])
                    })
            
            lang_results = {
                'total_substrings': len(substrings),
                'differences_count': len(differences),
                'differences_sample': differences[:10],  # Sample for inspection
                'agreement_rate': 1 - (len(differences) / len(substrings)) if substrings else 1.0
            }
            
            per_language_results[lang] = lang_results
            
            # Update global stats
            global_stats['total_substrings'] += len(substrings)
            global_stats['total_differences'] += len(differences)
            
            print(f"{lang}: {len(substrings)} substrings, "
                  f"{len(differences)} differences ({len(differences)/len(substrings):.1%}), "
                  f"agreement: {lang_results['agreement_rate']:.1%}")
        
        # Calculate global agreement rate
        global_agreement = 1 - (global_stats['total_differences'] / global_stats['total_substrings']) if global_stats['total_substrings'] > 0 else 1.0
        global_stats['agreement_rate'] = global_agreement
        
        print(f"Global: {global_stats['total_substrings']} substrings, "
              f"{global_stats['total_differences']} differences, "
              f"agreement: {global_agreement:.1%}")
        
        return {
            'per_language': per_language_results,
            'global': global_stats
        }
    
    def plot_comparisons(self, results: Dict, save_dir: str = "tokenizer_comparison_plots"):
        """Create visualization plots for the comparison results."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. Token length distributions
        if 'token_length' in results:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(results['token_length']['old_lengths'], bins=30, alpha=0.7, label='Old', density=True)
            plt.hist(results['token_length']['new_lengths'], bins=30, alpha=0.7, label='New', density=True)
            plt.xlabel('Token Length (bytes)')
            plt.ylabel('Density')
            plt.title('Byte-Level Token Length Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            old_lengths = np.array(results['token_length']['old_lengths'])
            new_lengths = np.array(results['token_length']['new_lengths'])
            plt.boxplot([old_lengths, new_lengths], labels=['Old', 'New'])
            plt.ylabel('Token Length (bytes)')
            plt.title('Token Length Box Plot')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/token_length_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Per-language segmentation comparison
        if 'segmentation' in results and 'per_language' in results['segmentation']:
            languages = list(results['segmentation']['per_language'].keys())
            old_avgs = [results['segmentation']['per_language'][lang]['avg_old_length'] for lang in languages]
            new_avgs = [results['segmentation']['per_language'][lang]['avg_new_length'] for lang in languages]
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(languages))
            width = 0.35
            
            plt.bar(x - width/2, old_avgs, width, label='Old', alpha=0.7)
            plt.bar(x + width/2, new_avgs, width, label='New', alpha=0.7)
            
            plt.xlabel('Language')
            plt.ylabel('Average Tokens per Text')
            plt.title('Per-Language Segmentation Comparison')
            plt.xticks(x, languages, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f"{save_dir}/per_language_segmentation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Per-language fertility comparison
        if 'fertility' in results and 'per_language' in results['fertility']:
            languages = list(results['fertility']['per_language'].keys())
            old_fertilities = [results['fertility']['per_language'][lang]['old_fertility_mean'] for lang in languages]
            new_fertilities = [results['fertility']['per_language'][lang]['new_fertility_mean'] for lang in languages]
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(languages))
            width = 0.35
            
            plt.bar(x - width/2, old_fertilities, width, label='Old', alpha=0.7)
            plt.bar(x + width/2, new_fertilities, width, label='New', alpha=0.7)
            
            plt.xlabel('Language')
            plt.ylabel('Average Fertility (tokens/char)')
            plt.title('Per-Language Fertility Comparison')
            plt.xticks(x, languages, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f"{save_dir}/per_language_fertility.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Global fertility scatter plot
        if 'fertility' in results and 'global' in results['fertility']:
            plt.figure(figsize=(10, 6))
            
            old_fert = results['fertility']['global']['old_fertilities']
            new_fert = results['fertility']['global']['new_fertilities']
            
            plt.scatter(old_fert, new_fert, alpha=0.6)
            plt.plot([min(old_fert), max(old_fert)], [min(old_fert), max(old_fert)], 'r--', label='Equal')
            plt.xlabel('Old Tokenizer Fertility (tokens/char)')
            plt.ylabel('New Tokenizer Fertility (tokens/char)')
            plt.title('Global Fertility Comparison (Tokens per Character)')
            plt.legend()
            
            plt.savefig(f"{save_dir}/global_fertility_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Script-specific analysis by language
        if 'script_analysis' in results:
            # Create a comprehensive plot showing script performance across languages
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Collect all script-language combinations
            all_data = []
            for lang, scripts in results['script_analysis'].items():
                for script, stats in scripts.items():
                    all_data.append({
                        'language': lang,
                        'script': script,
                        'old_fertility': stats['old_avg_fertility'],
                        'new_fertility': stats['new_avg_fertility'],
                        'fertility_change': stats['fertility_change'],
                        'compression_ratio': stats['compression_ratio'],
                        'sample_count': stats['sample_count']
                    })
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Plot 1: Fertility change by language
                axes[0,0].scatter(df['old_fertility'], df['new_fertility'], 
                                s=df['sample_count']/10, alpha=0.6, c=df['compression_ratio'], cmap='RdYlBu')
                axes[0,0].plot([df['old_fertility'].min(), df['old_fertility'].max()], 
                              [df['old_fertility'].min(), df['old_fertility'].max()], 'r--', alpha=0.5)
                axes[0,0].set_xlabel('Old Fertility')
                axes[0,0].set_ylabel('New Fertility')
                axes[0,0].set_title('Fertility Change (color=compression, size=samples)')
                
                # Plot 2: Compression ratio by language
                lang_compression = df.groupby('language')['compression_ratio'].mean()
                axes[0,1].bar(range(len(lang_compression)), lang_compression.values)
                axes[0,1].set_xticks(range(len(lang_compression)))
                axes[0,1].set_xticklabels(lang_compression.index, rotation=45)
                axes[0,1].set_ylabel('Average Compression Ratio')
                axes[0,1].set_title('Compression by Language')
                axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                
                # Plot 3: Fertility change by script
                script_change = df.groupby('script')['fertility_change'].mean()
                axes[1,0].bar(range(len(script_change)), script_change.values)
                axes[1,0].set_xticks(range(len(script_change)))
                axes[1,0].set_xticklabels(script_change.index, rotation=45)
                axes[1,0].set_ylabel('Average Fertility Change')
                axes[1,0].set_title('Fertility Change by Script')
                axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Plot 4: Sample distribution
                sample_counts = df.groupby('language')['sample_count'].sum()
                axes[1,1].pie(sample_counts.values, labels=sample_counts.index, autopct='%1.1f%%')
                axes[1,1].set_title('Sample Distribution by Language')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/script_language_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Unicode category distribution
        if 'multilingual' in results:
            # Plot byte pattern differences
            byte_patterns = results['multilingual']['byte_pattern_diff']
            if byte_patterns:
                plt.figure(figsize=(10, 6))
                patterns = list(byte_patterns.keys())
                changes = list(byte_patterns.values())
                
                colors = ['red' if x < 0 else 'green' for x in changes]
                plt.bar(patterns, changes, color=colors, alpha=0.7)
                plt.xlabel('Byte Pattern Type')
                plt.ylabel('Change in Proportion (New - Old)')
                plt.title('Change in Byte Pattern Distribution')
                plt.xticks(rotation=45)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(f"{save_dir}/byte_pattern_changes.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"\nPlots saved to {save_dir}/")


    def type_token_ratio_analysis(self, language_texts: Dict[str, List[str]], 
                                  all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Analyze Type-Token Ratio (TTR) for each tokenizer.
        TTR = (number of unique tokens) / (total number of tokens)
        """
        results = {
            'per_tokenizer': {},
            'per_language': {name: {} for name in self.tokenizer_names}
        }
        
        print(f"\n=== Type-Token Ratio Analysis ===")
        
        for name in self.tokenizer_names:
            # Global token usage
            global_token_counts = Counter()
            
            # Per-language token usage
            per_lang_token_counts = {lang: Counter() for lang in language_texts}
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[name]:
                    continue
                    
                encodings = all_encodings[name][lang]
                
                for encoding in encodings:
                    global_token_counts.update(encoding)
                    per_lang_token_counts[lang].update(encoding)
            
            # Calculate global TTR
            total_tokens = sum(global_token_counts.values())
            unique_tokens = len(global_token_counts)
            global_ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
            
            results['per_tokenizer'][name] = {
                'global_ttr': global_ttr,
                'total_tokens': total_tokens,
                'unique_tokens': unique_tokens
            }
            
            # Calculate per-language TTR
            for lang in language_texts:
                lang_total = sum(per_lang_token_counts[lang].values())
                lang_unique = len(per_lang_token_counts[lang])
                lang_ttr = lang_unique / lang_total if lang_total > 0 else 0
                
                results['per_language'][name][lang] = {
                    'ttr': lang_ttr,
                    'total_tokens': lang_total,
                    'unique_tokens': lang_unique
                }
            
            print(f"{name}: Global TTR = {global_ttr:.4f} ({unique_tokens:,}/{total_tokens:,})")
        
        return results


    def vocabulary_utilization_analysis(self, language_texts: Dict[str, List[str]], 
                                        all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Analyze vocabulary utilization - how much of each tokenizer's vocabulary is used per language.
        """
        results = {name: {} for name in self.tokenizer_names}
        
        print(f"\n=== Vocabulary Utilization Analysis ===")
        
        for name in self.tokenizer_names:
            vocab_size = len(self.vocabularies[name])
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[name]:
                    continue
                    
                # Count unique tokens used for this language
                unique_tokens_used = set()
                encodings = all_encodings[name][lang]
                
                for encoding in encodings:
                    unique_tokens_used.update(encoding)
                
                utilization_count = len(unique_tokens_used)
                utilization_percent = (utilization_count / vocab_size) * 100 if vocab_size > 0 else 0
                
                results[name][lang] = {
                    'tokens_used': utilization_count,
                    'vocab_size': vocab_size,
                    'utilization_percent': utilization_percent
                }
            
            # Print summary
            lang_utils = [f"{lang}:{results[name][lang]['utilization_percent']:.1f}%" 
                          for lang in sorted(results[name].keys())]
            print(f"{name}: {', '.join(lang_utils)}")
        
        return results


    def average_tokens_per_line_analysis(self, language_texts: Dict[str, List[str]], 
                                         all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Analyze average number of tokens per line (text sample) for each tokenizer and language.
        """
        results = {name: {} for name in self.tokenizer_names}
        
        print(f"\n=== Average Tokens per Line Analysis ===")
        
        for name in self.tokenizer_names:
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[name]:
                    continue
                    
                encodings = all_encodings[name][lang]
                token_lengths = [len(encoding) for encoding in encodings]
                
                if token_lengths:
                    avg_length = np.mean(token_lengths)
                    std_length = np.std(token_lengths)
                    min_length = np.min(token_lengths)
                    max_length = np.max(token_lengths)
                    
                    results[name][lang] = {
                        'avg': avg_length,
                        'std': std_length,
                        'min': min_length,
                        'max': max_length,
                        'lengths': token_lengths
                    }
                else:
                    results[name][lang] = {
                        'avg': 0, 'std': 0, 'min': 0, 'max': 0, 'lengths': []
                    }
            
            # Print summary
            lang_avgs = [f"{lang}:{results[name][lang]['avg']:.1f}" 
                         for lang in sorted(results[name].keys())]
            print(f"{name}: {', '.join(lang_avgs)}")
        
        return results

    
        
    def _compute_morphological_alignment(self, word: str, tokenizer_tokens: List[str], 
                                   language: str) -> Optional[Dict[str, float]]:
        """
        Compute morphological alignment metrics for a single word.
        
        Args:
            word: Original word
            tokenizer_tokens: Tokens produced by tokenizer for this word
            language: Language code
            
        Returns:
            Dictionary with alignment metrics or None if no morphological data
        """
        morphemes = self.morphological_loader.get_morphemes(word, language)
        if not morphemes:
            return None
        
        # Clean and prepare tokens
        clean_tokens = []
        token_chars = []  # Track character spans for each token
        
        for token in tokenizer_tokens:
            # Remove common subword indicators but track their presence
            clean_token = token
            
            # Handle different tokenizer conventions
            if token.startswith('##'):  # BERT-style
                clean_token = token[2:]
            elif token.startswith('▁'):  # SentencePiece style
                clean_token = token[1:]
            elif token.startswith('Ġ'):  # GPT-style
                clean_token = token[1:]
            
            # Remove any remaining special characters at start/end
            clean_token = clean_token.strip('##▁Ġ ')
            
            if clean_token:
                clean_tokens.append(clean_token)
                token_chars.append(clean_token)
        
        # Verify we can reconstruct the word
        reconstructed = ''.join(clean_tokens).lower()
        
        # Try alternative reconstruction if direct concatenation doesn't work
        if reconstructed != word.lower():
            # Try with original tokens (some tokenizers include spaces)
            alt_reconstructed = ''.join(t.replace('▁', ' ').replace('Ġ', ' ').strip() 
                                       for t in tokenizer_tokens).strip().lower()
            if alt_reconstructed == word.lower():
                # Re-parse with space handling
                clean_tokens = []
                for token in tokenizer_tokens:
                    if '▁' in token:
                        clean_token = token.replace('▁', '')
                    elif 'Ġ' in token:
                        clean_token = token.replace('Ġ', '')
                    else:
                        clean_token = token.strip('##')
                    if clean_token:
                        clean_tokens.append(clean_token)
                reconstructed = ''.join(clean_tokens).lower()
        
        if reconstructed != word.lower():
            return None  # Cannot align properly
        
        # Build character position mappings
        morpheme_boundaries = set()
        pos = 0
        morpheme_spans = []
        for morpheme in morphemes:
            start = pos
            # Handle potential stem changes by fuzzy matching
            if pos < len(word):
                # Look for morpheme in remaining word
                remaining = word[pos:].lower()
                if remaining.startswith(morpheme.lower()):
                    end = start + len(morpheme)
                else:
                    # Fuzzy match - find best alignment
                    best_match_len = 0
                    for i in range(1, min(len(morpheme), len(remaining)) + 1):
                        if morpheme[:i].lower() == remaining[:i].lower():
                            best_match_len = i
                    end = start + (best_match_len if best_match_len > 0 else len(morpheme))
            else:
                end = start + len(morpheme)
            
            morpheme_spans.append((start, min(end, len(word))))
            pos = min(end, len(word))
            if pos < len(word):
                morpheme_boundaries.add(pos)
        
        token_boundaries = set()
        pos = 0
        token_spans = []
        for token in clean_tokens:
            start = pos
            end = start + len(token)
            token_spans.append((start, end))
            pos = end
            if pos < len(word):
                token_boundaries.add(pos)
        
        # Compute boundary metrics
        if len(token_boundaries) == 0 and len(morpheme_boundaries) == 0:
            precision = recall = f1 = 1.0  # Perfect match: no internal boundaries
        elif len(token_boundaries) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        elif len(morpheme_boundaries) == 0:
            precision = 0.0
            recall = 1.0  # No morpheme boundaries to miss
            f1 = 0.0
        else:
            # Allow fuzzy boundary matching (within 1 character)
            correct_boundaries = 0
            for tb in token_boundaries:
                for mb in morpheme_boundaries:
                    if abs(tb - mb) <= 1:  # Fuzzy match
                        correct_boundaries += 1
                        break
            
            precision = correct_boundaries / len(token_boundaries)
            recall = correct_boundaries / len(morpheme_boundaries)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Morpheme preservation - check if morphemes appear as complete tokens
        preserved_morphemes = 0
        for morpheme in morphemes:
            morpheme_lower = morpheme.lower()
            # Check exact match
            if morpheme_lower in [t.lower() for t in clean_tokens]:
                preserved_morphemes += 1
            else:
                # Check if morpheme appears as a substring that aligns with token boundaries
                for token in clean_tokens:
                    if morpheme_lower == token.lower():
                        preserved_morphemes += 1
                        break
        
        morpheme_preservation = preserved_morphemes / len(morphemes) if morphemes else 0.0
        
        # Over/under-segmentation
        over_segmentation = max(0, len(clean_tokens) - len(morphemes)) / len(morphemes)
        under_segmentation = max(0, len(morphemes) - len(clean_tokens)) / len(morphemes)
        
        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1,
            'morpheme_preservation': morpheme_preservation,
            'over_segmentation': over_segmentation,
            'under_segmentation': under_segmentation
        }


    def _compute_renyi_entropy(self, token_counts: Counter, alpha: float) -> float:
        """
        Compute Rényi entropy of order alpha for token distribution.
        
        Args:
            token_counts: Counter of token occurrences
            alpha: Order of Rényi entropy
            
        Returns:
            Rényi entropy value
        """
        if len(token_counts) == 0:
            return 0.0
        
        total_count = sum(token_counts.values())
        probabilities = np.array([count / total_count for count in token_counts.values()])
        
        if alpha == 1.0:
            # Shannon entropy (limit case)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            # General Rényi entropy
            power_sum = np.sum(probabilities ** alpha)
            if power_sum <= 0:
                return 0.0
            return (1 / (1 - alpha)) * np.log2(power_sum)


    def morphological_alignment_analysis(self, language_texts: Dict[str, List[str]], 
                                       all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Analyze morphological alignment for all tokenizers.
        """
        results = {name: {
            'boundary_precision': {},
            'boundary_recall': {},
            'boundary_f1': {},
            'morpheme_preservation': {},
            'over_segmentation': {},
            'under_segmentation': {}
        } for name in self.tokenizer_names}
        
        print(f"\n=== Morphological Alignment Analysis ===")
        
        if not self.morphological_loader.morphological_data:
            print("No morphological data loaded. Skipping morphological analysis.")
            return results
        
        for name in self.tokenizer_names:
            total_words_analyzed = 0
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[name]:
                    continue
                
                print(f"Encoding morphological data for {lang} with {name}...")
                # Collect metrics for this language
                metrics_per_word = defaultdict(list)

                words = [t for text in texts[:500] for t in text.split()]
                words_clean = [re.sub(r'[^\w]', '', word.lower()) for word in words]
                word_encodings = encode_text(self.tokenizers[name], words_clean)["tokens"]
                assert len(words_clean) == len(word_encodings)
                
                for word_clean, word_encoding in zip(words_clean, word_encodings):
                    # if len(word_clean) < 3:  # Skip very short words
                    #     continue
                                                            
                    # Compute alignment metrics
                    alignment = self._compute_morphological_alignment(
                        word_clean, word_encoding, lang
                    )
                    
                    if alignment:
                        total_words_analyzed += 1
                        for metric, value in alignment.items():
                            metrics_per_word[metric].append(value)
                
                # Aggregate metrics for this language
                for metric in ['boundary_precision', 'boundary_recall', 'boundary_f1',
                              'morpheme_preservation', 'over_segmentation', 'under_segmentation']:
                    if metrics_per_word[metric]:
                        results[name][metric][lang] = {
                            'mean': np.mean(metrics_per_word[metric]),
                            'std': np.std(metrics_per_word[metric]),
                            'count': len(metrics_per_word[metric])
                        }
                    else:
                        results[name][metric][lang] = {
                            'mean': 0.0, 'std': 0.0, 'count': 0
                        }
            
            # Print summary for this tokenizer
            if total_words_analyzed > 0:
                avg_f1 = np.mean([results[name]['boundary_f1'][lang]['mean'] 
                                for lang in results[name]['boundary_f1'] 
                                if results[name]['boundary_f1'][lang]['count'] > 0])
                print(f"{name}: Analyzed {total_words_analyzed} words, avg boundary F1 = {avg_f1:.3f}")
        
        return results


    def renyi_efficiency_analysis(self, all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Compute Rényi efficiency metrics for all tokenizers.
        """
        results = {name: {} for name in self.tokenizer_names}
        
        print(f"\n=== Rényi Efficiency Analysis ===")
        
        for name in self.tokenizer_names:
            # Collect global token counts
            global_token_counts = Counter()
            per_lang_token_counts = defaultdict(Counter)
            
            for lang, encodings in all_encodings[name].items():
                for encoding in encodings:
                    global_token_counts.update(encoding)
                    per_lang_token_counts[lang].update(encoding)
            
            # Compute Rényi entropies for different alpha values
            for alpha in self.renyi_alphas:
                results[name][f'renyi_{alpha}'] = {}
                
                # Global entropy
                global_entropy = self._compute_renyi_entropy(global_token_counts, alpha)
                results[name][f'renyi_{alpha}']['overall'] = global_entropy
                
                # Per-language entropy
                for lang in per_lang_token_counts:
                    lang_entropy = self._compute_renyi_entropy(per_lang_token_counts[lang], alpha)
                    results[name][f'renyi_{alpha}'][lang] = lang_entropy
            
            # Print summary
            shannon_entropy = results[name]['renyi_1.0']['overall']
            print(f"{name}: Shannon entropy (α=1.0) = {shannon_entropy:.3f} bits")
        
        return results


    def plot_morphological_metrics(self, results: Dict, save_dir: str):
        """Generate plots for morphological alignment metrics."""
        if not any(any(results[name]['boundary_f1'].values()) for name in self.tokenizer_names):
            logger.info("No morphological data available for plotting")
            return
        
        metrics_to_plot = ['boundary_f1', 'morpheme_preservation', 'over_segmentation']
        
        for metric in metrics_to_plot:
            # Get languages that have data
            languages_with_data = set()
            for name in self.tokenizer_names:
                languages_with_data.update(
                    lang for lang in results[name][metric] 
                    if results[name][metric][lang]['count'] > 0
                )
            
            if not languages_with_data:
                continue
            
            languages = sorted(languages_with_data)
            
            plt.figure(figsize=(max(10, len(self.tokenizer_names) * len(languages) * 0.5), 6))
            
            x = np.arange(len(languages))
            width = 0.8 / len(self.tokenizer_names)
            
            for i, t_name in enumerate(self.tokenizer_names):
                values = []
                errors = []
                for lang in languages:
                    result = results[t_name][metric].get(lang, {'mean': 0, 'std': 0})
                    values.append(result['mean'])
                    errors.append(result['std'])
                
                plt.bar(x + i * width - (len(self.tokenizer_names)-1)*width/2, 
                       values, width, label=t_name, yerr=errors, capsize=3, alpha=0.8)
            
            plt.xlabel('Language')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'Morphological Alignment: {metric.replace("_", " ").title()}')
            plt.xticks(x, languages)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/morphological_{metric}.png", dpi=300, bbox_inches='tight')
            plt.close()


    def plot_renyi_efficiency(self, results: Dict, save_dir: str):
        """Generate plots for Rényi efficiency metrics."""
        # Plot 1: Rényi entropy vs alpha for each tokenizer
        plt.figure(figsize=(10, 6))
        
        for t_name in self.tokenizer_names:
            entropies = []
            for alpha in self.renyi_alphas:
                entropy = results[t_name][f'renyi_{alpha}']['overall']
                entropies.append(entropy)
            plt.plot(self.renyi_alphas, entropies, marker='o', linewidth=2, markersize=8, label=t_name)
        
        plt.xlabel('Rényi Alpha (α)', fontsize=12)
        plt.ylabel('Rényi Entropy (bits)', fontsize=12)
        plt.title('Rényi Efficiency: Entropy vs Alpha Parameter', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/renyi_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Per-language Rényi entropy heatmap (for α=2.0)
        if 2.0 in self.renyi_alphas:
            # Get languages that have data
            languages = set()
            for name in self.tokenizer_names:
                languages.update(k for k in results[name]['renyi_2.0'].keys() if k != 'overall')
            languages = sorted(languages)
            
            if languages:
                entropy_matrix = np.zeros((len(self.tokenizer_names), len(languages)))
                
                for i, name in enumerate(self.tokenizer_names):
                    for j, lang in enumerate(languages):
                        entropy_matrix[i, j] = results[name]['renyi_2.0'].get(lang, 0)
                
                plt.figure(figsize=(max(8, len(languages) * 0.8), max(6, len(self.tokenizer_names) * 0.8)))
                im = plt.imshow(entropy_matrix, aspect='auto', cmap='viridis')
                plt.colorbar(im, label='Rényi Entropy (α=2.0)')
                
                # Add text annotations
                for i in range(len(self.tokenizer_names)):
                    for j in range(len(languages)):
                        plt.text(j, i, f'{entropy_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='white', fontsize=10)
                
                plt.xticks(range(len(languages)), languages, rotation=45)
                plt.yticks(range(len(self.tokenizer_names)), self.tokenizer_names)
                plt.title('Rényi Entropy by Tokenizer and Language (α=2.0)')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/renyi_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()


    def whitespace_fertility_analysis(self, language_texts: Dict[str, List[str]], 
                                      all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Dict:
        """
        Analyze fertility as tokens per whitespace word (different from tokens per character).
        """
        results = {
            'per_tokenizer': {name: {} for name in self.tokenizer_names},
            'global': {name: {} for name in self.tokenizer_names}
        }
        
        print(f"\n=== Whitespace-based Fertility Analysis ===")
        
        for name in self.tokenizer_names:
            total_tokens_global = 0
            total_words_global = 0
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[name]:
                    continue
                    
                total_tokens = 0
                total_words = 0
                
                encodings = all_encodings[name][lang]
                
                for text, encoding in zip(texts, encodings):
                    word_count = len(text.split())
                    token_count = len(encoding)
                    
                    total_words += word_count
                    total_tokens += token_count
                
                # Calculate per-language fertility
                fertility = total_tokens / total_words if total_words > 0 else 0
                results['per_tokenizer'][name][lang] = {
                    'fertility': fertility,
                    'total_tokens': total_tokens,
                    'total_words': total_words
                }
                
                total_tokens_global += total_tokens
                total_words_global += total_words
            
            # Calculate global fertility
            global_fertility = total_tokens_global / total_words_global if total_words_global > 0 else 0
            results['global'][name] = {
                'fertility': global_fertility,
                'total_tokens': total_tokens_global,
                'total_words': total_words_global
            }
            
            # Print summary
            lang_ferts = [f"{lang}:{results['per_tokenizer'][name][lang]['fertility']:.2f}" 
                          for lang in sorted(results['per_tokenizer'][name].keys())]
            print(f"{name}: Global={global_fertility:.2f}, {', '.join(lang_ferts)}")
        
        return results


    def plot_multi_tokenizer_comparisons(self, results: Dict, save_dir: str = "tokenizer_comparison_plots"):
        """Enhanced plotting for multiple tokenizers."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. Type-Token Ratio comparison
        if 'ttr' in results:
            plt.figure(figsize=(max(8, len(self.tokenizer_names) * 1.5), 6))
            
            # Global TTR
            names = list(results['ttr']['per_tokenizer'].keys())
            ttrs = [results['ttr']['per_tokenizer'][name]['global_ttr'] * 100 for name in names]
            
            x = np.arange(len(names))
            plt.bar(x, ttrs)
            plt.xlabel('Tokenizer')
            plt.ylabel('Type-Token Ratio (%)')
            plt.title('Global Type-Token Ratio Comparison')
            plt.xticks(x, names, rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(ttrs):
                plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/ttr_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Per-language TTR heatmap
            if len(self.tokenizer_names) > 1:
                languages = sorted(next(iter(results['ttr']['per_language'].values())).keys())
                ttr_matrix = np.zeros((len(self.tokenizer_names), len(languages)))
                
                for i, name in enumerate(self.tokenizer_names):
                    for j, lang in enumerate(languages):
                        ttr_matrix[i, j] = results['ttr']['per_language'][name][lang]['ttr']
                
                plt.figure(figsize=(max(8, len(languages) * 0.8), max(6, len(self.tokenizer_names) * 0.8)))
                plt.imshow(ttr_matrix, aspect='auto', cmap='viridis')
                plt.colorbar(label='Type-Token Ratio')
                plt.xticks(range(len(languages)), languages, rotation=45)
                plt.yticks(range(len(self.tokenizer_names)), self.tokenizer_names)
                plt.title('Type-Token Ratio by Tokenizer and Language')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/ttr_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Vocabulary Utilization
        if 'vocab_utilization' in results:
            languages = sorted(next(iter(results['vocab_utilization'].values())).keys())
            
            plt.figure(figsize=(max(10, len(languages) * 1.2), 6))
            
            x = np.arange(len(languages))
            width = 0.8 / len(self.tokenizer_names)
            
            for i, name in enumerate(self.tokenizer_names):
                utils = [results['vocab_utilization'][name][lang]['utilization_percent'] 
                        for lang in languages]
                plt.bar(x + i * width - (len(self.tokenizer_names)-1)*width/2, 
                       utils, width, label=name, alpha=0.8)
            
            plt.xlabel('Language')
            plt.ylabel('Vocabulary Utilization (%)')
            plt.title('Vocabulary Utilization by Language')
            plt.xticks(x, languages, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/vocab_utilization.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Average tokens per line comparison
        if 'avg_tokens_per_line' in results:
            languages = sorted(next(iter(results['avg_tokens_per_line'].values())).keys())
            
            # Box plot style
            fig, ax = plt.subplots(figsize=(max(10, len(languages) * 1.2), 6))
            
            for lang_idx, lang in enumerate(languages):
                data_for_lang = []
                positions = []
                labels = []
                
                for tok_idx, name in enumerate(self.tokenizer_names):
                    if lang in results['avg_tokens_per_line'][name]:
                        lengths = results['avg_tokens_per_line'][name][lang]['lengths']
                        if lengths:
                            data_for_lang.append(lengths)
                            positions.append(lang_idx * (len(self.tokenizer_names) + 1) + tok_idx)
                            labels.append(f"{lang}\n{name}")
                
                if data_for_lang:
                    bp = ax.boxplot(data_for_lang, positions=positions, widths=0.6, patch_artist=True)
                    
                    # Color by tokenizer
                    colors = plt.cm.Set3(np.linspace(0, 1, len(self.tokenizer_names)))
                    for patch, tok_idx in zip(bp['boxes'], range(len(data_for_lang))):
                        patch.set_facecolor(colors[tok_idx % len(self.tokenizer_names)])
            
            ax.set_xlabel('Language / Tokenizer')
            ax.set_ylabel('Tokens per Line')
            ax.set_title('Token Length Distribution by Language and Tokenizer')
            
            # Create custom legend
            from matplotlib.patches import Rectangle
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.tokenizer_names)))
            legend_elements = [Rectangle((0,0),1,1, facecolor=colors[i], label=name) 
                              for i, name in enumerate(self.tokenizer_names)]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/avg_tokens_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Whitespace fertility comparison
        if 'whitespace_fertility' in results:
            # Global comparison
            plt.figure(figsize=(max(8, len(self.tokenizer_names) * 1.5), 6))
            
            names = list(results['whitespace_fertility']['global'].keys())
            fertilities = [results['whitespace_fertility']['global'][name]['fertility'] for name in names]
            
            x = np.arange(len(names))
            plt.bar(x, fertilities)
            plt.xlabel('Tokenizer')
            plt.ylabel('Tokens per Whitespace Word')
            plt.title('Global Whitespace-based Fertility')
            plt.xticks(x, names, rotation=45)
            
            # Add value labels
            for i, v in enumerate(fertilities):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/whitespace_fertility_global.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Per-language comparison
            languages = sorted(next(iter(results['whitespace_fertility']['per_tokenizer'].values())).keys())
            
            plt.figure(figsize=(max(10, len(languages) * 1.2), 6))
            
            x = np.arange(len(languages))
            width = 0.8 / len(self.tokenizer_names)
            
            for i, name in enumerate(self.tokenizer_names):
                ferts = [results['whitespace_fertility']['per_tokenizer'][name][lang]['fertility'] 
                        for lang in languages]
                plt.bar(x + i * width - (len(self.tokenizer_names)-1)*width/2, 
                       ferts, width, label=name, alpha=0.8)
            
            plt.xlabel('Language')
            plt.ylabel('Tokens per Whitespace Word')
            plt.title('Whitespace-based Fertility by Language')
            plt.xticks(x, languages, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/whitespace_fertility_by_language.png", dpi=300, bbox_inches='tight')
            plt.close()

    def run_full_comparison(self, language_files: Dict[str, str], num_samples_per_lang: int = 1000, save_plots: bool = True) -> Dict:
        """Run all comparison analyses with per-language data and return comprehensive results."""
        
        # Load language-specific texts
        language_texts = self._load_language_texts(language_files, num_samples_per_lang)
        
        results = {}
        
        # Encode all texts once for all tokenizers
        all_encodings = {}
        for name in self.tokenizer_names:
            all_encodings[name] = {}
            for lang, texts in language_texts.items():
                print(f"Encoding {lang} with {name}...")
                all_encodings[name][lang] = encode_text(self.tokenizers[name], texts)["tokens"]
        
        # Run original analyses if we have exactly 2 tokenizers
        if len(self.tokenizers) == 2:
            results['vocabulary_overlap'] = self.vocabulary_overlap_analysis()
            results['token_length'] = self.token_length_analysis()
            results['multilingual'] = self.multilingual_analysis()
            results['consistency'] = self.consistency_analysis_per_language(language_texts)
            
            names = list(self.tokenizers.keys())
            encoded_texts_old = all_encodings[names[0]]
            encoded_texts_new = all_encodings[names[1]]
            
            results['segmentation'] = self.segmentation_comparison_per_language(language_texts, encoded_texts_old, encoded_texts_new)
            results['fertility'] = self.fertility_analysis_per_language(language_texts, encoded_texts_old, encoded_texts_new)
            results['script_analysis'] = self.script_specific_analysis(language_texts, encoded_texts_old, encoded_texts_new)
        
        # Run multi-tokenizer analyses
        results['ttr'] = self.type_token_ratio_analysis(language_texts, all_encodings)
        results['vocab_utilization'] = self.vocabulary_utilization_analysis(language_texts, all_encodings)
        results['avg_tokens_per_line'] = self.average_tokens_per_line_analysis(language_texts, all_encodings)
        results['whitespace_fertility'] = self.whitespace_fertility_analysis(language_texts, all_encodings)
        
        # NEW: Run morphological and Rényi analyses
        results['morphological_alignment'] = self.morphological_alignment_analysis(language_texts, all_encodings)
        print(results['morphological_alignment'])
        results['renyi_efficiency'] = self.renyi_efficiency_analysis(all_encodings)
        
        # Create plots
        if save_plots:
            if len(self.tokenizers) == 2:
                self.plot_comparisons(results)
            self.plot_multi_tokenizer_comparisons(results)
            self.plot_morphological_metrics(results['morphological_alignment'], save_dir="tokenizer_comparison_plots")
            self.plot_renyi_efficiency(results['renyi_efficiency'], save_dir="tokenizer_comparison_plots")
        
        # Summary
        self._print_enhanced_summary(results)
        
        return results


    def _print_enhanced_summary(self, results: Dict) -> None:
        """Print enhanced summary including all metrics."""
        print(f"\n" + "="*60)
        print(f"MULTILINGUAL TOKENIZER COMPARISON SUMMARY")
        print(f"="*60)
        print(f"Tokenizers analyzed: {len(self.tokenizers)}")
        
        # Count languages with data
        languages_with_data = set()
        if 'ttr' in results:
            languages_with_data = set(results['ttr']['per_language'][self.tokenizer_names[0]].keys())
        print(f"Languages analyzed: {len(languages_with_data)}")
        
        # Original metrics if comparing exactly 2
        if len(self.tokenizers) == 2 and 'vocabulary_overlap' in results:
            print(f"\nPairwise Comparison ({self.tokenizer_names[0]} vs {self.tokenizer_names[1]}):")
            print(f"  Vocabulary overlap: {results['vocabulary_overlap']['overlap_percentage']:.1%}")
            print(f"  Average token length change: {results['token_length']['new_stats']['mean'] - results['token_length']['old_stats']['mean']:+.2f} bytes")
            print(f"  Global compression ratio: {results['segmentation']['global'].get('compression_ratio', 1):.3f}")
            print(f"  Global fertility change: {results['fertility']['global'].get('fertility_change', 0):+.4f} tokens/char")
            print(f"  Global segmentation agreement: {results['consistency']['global']['agreement_rate']:.1%}")
        
        # Multi-tokenizer metrics
        print(f"\n=== INFORMATION-THEORETIC METRICS ===")
        
        print(f"\nGlobal Type-Token Ratio:")
        for name in self.tokenizer_names:
            ttr = results['ttr']['per_tokenizer'][name]['global_ttr']
            print(f"  {name}: {ttr:.4f}")
        
        print(f"\nRényi Efficiency (Shannon Entropy, α=1.0):")
        for name in self.tokenizer_names:
            if 'renyi_efficiency' in results:
                shannon = results['renyi_efficiency'][name]['renyi_1.0']['overall']
                print(f"  {name}: {shannon:.3f} bits")

        print(f"\nRényi Efficiency (Renyi Entropy, α=3.0):")
        for name in self.tokenizer_names:
            if 'renyi_efficiency' in results:
                shannon = results['renyi_efficiency'][name]['renyi_3.0']['overall']
                print(f"  {name}: {shannon:.3f} bits")
        
        print(f"\nGlobal Whitespace-based Fertility (tokens/word):")
        for name in self.tokenizer_names:
            fert = results['whitespace_fertility']['global'][name]['fertility']
            print(f"  {name}: {fert:.3f}")
        
        print(f"\nAverage Vocabulary Utilization:")
        for name in self.tokenizer_names:
            utils = [results['vocab_utilization'][name][lang]['utilization_percent'] 
                    for lang in results['vocab_utilization'][name]]
            if utils:
                avg_util = np.mean(utils)
                print(f"  {name}: {avg_util:.1f}%")
        
        # Morphological metrics
        if 'morphological_alignment' in results and any(
            any(results['morphological_alignment'][name]['boundary_f1'].values())
            for name in self.tokenizer_names
        ):
            print(f"\n=== MORPHOLOGICAL ALIGNMENT METRICS ===")
            
            for name in self.tokenizer_names:
                # Calculate global averages
                f1_values = []
                preservation_values = []
                
                for lang in results['morphological_alignment'][name]['boundary_f1']:
                    if results['morphological_alignment'][name]['boundary_f1'][lang]['count'] > 0:
                        f1_values.append(results['morphological_alignment'][name]['boundary_f1'][lang]['mean'])
                        preservation_values.append(results['morphological_alignment'][name]['morpheme_preservation'][lang]['mean'])
                
                if f1_values:
                    print(f"\n{name}:")
                    print(f"  Avg Boundary F1: {np.mean(f1_values):.3f}")
                    print(f"  Avg Morpheme Preservation: {np.mean(preservation_values):.3f}")
        
        # Per-language compression if available
        if 'segmentation' in results:
            print(f"\nPer-language compression ratios (character-based):")
            for lang, lang_results in results['segmentation']['per_language'].items():
                print(f"  {lang}: {lang_results['compression_ratio']:.3f}")
        
        print(f"\n" + "="*60)


# Updated convenience function
def compare_multilingual_tokenizers(tokenizer_configs: Dict[str, Dict], 
                                  language_files: Dict[str, str],
                                  num_samples_per_lang: int = 1000,
                                  morphological_config: Optional[Dict[str, str]] = None,
                                  renyi_alphas: List[float] = [0.5, 1.0, 2.0, 3.0]):
    """
    Convenience function to compare multiple multilingual byte-level tokenizers.
    
    Args:
        tokenizer_configs: Dictionary mapping tokenizer names to their configs
        language_files: Dictionary mapping language codes to file paths
        num_samples_per_lang: Number of samples to use from each language file
        morphological_config: Configuration for morphological datasets
        renyi_alphas: List of alpha values for Rényi entropy computation
    
    Returns:
        Tuple of (results_dict, comparator_object)
    """
    
    # Create comparator and run analysis
    comparator = MultilingualTokenizerComparator(
        tokenizer_configs, 
        morphological_config=morphological_config,
        renyi_alphas=renyi_alphas
    )
    results = comparator.run_full_comparison(language_files, num_samples_per_lang)
    
    return results, comparator


if __name__ == "__main__":
    # Example usage for multilingual byte-level tokenizers
    language_files = {
        "en": "/pub/hofmann-scratch/meistecl/parallel/en/eval.txt",
    "es": "/pub/hofmann-scratch/meistecl/parallel/es/eval.txt",
    "ar": "/pub/hofmann-scratch/meistecl/parallel/ar/eval.txt",
    "cy": "/pub/hofmann-scratch/meistecl/parallel/cy/eval.txt",
    "de": "/pub/hofmann-scratch/meistecl/parallel/de/eval.txt",
    "zh": "/pub/hofmann-scratch/meistecl/parallel/zh/eval.txt",
    "ru": "/pub/hofmann-scratch/meistecl/parallel/ru/eval.txt",
    "hi": "/pub/hofmann-scratch/meistecl/parallel/hi/eval.txt",
    "sw": "/pub/hofmann-scratch/meistecl/parallel/sw/eval.txt",
    "yo": "/pub/hofmann-scratch/meistecl/parallel/yo/eval.txt",
    "el": "/pub/hofmann-scratch/meistecl/parallel/el/eval.txt",
    "gn": "/pub/hofmann-scratch/meistecl/parallel/gn/eval.txt",
    "id": "/pub/hofmann-scratch/meistecl/parallel/id/eval.txt",
    "ku": "/pub/hofmann-scratch/meistecl/parallel/ku/eval.txt",
    "mg": "/pub/hofmann-scratch/meistecl/parallel/mg/eval.txt",
    "pl": "/pub/hofmann-scratch/meistecl/parallel/pl/eval.txt",
    "tr": "/pub/hofmann-scratch/meistecl/parallel/tr/eval.txt",
    "vi": "/pub/hofmann-scratch/meistecl/parallel/vi/eval.txt",
    "ga": "/pub/hofmann-scratch/meistecl/parallel/ga/eval.txt"
    }
    dirr = "data_large_128k_byte/"
    tokenizer_configs = {
    'multigram-soft' : {
      "class": "standard",
      "path": dirr + "tokenizers_correct_loss/multigramlm_soft.json"
    },
    # 'multigram-hard' : {
    #   "class": "standard",
    #   "path": dirr + "tokenizers_correct_loss/multigramlm_hard.json"
    # },
    'standard-soft' : {
      "class": "standard",
      "path": dirr + "tokenizers_correct_loss/standard_unigramlm_soft.json"
    },
    'langspec-soft' :  {
      "class": "langspec",
      "base_path": dirr + "tokenizers_correct_loss/standard_unigramlm_soft.json",
      "language_paths": {
        "en": dirr + "tokenizers_correct_loss/langspec_soft_en_em_probs.json",
        "es": dirr + "tokenizers_correct_loss/langspec_soft_es_em_probs.json",
        "ar": dirr + "tokenizers_correct_loss/langspec_soft_ar_em_probs.json",
        "de": dirr + "tokenizers_correct_loss/langspec_soft_de_em_probs.json",
        "es": dirr + "tokenizers_correct_loss/langspec_soft_es_em_probs.json",
        "el": dirr + "tokenizers_correct_loss/langspec_soft_el_em_probs.json",
        "hi": dirr + "tokenizers_correct_loss/langspec_soft_hi_em_probs.json",
        "ru": dirr + "tokenizers_correct_loss/langspec_soft_ru_em_probs.json",
        "zh": dirr + "tokenizers_correct_loss/langspec_soft_zh_em_probs.json",
        "gn": dirr + "tokenizers_correct_loss/langspec_soft_gn_em_probs.json",
        "sw": dirr + "tokenizers_correct_loss/langspec_soft_sw_em_probs.json",
        "id": dirr + "tokenizers_correct_loss/langspec_soft_id_em_probs.json",
        "ku": dirr + "tokenizers_correct_loss/langspec_soft_ku_em_probs.json",
        "mg": dirr + "tokenizers_correct_loss/langspec_soft_mg_em_probs.json",
        "pl": dirr + "tokenizers_correct_loss/langspec_soft_pl_em_probs.json",
        "tr": dirr + "tokenizers_correct_loss/langspec_soft_tr_em_probs.json",
        "vi": dirr + "tokenizers_correct_loss/langspec_soft_vi_em_probs.json",
        "ga": dirr + "tokenizers_correct_loss/langspec_soft_ga_em_probs.json",
        "yo": dirr + "tokenizers_correct_loss/langspec_soft_yo_em_probs.json"
      }
    },
    # 'langspec-hard' : {
    #   "class": "langspec",
    #   "base_path": dirr + "tokenizers_correct_loss/standard_unigramlm_hard.json",
    #   "language_paths": {
    #     "en": dirr + "tokenizers_correct_loss/langspec_hard_en_em_probs.json",
    #     "es": dirr + "tokenizers_correct_loss/langspec_hard_es_em_probs.json",
    #     "ar": dirr + "tokenizers_correct_loss/langspec_hard_ar_em_probs.json",
    #     "de": dirr + "tokenizers_correct_loss/langspec_hard_de_em_probs.json",
    #     "es": dirr + "tokenizers_correct_loss/langspec_hard_es_em_probs.json",
    #     "el": dirr + "tokenizers_correct_loss/langspec_hard_el_em_probs.json",
    #     "hi": dirr + "tokenizers_correct_loss/langspec_hard_hi_em_probs.json",
    #     "ru": dirr + "tokenizers_correct_loss/langspec_hard_ru_em_probs.json",
    #     "zh": dirr + "tokenizers_correct_loss/langspec_hard_zh_em_probs.json",
    #     "gn": dirr + "tokenizers_correct_loss/langspec_hard_gn_em_probs.json",
    #     "sw": dirr + "tokenizers_correct_loss/langspec_hard_sw_em_probs.json",
    #     "id": dirr + "tokenizers_correct_loss/langspec_hard_id_em_probs.json",
    #     "ku": dirr + "tokenizers_correct_loss/langspec_hard_ku_em_probs.json",
    #     "mg": dirr + "tokenizers_correct_loss/langspec_hard_mg_em_probs.json",
    #     "pl": dirr + "tokenizers_correct_loss/langspec_hard_pl_em_probs.json",
    #     "tr": dirr + "tokenizers_correct_loss/langspec_hard_tr_em_probs.json",
    #     "vi": dirr + "tokenizers_correct_loss/langspec_hard_vi_em_probs.json",
    #     "ga": dirr + "tokenizers_correct_loss/langspec_hard_ga_em_probs.json",
    #     "yo": dirr + "tokenizers_correct_loss/langspec_hard_yo_em_probs.json"
    #   }
    # },
    'tokmix': {
      "class": "standard",
      "path": dirr + "tokenizers_correct_loss/tokmix.json"
    }}
        # Add more languages as needed

    morphological_config = {
      "morphynet": "/local/home/meistecl/multilingual_tokenization/morph_data/MORPHYNET/morphynet_multilingual.tsv"
    }
    # Example usage
    compare_multilingual_tokenizers(
        tokenizer_configs,
        num_samples_per_lang=2000,
        language_files=language_files,
        morphological_config=morphological_config
    )