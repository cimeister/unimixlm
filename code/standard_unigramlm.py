import os
import logging
import json
import math
import multiprocessing

from collections import defaultdict
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from code.utils import _accumulate_usage, get_baseline_characters
from code.utils import SPECIAL_TOKENS, INITIAL_VOCAB_MULT_FACTOR, NUM_TRIMMING_ITERATIONS, MIN_CHAR_OCCURENCE_THRESHOLD
from code.language_selection import SELECTED_LANGUAGES, LANGUAGES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKENIZERS_DIR = os.path.join(DATA_DIR, "tokenizers")
os.makedirs(TOKENIZERS_DIR, exist_ok=True)


class StandardUnigramLMTokenizer:
    """
    A 'standard' UnigramLM tokenizer that can be trained in three modes:
      1) HF mode ("em_mode='hf'") => use the built-in HuggingFace UnigramTrainer
      2) Hard EM ("em_mode='hard'") => single-language custom approach with Viterbi E-step
      3) Soft EM ("em_mode='soft'") => single-language custom approach with forward-backward E-step
    """
    def __init__(
        self, 
        vocab_size=8000, 
        unk_token=SPECIAL_TOKENS['unk_token'], 
        special_tokens=None,
        em_mode="hf",            # "hf", "hard", or "soft"
        num_iterations=10,        # If em_mode != "hf", how many EM iterations
        max_sub_len=15          # for forward-backward, to speed up
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        if special_tokens is None:
            special_tokens = list(SPECIAL_TOKENS.values())
        self.special_tokens = special_tokens

        self.base_tokenizer = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.max_sub_len = max_sub_len
    
    def train(self, corpus_files, output_path=None, load_if_exists=False):
        """
        Variant of train that uses _accumulate_usage_soft / _accumulate_usage_hard
        for custom EM if em_mode is 'soft' or 'hard'.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import Unigram

        if load_if_exists and output_path and os.path.isfile(output_path):
            logger.info(f"Loading existing tokenizer from {output_path}")
            self.base_tokenizer = Tokenizer.from_file(output_path)
            return self.base_tokenizer

        if self.em_mode == "hf":
            # ============= HF Approach =============
            logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
            tok = Tokenizer(Unigram())
            tok.pre_tokenizer = Whitespace()

            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens,
                unk_token=self.unk_token
            )
            tok.train(corpus_files, trainer)
            self.base_tokenizer = tok

            if output_path:
                self.base_tokenizer.save(output_path)
                logger.info(f"Saved HF-based standard tokenizer to {output_path}")
            return self.base_tokenizer

        else:
            # ============= Custom EM Approach =============
            logger.info(f"Custom EM approach ({self.em_mode}). Building an initial oversized vocab with HF tools...")

            chars_to_keep, char_freq = get_baseline_characters(corpus_files, MIN_CHAR_OCCURENCE_THRESHOLD)
            logger.info(f"{len(char_freq)} unique characters found in dataset. Attempting to keep all with "
                f"at least {MIN_CHAR_OCCURENCE_THRESHOLD} occurences, which is {len(chars_to_keep)} characters.")
            # 1) Build an initial big Unigram model
            init_tok = Tokenizer(Unigram())
            init_tok.pre_tokenizer = Whitespace()
            trainer = UnigramTrainer(
                vocab_size=self.vocab_size * INITIAL_VOCAB_MULT_FACTOR,
                special_tokens=self.special_tokens,
                unk_token=self.unk_token
            )
            init_tok.train(corpus_files, trainer)
            vocab_dict = init_tok.get_vocab()
            all_tokens = set(vocab_dict.keys())
            all_tokens.update(chars_to_keep)
            all_tokens.update(self.special_tokens)

            # uniform distribution initially
            p_token = {}
            for tk in all_tokens:
                p_token[tk] = 1.0/ len(all_tokens)
            
            # 2) EM loop
            num_tokens_to_trim = self.vocab_size * INITIAL_VOCAB_MULT_FACTOR - self.vocab_size
            tokens_to_remove_per_loop = [math.floor(num_tokens_to_trim/NUM_TRIMMING_ITERATIONS)] * NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - NUM_TRIMMING_ITERATIONS)
            keys_to_keep = set(self.special_tokens + chars_to_keep)
            for it in range(self.num_iterations):
                logger.info(f"EM iteration {it+1}/{self.num_iterations} for single-language ({self.em_mode})")

                usage_counts = {tk: 0.0 for tk in all_tokens}
                total_subwords = 0.0

                # keep UNK prob small
                min_prob = min(p_token.values())
                if self.unk_token in p_token:
                    p_token[self.unk_token] = min_prob / 1000.0
                logger.info(f"Reseting UNK prob to {p_token[self.unk_token]}")

                # For each corpus file
                for fpath in corpus_files:
                    c_dict, tot = _accumulate_usage(fpath, p_token, self.unk_token, mode=self.em_mode)

                    for tk, val in c_dict.items():
                        usage_counts[tk]+= val
                    total_subwords+= tot

                # M-step
                # prune if needed
                if tokens_to_remove_per_loop[it] > 0:                    
                    tokens_to_keep = []
                    remaining_items = []
                    for tk, pr in p_token.items():
                        if tk in keys_to_keep:
                            tokens_to_keep.append((tk, pr))
                        else:
                            remaining_items.append((tk, pr))
                    # Get the rest of the tuples that are not in items_to_move.
                    sorted_final = tokens_to_keep + sorted(remaining_items, key=lambda x: x[1], reverse=True)
                    all_tokens = set([tk for tk, sc in sorted_final[:-tokens_to_remove_per_loop[it]]])
                    logger.info(f"Current vocab size: " + str(len(all_tokens)))

                sum_counts = sum([usage_counts[t] for t in all_tokens])
                p_token = {}
                if sum_counts>0:
                    for tk in all_tokens:
                        p_token[tk] = usage_counts[tk]/ sum_counts
                else:
                    logger.warning("No usage => fallback uniform distribution.")
                    for tk in all_tokens:
                        p_token[tk] = 1.0/ len(all_tokens)

                logger.info(f"Current UNK prob: {p_token[self.unk_token]}" )
            
            if len(p_token) > self.vocab_size:
                tokens_to_keep = []
                remaining_items = []
                for tk, pr in p_token.items():
                    if tk in keys_to_keep:
                        tokens_to_keep.append((tk, pr))
                    else:
                        remaining_items.append((tk, pr))
                # Get the rest of the tuples that are not in items_to_move.
                sorted_final = tokens_to_keep + sorted(remaining_items, key=lambda x: x[1], reverse=True)
                sorted_final = sorted_final[:self.vocab_size]
                remaining_prob = sum([pr for tk, pr in sorted_final])
                sorted_final =[(tk, pr/remaining_prob) for tk, pr in sorted_final]
            else:
                sorted_final = sorted(p_token.items(), key=lambda x: x[1], reverse=True)

            total_prob_coverage = sum([p for tk, p in sorted_final])
            logger.info(f"Token probabilities sum to {total_prob_coverage}")
            
            # build final Unigram model, keeping special tokens at beginning
            vocab_list = []
            special_tokens = []
            for tk, prob in sorted_final:
                score = math.log(max(prob,1e-12))
                if tk in self.special_tokens:
                    special_tokens.append((tk, score))
                else:
                    vocab_list.append((tk, score))
            vocab_list = special_tokens + vocab_list

            logger.info(f"Custom EM done. final vocab size= {len(vocab_list)}")
            # figure out unk_id
            unk_id = 0
            for i,(tk,sc) in enumerate(vocab_list):
                if tk == self.unk_token:
                    logger.info(f"UNK score: {sc}")
                    unk_id= i
                    break

            final_tok = Tokenizer(Unigram(tuple(vocab_list), unk_id))
            final_tok.pre_tokenizer = Whitespace()
            self.base_tokenizer = final_tok
            vocab = self.base_tokenizer.get_vocab()
            id_check = {k: vocab[k] for k in self.special_tokens}
            logger.info(f"Special tokens: "+ str(id_check))

            if output_path:
                self.base_tokenizer.save(output_path)
                logger.info(f"Saved custom-EM standard tokenizer to {output_path}")

            return self.base_tokenizer


    def finalize_vocab_indices(self):
        """
        Re-map tokens in self.base_tokenizer => IDs [0..N-1] 
        as a standard Unigram. 
        """
        if not self.base_tokenizer:
            logger.warning("No base_tokenizer to finalize in StandardUnigramLMTokenizer.")
            return

        unigram_model = self.base_tokenizer.model
        state = unigram_model.__getstate__()
        attributes = json.loads(state.decode('utf-8'))
        old_vocab = attributes['vocab']
        unk_id    = attributes['unk_id']

        from tokenizers import Tokenizer
        from tokenizers.models import Unigram

        new_tok = Tokenizer(Unigram(tuple((x,y) for x,y in old_vocab), unk_id))
        vocab = new_tok.get_vocab()
        id_check = {k: vocab[k] for k in self.special_tokens}
        logger.info(f"Special tokens: "+ str(id_check))
        new_tok.pre_tokenizer = self.base_tokenizer.pre_tokenizer
        self.base_tokenizer = new_tok

    def encode(self, text):
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded or trained.")
        return self.base_tokenizer.encode(text).ids

    def decode(self, ids):
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer loaded or trained.")
        return self.base_tokenizer.decode(ids)

    def get_vocab(self):
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer.")
        return self.base_tokenizer.get_vocab()

    def get_vocab_size(self):
        if not self.base_tokenizer:
            return 0
        return len(self.base_tokenizer.get_vocab())

def train_standard_tokenizer(corpus_info, vocab_size=8000, em_mode='hard'):
    """
    Train a standard UnigramLM tokenizer on all languages.
    
    Args:
        corpus_info: Dictionary with corpus information for all languages
        vocab_size: Size of the vocabulary
    
    Returns:
        Path to the trained tokenizer
    """
    # Collect all training files
    train_files = []
    for lang_code, info in corpus_info.items():
        if 'splits' in info and 'train' in info['splits']:
            train_files.append(info['splits']['train'])
    
    if not train_files:
        logger.error("No training files found")
        return None
    
    # Create and train the tokenizer
    tokenizer = StandardUnigramLMTokenizer(vocab_size=vocab_size, em_mode=em_mode)
    output_path = os.path.join(TOKENIZERS_DIR, f"standard_unigramlm_{em_mode}.json")
    tokenizer.train(train_files, output_path)
    
    return output_path

if __name__ == "__main__":
    # For testing purposes
    from code.corpus_downloader import process_all_corpora
    
    # Process corpora
    corpus_info = process_all_corpora()
    
    # Train standard tokenizer
    tokenizer_path = train_standard_tokenizer(corpus_info)
    
    # Test the tokenizer
    if tokenizer_path:
        tokenizer = StandardUnigramLMTokenizer()
        tokenizer.load(tokenizer_path)
        
        # Test tokenization
        test_texts = {
            'en': "Hello world, this is a test.",
            'es': "Hola mundo, esta es una prueba.",
            'fr': "Bonjour le monde, c'est un test."
        }
        
        print("\nTokenization Test:")
        print("-----------------")
        for lang, text in test_texts.items():
            tokens = tokenizer.tokenize(text)
            print(f"{lang}: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {tokenizer.encode(text)}")
            print()