import random
import math
from collections import OrderedDict


SPECIAL_TOKENS = OrderedDict([("bos_token", "<s>"), ("eos_token", "</s>"), ("pad_token","<pad>"), ("unk_token","<unk>")])
NUM_TRIMMING_ITERATIONS = 3
INITIAL_VOCAB_MULT_FACTOR = 3
MIN_CHAR_OCCURENCE_THRESHOLD = 2
MIN_TOKEN_PROB = 1e-12
MIN_TOKEN_LOG_PROB = math.log(MIN_TOKEN_PROB)
MAX_BYTE_TOKEN_LEN = 50
MAX_CHAR_TOKEN_LEN = 35
MIN_TOKENIZER_LANG_TRAINING_LINES = 5000
TOKENIZER_TOTAL_TRAINING_LINES = 2_000_000
MAX_VAL_TEST_SIZE = 3000
_RNG = random.Random(42)
SP_DEFAULT_ARGS = ["--model_type=unigram",
                    "--normalization_rule_name=identity",
                    "--remove_extra_whitespaces=false",
                    "--allow_whitespace_only_pieces=true",
                    "--unk_id=0",
                    "--add_dummy_prefix=false",
                    "--split_by_whitespace=false",  # handled by write_hf_bytelevel_corpus 
                    "--split_by_unicode_script=false",
                    "--split_by_number=false"]