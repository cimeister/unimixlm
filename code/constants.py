import random
import math
from collections import OrderedDict


SPECIAL_TOKENS = OrderedDict([("bos_token", "<s>"), ("eos_token", "</s>"), ("pad_token","<pad>"), ("unk_token","<unk>")])
NUM_TRIMMING_ITERATIONS = 3
INITIAL_VOCAB_MULT_FACTOR = 3
MIN_CHAR_OCCURENCE_THRESHOLD = 2
MIN_TOKEN_PROB = 1e-12
MIN_TOKEN_LOG_PROB = math.log(MIN_TOKEN_PROB)
_TOTAL_SAMPLE_LINES = 2_000_000
MAX_VAL_TEST_SIZE = 3000
_RNG = random.Random(42)