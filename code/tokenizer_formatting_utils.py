import os
import json
import math

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import ByteLevel, Sequence
from typing import List, Tuple, Any
import code.constants as constants


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------
def _apply_pretok_mapping_hf(text: str, pretok: Any, return_individual_pieces: bool = False) -> str:
    
    normalized = pretok.pre_tokenize_str(text)
    pieces = [piece[0] for piece in normalized]
    if return_individual_pieces:
        return pieces
    return ["".join(pieces)]


def _hf_to_sp_piece(token: str) -> str:
    """
    HF ByteLevel -> SentencePiece seed text:
      - GPT-2 space marker (byte 0x20) -> '▁'
      - printable bytes -> literal char
      - other bytes -> '<0xXX>'
    """ 
    out = []
    for ch in token:
        b = constants._UNI_TO_BYTE.get(ch)  # GPT-2 byte-unicode -> raw byte (0..255) or None
        if b is None:
            out.append("\u2581" if ch == " " else ch)
            continue
        if b == 0x20:
            out.append("\u2581")
        elif 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255:
            out.append(chr(b))
        else:
            out.append(f"<0x{b:02X}>")
    return "".join(out)


def set_special_tokens(tokenizer):
    vocab = tokenizer.get_vocab()

    for token_name, token in constants.SPECIAL_TOKENS.items():
        if token in vocab:
                # This sets the pad_token to <pad> if the user wants it recognized as special
            setattr(tokenizer, token_name, token)
            setattr(tokenizer, token_name + '_id', vocab[token])


def _load_vocab_from_hf_tokenizer(file_path: str) -> List[str]:
    """Load vocabulary from HuggingFace tokenizer JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'model' not in data or 'vocab' not in data['model']:
        raise ValueError(f"File does not appear to be a HuggingFace tokenizer: {file_path}")
    
    vocab_data = data['model']['vocab']
    if isinstance(vocab_data, list):
        # Format: [["token", score], ...]
        tokens = [item[0] for item in vocab_data if isinstance(item, list) and len(item) >= 1]
    elif isinstance(vocab_data, dict):
        # Format: {"token": id, ...}
        tokens = sorted(vocab_data, key=vocab_data.get)
    else:
        raise ValueError(f"Unsupported vocab format in HF tokenizer file: {file_path}")
    
    return tokens


def _get_hf_unigram_tokenizer_vocab(tokenizer):
    logger.warning("`_get_hf_unigram_tokenizer_vocab` is a deprecated function; use _load_vocab_from_hf_tokenizer instead")
    state = tokenizer.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    vocab = attributes["vocab"]
    if isinstance(vocab, dict):
        # This is the format the BPE vocab comes in
        tuples = sorted(vocab.items(), key=lambda x: x[1])
        return [(tk, constants.MIN_TOKEN_LOG_PROB) for tk, idx in tuples]
    return vocab


def write_hf_bytelevel_corpus(in_path: str, out_path: str, pretok: Any):
    """Convert raw UTF-8 text to HF-style text (e.g., with 'Ġ' markers)."""
    # TODO: make sure this works for larger corpora
    with open(in_path, "r", encoding="utf-8", errors="strict") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            line = line.rstrip("\n")
            # Pre-tokenize to HF pieces, then concatenate the HF strings.
            pieces = _apply_pretok_mapping_hf(line, pretok, return_individual_pieces=True)
            # The pieces already contain 'Ġ' where spaces were in the *raw* text
            fout.write("".join(pieces) + "\n")

def write_hf_bytelevel_corpus_optimized(in_path: str, out_path: str, pretok: Any, chunk_size_bytes: int = 1024 * 1024):
    """
    An optimized version of the corpus writing function that reads and writes
    in larger chunks to reduce I/O overhead.

    Args:
        in_path (str): Path to the input corpus file.
        out_path (str): Path to write the processed output file.
        pretok (Any): A HuggingFace pre-tokenizer object.
        chunk_size_bytes (int): The approximate size of chunks (in bytes) to
                                read from the input file at a time.
    """
    logger.info(f"Processing {os.path.basename(in_path)} -> {os.path.basename(out_path)}")
    empty = True
    with open(in_path, "r", encoding="utf-8", errors="strict") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        while True:
            # fin.readlines(hint) reads an approximate number of bytes,
            # but will always read until the end of a line.
            lines = fin.readlines(chunk_size_bytes)
            if not lines:
                break
            
            # Use a generator expression for memory efficiency. It processes
            # one line and passes it to writelines, avoiding storing all
            # processed lines in memory at once.
            processed_lines_generator = (
                "".join(_apply_pretok_mapping_hf(line.rstrip("\n"), pretok, return_individual_pieces=True)) + "\n"
                for line in lines
            )
            
            fout.writelines(processed_lines_generator)
            empty = False
    if empty:
        logger.warning(f"No lines produced when processing {in_path}")
    logger.info(f"Finished processing {os.path.basename(in_path)}")

def _process_file_worker(args: Tuple[str, str, Any]) -> str:
    """
    Helper function for multiprocessing.Pool. It unpacks arguments to call
    the optimized processing function. This function must be at the top level
    of the module so it can be pickled.
    """
    train_file, output_dir, pretok, i = args
    base_name = os.path.basename(train_file)
    output_file_path = os.path.join(output_dir, f"processed_{i}_{base_name}")
    write_hf_bytelevel_corpus_optimized(train_file, output_file_path, pretok)
    return output_file_path
# ---------------------------------------------------------------------------
# SentencePiece helpers
# ---------------------------------------------------------------------------

def _write_sp_seed_vocab_file_with_conversion(
    vocab_with_scores: list[tuple[str, float]],
    output_path: str,
    unk_token: str,
) -> int:
    """
    Write TSV for --seed_sentencepieces_file with *trainable* pieces only.
    UNK/BOS/EOS/<pad> are excluded (SP will add meta if enabled).
    Writes raw 'piece<TAB>log_score' lines (no CSV quoting/escaping).
    """
    specials = set(constants.SPECIAL_TOKENS.values()) | {unk_token}
    num_pieces = len([t for t, _ in vocab_with_scores if t not in specials])
    initial_log_score = math.log(1.0 / num_pieces) if num_pieces > 0 else -10.0

    count = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        for token, _ in vocab_with_scores:
            if token in specials:
                continue
            sp_piece = _hf_to_sp_piece(token)

            # sanity: SP seed must not contain tab/newline
            if ("\t" in sp_piece) or ("\n" in sp_piece) or ("\r" in sp_piece):
                # You can choose to replace with a safe printable surrogate or error out.
                raise ValueError(f"Token contains tab/newline: {repr(token)} -> {repr(sp_piece)}")

            f.write(f"{sp_piece}\t{initial_log_score:.6f}\n")
            count += 1
    return count


def _write_sp_seed_vocab_file(vocab_with_scores: list[tuple[str, float]], out_path: str, unk_token: str):
    """One token per line: token<TAB>log_score (uniform log prob)."""
    specials = set(constants.SPECIAL_TOKENS.values()) | {unk_token}
    count = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for t, score in vocab_with_scores:
            if t in specials:
                continue
            if ("\t" in t) or ("\n" in t) or ("\r" in t):
                raise ValueError(f"Token contains tab/newline: {repr(t)}")
            f.write(f"{t}\t{score}\n")
            count += 1
    return count


def sp_piece_to_hf_basic(piece: str) -> str:
    """Map an SP piece to the HF ByteLevel representation."""
    # 1) Map every embedded <0xXX> to the GPT-2 unicode char
    def _byte_repl(m):
        b = int(m.group(1), 16)
        return constants._BYTE_TO_UNI[b]
    piece = constants._SP_BYTE_ANY_RE.sub(_byte_repl, piece)

    # 2) Map SP's whitespace marker to HF's space token char
    #    (SP uses '▁' to mark a preceding space; HF uses the GPT-2 'Ġ' char)
    piece = piece.replace("\u2581", constants._HF_SPACE)

    return piece

def sp_piece_to_hf(piece: str) -> str:
    # 1) Replace leading ▁… with Ġ… (keep count)
    i = 0
    n = len(piece)
    while i < n and piece[i] == constants._SP_SPACE:
        i += 1
    prefix = _HF_SPACE * i
    rest = piece[i:]

    # 2) Walk the remainder, expanding <0xNN> and mapping other literals via UTF-8 -> ByteLevel
    out = [prefix]
    pos = 0
    for m in constants._SP_BYTE_ANY_RE.finditer(rest):
        # map literal region before the match
        lit = rest[pos:m.start()]
        if lit:
            b = lit.encode("utf-8", errors="strict")
            out.append("".join(constants._BYTE_TO_UNI[x] for x in b))
        # map the byte token
        out.append(constants._BYTE_TO_UNI[int(m.group(1), 16)])
        pos = m.end()
    # tail literal
    tail = rest[pos:]
    if tail:
        b = tail.encode("utf-8", errors="strict")
        out.append("".join(constants._BYTE_TO_UNI[x] for x in b))
    return "".join(out)


def load_tokenizer_with_mode(path: str):
    tok = Tokenizer.from_file(path)
    byte_level = isinstance(tok.pre_tokenizer, ByteLevel) or (isinstance(tok.pre_tokenizer, Sequence) and isinstance(tok.pre_tokenizer[1], ByteLevel))
    try:
        # sanity-check the model as well
        byte_fallback = bool(tok.model.byte_fallback)
    except AttributeError:
        byte_fallback = False
    if byte_level and not byte_fallback:
        logger.warning(f"{path} looks byte-level but byte_fallback==False (currently known issue for HF Unigram Tokenizers).")
    return tok, byte_level


def _build_unigramlm_hf_tokenizer_from_lprobs(
    lprobs_tuples: List[Tuple[str, float]],
    unk_id: int,
    pretokenizer: Any = None,
    decoder: Any = None,
    base_tokenizer: Any = None
) -> Tokenizer:
    """
    Turn a plain log probability dictionary into a HF Unigram tokenizer.
    """
    model = Unigram(lprobs_tuples, unk_id)
    tok = Tokenizer(model)
    if base_tokenizer is not None:
        # 1) Clone base config
        base_json = json.loads(base_tokenizer.to_str())
        # 2) Serialize a *temporary* tokenizer holding just this model,
        #    then extract its 'model' JSON block.
        model_json = json.loads(tok.to_str())["model"]

        # 2) Replace only the model (keep *everything else* intact)
        base_json["model"] = model_json

        # 3) Rebuild tokenizer from modified JSON
        tok = Tokenizer.from_str(json.dumps(base_json))
        # NOTE: All of these are preserved automatically from base:
        # - normalizer
        # - pre_tokenizer
        # - decoder
        # - post_processor (templates for BOS/EOS/CLS/SEP, etc.)
        # - added_tokens (special + regular added tokens)
        # - truncation & padding settings

    if pretokenizer:
        tok.pre_tokenizer = pretokenizer
    if decoder:
        tok.decoder = decoder
    return tok


def _detect_vocab_file_format(file_path: str) -> str:
    """Automatically detect the format of a vocabulary file."""
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it's HF tokenizer format
                if isinstance(data, dict) and 'model' in data and 'vocab' in data.get('model', {}):
                    return 'hf_tokenizer'
                # Check if it's a simple JSON list
                elif isinstance(data, list):
                    return 'json_list'
                else:
                    raise ValueError(f"Unsupported JSON format in {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {file_path}")
    else:
        return 'text_file'


def _load_vocab_from_text_file(file_path: str) -> List[str]:
    """Load vocabulary from plain text file (one token per line)."""
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            token = line.rstrip('\n\r')
            if token:  # Skip empty lines
                tokens.append(token)
            elif line.strip() == '':
                logger.warning(f"Empty line {line_num} in vocab file {file_path}")
    return tokens


def _load_vocab_from_json_file(file_path: str) -> List[str]:
    """Load vocabulary from JSON list file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of strings: {file_path}")
    
    tokens = []
    for i, token in enumerate(data):
        if not isinstance(token, str):
            logger.warning(f"Non-string token at index {i} in {file_path}: {token}")
            continue
        if token:  # Skip empty strings
            tokens.append(token)
    
    return tokens


def _load_custom_vocab_from_file(file_path: str) -> List[str]:
    """Main method to load vocabulary from file, with automatic format detection."""
    format_type = _detect_vocab_file_format(file_path)
    
    if format_type == 'text_file':
        tokens = _load_vocab_from_text_file(file_path)
    elif format_type == 'json_list':
        tokens = _load_vocab_from_json_file(file_path)
    elif format_type == 'hf_tokenizer':
        tokens = _load_vocab_from_hf_tokenizer(file_path)
    else:
        raise ValueError(f"Unsupported vocabulary file format: {format_type}")
    
    if not tokens:
        raise ValueError(f"No valid tokens found in vocabulary file: {file_path}")
    
    logger.info(f"Loaded {len(tokens)} tokens from {file_path} (format: {format_type})")
    return tokens
