#!/usr/bin/env python3
"""A4 (F15) vocabulary-extension baseline: BPE vocabulary EXTENSION + FOCUS-style init.

Two pieces live here:

1. ``build_extended_tokenizer`` — takes the model's own ``tokenizer.json`` and a
   monolingual BPE ``tokenizer.json`` (fit elsewhere with
   ``las_experiment.tokenizer_training.training.train_bpe_tokenizer``) and
   appends the top-k monolingual pieces that are absent from the base vocab as
   ``added_tokens``. Segmentation of the result = base BPE everywhere, with
   greedy longest-match on the appended pieces (that is what the ``tokenizers``
   AddedVocabulary does: it splits the raw input on added-token matches first,
   and only the residue reaches the normalizer/pre-tokenizer/BPE model).

2. ``extend_and_init_embeddings`` — resizes the model's embedding matrix to the
   extended vocabulary and initializes every new row to the mean of the OLD
   embedding rows of the piece's decomposition under the OLD tokenizer
   (FOCUS-style). Deterministic; no RNG.

DECODING CAVEAT (measured with tokenizers 0.22.1) — read before using the
emitted file directly. ``added_tokens`` are matched against RAW text, so their
``content`` must be the raw surface string (" arabalar"), not the byte-level
form ("Ġarabalar"). But ``Tokenizer.decode`` sends every token, added ones
included, through the ByteLevel decoder, which maps each character back to a
single byte via the GPT-2 byte<->char table. For a raw surface string whose
characters all happen to be in that table (e.g. "ları": 'l','a','r','ı' with
'ı' = U+0131) the mapping succeeds and produces the WRONG bytes; only tokens
containing a character outside the table (a space, or any character at or above
U+0144) fall back to the correct raw UTF-8 bytes. Measured on Qwen2.5-0.5B:
``decode([id_of("ları")])`` returns ``'lar�'``.

Consequence: ``tokenizers.Tokenizer.from_file(vocabext_*.json).decode(...)`` is
NOT safe for these files. Use ``ExtendedTokenizer`` (below), which rebuilds the
byte stream itself and is byte-exact. Encoding via the raw ``Tokenizer`` is
fine, so ``ExtendedTokenizer`` delegates encoding unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default number of monolingual pieces appended to the base vocabulary: the
# top-16k new per-language pieces from a monolingual 32k BPE fit. This is a
# free design choice of the experiment.
DEFAULT_TOP_K = 16000

# Highest codepoint the GPT-2 byte<->char table maps (see _bytes_to_unicode:
# the 68 bytes with no printable representation get chars U+0100..U+0143).
# Only used for the diagnostic that counts pieces the native ByteLevel decoder
# would mis-decode.
_BYTELEVEL_MAX_MAPPED_CODEPOINT = 0x143


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / ByteLevel byte -> printable-char table (the standard one)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR: dict[int, str] = _bytes_to_unicode()
CHAR_TO_BYTE: dict[str, int] = {c: b for b, c in BYTE_TO_CHAR.items()}


def bytelevel_piece_to_bytes(piece: str) -> bytes:
    """Byte-level-encoded piece ("Ġarabalar") -> the bytes it stands for.

    Raises on any character outside the byte-level alphabet: that means the
    caller handed in something that is not a byte-level piece, which would
    otherwise be silently mis-decoded.
    """
    out = bytearray()
    for ch in piece:
        b = CHAR_TO_BYTE.get(ch)
        if b is None:
            raise ValueError(
                f"Character {ch!r} (U+{ord(ch):04X}) in piece {piece!r} is not in "
                f"the ByteLevel alphabet — this is not a byte-level BPE piece."
            )
        out.append(b)
    return bytes(out)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> dict:
    """{"commit": <sha or None>, "dirty": <bool or None>} -- mirrors the eval
    drivers' git_state() (scripts/eval_f14_multiblimp.py, scripts/eval_f15_bpb.py)
    so a tokenizer-build manifest carries the same working-tree provenance
    shape used everywhere else in the pipeline. Manifests written before this
    change carry a bare commit string under the same "git_commit" key; this
    function is not retroactively applied to them (do not regenerate existing
    manifests)."""
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        ).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _native_decode_is_correct(surface: str) -> bool:
    """Would tokenizers' ByteLevel decoder return `surface` for this content?

    True when at least one character is outside the byte<->char table (the
    decoder then falls back to the token's raw UTF-8 bytes) or when every
    character is ASCII-printable (the table maps those to themselves).
    """
    if any(ch not in CHAR_TO_BYTE for ch in surface):
        return True
    return all(0x21 <= ord(ch) <= 0x7E for ch in surface)


# ---------------------------------------------------------------------------
# 1. Building the extended tokenizer file
# ---------------------------------------------------------------------------
def build_extended_tokenizer(
    model_tokenizer_json_path,
    mono_bpe_json_path,
    out_path,
    top_k: int = DEFAULT_TOP_K,
):
    """Append the top-`top_k` new monolingual BPE pieces to the model's tokenizer.

    Args:
        model_tokenizer_json_path: the model's own ``tokenizer.json``.
        mono_bpe_json_path: a monolingual BPE ``tokenizer.json``
            (``train_bpe_tokenizer`` output). Its vocab id order is the
            frequency/merge rank, so ids ascending == rank descending.
        out_path: where the extended ``tokenizer.json`` is written. A
            ``<stem>.manifest.json`` is written alongside it.
        top_k: how many new pieces to append. Exactly this many are appended or
            the call raises.

    Returns:
        the manifest dict (also written to disk).
    """
    if isinstance(out_path, int) or isinstance(top_k, (str, Path)):
        raise TypeError(
            "build_extended_tokenizer signature is "
            "(model_tokenizer_json_path, mono_bpe_json_path, out_path, top_k=16000) "
            f"— got out_path={out_path!r}, top_k={top_k!r}. Fix the argument order."
        )
    model_tokenizer_json_path = Path(model_tokenizer_json_path)
    mono_bpe_json_path = Path(mono_bpe_json_path)
    out_path = Path(out_path)
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    for p in (model_tokenizer_json_path, mono_bpe_json_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing tokenizer JSON: {p}")

    with open(model_tokenizer_json_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(mono_bpe_json_path, encoding="utf-8") as f:
        mono = json.load(f)

    if base.get("model", {}).get("type") != "BPE":
        raise ValueError(
            f"{model_tokenizer_json_path} is a "
            f"{base.get('model', {}).get('type')!r} tokenizer; A4 extends a BPE "
            f"vocabulary only."
        )
    if mono.get("model", {}).get("type") != "BPE":
        raise ValueError(
            f"{mono_bpe_json_path} is a {mono.get('model', {}).get('type')!r} "
            f"tokenizer; the monolingual source must be BPE, not Unigram, so "
            f"the known Unigram BpB cost does not bias A4 vs A2."
        )

    base_vocab: dict[str, int] = base["model"]["vocab"]
    base_added = base.get("added_tokens", []) or []
    base_added_contents = {a["content"] for a in base_added}

    # Id space: the next free id must be exactly the number of entries, i.e. the
    # base id space has no holes. A hole would make `resize_token_embeddings`
    # and the new-row init disagree about which rows are new.
    all_base_ids = set(base_vocab.values()) | {int(a["id"]) for a in base_added}
    n_base_entries = len(base_vocab) + len(base_added)
    if len(all_base_ids) != n_base_entries:
        raise ValueError(
            f"{model_tokenizer_json_path}: duplicate ids between model vocab and "
            f"added_tokens ({n_base_entries} entries, {len(all_base_ids)} distinct ids)."
        )
    first_new_id = max(all_base_ids) + 1
    if first_new_id != n_base_entries:
        raise ValueError(
            f"{model_tokenizer_json_path} has a hole in its id space: "
            f"{n_base_entries} entries but max id {first_new_id - 1}. Refusing to "
            f"extend (new rows could not be identified by id range)."
        )

    mono_vocab: dict[str, int] = mono["model"]["vocab"]
    mono_added_contents = {a["content"] for a in (mono.get("added_tokens") or [])}

    # Rank order = ascending id in the monolingual vocab (BpeTrainer emits
    # specials, then the initial byte alphabet, then merges by frequency).
    candidates = sorted(mono_vocab.items(), key=lambda kv: kv[1])

    appended: list[tuple[str, str]] = []   # (bytelevel_piece, raw_surface)
    seen_surface: dict[str, str] = {}
    n_skip_in_base = 0
    n_skip_mono_special = 0
    n_skip_undecodable = 0
    n_skip_already_added = 0
    n_skip_surface_collides_base_vocab = 0
    n_excluded_digit = 0
    n_examined = 0
    skipped_undecodable_examples: list[str] = []
    excluded_digit_examples: list[str] = []

    for piece, _rank in candidates:
        n_examined += 1
        if len(appended) >= top_k:
            break
        if piece in mono_added_contents:
            n_skip_mono_special += 1
            continue
        if piece in base_vocab:
            n_skip_in_base += 1
            continue
        try:
            raw_bytes = bytelevel_piece_to_bytes(piece)
        except ValueError:
            # Not a byte-level piece at all (a stray special the trainer wrote
            # into the vocab). Count and skip; recorded in the manifest.
            n_skip_mono_special += 1
            continue
        try:
            surface = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Byte-level BPE can produce pieces that are a partial UTF-8
            # sequence. An added token's content must be a str, so such pieces
            # cannot be represented. Counted, listed in the manifest, skipped.
            n_skip_undecodable += 1
            if len(skipped_undecodable_examples) < 20:
                skipped_undecodable_examples.append(piece)
            continue
        if not surface:
            n_skip_undecodable += 1
            continue
        # A4 digit exclusion: a piece whose surface
        # is ALL digits (ignoring one leading space, since byte-level BPE marks
        # word-initial pieces that way) is excluded from the appended set. Pure
        # number strings are not a language-specific morphological unit and
        # would just burn top-k budget on frequent numerals; unicode isdigit()
        # so this also catches non-ASCII digit scripts (e.g. Devanagari,
        # Arabic-Indic digits), matching the general spirit of the rule.
        _digit_probe = surface[1:] if surface.startswith(" ") else surface
        if _digit_probe.isdigit():
            n_excluded_digit += 1
            if len(excluded_digit_examples) < 20:
                excluded_digit_examples.append(surface)
            continue
        # A raw UTF-8 surface can coincidentally be the IDENTICAL string to an
        # unrelated base-vocab BYTE-LEVEL piece (that piece's byte-level chars
        # happen to spell the same text as this surface's raw bytes, without
        # representing the same underlying bytes at all), observed
        # on tur: surface "Çİ" collided with an existing Qwen2.5 BPE piece).
        # `tokenizers.Tokenizer`'s merged vocab (model.vocab ∪ added_tokens)
        # is a single string->id namespace, so appending this surface as an
        # added_token would silently shadow/collide with that base piece and
        # the written file would report a vocab size one short of what the
        # manifest claims (caught by the post-write size check below if not
        # skipped here). Skip and count rather than append a colliding string.
        if surface in base_vocab:
            n_skip_surface_collides_base_vocab += 1
            continue
        if surface in base_added_contents:
            n_skip_already_added += 1
            continue
        if surface in seen_surface:
            raise ValueError(
                f"Two monolingual pieces decode to the same surface "
                f"{surface!r}: {seen_surface[surface]!r} and {piece!r}. "
                f"Refusing to append a duplicate added token."
            )
        seen_surface[surface] = piece
        appended.append((piece, surface))

    if len(appended) < top_k:
        raise ValueError(
            f"Only {len(appended)} eligible new pieces found in "
            f"{mono_bpe_json_path} (requested top_k={top_k}); the monolingual "
            f"vocab has {len(mono_vocab)} pieces of which {n_skip_in_base} are "
            f"already in the base vocab, {n_skip_undecodable} are not valid "
            f"UTF-8, {n_skip_mono_special} are specials, {n_excluded_digit} are "
            f"all-digit surfaces, {n_skip_surface_collides_base_vocab} decode to "
            f"a string that collides with an unrelated base-vocab piece. Refit "
            f"the monolingual BPE with a larger vocab_size or lower top_k."
        )

    new_entries = []
    for i, (_piece, surface) in enumerate(appended):
        new_entries.append(
            {
                "id": first_new_id + i,
                "content": surface,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": False,
            }
        )
    last_new_id = first_new_id + len(new_entries) - 1
    n_total_vocab = last_new_id + 1

    extended = copy.deepcopy(base)
    extended["added_tokens"] = list(base_added) + new_entries

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(extended, f, ensure_ascii=False)
    tmp.replace(out_path)

    # ---- Verify the written file, not the in-memory dict ----
    tok = Tokenizer.from_file(str(out_path))
    if tok.get_vocab_size(with_added_tokens=True) != n_total_vocab:
        raise RuntimeError(
            f"Extended tokenizer reports vocab size "
            f"{tok.get_vocab_size(with_added_tokens=True)} != expected {n_total_vocab}."
        )
    n_native_decode_wrong = 0
    for entry in new_entries:
        got_id = tok.token_to_id(entry["content"])
        if got_id != entry["id"]:
            raise RuntimeError(
                f"Id collision: token {entry['content']!r} resolves to {got_id}, "
                f"expected {entry['id']}."
            )
        if not _native_decode_is_correct(entry["content"]):
            n_native_decode_wrong += 1
    # Base pieces must keep their ids.
    for probe_piece, probe_id in list(base_vocab.items())[:512]:
        if tok.token_to_id(probe_piece) != probe_id:
            raise RuntimeError(
                f"Base piece {probe_piece!r} moved from id {probe_id} to "
                f"{tok.token_to_id(probe_piece)} in the extended tokenizer."
            )
    for a in base_added:
        if tok.token_to_id(a["content"]) != int(a["id"]):
            raise RuntimeError(
                f"Base added token {a['content']!r} moved from id {a['id']} to "
                f"{tok.token_to_id(a['content'])} in the extended tokenizer."
            )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "kind": "vocab_extend",
        "base_tokenizer_json": str(model_tokenizer_json_path.resolve()),
        "mono_bpe_json": str(mono_bpe_json_path.resolve()),
        "extended_tokenizer_json": str(out_path.resolve()),
        "sha256": {
            "base_tokenizer_json": _sha256(model_tokenizer_json_path),
            "mono_bpe_json": _sha256(mono_bpe_json_path),
            "extended_tokenizer_json": _sha256(out_path),
        },
        "top_k_requested": top_k,
        "n_appended": len(new_entries),
        "first_new_id": first_new_id,
        "last_new_id": last_new_id,
        "n_total_vocab": n_total_vocab,
        "n_base_model_vocab": len(base_vocab),
        "n_base_added_tokens": len(base_added),
        "n_mono_vocab": len(mono_vocab),
        # Top-level (digit exclusion), alongside
        # the other top-level counters (n_appended,
        # n_appended_native_bytelevel_decode_wrong) that scripts read directly
        # without reaching into "selection".
        "n_excluded_digit": n_excluded_digit,
        "selection": {
            "n_candidates_examined": n_examined,
            "n_skipped_already_in_base_vocab": n_skip_in_base,
            "n_skipped_already_added_token": n_skip_already_added,
            "n_skipped_mono_special_or_non_bytelevel": n_skip_mono_special,
            "n_skipped_not_valid_utf8": n_skip_undecodable,
            "skipped_not_valid_utf8_examples": skipped_undecodable_examples,
            "n_excluded_digit": n_excluded_digit,
            "excluded_digit_examples": excluded_digit_examples,
            "n_skipped_surface_collides_base_vocab": n_skip_surface_collides_base_vocab,
        },
        # See the module docstring: the emitted file must be decoded with
        # ExtendedTokenizer, never with a bare tokenizers.Tokenizer.
        "decode_requires_ExtendedTokenizer": True,
        "n_appended_native_bytelevel_decode_wrong": n_native_decode_wrong,
        "first_appended_examples": [
            {"id": e["id"], "content": e["content"], "bytelevel_piece": p}
            for e, (p, _s) in list(zip(new_entries, appended))[:20]
        ],
    }
    manifest_path = out_path.with_suffix("")
    manifest_path = manifest_path.with_name(manifest_path.name + ".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path.resolve())

    logging.info(
        "build_extended_tokenizer: appended %d pieces to %s -> ids %d..%d "
        "(n_total_vocab=%d); skipped in-base=%d non-utf8=%d specials=%d "
        "all-digit=%d surface-collides-base=%d; %d appended pieces would be "
        "mis-decoded by a bare ByteLevel decoder (use ExtendedTokenizer). "
        "Manifest: %s",
        len(new_entries), model_tokenizer_json_path.name, first_new_id,
        last_new_id, n_total_vocab, n_skip_in_base, n_skip_undecodable,
        n_skip_mono_special, n_excluded_digit, n_skip_surface_collides_base_vocab,
        n_native_decode_wrong, manifest_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# 2. The tokenizer wrapper with a byte-exact decode
# ---------------------------------------------------------------------------
class ExtendedTokenizer:
    """`tokenizers.Tokenizer` with a decode that is correct for added tokens.

    Encoding is delegated unchanged. Decoding rebuilds the byte stream from an
    id -> bytes table (byte-level pieces via the GPT-2 table, added tokens via
    their content's UTF-8 bytes) and decodes once at the end, which is what the
    ByteLevel decoder does for model pieces and what it fails to do for added
    tokens (see the module docstring).
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_json_path):
        self._tok = tokenizer
        self.path = Path(tokenizer_json_path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        n_total = tokenizer.get_vocab_size(with_added_tokens=True)
        id_to_bytes: list[bytes | None] = [None] * n_total
        is_special = [False] * n_total
        for piece, i in data["model"]["vocab"].items():
            id_to_bytes[i] = bytelevel_piece_to_bytes(piece)
        for a in data.get("added_tokens", []) or []:
            i = int(a["id"])
            id_to_bytes[i] = a["content"].encode("utf-8")
            is_special[i] = bool(a.get("special", False))
        missing = [i for i, b in enumerate(id_to_bytes) if b is None]
        if missing:
            raise ValueError(
                f"{self.path}: {len(missing)} ids have no piece "
                f"(first: {missing[:10]}) — id space is not dense."
            )
        self._id_to_bytes = id_to_bytes
        self._is_special = is_special
        self._n_total = n_total

    @classmethod
    def from_file(cls, path) -> "ExtendedTokenizer":
        return cls(Tokenizer.from_file(str(path)), path)

    # ---- encoding: delegate ----
    def encode(self, text, add_special_tokens: bool = True):
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(self, texts, add_special_tokens: bool = True):
        return self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)

    def token_to_id(self, token):
        return self._tok.token_to_id(token)

    def id_to_token(self, i):
        return self._tok.id_to_token(i)

    def get_vocab_size(self, with_added_tokens: bool = True):
        return self._tok.get_vocab_size(with_added_tokens=with_added_tokens)

    def __len__(self):
        return self._n_total

    @property
    def backend(self) -> Tokenizer:
        return self._tok

    # ---- decoding: byte-exact reimplementation ----
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < 0 or i >= self._n_total:
                raise ValueError(
                    f"id {i} out of range for extended tokenizer "
                    f"(size {self._n_total}, {self.path})"
                )
            if skip_special_tokens and self._is_special[i]:
                continue
            buf += self._id_to_bytes[i]
        return buf.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 3. Resize + FOCUS-style init of the new embedding rows
# ---------------------------------------------------------------------------
def extend_and_init_embeddings(model, hf_base_tokenizer, extended_tok_path):
    """Resize `model` to the extended vocabulary and initialize the new rows.

    Each new row is set to the mean of the OLD embedding rows of the piece's
    decomposition under the OLD (base) tokenizer — FOCUS-style, deterministic,
    no RNG.

    MUST be called before any PEFT wrapping: `get_peft_model` deep-copies the
    embedding module for `modules_to_save`, so a later resize would leave the
    trained copy and the base weights inconsistent. Writes go through
    `get_input_embeddings()` so a tied output head follows.

    Returns a stats dict (also suitable for run_metadata).
    """
    extended_tok_path = Path(extended_tok_path)
    if not extended_tok_path.is_file():
        raise FileNotFoundError(
            f"Extended tokenizer not found: {extended_tok_path}. Build it with "
            f"vocab_extend.build_extended_tokenizer — no fallback."
        )
    with open(extended_tok_path, encoding="utf-8") as f:
        data = json.load(f)

    old_len = len(hf_base_tokenizer)
    added = data.get("added_tokens", []) or []
    new_entries = sorted(
        (a for a in added if int(a["id"]) >= old_len), key=lambda a: int(a["id"])
    )
    if not new_entries:
        raise ValueError(
            f"{extended_tok_path} has no added tokens with id >= {old_len} "
            f"(len of the base tokenizer) — nothing to extend. Wrong file?"
        )
    ids = [int(a["id"]) for a in new_entries]
    if ids != list(range(old_len, old_len + len(ids))):
        raise ValueError(
            f"{extended_tok_path}: new-token ids are not contiguous from "
            f"{old_len} (got {ids[0]}..{ids[-1]}, n={len(ids)}). The new-row "
            f"range must be exactly [len(base_tokenizer), n_total)."
        )
    n_total = old_len + len(ids)

    # Every base entry must keep its id, otherwise the pretrained rows below
    # `old_len` no longer correspond to the pieces they were trained for.
    for a in added:
        if int(a["id"]) < old_len:
            got = hf_base_tokenizer.convert_tokens_to_ids(a["content"])
            if got != int(a["id"]):
                raise ValueError(
                    f"{extended_tok_path}: base added token {a['content']!r} sits "
                    f"at id {a['id']} but the base tokenizer has it at {got}."
                )

    emb = model.get_input_embeddings()
    old_rows, hidden = emb.weight.shape
    if old_rows < old_len:
        raise ValueError(
            f"Model has {old_rows} embedding rows but the base tokenizer has "
            f"{old_len} entries — model/tokenizer mismatch, refusing to extend."
        )
    old_weight = emb.weight.detach().to(torch.float32).cpu()[:old_len].clone()

    # ---- decompose each new surface with the OLD tokenizer ----
    surfaces = [a["content"] for a in new_entries]
    decomp_lens: list[int] = []
    n_surface_mismatch = 0
    rows = torch.empty(len(surfaces), hidden, dtype=torch.float32)
    for j, surface in enumerate(surfaces):
        piece_ids = hf_base_tokenizer(surface, add_special_tokens=False)["input_ids"]
        if len(piece_ids) == 0:
            raise ValueError(
                f"New piece {surface!r} (id {old_len + j}) decomposes to ZERO "
                f"tokens under the base tokenizer — cannot initialize its row."
            )
        bad = [p for p in piece_ids if p >= old_len]
        if bad:
            raise ValueError(
                f"New piece {surface!r} decomposes to id(s) {bad} >= "
                f"{old_len}; the base tokenizer must only emit base ids."
            )
        decomp_lens.append(len(piece_ids))
        rows[j] = old_weight[piece_ids].mean(dim=0)
        if hf_base_tokenizer.decode(piece_ids) != surface:
            n_surface_mismatch += 1

    # ---- resize, then overwrite every new row ----
    if n_total < old_rows:
        logging.warning(
            "extend_and_init_embeddings: extended vocab %d is SMALLER than the "
            "model's %d embedding rows; the matrix shrinks (spare rows dropped).",
            n_total, old_rows,
        )
    # mean_resizing=False: the fresh rows are overwritten below, and the
    # covariance-based initializer would put RNG in a path that must stay
    # deterministic.
    model.resize_token_embeddings(n_total, mean_resizing=False)
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] != n_total:
        raise RuntimeError(
            f"resize_token_embeddings gave {emb.weight.shape[0]} input rows, "
            f"expected {n_total}."
        )
    with torch.no_grad():
        emb.weight.data[old_len:n_total] = rows.to(
            device=emb.weight.device, dtype=emb.weight.dtype
        )

    out_emb = model.get_output_embeddings()
    tied = (
        out_emb is not None
        and out_emb.weight.data_ptr() == emb.weight.data_ptr()
    )
    if out_emb is not None:
        if out_emb.weight.shape[0] != n_total:
            raise RuntimeError(
                f"Output embedding has {out_emb.weight.shape[0]} rows after "
                f"resize, expected {n_total}."
            )
        if bool(getattr(model.config, "tie_word_embeddings", False)) and not tied:
            raise RuntimeError(
                "config.tie_word_embeddings is True but the input and output "
                "embeddings do not share storage after resize — refusing to "
                "train a model whose tie was silently broken by the resize."
            )
    if int(getattr(model.config, "vocab_size", n_total)) != n_total:
        raise RuntimeError(
            f"model.config.vocab_size={model.config.vocab_size} != {n_total} "
            f"after resize."
        )

    new_norms = rows.norm(dim=1)
    old_norms = old_weight.norm(dim=1)
    stats = {
        "extended_tokenizer_json": str(extended_tok_path.resolve()),
        "sha256_extended_tokenizer_json": _sha256(extended_tok_path),
        "old_tokenizer_len": int(old_len),
        "old_embedding_rows": int(old_rows),
        "n_new_rows": int(len(surfaces)),
        "n_total_vocab": int(n_total),
        "first_new_id": int(old_len),
        "last_new_id": int(n_total - 1),
        "init": "focus_mean_of_old_decomposition",
        "mean_decomposition_len": float(sum(decomp_lens) / len(decomp_lens)),
        "min_decomposition_len": int(min(decomp_lens)),
        "max_decomposition_len": int(max(decomp_lens)),
        "n_decomposition_surface_mismatch": int(n_surface_mismatch),
        "new_row_norm_mean": float(new_norms.mean()),
        "new_row_norm_std": float(new_norms.std()),
        "old_row_norm_mean": float(old_norms.mean()),
        "old_row_norm_std": float(old_norms.std()),
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "input_output_share_storage": bool(tied),
    }
    logging.info(
        "extend_and_init_embeddings: %d new rows (ids %d..%d), vocab %d -> %d, "
        "mean decomposition length %.2f (min %d, max %d), %d surfaces changed by "
        "the base tokenizer's normalizer; new-row |w| %.4f+-%.4f vs old-row "
        "|w| %.4f+-%.4f; tied=%s (shared storage=%s).",
        stats["n_new_rows"], stats["first_new_id"], stats["last_new_id"],
        old_rows, n_total, stats["mean_decomposition_len"],
        stats["min_decomposition_len"], stats["max_decomposition_len"],
        stats["n_decomposition_surface_mismatch"],
        stats["new_row_norm_mean"], stats["new_row_norm_std"],
        stats["old_row_norm_mean"], stats["old_row_norm_std"],
        stats["tie_word_embeddings"], stats["input_output_share_storage"],
    )
    return stats
