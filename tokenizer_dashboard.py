"""
Gradio Tokenizer Visualization Dashboard for LAS tokenizers.

Compares tokenization across the original BPE model tokenizer,
the Unigram base, and each language-adapted variant.

Requires: pip install gradio
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import gradio as gr
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKENIZER_DIR = Path(__file__).resolve().parent / "olmo3_las_tokenizers"
MODEL_TOKENIZER_DIR = TOKENIZER_DIR / "model_tokenizer"

# 20-color vibrant palette for token coloring (better contrast)
COLORS = [
    "#FFB3BA",
    "#FFDFBA",
    "#FFFFBA",
    "#BAFFC9",
    "#BAE1FF",
    "#FFBAF3",
    "#FFD4D4",
    "#C9FFD4",
    "#FFE4BA",
    "#D4D4FF",
    "#FFB3E6",
    "#B3E6FF",
    "#E6FFB3",
    "#FFE6B3",
    "#E6B3FF",
    "#B3FFE6",
    "#FFCCB3",
    "#CCE6FF",
    "#E6FFCC",
    "#FFCCE6",
]

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_model_tokenizer_cache: dict = {}
_las_tokenizer_cache: dict[str, LanguageSpecificUnigramLMTokenizer] = {}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_tokenizer_groups() -> list[str]:
    """Return directory names inside TOKENIZER_DIR that contain a base_tokenizer.json."""
    if not TOKENIZER_DIR.is_dir():
        return []
    groups = []
    for entry in sorted(TOKENIZER_DIR.iterdir()):
        if entry.is_dir() and (entry / "base_tokenizer.json").is_file():
            groups.append(entry.name)
    return groups


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def get_model_tokenizer():
    """Load and cache the HF BPE model tokenizer."""
    if "tok" not in _model_tokenizer_cache:
        _model_tokenizer_cache["tok"] = AutoTokenizer.from_pretrained(
            str(MODEL_TOKENIZER_DIR)
        )
    return _model_tokenizer_cache["tok"]


def get_las_tokenizer(group_name: str) -> LanguageSpecificUnigramLMTokenizer:
    """Load and cache a LAS tokenizer group (base + language-specific)."""
    if group_name in _las_tokenizer_cache:
        return _las_tokenizer_cache[group_name]

    group_dir = TOKENIZER_DIR / group_name
    base_path = str(group_dir / "base_tokenizer.json")

    # Discover language-specific tokenizer files
    lang_paths: dict[str, str] = {}
    for f in sorted(group_dir.iterdir()):
        if f.name.startswith("langspec_") and f.suffix == ".json":
            # e.g. "langspec_yor_Latn.json" -> lang_code = "yor_Latn"
            lang_code = f.stem[len("langspec_") :]
            lang_paths[lang_code] = str(f)

    tok = LanguageSpecificUnigramLMTokenizer()
    tok.load(base_path, language_paths=lang_paths if lang_paths else None)

    _las_tokenizer_cache[group_name] = tok
    return tok


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_all(text: str, group_name: str):
    """
    Run all tokenizers on the input text.

    Returns a list of dicts:
        [{"label": str, "tokens": [str, ...], "ids": [int, ...], "decoded": [str, ...], "kind": str}, ...]
        - tokens: raw token strings (byte-encoded for byte-level tokenizers)
        - decoded: actual text each token represents
    """
    if not text.strip():
        return []

    results: list[dict] = []

    # 1. Model tokenizer (BPE)
    try:
        model_tok = get_model_tokenizer()
        # Get encoding with offset mapping to extract actual text spans
        enc = model_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = enc.get("offset_mapping", [])
        tokens = model_tok.convert_ids_to_tokens(ids)

        # Extract the actual text span for each token using offsets
        decoded = []
        for start, end in offsets:
            decoded.append(text[start:end] if start < end else "")

        results.append(
            {
                "label": "Model BPE",
                "tokens": list(tokens),
                "ids": list(ids),
                "decoded": decoded,
                "kind": "bpe",
            }
        )
    except Exception as e:
        results.append(
            {
                "label": "Model BPE",
                "tokens": [f"[error: {e}]"],
                "ids": [],
                "decoded": [],
                "kind": "bpe",
            }
        )

    # 2. LAS tokenizers
    if not group_name:
        return results

    try:
        las_tok = get_las_tokenizer(group_name)
    except Exception as e:
        results.append(
            {
                "label": "Base Unigram",
                "tokens": [f"[error loading: {e}]"],
                "ids": [],
                "kind": "base",
            }
        )
        return results

    # Base tokenizer
    try:
        enc = las_tok.base_tokenizer.encode(text)
        # Extract actual text spans using offsets
        decoded = [text[start:end] if start < end else "" for start, end in enc.offsets]
        results.append(
            {
                "label": "Base Unigram",
                "tokens": list(enc.tokens),
                "ids": list(enc.ids),
                "decoded": decoded,
                "kind": "base",
            }
        )
    except Exception as e:
        results.append(
            {
                "label": "Base Unigram",
                "tokens": [f"[error: {e}]"],
                "ids": [],
                "decoded": [],
                "kind": "base",
            }
        )

    # Determine best language if there are 2+ languages
    best_lang = None
    if len(las_tok.per_lang_tok.keys()) >= 2:
        try:
            best_langs, _ = las_tok.best_language_encode_batch([text])
            best_lang = best_langs[0]
        except Exception:
            pass

    # Per-language tokenizers
    lang_codes = sorted(las_tok.per_lang_tok.keys())
    for lang_code in lang_codes:
        try:
            enc = las_tok.encode_lang(text, lang_code)
            # Extract actual text spans using offsets
            decoded = [
                text[start:end] if start < end else "" for start, end in enc.offsets
            ]
            results.append(
                {
                    "label": f"LangSpec {lang_code}",
                    "tokens": list(enc.tokens),
                    "ids": list(enc.ids),
                    "decoded": decoded,
                    "kind": "langspec",
                    "is_best": lang_code == best_lang,
                }
            )
        except Exception as e:
            results.append(
                {
                    "label": f"LangSpec {lang_code}",
                    "tokens": [f"[error: {e}]"],
                    "ids": [],
                    "decoded": [],
                    "kind": "langspec",
                    "is_best": False,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SPACE_MARKER_RE = re.compile(r"\u0120")  # GPT-2 Ġ


def _display_token(token: str) -> str:
    """Replace the GPT-2 space marker with a visible symbol for display."""
    return _SPACE_MARKER_RE.sub("\u2423", token)  # ␣


def render_tokens_html(
    tokens: list[str], ids: list[int], decoded: list[str], show_raw_tokens: bool = False
) -> str:
    """Return HTML with colour-coded token spans and custom hover tooltips.

    Args:
        tokens: Raw token strings
        ids: Token IDs
        decoded: Actual text each token represents
        show_raw_tokens: If True, display raw tokens and show decoded text on hover.
                        If False, display decoded text and show raw tokens on hover.
    """
    # Group consecutive tokens with the same decoded text (multi-byte chars split across tokens)
    grouped = []
    i = 0
    while i < len(decoded):
        current_decoded = decoded[i]
        current_tokens = [tokens[i] if i < len(tokens) else "?"]
        current_ids = [ids[i] if i < len(ids) else "?"]
        start_pos = i

        # Merge consecutive tokens with same decoded text
        j = i + 1
        while j < len(decoded) and decoded[j] == current_decoded and current_decoded:
            current_tokens.append(tokens[j] if j < len(tokens) else "?")
            current_ids.append(ids[j] if j < len(ids) else "?")
            j += 1

        grouped.append(
            {
                "decoded": current_decoded,
                "tokens": current_tokens,
                "ids": current_ids,
                "pos": start_pos,
            }
        )
        i = j

    # CSS for custom tooltips
    tooltip_css = """
    <style>
    .token-span {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .token-span .tooltip-content {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: #2d3748;
        color: #f7fafc;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: opacity 0.2s, visibility 0.2s;
        pointer-events: none;
    }
    .token-span .tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #2d3748 transparent transparent transparent;
    }
    .token-span:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
    }
    .tooltip-label {
        color: #a0aec0;
        font-weight: 600;
        margin-right: 4px;
    }
    </style>
    """

    # Render grouped tokens
    parts: list[str] = [tooltip_css]
    for i, group in enumerate(grouped):
        bg = COLORS[i % len(COLORS)]

        if show_raw_tokens:
            # Display raw tokens, show decoded text in tooltip
            if len(group["tokens"]) == 1:
                display_text = _display_token(group["tokens"][0])
                display = html.escape(display_text)
                decoded_text = group["decoded"].replace(" ", "␣")
                tooltip_html = (
                    f'<span class="tooltip-label">Decoded:</span>{html.escape(decoded_text or "[empty]")} '
                    f'<span class="tooltip-label">ID:</span>{group["ids"][0]} '
                    f'<span class="tooltip-label">Pos:</span>{group["pos"]}'
                )
            else:
                # Multiple raw tokens - show them all
                display_text = " + ".join(_display_token(t) for t in group["tokens"])
                display = html.escape(display_text)
                decoded_text = group["decoded"].replace(" ", "␣")
                id_list = ", ".join(str(i) for i in group["ids"])
                tooltip_html = (
                    f'<span class="tooltip-label">Decoded:</span>{html.escape(decoded_text or "[empty]")} '
                    f'<span class="tooltip-label">IDs:</span>[{id_list}] '
                    f'<span class="tooltip-label">Pos:</span>{group["pos"]}'
                )
        else:
            # Display decoded text, show raw tokens in tooltip (current behavior)
            display_text = group["decoded"].replace(" ", "␣")  # Show spaces as ␣
            display = html.escape(display_text) if display_text else "[empty]"

            # Build custom tooltip content
            if len(group["tokens"]) == 1:
                tooltip_html = (
                    f'<span class="tooltip-label">Raw:</span>{html.escape(repr(group["tokens"][0]))} '
                    f'<span class="tooltip-label">ID:</span>{group["ids"][0]} '
                    f'<span class="tooltip-label">Pos:</span>{group["pos"]}'
                )
            else:
                raw_list = ", ".join(f"{t!r}" for t in group["tokens"])
                id_list = ", ".join(str(i) for i in group["ids"])
                tooltip_html = (
                    f'<span class="tooltip-label">Raw:</span>[{html.escape(raw_list)}] '
                    f'<span class="tooltip-label">IDs:</span>[{id_list}] '
                    f'<span class="tooltip-label">Pos:</span>{group["pos"]}'
                )

        parts.append(
            f'<span class="token-span" style="'
            f"background:{bg};padding:3px 6px;margin:2px;"
            f"border:1px solid #888;border-radius:4px;"
            f'font-family:monospace;font-size:14px;font-weight:600;color:#222;">'
            f"{display}"
            f'<span class="tooltip-content">{tooltip_html}</span>'
            f"</span>"
        )
    return "".join(parts)


def render_result_card(result: dict, show_raw_tokens: bool = False) -> str:
    """Wrap a single tokenizer result in a card div."""
    label = html.escape(result["label"])
    n_tokens = len(result["tokens"])
    token_html = render_tokens_html(
        result["tokens"],
        result["ids"],
        result.get("decoded", result["tokens"]),
        show_raw_tokens,
    )

    kind_colors = {
        "bpe": "#4A90D9",
        "base": "#7B68EE",
        "langspec": "#2ECC71",
    }
    badge_color = kind_colors.get(result["kind"], "#888")

    # Add "BEST" indicator if this is the best language
    best_badge = ""
    if result.get("is_best", False):
        best_badge = (
            '<span style="background:#E67E22;color:#fff;padding:2px 8px;'
            'border-radius:4px;font-size:13px;font-weight:600;margin-left:8px;">⭐ BEST</span>'
        )

    return (
        f'<div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:8px 0;">'
        f'<div style="margin-bottom:6px;">'
        f'<span style="background:{badge_color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:13px;font-weight:600;">{label}</span>'
        f"{best_badge}"
        f'<span style="color:#888;font-size:12px;margin-left:8px;">'
        f"{n_tokens} tokens</span></div>"
        f'<div style="line-height:2;">{token_html}</div>'
        f"</div>"
    )


def render_all_results(
    text: str, group_name: str, show_raw_tokens: bool = False
) -> str:
    """Tokenize and render all results as HTML cards."""
    if not text.strip():
        return '<p style="color:#888;">Enter some text above and click Tokenize.</p>'

    results = tokenize_all(text, group_name)
    if not results:
        return '<p style="color:#888;">No results.</p>'

    cards = [render_result_card(r, show_raw_tokens) for r in results]
    return "\n".join(cards)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    groups = discover_tokenizer_groups()
    default_group = groups[0] if groups else ""

    with gr.Blocks(title="LAS Tokenizer Dashboard") as app:
        gr.Markdown("# LAS Tokenizer Visualization Dashboard")
        gr.Markdown(
            "Type or paste text to see how different tokenizers segment it. "
            "Tokenization updates automatically as you type. "
            "Hover over tokens to see detailed information."
        )

        with gr.Row():
            group_dd = gr.Dropdown(
                choices=groups,
                value=default_group,
                label="Tokenizer group",
                interactive=True,
            )
            show_raw_toggle = gr.Checkbox(
                value=False,
                label="Show raw tokens (instead of decoded text)",
                interactive=True,
            )

        text_input = gr.Textbox(
            lines=4,
            placeholder="Start typing to see tokenization...",
            label="Input text",
        )

        output_html = gr.HTML(
            value='<p style="color:#888;">Start typing to see tokenization.</p>'
        )

        # Auto-tokenize on text change
        text_input.change(
            fn=render_all_results,
            inputs=[text_input, group_dd, show_raw_toggle],
            outputs=output_html,
        )

        # Also tokenize when group changes
        group_dd.change(
            fn=render_all_results,
            inputs=[text_input, group_dd, show_raw_toggle],
            outputs=output_html,
        )

        # Also tokenize when toggle changes
        show_raw_toggle.change(
            fn=render_all_results,
            inputs=[text_input, group_dd, show_raw_toggle],
            outputs=output_html,
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch()
