#!/usr/bin/env bash
# =============================================================================
# UniMixLM Environment Setup
# =============================================================================
# Sets up all dependencies for tokenizer training, finetuning, and evaluation.
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - CMake and a C++ compiler (for SentencePiece)
#   - Git (with submodules)
#
# What this script does:
#   1. Initialise git submodules (tokenizers, sentencepiece)
#   2. Install Python dependencies from requirements.txt
#   3. Build the HuggingFace tokenizers library from the local submodule
#      (provides custom modifications needed for LAS tokenizer training)
#   4. Build the SentencePiece C++ binary from the local submodule
#      (provides spm_train with --seed_sentencepieces_file warm-start flag,
#       which is NOT available in the pip release)
#   5. Verify everything works
#
# NOTE on SentencePiece dual install:
#   - `pip install sentencepiece` provides the Python bindings (import sentencepiece)
#   - Building from the submodule provides the spm_train CLI binary with warm-start
#     support (v0.2.2+ feature not in the pip release). Both are needed.
# =============================================================================

set -euo pipefail

echo "========================================"
echo "UniMixLM Setup"
echo "========================================"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$HOME/.local"
PYTHON="${PYTHON:-python3}"

# ── 1. Git submodules ────────────────────────────────────────────────────────

echo "[1/5] Initialising git submodules..."
cd "$REPO_DIR"
git submodule update --init --recursive

# ── 2. Python dependencies ───────────────────────────────────────────────────

echo "[2/5] Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip setuptools wheel
$PYTHON -m pip install -r requirements.txt

# Maturin is needed to build the tokenizers Rust bindings.
$PYTHON -m pip install "maturin>=1.0" setuptools_rust

# ── 3. Build HuggingFace tokenizers from submodule ───────────────────────────

echo "[3/5] Building tokenizers (Rust bindings)..."
export PATH="$HOME/.cargo/bin:$PATH"
command -v rustc >/dev/null 2>&1 || {
    echo "  Rust not found, installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
}

cd "$REPO_DIR/tokenizers/bindings/python"
maturin develop --release
cd "$REPO_DIR"

# ── 4. Build SentencePiece C++ from submodule ────────────────────────────────

echo "[4/5] Building SentencePiece C++ (static, with warm-start support)..."
SP_DIR="$REPO_DIR/sentencepiece"
mkdir -p "$SP_DIR/build"
cd "$SP_DIR/build"

cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DSPM_ENABLE_SHARED=OFF \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make install
cd "$REPO_DIR"

# ── 5. Verification ─────────────────────────────────────────────────────────

echo "[5/5] Verifying installation..."
export PATH="$INSTALL_PREFIX/bin:$PATH"
hash -r

# Check spm_train has warm-start support
SPM_BIN="$INSTALL_PREFIX/bin/spm_train"
HELP_TEXT=$("$SPM_BIN" --help 2>&1 || true)

if [ -n "$HELP_TEXT" ] && echo "$HELP_TEXT" | grep -q "seed_sentencepieces_file"; then
    echo "  spm_train: OK (warm-start support verified)"
else
    echo "  ERROR: spm_train missing --seed_sentencepieces_file flag."
    echo "  The SentencePiece build may have failed."
    exit 1
fi

# Check Python imports
$PYTHON -c "import sentencepiece; print('  sentencepiece: OK')"
$PYTHON -c "import tokenizers; print('  tokenizers: OK')"
$PYTHON -c "import transformers; print('  transformers: OK')"
$PYTHON -c "import peft; print('  peft: OK')"
$PYTHON -c "import torch; print(f'  torch: OK (CUDA: {torch.cuda.is_available()})')"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Add to your shell profile if not already present:"
echo "  export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login"
echo "  2. See README.md for usage instructions"
echo "========================================"
