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
# Usage:
#   # Option A: Run inside an existing virtualenv or conda env (recommended)
#   conda activate myenv   # or: source .venv/bin/activate
#   bash setup.sh
#
#   # Option B: Let the script create a venv for you
#   bash setup.sh --create-venv
#
# What this script does:
#   1. Initialise git submodules (tokenizers, sentencepiece)
#   2. (Optional) Create a Python venv if --create-venv is passed
#   3. Install Python dependencies from requirements.txt
#   4. Build the HuggingFace tokenizers library from the local submodule
#      (provides custom modifications needed for LAS tokenizer training)
#   5. Build the SentencePiece C++ binary from the local submodule
#      (provides spm_train with --seed_sentencepieces_file warm-start flag,
#       which is NOT available in the pip release)
#   6. Verify everything works
#
# NOTE on SentencePiece dual install:
#   - `pip install sentencepiece` provides the Python bindings (import sentencepiece)
#   - Building from the submodule provides the spm_train CLI binary with warm-start
#     support (v0.2.2+ feature not in the pip release). Both are needed.
#
# NOTE on tokenizers build:
#   - maturin requires either VIRTUAL_ENV or CONDA_PREFIX to be set.
#   - If both are set (common in conda + pip workflows), the script unsets
#     CONDA_PREFIX so maturin doesn't get confused.
#   - If neither is set, use --create-venv or activate an environment first.
# =============================================================================

set -euo pipefail

echo "========================================"
echo "UniMixLM Setup"
echo "========================================"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$HOME/.local"
PYTHON="${PYTHON:-python3}"
CREATE_VENV=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --create-venv) CREATE_VENV=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── 1. Git submodules ────────────────────────────────────────────────────────

echo "[1/6] Initialising git submodules..."
cd "$REPO_DIR"
git submodule update --init --recursive

# ── 2. Python environment ────────────────────────────────────────────────────

if [ "$CREATE_VENV" = true ]; then
    echo "[2/6] Creating Python virtual environment at .venv ..."
    $PYTHON -m venv "$REPO_DIR/.venv"
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
    PYTHON="python"
    echo "  Activated .venv (VIRTUAL_ENV=$VIRTUAL_ENV)"
else
    echo "[2/6] Checking Python environment..."
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
        echo ""
        echo "  WARNING: No virtualenv or conda environment detected."
        echo "  maturin (used to build tokenizers) requires one."
        echo ""
        echo "  Options:"
        echo "    1. Activate your env first:  conda activate myenv"
        echo "    2. Let this script create one:  bash setup.sh --create-venv"
        echo ""
        echo "  Continuing anyway — the tokenizers build step may fail."
        echo ""
    fi
fi

# ── 3. Python dependencies ───────────────────────────────────────────────────

echo "[3/6] Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip setuptools wheel
$PYTHON -m pip install -r requirements.txt

# Maturin is needed to build the tokenizers Rust bindings.
$PYTHON -m pip install "maturin>=1.0" setuptools_rust

# ── 4. Build HuggingFace tokenizers from submodule ───────────────────────────

echo "[4/6] Building tokenizers (Rust bindings)..."

# Install Rust if not present, and source the env so cargo is available
# in this shell session (not just in future shells).
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustc >/dev/null 2>&1; then
    echo "  Rust not found, installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # Source cargo env so rustc/cargo are available immediately
    # (rustup installer tells you to do this but setup scripts often miss it)
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
fi
echo "  rustc version: $(rustc --version)"

# Handle VIRTUAL_ENV / CONDA_PREFIX conflicts for maturin.
# If both are set, maturin refuses to proceed. Unset CONDA_PREFIX
# so maturin uses VIRTUAL_ENV (the pip-based env).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    echo "  Both VIRTUAL_ENV and CONDA_PREFIX are set — unsetting CONDA_PREFIX for maturin."
    unset CONDA_PREFIX
fi

cd "$REPO_DIR/tokenizers/bindings/python"
maturin develop --release
cd "$REPO_DIR"

echo "  tokenizers built successfully."

# ── 5. Build SentencePiece C++ from submodule ────────────────────────────────

echo "[5/6] Building SentencePiece C++ (static, with warm-start support)..."
SP_DIR="$REPO_DIR/sentencepiece"
mkdir -p "$SP_DIR/build"
cd "$SP_DIR/build"

cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DSPM_ENABLE_SHARED=OFF \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make install
cd "$REPO_DIR"

# ── 6. Verification ─────────────────────────────────────────────────────────

echo "[6/6] Verifying installation..."
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
