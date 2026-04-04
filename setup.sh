#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "🚀 UnimixLM HPC Setup: FINAL STABLE"
echo "========================================"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$HOME/.local"
PYTHON=python3

# 1. System Dependencies (Assuming already installed as root)
# 2. Python Environment & Rust
$PYTHON -m pip install --upgrade pip setuptools wheel maturin>=1.0 setuptools_rust
$PYTHON -m pip install sentencepiece 

export PATH="$HOME/.cargo/bin:$PATH"
command -v rustc >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 3. Build Tokenizers
cd "$REPO_DIR/tokenizers/bindings/python"
maturin develop --release

# 4. Build SentencePiece C++ (STATIC)
echo "🔤 Building SentencePiece C++ (Static)..."
SP_DIR="$REPO_DIR/sentencepiece"
mkdir -p "$SP_DIR/build"
cd "$SP_DIR/build"

cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
         -DSPM_ENABLE_SHARED=OFF \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON
         
make -j "$(nproc)"
make install

# 5. Verification (The robust version)
echo "🧪 Final verification..."
export PATH="$INSTALL_PREFIX/bin:$PATH"
hash -r

SPM_BIN="$INSTALL_PREFIX/bin/spm_train"

# We capture the output and ignore the exit code (|| true) 
# so the script doesn't trip on gflags' "exit 1 on help" behavior.
HELP_TEXT=$("$SPM_BIN" --help 2>&1 || true)

if [ -n "$HELP_TEXT" ] && echo "$HELP_TEXT" | grep -q "seed_sentencepieces_file"; then
    echo "✅ SUCCESS: spm_train (v0.2.2+) verified with warm-start support."
else
    echo "❌ ERROR: Verification failed. The flag was not found in the help output."
    exit 1
fi

$PYTHON -c "import sentencepiece; print('✅ Python library loaded successfully')"

echo "========================================"
echo "🎉 Setup complete! Proceed to training."
echo "========================================"