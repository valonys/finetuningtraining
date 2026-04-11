#!/usr/bin/env bash
# ValonyLabs Studio v3.0 — hardware-aware installer
# Usage: bash scripts/install.sh
set -euo pipefail

echo "🔍 Detecting hardware..."
OS="$(uname -s)"
ARCH="$(uname -m)"

STACK=""

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    STACK="mlx"
elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        STACK="cuda"
    elif command -v rocminfo &>/dev/null; then
        STACK="cuda"   # use CUDA stack — vLLM + torch ROCm wheels
    else
        STACK="cpu"
    fi
else
    STACK="cpu"
fi

echo "🎯 Stack selected: $STACK"

# Prefer uv if it's on the PATH (fast pip replacement, matches v2.0's install path)
if command -v uv &>/dev/null; then
    PIP_CMD=(uv pip install)
else
    PIP_CMD=(pip install --upgrade)
fi

REQ_FILE="requirements-${STACK}.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    echo "❌ Missing $REQ_FILE"; exit 1
fi

echo "📦 Installing from $REQ_FILE ..."
"${PIP_CMD[@]}" -r "$REQ_FILE"

echo "🧪 Running preflight check..."
python scripts/preflight.py || true

echo "✅ Done. Launch the studio with: bash scripts/run_studio.sh"
