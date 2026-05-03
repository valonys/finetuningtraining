#!/usr/bin/env bash
# Install / update the vendored llama.cpp toolchain that A1's GGUF export
# (app/trainers/export.py) depends on. Pins a commit so the
# convert_hf_to_gguf.py + llama-quantize behavior stays reproducible.
#
# Override target dir with VALONY_LLAMA_CPP_PATH and pin commit/tag with
# LLAMA_CPP_REF.
set -euo pipefail

LLAMA_CPP_DIR="${VALONY_LLAMA_CPP_PATH:-$HOME/.local/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"   # pin to a tag in production

if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
  echo "==> Updating llama.cpp at $LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" fetch --tags origin
else
  echo "==> Cloning llama.cpp into $LLAMA_CPP_DIR"
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_REF"

echo "==> Building llama-quantize"
cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" \
      -DLLAMA_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j

echo "==> Installing python deps for convert_hf_to_gguf.py"
python -m pip install --quiet --upgrade numpy gguf sentencepiece protobuf

cat <<EOF

✅ llama.cpp ready
   convert script:  $LLAMA_CPP_DIR/convert_hf_to_gguf.py
   quantize binary: $LLAMA_CPP_DIR/build/bin/llama-quantize

Use it via either:
  export VALONY_LLAMA_CPP_PATH=$LLAMA_CPP_DIR
  python scripts/export_gguf.py --base-model ... --adapter ...
EOF
