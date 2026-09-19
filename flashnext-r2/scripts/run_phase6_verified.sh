#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
LAYER_BIN="$RUNTIME_ROOT/layer/bin"
TENSOR_BIN="$RUNTIME_ROOT/tensor-1x1/bin"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# Stable Phase-6 entrypoint. Normalize both staged bundles first so an absolute
# CMake RUNPATH cannot survive merely because its old build directory still
# happens to exist. Then run the same-ELF/wrapper/device audit before model load.
OUT="$LOG_DIR/flashnext-r2-phase6-layer-normalize-$STAMP.log" \
  bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$LAYER_BIN"
OUT="$LOG_DIR/flashnext-r2-phase6-tensor-normalize-$STAMP.log" \
  bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$TENSOR_BIN"

bash "$SCRIPT_DIR/verify_phase6_split_runtime.sh"
exec bash "$SCRIPT_DIR/run_phase6_tensor_split_ab.sh"
