#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"

for name in tensor-1x1 tensor-mid tensor-cap tensor-inv-mid tensor-inv-cap; do
  bin="$RUNTIME_ROOT/$name/bin"
  [[ -d "$bin" ]] || { echo "ERROR Phase-6b runtime missing: $bin" >&2; exit 2; }
  OUT="$LOG_DIR/flashnext-r2-phase6b-${name}-normalize-$STAMP.log" \
    bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$bin"
done

bash "$SCRIPT_DIR/verify_phase6b_ratio_runtime.sh"
exec bash "$SCRIPT_DIR/run_phase6b_tensor_ratio_sweep.sh"
