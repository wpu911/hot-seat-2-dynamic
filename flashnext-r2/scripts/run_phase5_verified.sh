#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-gdn-microfusion/bin}"

[[ -x "$RUNTIME/llama-server" ]] || {
  echo "ERROR Phase-5 runtime missing: $RUNTIME/llama-server" >&2
  echo "Run prepare_phase5_gdn_microfusion.sh first." >&2
  exit 2
}
AUDIT_SERVER="$RUNTIME/llama-server"
[[ -x "$RUNTIME/llama-server.real" ]] && AUDIT_SERVER="$RUNTIME/llama-server.real"
SERVER="$AUDIT_SERVER" REQUIRE_BOTH_GPUS=1 \
  bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$RUNTIME"

exec bash "$SCRIPT_DIR/run_phase5_gdn_microfusion_ab.sh"
