#!/usr/bin/env bash
set -euo pipefail

# Canonical FR-Spec entry after the Sep18 foundation correction.
# Reuse the audited Stage-12 FR-Spec implementation, but force its base to the
# corrected modern-MTP candidate rather than the retired Sep11 approximation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
export R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-frspec-20260918}"
export RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-frspec}"
export SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
export FULL_ALIAS="${FULL_ALIAS:-qwen3.8-flash-next-r2-modern-frspec-full:256k}"
export FR_ALIAS="${FR_ALIAS:-qwen3.8-flash-next-r2-modern-frspec-65k:256k}"

[[ -f "$BASE_SRC/r2-meta/stage10-modern-mtp-base.txt" ]] || {
  echo "ERROR: corrected modern-MTP base is not ready: $BASE_SRC" >&2
  echo "Run prepare_stage10_upstream_mtp.sh and its A/B first." >&2
  exit 2
}

exec bash "$SCRIPT_DIR/prepare_stage12_frspec.sh" "$@"
