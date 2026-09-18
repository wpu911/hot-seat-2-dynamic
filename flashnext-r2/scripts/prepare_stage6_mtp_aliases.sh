#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-6: MTP n-max 2/3/4 sweep.
# Uses one selected runtime and holds every kernel/cache setting constant.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-q8dedup}"
HC_MODE="${HC_MODE:-1}"
GDN_MODE="${GDN_MODE:-1}"
Q8_MODE="${Q8_MODE:-1}"
GRAPH_DISABLE="${GRAPH_DISABLE:-0}"
JMAX="${JMAX:-0}"

A2="${A2:-qwen3.8-flash-next-r2-mtp2:256k}"
A3="${A3:-qwen3.8-flash-next-r2-mtp3:256k}"
A4="${A4:-qwen3.8-flash-next-r2-mtp4:256k}"

if [[ ! -x "$RUNTIME/llama-server" ]]; then
  echo "ERROR: selected runtime missing: $RUNTIME/llama-server" >&2
  exit 2
fi

COMMON_ENV=(
  --env "GGML_JOHNV8_HC_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_MIX_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_GDN_PROLOG=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2_FMA=1"
  --env "GGML_JOHNV8_Q8_DEDUP=$Q8_MODE"
  --env "GGML_CUDA_DISABLE_GRAPHS=$GRAPH_DISABLE"
)

make_alias() {
  local alias="$1" n="$2" validate="$3"
  local args=(
    --alias "$alias"
    --r2-bin "$RUNTIME"
    --jmax "$JMAX"
    --spec-draft-n-max "$n"
    "${COMMON_ENV[@]}"
    --replace
  )
  if [[ "$validate" == 1 ]]; then args+=(--validate); fi
  python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" "${args[@]}"
}

make_alias "$A2" 2 0
make_alias "$A3" 3 0
make_alias "$A4" 4 1

echo
echo "Stage-6 MTP aliases ready"
echo "MTP2: $A2"
echo "MTP3: $A3"
echo "MTP4: $A4"
echo "runtime=$RUNTIME"
echo "HC=$HC_MODE GDN=$GDN_MODE Q8=$Q8_MODE GRAPH_DISABLE=$GRAPH_DISABLE JMAX=$JMAX"
echo "Next: bash $SCRIPT_DIR/run_stage6_mtp_sweep.sh"
