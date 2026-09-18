#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-5: HIP Graph ON/OFF using the SAME already-built runtime.
# No source rebuild is needed because prior R2 builds use -DGGML_CUDA_GRAPHS=ON.
# Runtime graph replay is toggled with GGML_CUDA_DISABLE_GRAPHS.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-q8dedup}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-graph-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-graph-on:256k}"
HC_MODE="${HC_MODE:-1}"
GDN_MODE="${GDN_MODE:-1}"
Q8_MODE="${Q8_MODE:-1}"
JMAX="${JMAX:-0}"

if [[ ! -x "$RUNTIME/llama-server" ]]; then
  echo "ERROR: runtime missing: $RUNTIME/llama-server" >&2
  echo "Run the selected prior stage build first, or pass RUNTIME=/path/to/bin-dir." >&2
  exit 2
fi

COMMON_ENV=(
  --env "GGML_JOHNV8_HC_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_MIX_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_GDN_PROLOG=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2_FMA=1"
  --env "GGML_JOHNV8_Q8_DEDUP=$Q8_MODE"
)

# Graph OFF baseline.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax "$JMAX" \
  "${COMMON_ENV[@]}" \
  --env GGML_CUDA_DISABLE_GRAPHS=1 \
  --replace

# Graph ON: compile-time support remains enabled and the disable flag is zero.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax "$JMAX" \
  "${COMMON_ENV[@]}" \
  --env GGML_CUDA_DISABLE_GRAPHS=0 \
  --replace --validate

echo "Stage-5 HIP Graph aliases ready"
echo "OFF: $OFF_ALIAS"
echo " ON: $ON_ALIAS"
echo "runtime=$RUNTIME"
echo "HC=$HC_MODE GDN=$GDN_MODE Q8=$Q8_MODE JMAX=$JMAX held constant"
echo "Next: bash $SCRIPT_DIR/run_stage5_graph_ab.sh"
