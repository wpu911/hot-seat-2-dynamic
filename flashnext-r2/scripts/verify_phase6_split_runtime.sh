#!/usr/bin/env bash
set -euo pipefail

# Hard audit for Phase-6 layer-vs-tensor runtimes before any expensive A/B.
# It verifies that both wrappers point at the same real ELF, both RDNA devices
# are visible, llama/ggml shared libraries are self-contained enough for the
# staged runtime, and the wrapper-enforced split parameters are exactly the
# intended experimental variable.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
LAYER_ALIAS="${LAYER_ALIAS:-qwen3.8-flash-next-r2-split-layer:256k}"
TENSOR_ALIAS="${TENSOR_ALIAS:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
LAYER_BIN="$RUNTIME_ROOT/layer/bin"
TENSOR_BIN="$RUNTIME_ROOT/tensor-1x1/bin"

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
for a in "$LAYER_ALIAS" "$TENSOR_ALIAS"; do
  grep -qE "^[[:space:]]*${a//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR alias missing: $a" >&2; exit 3;
  }
done
for d in "$LAYER_BIN" "$TENSOR_BIN"; do
  [[ -x "$d/llama-server" && -x "$d/llama-server.real" ]] || {
    echo "ERROR incomplete Phase-6 runtime: $d" >&2; exit 4;
  }
done

S_LAYER="$(sha256sum "$LAYER_BIN/llama-server.real" | awk '{print $1}')"
S_TENSOR="$(sha256sum "$TENSOR_BIN/llama-server.real" | awk '{print $1}')"
[[ "$S_LAYER" == "$S_TENSOR" ]] || {
  echo "ERROR layer/tensor real ELF differs: $S_LAYER vs $S_TENSOR" >&2
  exit 5
}

# Wrapper intent must be explicit. Do not trust alias names as evidence.
grep -Fq -- 'extra=(--split-mode layer)' "$LAYER_BIN/llama-server" || {
  echo "ERROR layer wrapper does not force --split-mode layer" >&2; exit 6;
}
grep -Fq -- '--split-mode tensor' "$TENSOR_BIN/llama-server" || {
  echo "ERROR tensor wrapper does not force --split-mode tensor" >&2; exit 7;
}
grep -Fq -- '--tensor-split "1,1"' "$TENSOR_BIN/llama-server" || {
  echo "ERROR tensor wrapper is not the 1,1 arm" >&2; exit 8;
}
grep -Fq -- '--fit off' "$TENSOR_BIN/llama-server" || {
  echo "ERROR tensor wrapper must force --fit off" >&2; exit 9;
}

# A wrapper's --version/--list-devices also carries split args, so verify the real
# ELF bundle directly. Both dirs must independently resolve local libraries and
# expose both gfx1100 and gfx1201 devices.
REQUIRE_BOTH_GPUS=1 bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$LAYER_BIN"
REQUIRE_BOTH_GPUS=1 bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$TENSOR_BIN"

echo "PHASE6_RUNTIME_AUDIT=PASS"
echo "REAL_BINARY_SHA256=$S_LAYER"
echo "LAYER_BIN=$LAYER_BIN"
echo "TENSOR_BIN=$TENSOR_BIN"
