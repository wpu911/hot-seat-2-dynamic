#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
PHASE6_SRC="${PHASE6_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
EVEN_ALIAS="${EVEN_ALIAS:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
MID_ALIAS="${MID_ALIAS:-qwen3.8-flash-next-r2-split-tensor-mid:256k}"
CAP_ALIAS="${CAP_ALIAS:-qwen3.8-flash-next-r2-split-tensor-cap:256k}"
INV_MID_ALIAS="${INV_MID_ALIAS:-qwen3.8-flash-next-r2-split-tensor-inv-mid:256k}"
INV_CAP_ALIAS="${INV_CAP_ALIAS:-qwen3.8-flash-next-r2-split-tensor-inv-cap:256k}"
META="$PHASE6_SRC/r2-meta/phase6b-ratios.env"

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
[[ -f "$META" ]] || { echo "ERROR ratio manifest missing: $META" >&2; exit 3; }
# shellcheck disable=SC1090
source "$META"

for a in "$EVEN_ALIAS" "$MID_ALIAS" "$CAP_ALIAS" "$INV_MID_ALIAS" "$INV_CAP_ALIAS"; do
  grep -qE "^[[:space:]]*${a//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR alias missing: $a" >&2; exit 4;
  }
done

names=(tensor-1x1 tensor-mid tensor-cap tensor-inv-mid tensor-inv-cap)
ratios=("${EVEN_RATIO:-1,1}" "$MID_RATIO" "$CAP_RATIO" "$INV_MID_RATIO" "$INV_CAP_RATIO")
base_sha=""
for i in "${!names[@]}"; do
  bin="$RUNTIME_ROOT/${names[$i]}/bin"
  [[ -x "$bin/llama-server" && -x "$bin/llama-server.real" ]] || {
    echo "ERROR incomplete ratio runtime: $bin" >&2; exit 5;
  }
  s="$(sha256sum "$bin/llama-server.real" | awk '{print $1}')"
  if [[ -z "$base_sha" ]]; then base_sha="$s"; fi
  [[ "$s" == "$base_sha" ]] || {
    echo "ERROR ratio arms do not share the same real ELF: ${names[$i]}=$s base=$base_sha" >&2
    exit 6
  }
  grep -Fq -- '--split-mode tensor' "$bin/llama-server" || {
    echo "ERROR ${names[$i]} wrapper does not force tensor mode" >&2; exit 8;
  }
  grep -Fq -- "--tensor-split \"${ratios[$i]}\"" "$bin/llama-server" || {
    echo "ERROR ${names[$i]} ratio mismatch, expected ${ratios[$i]}" >&2; exit 9;
  }
  grep -Fq -- '--fit off' "$bin/llama-server" || {
    echo "ERROR ${names[$i]} wrapper must force --fit off" >&2; exit 10;
  }
  SERVER="$bin/llama-server.real" REQUIRE_BOTH_GPUS=1 \
    bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$bin"
done

[[ -z "${REAL_BINARY_SHA256:-}" || "$base_sha" == "$REAL_BINARY_SHA256" ]] || {
  echo "ERROR ratio manifest ELF sha mismatch: manifest=$REAL_BINARY_SHA256 actual=$base_sha" >&2
  exit 7
}

echo "PHASE6B_RUNTIME_AUDIT=PASS"
echo "REAL_BINARY_SHA256=$base_sha"
echo "DEVICE_ORDER=${DEVICE0:-unknown},${DEVICE1:-unknown}"
echo "RATIOS=${EVEN_RATIO:-1,1};$MID_RATIO;$CAP_RATIO;$INV_MID_RATIO;$INV_CAP_RATIO"
