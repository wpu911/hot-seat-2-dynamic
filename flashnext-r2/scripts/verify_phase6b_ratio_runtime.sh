#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
PHASE6_SRC="${PHASE6_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
EVEN_ALIAS="${EVEN_ALIAS:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
MID_ALIAS="${MID_ALIAS:-qwen3.8-flash-next-r2-split-tensor-mid:256k}"
CAP_ALIAS="${CAP_ALIAS:-qwen3.8-flash-next-r2-split-tensor-cap:256k}"
META="$PHASE6_SRC/r2-meta/phase6b-ratios.env"

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
[[ -f "$META" ]] || { echo "ERROR ratio manifest missing: $META" >&2; exit 3; }
# shellcheck disable=SC1090
source "$META"

for a in "$EVEN_ALIAS" "$MID_ALIAS" "$CAP_ALIAS"; do
  grep -qE "^[[:space:]]*${a//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR alias missing: $a" >&2; exit 4;
  }
done

EVEN_BIN="$RUNTIME_ROOT/tensor-1x1/bin"
MID_BIN="$RUNTIME_ROOT/tensor-mid/bin"
CAP_BIN="$RUNTIME_ROOT/tensor-cap/bin"
for d in "$EVEN_BIN" "$MID_BIN" "$CAP_BIN"; do
  [[ -x "$d/llama-server" && -x "$d/llama-server.real" ]] || {
    echo "ERROR incomplete ratio runtime: $d" >&2; exit 5;
  }
done

S0="$(sha256sum "$EVEN_BIN/llama-server.real" | awk '{print $1}')"
S1="$(sha256sum "$MID_BIN/llama-server.real" | awk '{print $1}')"
S2="$(sha256sum "$CAP_BIN/llama-server.real" | awk '{print $1}')"
[[ "$S0" == "$S1" && "$S0" == "$S2" ]] || {
  echo "ERROR ratio arms do not share the same real ELF" >&2
  echo "even=$S0 mid=$S1 cap=$S2" >&2
  exit 6
}
[[ -z "${REAL_BINARY_SHA256:-}" || "$S0" == "$REAL_BINARY_SHA256" ]] || {
  echo "ERROR ratio manifest ELF sha mismatch: manifest=$REAL_BINARY_SHA256 actual=$S0" >&2
  exit 7
}

check_wrapper() {
  local bin="$1" ratio="$2" label="$3"
  grep -Fq -- '--split-mode tensor' "$bin/llama-server" || {
    echo "ERROR $label wrapper does not force tensor mode" >&2; exit 8;
  }
  grep -Fq -- "--tensor-split \"$ratio\"" "$bin/llama-server" || {
    echo "ERROR $label wrapper ratio mismatch, expected $ratio" >&2; exit 9;
  }
  grep -Fq -- '--fit off' "$bin/llama-server" || {
    echo "ERROR $label wrapper must force --fit off" >&2; exit 10;
  }
  SERVER="$bin/llama-server.real" REQUIRE_BOTH_GPUS=1 \
    bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$bin"
}

check_wrapper "$EVEN_BIN" "${EVEN_RATIO:-1,1}" even
check_wrapper "$MID_BIN" "$MID_RATIO" mid
check_wrapper "$CAP_BIN" "$CAP_RATIO" cap

echo "PHASE6B_RUNTIME_AUDIT=PASS"
echo "REAL_BINARY_SHA256=$S0"
echo "DEVICE_ORDER=${DEVICE0:-unknown},${DEVICE1:-unknown}"
echo "RATIOS=${EVEN_RATIO:-1,1};$MID_RATIO;$CAP_RATIO"
