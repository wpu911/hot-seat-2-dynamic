#!/usr/bin/env bash
set -euo pipefail

# Stage 7A: verify that the production source/build already contains the merged
# ROCm long-row TOP_K path. PR #27466 merged on 2026-08-31, so a 2026-09-11
# production base should already have it. Do not re-patch what is already there.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
RUNTIME_BIN="${RUNTIME_BIN:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/20260911-02/bin}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/flashnext-stage7-topk-verify-$STAMP.log"

exec > >(tee "$LOG") 2>&1

echo "=== Flash Next Stage 7A ROCm TOP_K verify ==="
echo "date=$(date -Is)"
echo "source=$PROD_SRC"
echo "runtime=$RUNTIME_BIN"

test -e "$PROD_SRC/.git" || { echo "ERROR: not a git worktree: $PROD_SRC"; exit 2; }

echo
printf 'HEAD='; git -C "$PROD_SRC" rev-parse HEAD
printf 'STATUS='; git -C "$PROD_SRC" status --short || true

echo
echo "=== source probes ==="
# PR #27466 added radix TOP_K support in ggml-cuda TOP_K sources. We probe for
# the relevant implementation names and for the old hard <=1024-only gate.
grep -RniE 'radix.*top.?k|top.?k.*radix|radix_select|top_k_radix' \
  "$PROD_SRC/ggml/src/ggml-cuda" | head -n 80 || true

echo
echo "=== old <=1024-only support gates, if any ==="
grep -RniE 'TOP_K|top_k' "$PROD_SRC/ggml/src/ggml-cuda" \
  | grep -E '1024|supports_op' | head -n 120 || true

echo
echo "=== compile/runtime identity ==="
if [[ -x "$RUNTIME_BIN/llama-server" ]]; then
  "$RUNTIME_BIN/llama-server" --version || true
  sha256sum "$RUNTIME_BIN/llama-server"
else
  echo "WARNING: production llama-server not found at $RUNTIME_BIN/llama-server"
fi

echo
echo "=== backend TOP_K test ==="
TEST_BIN=""
for p in \
  "$PROD_SRC/build/bin/test-backend-ops" \
  "$PROD_SRC/build-hip/bin/test-backend-ops" \
  "$PROD_SRC/build-hip-rocm10/bin/test-backend-ops" \
  "$PROD_SRC/build-hip-rocm714/bin/test-backend-ops"; do
  if [[ -x "$p" ]]; then TEST_BIN="$p"; break; fi
done

if [[ -n "$TEST_BIN" ]]; then
  echo "test_bin=$TEST_BIN"
  # Backend names differ across builds. Try ROCm0 first, then HIP0, then generic.
  if "$TEST_BIN" test -o TOP_K -b ROCm0; then
    echo "TOPK_TEST=PASS backend=ROCm0"
  elif "$TEST_BIN" test -o TOP_K -b HIP0; then
    echo "TOPK_TEST=PASS backend=HIP0"
  elif "$TEST_BIN" test -o TOP_K; then
    echo "TOPK_TEST=PASS backend=auto"
  else
    echo "TOPK_TEST=FAIL"
    exit 10
  fi
else
  echo "TOPK_TEST=SKIP no test-backend-ops binary found in known build dirs"
  echo "This is not a failure by itself; Stage 7B runtime build will provide one."
fi

echo
echo "=== verdict ==="
if grep -RqiE 'radix.*top.?k|top.?k.*radix|radix_select|top_k_radix' "$PROD_SRC/ggml/src/ggml-cuda"; then
  echo "SOURCE_RADIX_TOPK=PRESENT"
else
  echo "SOURCE_RADIX_TOPK=NOT_CONFIRMED"
  echo "Do not patch production automatically. Inspect current ggml-cuda TOP_K implementation first."
  exit 20
fi

echo "LOG=$LOG"
