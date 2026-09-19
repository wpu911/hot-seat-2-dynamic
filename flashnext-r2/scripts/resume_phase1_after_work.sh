#!/usr/bin/env bash
set -euo pipefail

# Resume the real-machine R2 work after an interrupted ChatGPT Work run.
# Assumption: Modern Foundation has already been built and registered in llama-swap.
# This script never promotes or rewrites the production alias.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
FOUNDATION_ALIAS="${FOUNDATION_ALIAS:-qwen3.8-flash-next-r2-modern-foundation:256k}"
FOUNDATION_SRC="${FOUNDATION_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"
FOUNDATION_RUNTIME="${FOUNDATION_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-foundation}"
FOUNDATION_RESULT="${FOUNDATION_RESULT:-$LOG_DIR/flashnext-r2-modern-foundation-ab.json}"
FOUNDATION_ANALYSIS="${FOUNDATION_ANALYSIS:-${FOUNDATION_RESULT%.json}.foundation-analysis.json}"

require_alias() {
  local a="$1"
  grep -qE "^[[:space:]]*${a//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR: llama-swap alias missing: $a" >&2
    exit 2
  }
}

[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 3; }
[[ -e "$FOUNDATION_SRC/.git" ]] || { echo "ERROR: foundation source missing: $FOUNDATION_SRC" >&2; exit 4; }
[[ -f "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt" ]] || {
  echo "ERROR: foundation manifest missing; this is not a resumable prepared candidate" >&2
  exit 5
}
[[ -x "$FOUNDATION_RUNTIME/llama-server" ]] || {
  echo "ERROR: foundation runtime missing: $FOUNDATION_RUNTIME/llama-server" >&2
  exit 5
}
require_alias "$PROD_ALIAS"
require_alias "$FOUNDATION_ALIAS"

curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models >/dev/null || {
  echo "ERROR: llama-swap :8090 unhealthy" >&2
  exit 6
}

# Runtime packaging is part of correctness. A binary that quietly resolves local
# libllama/libggml from the disposable CMake build tree is not a real independent
# experiment runtime and may die after cleanup.
REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
  OUT="$LOG_DIR/flashnext-r2-foundation-runtime-verify.log" \
  bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$FOUNDATION_RUNTIME"

# Re-run the two source/runtime gates before trusting an experiment prepared by an
# interrupted session. This is cheap compared with loading the 190+ GiB model.
bash "$SCRIPT_DIR/verify_modern_foundation_carryover.sh"
env SRC="$FOUNDATION_SRC" CONFIG="$CONFIG" ALIAS="$FOUNDATION_ALIAS" \
  bash "$SCRIPT_DIR/verify_qwen4exp_native_rs_rollback.sh"

# Never trust a stale analysis file on its own. Work may have produced it with an
# older benchmark that allowed early EOS or failed to read timings.draft_n. Re-run
# the CURRENT analyzer over the raw JSON. Legacy raw results intentionally fail the
# fixed_tg schema gate and trigger one fresh A/B instead of being silently reused.
revalidate_existing_result() {
  [[ -f "$FOUNDATION_RESULT" ]] || return 1
  echo "FOUNDATION_AB_REVALIDATE=$FOUNDATION_RESULT"
  set +e
  python3 "$SCRIPT_DIR/analyze_modern_foundation.py" \
    "$FOUNDATION_RESULT" --baseline "$PROD_ALIAS" --foundation "$FOUNDATION_ALIAS" \
    --max-median-tg-loss "${MAX_MEDIAN_TG_LOSS:-2.0}" \
    --max-workload-tg-loss "${MAX_WORKLOAD_TG_LOSS:-3.0}" \
    --max-median-pp-loss "${MAX_MEDIAN_PP_LOSS:-3.0}" \
    --max-acceptance-drop-pp "${MAX_ACCEPTANCE_DROP_PP:-2.0}"
  local rc=$?
  set -e
  return "$rc"
}

if revalidate_existing_result; then
  echo "FOUNDATION_AB_REUSE=PASS"
  echo "analysis=$FOUNDATION_ANALYSIS"
else
  if [[ -f "$FOUNDATION_RESULT" ]]; then
    STALE="$FOUNDATION_RESULT.stale-$(date +%Y%m%d-%H%M%S)"
    cp -a "$FOUNDATION_RESULT" "$STALE"
    echo "FOUNDATION_AB_OLD_RESULT_SAVED=$STALE"
  fi
  echo "FOUNDATION_AB_REUSE=NO"
  echo "Running a fresh fixed-length foundation A/B through llama-swap :8090"
  OUT="$FOUNDATION_RESULT" bash "$SCRIPT_DIR/run_modern_foundation_ab.sh"
  revalidate_existing_result || {
    echo "ERROR: Modern Foundation did not pass the current gate; stop before MTP." >&2
    exit 10
  }
fi

# Foundation is now known-good. Continue only the uncompleted Phase-1 stages.
# START_AT=mtp deliberately avoids rebuilding the snapshot/foundation that Work
# already prepared. The phase runner still performs preflight, config backup and
# rollback audit before touching the MTP candidate.
START_AT=mtp STOP_AFTER=topk TOPK_FULL="${TOPK_FULL:-0}" \
  bash "$SCRIPT_DIR/run_phase1_real_ab.sh"

echo
printf '%s\n' \
  "RESUME_PHASE1_COMPLETE=1" \
  "PRODUCTION_PROMOTED=NO" \
  "NEXT=inspect latest flashnext-r2-phase1-*/summary.env, then run Phase-2 only from the recorded winner"
