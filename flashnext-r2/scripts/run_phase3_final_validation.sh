#!/usr/bin/env bash
set -euo pipefail

# Validate the composed pre-sweep candidate through the real llama-swap :8090 path.
# This is a non-regression/correctness gate before parameter sweeps. It still does
# not promote production.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
FINAL_ALIAS="${FINAL_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
FINAL_SRC="${FINAL_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919}"
FINAL_RUNTIME="${FINAL_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-final-pre-sweep/bin}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase3-validation-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/validation.log") 2>&1

note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias() {
  grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"
}

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
[[ -f "$FINAL_SRC/r2-meta/phase3-compose-manifest.txt" ]] || {
  echo "ERROR final compose manifest missing. Run prepare_phase3_final_candidate.sh first." >&2
  exit 3
}
[[ -x "$FINAL_RUNTIME/llama-server" ]] || {
  echo "ERROR final runtime missing: $FINAL_RUNTIME/llama-server" >&2; exit 4;
}
require_alias "$PROD_ALIAS" || { echo "ERROR production alias missing" >&2; exit 5; }
require_alias "$FINAL_ALIAS" || { echo "ERROR final candidate alias missing" >&2; exit 6; }
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 7;
}

# Gate -1: staged runtime itself. If PLE won, llama-server is a shell wrapper and
# the ELF is llama-server.real; ldd/readelf must inspect the real executable.
echo "=== Gate -1: final runtime bundle / dual-GPU audit ==="
AUDIT_SERVER="$FINAL_RUNTIME/llama-server"
[[ -x "$FINAL_RUNTIME/llama-server.real" ]] && AUDIT_SERVER="$FINAL_RUNTIME/llama-server.real"
SERVER="$AUDIT_SERVER" REQUIRE_BOTH_GPUS=1 \
  bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$FINAL_RUNTIME"
note "RUNTIME_BUNDLE=PASS"
note "RUNTIME_SERVER_SHA256=$(sha256sum "$AUDIT_SERVER" | awk '{print $1}')"

# Gate 0: native rollback survived composition.
echo "=== Gate 0: native qwen4exp recurrent rollback ==="
SRC="$FINAL_SRC" CONFIG="$CONFIG" ALIAS="$FINAL_ALIAS" \
  bash "$SCRIPT_DIR/verify_qwen4exp_native_rs_rollback.sh"
note "NATIVE_RS_ROLLBACK=PASS"

# 1) Short/normal workload, exact greedy output, PP/TG non-regression.
SHORT_JSON="$RUN_DIR/short-ab.json"
echo "=== Gate 1: short/normal PP+TG exact A/B ==="
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$PROD_ALIAS" --r2 "$FINAL_ALIAS" \
  --rounds "${SHORT_ROUNDS:-4}" --repeat "${SHORT_REPEAT:-3}" \
  --tg "${SHORT_TG:-512}" --pp "${SHORT_PP:-512,2048,8192}" \
  --out "$SHORT_JSON"
python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$SHORT_JSON" \
  --baseline "$PROD_ALIAS" --r2 "$FINAL_ALIAS" --label phase3-short \
  --min-median-tg-gain "${MIN_SHORT_TG_GAIN:--1.0}" \
  --max-workload-tg-loss "${MAX_SHORT_TG_LOSS:-2.0}" \
  --max-median-pp-loss "${MAX_SHORT_PP_LOSS:-3.0}" \
  --max-acceptance-drop-pp "${MAX_SHORT_ACC_DROP_PP:-3.0}"
note "SHORT_GATE=PASS"
note "SHORT_RESULT=$SHORT_JSON"

# 2) Full long-context ladder.
LONG_JSON="$RUN_DIR/long-context.json"
echo
echo "=== Gate 2: 4K/32K/64K/128K long-context ladder ==="
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$PROD_ALIAS" --r2 "$FINAL_ALIAS" \
  --depths "${LONG_DEPTHS:-4096,32768,65536,131072}" \
  --rounds "${LONG_ROUNDS:-4}" --tg "${LONG_TG:-256}" --out "$LONG_JSON"
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" "$LONG_JSON" \
  --baseline "$PROD_ALIAS" --r2 "$FINAL_ALIAS" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_LONG_GAIN:--1.5}" \
  --max-deep-loss "${MAX_LONG_LOSS:-3.0}" \
  --max-pp-loss "${MAX_LONG_PP_LOSS:-5.0}"
note "LONG_GATE=PASS"
note "LONG_RESULT=$LONG_JSON"

# 3) Cached Large-PP / high-LCP branch.
CACHED_JSON="$RUN_DIR/cached-largepp.json"
echo
echo "=== Gate 3: cached Large-PP / high-LCP ==="
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$PROD_ALIAS" --r2 "$FINAL_ALIAS" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" --repeats "${CACHED_REPEATS:-2}" \
  --absolute-tg-floor "${CACHED_TG_FLOOR:-5.0}" \
  --min-self-retention "${CACHED_SELF_RETENTION:-0.50}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-5.0}" --out "$CACHED_JSON"
note "CACHED_GATE=PASS"
note "CACHED_RESULT=$CACHED_JSON"

# 4) 64K rollback/checkpoint-style stress.
ROLLBACK_JSON="$RUN_DIR/rollback-stress.json"
echo
echo "=== Gate 4: long-context rollback/determinism stress ==="
python3 "$SCRIPT_DIR/bench_stage8_rollback_stress.py" \
  --off "$PROD_ALIAS" --on "$FINAL_ALIAS" \
  --depth "${ROLLBACK_DEPTH:-65536}" --n-predict "${ROLLBACK_TG:-512}" \
  --compare-first "${ROLLBACK_COMPARE:-256}" --rounds "${ROLLBACK_ROUNDS:-4}" \
  --out "$ROLLBACK_JSON"
note "ROLLBACK_GATE=PASS"
note "ROLLBACK_RESULT=$ROLLBACK_JSON"

require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }
note "PRODUCTION_ALIAS=$PROD_ALIAS"
note "FINAL_ALIAS=$FINAL_ALIAS"
note "PRODUCTION_PROMOTED=NO"
note "VALIDATION=PASS"
note "FINISHED=$(date -Is)"

echo
echo "================================================================"
echo "PHASE-3 FINAL PRE-SWEEP VALIDATION PASSED"
echo "================================================================"
cat "$SUMMARY"
echo
echo "The candidate is now eligible for MTP-depth/HIP-Graph/gfx1201 microfusion sweeps. Production is still untouched."
