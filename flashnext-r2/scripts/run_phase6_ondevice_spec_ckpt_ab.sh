#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
OFF="${OFF:-qwen3.8-flash-next-r2-ckpt-host:256k}"
ON="${ON:-qwen3.8-flash-next-r2-ckpt-device:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase6-ondevice-ckpt-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase6.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
for a in "$PROD_ALIAS" "$OFF" "$ON"; do
  require_alias "$a" || { echo "ERROR alias missing: $a" >&2; exit 2; }
done
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 3;
}

# Fresh-prompt A/B. Long generations amplify per-round checkpoint cost and are
# therefore more informative than tg32.
MAIN="$RUN_DIR/host-vs-device.json"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$OFF" --r2 "$ON" \
  --rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" \
  --tg "${TG:-1024}" --pp "${PP:-512}" --out "$MAIN"

set +e
python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$MAIN" \
  --baseline "$OFF" --r2 "$ON" --label phase6-ondevice-ckpt \
  --min-median-tg-gain "${MIN_TG_GAIN:-3.0}" \
  --max-workload-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-median-pp-loss "${MAX_PP_LOSS:-2.0}"
RC_MAIN=$?
set -e
note "FRESH_AB_RC=$RC_MAIN"
if [[ "$RC_MAIN" != 0 ]]; then
  note "CACHED_GATE=NOT_RUN"
  note "ROLLBACK_GATE=NOT_RUN"
  note "PHASE6_RESULT=REJECT"
  note "PROMOTION_ELIGIBLE=NO"
  note "PRODUCTION_PROMOTED=NO"
  echo "Phase-6 rejected at the fresh exact-output/TG gate."
  exit 0
fi

# High-LCP cached branch. Besides speed, an HTTP/model failure here is useful: it
# can expose checkpoint invalidation/range issues before the deeper rollback run.
CACHED="$RUN_DIR/cached-largepp.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$OFF" --r2 "$ON" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" \
  --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" \
  --repeats "${CACHED_REPEATS:-2}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-3.0}" \
  --out "$CACHED"
RC_CACHED=$?
set -e
if [[ "$RC_CACHED" != 0 ]]; then
  note "CACHED_GATE=REJECT"
  note "ROLLBACK_GATE=NOT_RUN"
  note "PHASE6_RESULT=REJECT"
  note "PROMOTION_ELIGIBLE=NO"
  note "PRODUCTION_PROMOTED=NO"
  exit 0
fi
note "CACHED_GATE=PASS"

# 64K sequence-removal/replay stress. This still cannot prove every possible
# fragmented recurrent-cache topology, but a failure is an immediate hard reject.
ROLL="$RUN_DIR/rollback-64k.json"
set +e
OFF="$OFF" ON="$ON" OUT="$ROLL" DEPTH="${ROLLBACK_DEPTH:-65536}" \
  N_PREDICT="${ROLLBACK_TG:-512}" COMPARE_FIRST="${ROLLBACK_COMPARE:-256}" \
  ROUNDS="${ROLLBACK_ROUNDS:-4}" bash "$SCRIPT_DIR/run_stage8_rollback_stress.sh"
RC_ROLL=$?
set -e
if [[ "$RC_ROLL" != 0 ]]; then
  note "ROLLBACK_GATE=REJECT"
  note "PHASE6_RESULT=REJECT"
  note "PROMOTION_ELIGIBLE=NO"
else
  note "ROLLBACK_GATE=PASS"
  note "PHASE6_RESULT=PERF_PASS_EXPERIMENTAL"
  # Deliberately NO: upstream PR28118 itself documents a multi-range hard abort.
  note "PROMOTION_ELIGIBLE=NO_KNOWN_MULTIRANGE_ABORT"
fi

note "MAIN_RESULT=$MAIN"
note "CACHED_RESULT=$CACHED"
note "ROLLBACK_RESULT=$ROLL"
note "FORCED_PARALLEL=1"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"
require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-6 ON-DEVICE SPEC CHECKPOINT COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "Even a performance PASS is experimental only until the multi-range recurrent-state abort has a safe fallback."
