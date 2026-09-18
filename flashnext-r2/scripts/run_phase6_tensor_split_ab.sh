#!/usr/bin/env bash
set -euo pipefail

# Phase-6 real-machine A/B: same patched qwen4exp binary, layer split vs tensor
# split 1,1. Tensor mode is experimental upstream, so a load/OOM/HTTP failure is
# a clean reject, not a reason to disturb production.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
LAYER="${LAYER:-qwen3.8-flash-next-r2-split-layer:256k}"
TENSOR="${TENSOR:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase6-tensor-split-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase6.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
for a in "$PROD_ALIAS" "$LAYER" "$TENSOR"; do
  require_alias "$a" || { echo "ERROR alias missing: $a" >&2; exit 2; }
done
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 3;
}

note "LAYER_ALIAS=$LAYER"
note "TENSOR_ALIAS=$TENSOR"
note "TENSOR_SPLIT=1,1"
note "TENSOR_FIT=off"

# Gate 1: cheap short/normal smoke. Upstream ROCm reports tensor mode can be
# within noise at short context, so this gate asks for exact greedy output and
# only rejects a catastrophic (>10%) throughput/PP regression.
SHORT="$RUN_DIR/short-layer-vs-tensor.json"
set +e
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$LAYER" --r2 "$TENSOR" \
  --rounds "${SHORT_ROUNDS:-4}" --repeat "${SHORT_REPEAT:-3}" \
  --tg "${SHORT_TG:-512}" --pp "${SHORT_PP:-512,2048}" --out "$SHORT"
RC_SHORT_RUN=$?
set -e
if [[ "$RC_SHORT_RUN" != 0 ]]; then
  note "SHORT_RUN=FAIL_LOAD_OR_BENCH"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
  note "PRODUCTION_PROMOTED=NO"
  echo "Tensor candidate failed to load or benchmark. Layer split remains the winner."
  exit 0
fi

set +e
python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$SHORT" \
  --baseline "$LAYER" --r2 "$TENSOR" --label phase6-tensor-short \
  --min-median-tg-gain "${SHORT_MIN_TG_GAIN:--10.0}" \
  --max-workload-tg-loss "${SHORT_MAX_TG_LOSS:-10.0}" \
  --max-median-pp-loss "${SHORT_MAX_PP_LOSS:-10.0}"
RC_SHORT=$?
set -e
note "SHORT_GATE_RC=$RC_SHORT"
if [[ "$RC_SHORT" != 0 ]]; then
  note "LONG_GATE=NOT_RUN"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
  note "PRODUCTION_PROMOTED=NO"
  echo "Tensor split rejected by short exactness/non-catastrophic gate."
  exit 0
fi

# Gate 2: this is the reason to test tensor mode at all. Require a real gain at
# depth, not an attractive one-off short TG sample. The ladder also carries a
# needle retrieval check at every depth.
LONG="$RUN_DIR/long-layer-vs-tensor.json"
set +e
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$LAYER" --r2 "$TENSOR" \
  --depths "${LONG_DEPTHS:-4096,32768,65536,131072}" \
  --rounds "${LONG_ROUNDS:-4}" --tg "${LONG_TG:-256}" --out "$LONG"
RC_LONG_RUN=$?
set -e
if [[ "$RC_LONG_RUN" != 0 ]]; then
  note "LONG_RUN=FAIL_LOAD_OR_BENCH"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
  note "PRODUCTION_PROMOTED=NO"
  echo "Tensor split failed during the long-context ladder. Layer split remains the winner."
  exit 0
fi

set +e
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" "$LONG" \
  --baseline "$LAYER" --r2 "$TENSOR" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_DEEP_GAIN:-2.0}" \
  --max-deep-loss "${MAX_DEEP_LOSS:-5.0}" \
  --max-pp-loss "${MAX_LONG_PP_LOSS:-10.0}"
RC_LONG=$?
set -e
note "LONG_GATE_RC=$RC_LONG"
if [[ "$RC_LONG" != 0 ]]; then
  note "CACHED_GATE=NOT_RUN"
  note "ROLLBACK_GATE=NOT_RUN"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
  note "PRODUCTION_PROMOTED=NO"
  echo "Tensor split did not deliver a safe >=2% deep-context median gain."
  exit 0
fi

# Gate 3: cached high-LCP path. This catches the historical Flash Next family of
# 'fresh prompt looks normal, reused prefix falls off a cliff' regressions.
CACHED="$RUN_DIR/cached-layer-vs-tensor.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$LAYER" --r2 "$TENSOR" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" \
  --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" \
  --repeats "${CACHED_REPEATS:-2}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-5.0}" \
  --out "$CACHED"
RC_CACHED=$?
set -e
if [[ "$RC_CACHED" != 0 ]]; then
  note "CACHED_GATE=REJECT"
  note "ROLLBACK_GATE=NOT_RUN"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
  note "PRODUCTION_PROMOTED=NO"
  exit 0
fi
note "CACHED_GATE=PASS"

# Gate 4: recurrent-state rollback with MTP. Tensor split moves weights and KV
# across both devices, so this is mandatory before considering it production-safe.
ROLL="$RUN_DIR/rollback-64k.json"
set +e
OFF="$LAYER" ON="$TENSOR" OUT="$ROLL" DEPTH="${ROLLBACK_DEPTH:-65536}" \
  N_PREDICT="${ROLLBACK_TG:-512}" COMPARE_FIRST="${ROLLBACK_COMPARE:-256}" \
  ROUNDS="${ROLLBACK_ROUNDS:-4}" bash "$SCRIPT_DIR/run_stage8_rollback_stress.sh"
RC_ROLL=$?
set -e
if [[ "$RC_ROLL" != 0 ]]; then
  note "ROLLBACK_GATE=REJECT"
  note "PHASE6_WINNER_ALIAS=$LAYER"
  note "PHASE6_WINNER_MODE=LAYER"
else
  note "ROLLBACK_GATE=PASS"
  note "PHASE6_WINNER_ALIAS=$TENSOR"
  note "PHASE6_WINNER_MODE=TENSOR_1x1"
fi

note "SHORT_RESULT=$SHORT"
note "LONG_RESULT=$LONG"
note "CACHED_RESULT=$CACHED"
note "ROLLBACK_RESULT=$ROLL"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"
require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-6 TENSOR-SPLIT A/B COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "No production promotion was performed. A heterogeneous 24G/32G ratio sweep is intentionally deferred until live device order and per-card headroom are recorded."
