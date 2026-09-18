#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
OFF="${OFF:-qwen3.8-flash-next-r2-gdn-modern-off:256k}"
PROLOG="${PROLOG:-qwen3.8-flash-next-r2-gdn-prolog:256k}"
FULL="${FULL:-qwen3.8-flash-next-r2-gdn-prolog-l2:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase5-gdn-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase5.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
for a in "$PROD_ALIAS" "$OFF" "$PROLOG" "$FULL"; do
  require_alias "$a" || { echo "ERROR alias missing: $a" >&2; exit 2; }
done
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 3;
}

COMMON=(--rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" --tg "${TG:-512}" --pp "${PP:-512,2048}")
R_OP="$RUN_DIR/off-vs-prolog.json"
R_OF="$RUN_DIR/off-vs-full.json"
R_PF="$RUN_DIR/prolog-vs-full.json"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$OFF" --r2 "$PROLOG" "${COMMON[@]}" --out "$R_OP"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$OFF" --r2 "$FULL"   "${COMMON[@]}" --out "$R_OF"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$PROLOG" --r2 "$FULL" "${COMMON[@]}" --out "$R_PF"

analyze_pair(){
  local result="$1" base="$2" cand="$3" label="$4" var="$5"
  set +e
  python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$result" \
    --baseline "$base" --r2 "$cand" --label "$label" \
    --min-median-tg-gain "${MIN_TG_GAIN:-0.5}" \
    --max-workload-tg-loss "${MAX_TG_LOSS:-1.5}" \
    --max-median-pp-loss "${MAX_PP_LOSS:-2.0}"
  local rc=$?
  set -e
  printf -v "$var" '%s' "$rc"
}
analyze_pair "$R_OP" "$OFF" "$PROLOG" phase5-prolog RC_PROLOG
analyze_pair "$R_OF" "$OFF" "$FULL"   phase5-full   RC_FULL
analyze_pair "$R_PF" "$PROLOG" "$FULL" phase5-l2-increment RC_L2

note "PROLOG_AB_RC=$RC_PROLOG"
note "FULL_AB_RC=$RC_FULL"
note "L2_INCREMENT_AB_RC=$RC_L2"

# Prefer the complete prolog+L2 path only if it beats OFF directly. If it does
# not, a prolog-only win may still survive. No candidate earns a win from an
# indirect comparison alone.
if [[ "$RC_FULL" == 0 ]]; then
  CAND="$FULL"
  MODE="PROLOG_L2"
elif [[ "$RC_PROLOG" == 0 ]]; then
  CAND="$PROLOG"
  MODE="PROLOG_ONLY"
else
  CAND="$OFF"
  MODE="OFF"
fi
note "THROUGHPUT_CANDIDATE=$CAND"
note "THROUGHPUT_MODE=$MODE"

if [[ "$CAND" == "$OFF" ]]; then
  note "CACHED_GATE=SKIP"
  note "ROLLBACK_GATE=SKIP"
  note "PHASE5_WINNER_ALIAS=$OFF"
  note "PHASE5_WINNER_MODE=OFF"
  note "PRODUCTION_PROMOTED=NO"
  echo "Phase-5 result: no GDN microfusion survived the throughput/exactness gate."
  exit 0
fi

# Recurrent-operator changes must survive the two failure modes that mattered most
# in this deployment: cached high-LCP speculative decode and recurrent rollback.
CACHED="$RUN_DIR/cached-largepp.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$OFF" --r2 "$CAND" \
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
  note "PHASE5_WINNER_ALIAS=$OFF"
  note "PHASE5_WINNER_MODE=OFF"
  note "PRODUCTION_PROMOTED=NO"
  echo "Phase-5 candidate rejected by cached Large-PP/high-LCP gate."
  exit 0
fi
note "CACHED_GATE=PASS"

ROLL="$RUN_DIR/rollback-64k.json"
set +e
OFF="$OFF" ON="$CAND" OUT="$ROLL" DEPTH="${ROLLBACK_DEPTH:-65536}" \
  N_PREDICT="${ROLLBACK_TG:-512}" COMPARE_FIRST="${ROLLBACK_COMPARE:-256}" \
  ROUNDS="${ROLLBACK_ROUNDS:-4}" bash "$SCRIPT_DIR/run_stage8_rollback_stress.sh"
RC_ROLL=$?
set -e
if [[ "$RC_ROLL" != 0 ]]; then
  note "ROLLBACK_GATE=REJECT"
  note "PHASE5_WINNER_ALIAS=$OFF"
  note "PHASE5_WINNER_MODE=OFF"
else
  note "ROLLBACK_GATE=PASS"
  note "PHASE5_WINNER_ALIAS=$CAND"
  note "PHASE5_WINNER_MODE=$MODE"
fi
note "OFF_VS_PROLOG=$R_OP"
note "OFF_VS_FULL=$R_OF"
note "PROLOG_VS_FULL=$R_PF"
note "CACHED_RESULT=$CACHED"
note "ROLLBACK_RESULT=$ROLL"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"

require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-5 GDN MICROFUSION COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "No production promotion was performed."
