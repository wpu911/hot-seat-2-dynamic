#!/usr/bin/env bash
set -euo pipefail

# Phase-4b: deep validation of the MTP-depth/HIP-Graph winner before any GDN
# source experiment is layered on top. Phase-4 is intentionally cheap enough to
# tune parameters; this gate proves the chosen parameter set survives depth,
# cached-prefix reuse and recurrent rollback.
#
# Failure is not fatal to the R2 program. It rejects the Phase-4 parameter arm
# and records FINAL_ALIAS (the already Phase-3-validated candidate) as the safe
# winner for Phase-5. Production is never promoted.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
FINAL_ALIAS="${FINAL_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase4b-param-validation-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase4b.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
summary_value(){ awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"; }
latest_phase4_summary(){
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase4-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
finish_fallback(){
  local why="$1"
  note "PARAM_ACCEPTED=NO"
  note "FALLBACK_REASON=$why"
  note "PHASE4B_WINNER_ALIAS=$FINAL_ALIAS"
  note "VALIDATION=PASS"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  echo
  echo "Phase-4 parameter candidate rejected: $why"
  echo "Phase-5 base falls back to the already validated final alias: $FINAL_ALIAS"
  exit 0
}

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
require_alias "$PROD_ALIAS" || { echo "ERROR production alias missing" >&2; exit 3; }
require_alias "$FINAL_ALIAS" || { echo "ERROR final alias missing" >&2; exit 4; }
P4="${PHASE4_SUMMARY:-$(latest_phase4_summary || true)}"
[[ -n "$P4" && -f "$P4" ]] || { echo "ERROR Phase-4 summary missing" >&2; exit 5; }
[[ "$(summary_value "$P4" PRODUCTION_PROMOTED)" == NO ]] || {
  echo "ERROR Phase-4 summary does not prove production remained untouched" >&2; exit 6;
}
PARAM_ALIAS="${PARAM_ALIAS:-$(summary_value "$P4" PARAM_WINNER_ALIAS)}"
[[ -n "$PARAM_ALIAS" ]] || { echo "ERROR PARAM_WINNER_ALIAS missing" >&2; exit 7; }
require_alias "$PARAM_ALIAS" || { echo "ERROR Phase-4 winner alias missing: $PARAM_ALIAS" >&2; exit 8; }
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 9;
}

note "PHASE4_SUMMARY=$P4"
note "PARAM_ALIAS=$PARAM_ALIAS"
note "BASELINE_ALIAS=$FINAL_ALIAS"

# Gate 1: the tuning winner has to retain the 32K/64K/128K slope. A small
# aggregate loss is tolerated because Graph/MTP depth can trade tiny short-run
# noise, but a real depth regression is rejected.
LONG="$RUN_DIR/long-final-vs-param.json"
set +e
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$FINAL_ALIAS" --r2 "$PARAM_ALIAS" \
  --depths "${LONG_DEPTHS:-32768,65536,131072}" \
  --rounds "${LONG_ROUNDS:-4}" --tg "${LONG_TG:-256}" --out "$LONG"
RC_LONG_RUN=$?
set -e
[[ "$RC_LONG_RUN" == 0 ]] || finish_fallback "LONG_BENCH_FAILED"

set +e
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" "$LONG" \
  --baseline "$FINAL_ALIAS" --r2 "$PARAM_ALIAS" \
  --deep-from 32768 \
  --min-deep-median-gain "${MIN_DEEP_GAIN:--1.0}" \
  --max-deep-loss "${MAX_DEEP_LOSS:-3.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-5.0}" \
  --max-acceptance-drop-pp "${MAX_ACC_DROP_PP:-3.0}"
RC_LONG=$?
set -e
[[ "$RC_LONG" == 0 ]] || finish_fallback "LONG_GATE_REJECT"
note "LONG_GATE=PASS"
note "LONG_RESULT=$LONG"

# Gate 2: historical high-LCP/cached Large-PP failure family. This is mandatory
# for every speculative-depth winner because a fresh-prompt TG win can hide a
# disastrous cache branch.
CACHED="$RUN_DIR/cached-final-vs-param.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$FINAL_ALIAS" --r2 "$PARAM_ALIAS" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" \
  --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" \
  --repeats "${CACHED_REPEATS:-2}" \
  --absolute-tg-floor "${CACHED_TG_FLOOR:-5.0}" \
  --min-self-retention "${CACHED_SELF_RETENTION:-0.50}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-3.0}" \
  --max-acceptance-drop-pp "${CACHED_MAX_ACC_DROP_PP:-5.0}" \
  --out "$CACHED"
RC_CACHED=$?
set -e
[[ "$RC_CACHED" == 0 ]] || finish_fallback "CACHED_LARGEPP_REJECT"
note "CACHED_GATE=PASS"
note "CACHED_RESULT=$CACHED"

# Gate 3: 64K recurrent rollback with MTP. This catches parameter combinations
# that look fast until a reject/seq_rm path is exercised.
ROLL="$RUN_DIR/rollback-final-vs-param.json"
set +e
python3 "$SCRIPT_DIR/bench_stage8_rollback_stress.py" \
  --off "$FINAL_ALIAS" --on "$PARAM_ALIAS" \
  --depth "${ROLLBACK_DEPTH:-65536}" \
  --n-predict "${ROLLBACK_TG:-512}" \
  --compare-first "${ROLLBACK_COMPARE:-256}" \
  --rounds "${ROLLBACK_ROUNDS:-4}" \
  --out "$ROLL"
RC_ROLL=$?
set -e
[[ "$RC_ROLL" == 0 ]] || finish_fallback "ROLLBACK_REJECT"
note "ROLLBACK_GATE=PASS"
note "ROLLBACK_RESULT=$ROLL"

require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }
note "PARAM_ACCEPTED=YES"
note "PHASE4B_WINNER_ALIAS=$PARAM_ALIAS"
note "VALIDATION=PASS"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-4B PARAMETER VALIDATION PASSED"
echo "================================================================"
cat "$SUMMARY"
