#!/usr/bin/env bash
set -euo pipefail

# Phase-4 parameter sweep on the fully composed/validated pre-sweep runtime.
#
# Modern Foundation already has RDNA3/RDNA4 ncols_opt MoE tile selection and
# broadcast Q8 activation dedup. Therefore this sweep deliberately does NOT
# resurrect the old manual JMAX or Q8-dedup environment experiments.
#
# Sweep order:
#   1) MTP n-max 2 / 3 / 4 with all else identical
#   2) cached Large-PP regression on the throughput winner
#   3) HIP Graph ON / OFF with the accepted MTP winner
#
# No production promotion.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
FINAL_ALIAS="${FINAL_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
FINAL_RUNTIME="${FINAL_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-final-pre-sweep/bin}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase4-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase4.log") 2>&1

note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias() {
  grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"
}
latest_validation_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase3-validation-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value() {
  awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"
}

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 2; }
[[ -x "$FINAL_RUNTIME/llama-server" ]] || { echo "ERROR final runtime missing: $FINAL_RUNTIME" >&2; exit 3; }
require_alias "$FINAL_ALIAS" || { echo "ERROR final pre-sweep alias missing" >&2; exit 4; }
require_alias "$PROD_ALIAS" || { echo "ERROR production alias missing" >&2; exit 5; }

VALIDATION_SUMMARY="${VALIDATION_SUMMARY:-$(latest_validation_summary || true)}"
[[ -n "$VALIDATION_SUMMARY" && -f "$VALIDATION_SUMMARY" ]] || {
  echo "ERROR Phase-3 validation summary not found." >&2
  exit 6
}
[[ "$(summary_value "$VALIDATION_SUMMARY" VALIDATION)" == PASS ]] || {
  echo "ERROR final pre-sweep candidate has not passed Phase-3 validation." >&2
  exit 7
}
note "VALIDATION_SUMMARY=$VALIDATION_SUMMARY"

# Ensure modern carry-over is understood before tuning. This prevents an old
# experiment script from accidentally reintroducing superseded kernel patches.
if [[ -x "$SCRIPT_DIR/verify_modern_kernel_carryover.sh" || -f "$SCRIPT_DIR/verify_modern_kernel_carryover.sh" ]]; then
  SRC="${FOUNDATION_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}" \
    bash "$SCRIPT_DIR/verify_modern_kernel_carryover.sh"
fi

# ---------------- MTP depth 2/3/4 ----------------
A2="qwen3.8-flash-next-r2-final-mtp2:256k"
A3="qwen3.8-flash-next-r2-final-mtp3:256k"
A4="qwen3.8-flash-next-r2-final-mtp4:256k"

make_mtp_alias() {
  local alias="$1" n="$2" validate="$3"
  args=(python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py"
    --config "$CONFIG"
    --source-alias "$FINAL_ALIAS"
    --alias "$alias"
    --r2-bin "$FINAL_RUNTIME"
    --jmax keep
    --spec-draft-n-max "$n"
    --replace)
  [[ "$validate" == 1 ]] && args+=(--validate)
  "${args[@]}"
}
make_mtp_alias "$A2" 2 0
make_mtp_alias "$A3" 3 0
make_mtp_alias "$A4" 4 1

COMMON=(--rounds "${MTP_ROUNDS:-4}" --repeat "${MTP_REPEAT:-3}" --tg "${MTP_TG:-512}" --pp "${MTP_PP:-512}")
R23="$RUN_DIR/mtp2-vs3.json"
R24="$RUN_DIR/mtp2-vs4.json"
R34="$RUN_DIR/mtp3-vs4.json"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A2" --r2 "$A3" "${COMMON[@]}" --out "$R23"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A2" --r2 "$A4" "${COMMON[@]}" --out "$R24"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A3" --r2 "$A4" "${COMMON[@]}" --out "$R34"
MTP_ANALYSIS="$RUN_DIR/mtp-analysis.json"
python3 "$SCRIPT_DIR/analyze_mtp_sweep.py" "$R23" "$R24" "$R34" --out "$MTP_ANALYSIS"

MTP_WINNER="$(python3 - "$MTP_ANALYSIS" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
for r in j.get('ranking',[]):
    if r.get('exact') and isinstance(r.get('median_tg'),(int,float)):
        print(r['model']); break
else:
    raise SystemExit('ERROR: no exact MTP sweep winner')
PY
)"
note "MTP_THROUGHPUT_WINNER=$MTP_WINNER"
note "MTP_ANALYSIS=$MTP_ANALYSIS"

# Speculative depth must survive cached-prefix/high-LCP behavior, not merely fresh
# TG. If the throughput winner fails, fall back to the already Phase-3-validated
# final alias rather than promoting a flashy fresh-prompt regression.
MTP_CACHED="$RUN_DIR/mtp-winner-cached.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$FINAL_ALIAS" \
  --r2 "$MTP_WINNER" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" \
  --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" \
  --repeats "${CACHED_REPEATS:-2}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-5.0}" \
  --out "$MTP_CACHED"
MTP_CACHED_RC=$?
set -e
if [[ "$MTP_CACHED_RC" == 0 ]]; then
  MTP_ACCEPTED="$MTP_WINNER"
  note "MTP_CACHED_GATE=PASS"
else
  MTP_ACCEPTED="$FINAL_ALIAS"
  note "MTP_CACHED_GATE=REJECT_WINNER_FALLBACK_FINAL"
fi
note "MTP_ACCEPTED_ALIAS=$MTP_ACCEPTED"
note "MTP_CACHED_RESULT=$MTP_CACHED"

# ---------------- HIP Graph ON/OFF ----------------
GRAPH_ON="qwen3.8-flash-next-r2-final-graph-on:256k"
GRAPH_OFF="qwen3.8-flash-next-r2-final-graph-off:256k"
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$MTP_ACCEPTED" \
  --alias "$GRAPH_ON" --r2-bin "$FINAL_RUNTIME" --jmax keep \
  --unset-env GGML_CUDA_DISABLE_GRAPHS --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$MTP_ACCEPTED" \
  --alias "$GRAPH_OFF" --r2-bin "$FINAL_RUNTIME" --jmax keep \
  --env GGML_CUDA_DISABLE_GRAPHS=1 --replace --validate

GRAPH_JSON="$RUN_DIR/graph-on-vs-off.json"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$GRAPH_ON" --r2 "$GRAPH_OFF" \
  --rounds "${GRAPH_ROUNDS:-4}" --repeat "${GRAPH_REPEAT:-3}" \
  --tg "${GRAPH_TG:-512}" --pp "${GRAPH_PP:-512,2048}" \
  --out "$GRAPH_JSON"

set +e
python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$GRAPH_JSON" \
  --baseline "$GRAPH_ON" --r2 "$GRAPH_OFF" --label phase4-graph-off \
  --min-median-tg-gain "${GRAPH_OFF_MIN_GAIN:-0.5}" \
  --max-workload-tg-loss "${GRAPH_OFF_MAX_TG_LOSS:-1.5}" \
  --max-median-pp-loss "${GRAPH_OFF_MAX_PP_LOSS:-2.0}"
GRAPH_OFF_RC=$?
set -e
if [[ "$GRAPH_OFF_RC" == 0 ]]; then
  PARAM_WINNER="$GRAPH_OFF"
  note "GRAPH_WINNER=OFF"
else
  PARAM_WINNER="$GRAPH_ON"
  note "GRAPH_WINNER=ON"
fi
note "PARAM_WINNER_ALIAS=$PARAM_WINNER"
note "GRAPH_RESULT=$GRAPH_JSON"

require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-4 PARAMETER SWEEP COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "Manual JMAX and old Q8-dedup sweeps were intentionally skipped because Modern Foundation already contains their upstream equivalents."
echo "Next: validate PARAM_WINNER_ALIAS on 32K/64K/128K + cached rollback before any gfx1201-only kernel experiment."
