#!/usr/bin/env bash
set -euo pipefail

# Stage 14 end-to-end A/B through the real llama-swap :8090 path.
#
# Important Qwen4Exp detail: the long-context QSA selector is roughly k~2048
# (often 2051 after local-cell accounting) over multiple query rows. In PR #28313
# that does NOT use the flashy small-k path; it falls through to the parallel
# radix path. Therefore do a cheap 32K/64K smoke A/B first instead of spending
# the evening prefilling 128K four times for a kernel that may be neutral here.
#
# Default smoke:
#   32K,64K / OFF-ON once for graph ON and graph OFF
# Full confirmation after a promising smoke:
#   FULL=1 bash run_stage14_rocm_topk_ab.sh
# which uses 16K,32K,64K,128K and 4 alternating legs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_ON="${BASE_ON:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_ON="${R2_ON:-qwen3.8-flash-next-r2-rocm-topk:256k}"
BASE_OFF="${BASE_OFF:-qwen3.8-flash-next-r2-topk-base-nograph:256k}"
R2_OFF="${R2_OFF:-qwen3.8-flash-next-r2-topk-nograph:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"

if [[ "${FULL:-0}" == "1" ]]; then
  DEPTHS="${DEPTHS:-16384,32768,65536,131072}"
  ROUNDS="${ROUNDS:-4}"
  MODE=full
else
  DEPTHS="${DEPTHS:-32768,65536}"
  ROUNDS="${ROUNDS:-2}"
  MODE=smoke
fi
TG="${TG:-128}"
FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG=(--force)
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
ON_OUT="${ON_OUT:-$LOG_DIR/flashnext-stage14-topk-${MODE}-graph-on-$STAMP.json}"
OFF_OUT="${OFF_OUT:-$LOG_DIR/flashnext-stage14-topk-${MODE}-graph-off-$STAMP.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$LOG_DIR/flashnext-stage14-topk-${MODE}-analysis-$STAMP.json}"

echo "=== Stage 14 ROCm TOP_K / $MODE ==="
echo "depths=$DEPTHS rounds=$ROUNDS tg=$TG"
echo "NOTE: qwen4exp QSA k~2048 multi-row is expected to exercise parallel radix, not the small-k fast path."

echo
echo "=== pass 1: HIP graphs ON ==="
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASE_ON" \
  --r2 "$R2_ON" \
  --depths "$DEPTHS" \
  --tg "$TG" \
  --rounds "$ROUNDS" \
  --out "$ON_OUT" \
  "${FORCE_ARG[@]}"

echo
echo "=== pass 2: HIP graphs OFF ==="
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASE_OFF" \
  --r2 "$R2_OFF" \
  --depths "$DEPTHS" \
  --tg "$TG" \
  --rounds "$ROUNDS" \
  --out "$OFF_OUT" \
  "${FORCE_ARG[@]}"

echo
echo "=== Stage 14 analysis ==="
set +e
python3 "$SCRIPT_DIR/analyze_stage14_rocm_topk.py" \
  --on "$ON_OUT" \
  --off "$OFF_OUT" \
  --base-on "$BASE_ON" \
  --r2-on "$R2_ON" \
  --base-off "$BASE_OFF" \
  --r2-off "$R2_OFF" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-on-gain "${MIN_ON_GAIN:-2.0}" \
  --min-off-gain-for-interaction "${MIN_OFF_GAIN:-3.0}" \
  --max-deep-loss "${MAX_DEEP_LOSS:-2.0}" \
  --max-acceptance-drop-pp "${MAX_ACC_DROP_PP:-3.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-5.0}" \
  --out "$ANALYSIS_OUT"
RC=$?
set -e

echo
echo "GRAPH_ON_RESULT=$ON_OUT"
echo "GRAPH_OFF_RESULT=$OFF_OUT"
echo "ANALYSIS=$ANALYSIS_OUT"
case "$RC" in
  0)
    echo "STAGE14=PASS"
    if [[ "$MODE" == smoke ]]; then
      echo "Smoke looks promising. Confirm with: FULL=1 bash $SCRIPT_DIR/run_stage14_rocm_topk_ab.sh"
    else
      echo "ROCm TOP_K is a candidate winner on the normal llama-swap production path."
    fi
    ;;
  3)
    echo "STAGE14=HIP_GRAPH_INTERACTION"
    echo "TOP_K wins graph-off but not cleanly graph-on. Profile graph interaction before promotion."
    ;;
  *)
    echo "STAGE14=FAIL"
    echo "No reason to spend 128K-prefill time on this candidate unless profiling shows a hidden bottleneck."
    ;;
esac

exit "$RC"
