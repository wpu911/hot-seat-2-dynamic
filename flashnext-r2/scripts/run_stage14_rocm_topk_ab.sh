#!/usr/bin/env bash
set -euo pipefail

# Stage 14 end-to-end A/B through the real llama-swap :8090 path.
# Run the long-context ladder twice:
#   A. normal HIP graph policy
#   B. both engines with HIP graphs disabled
# Then classify the result as PASS / HIP_GRAPH_INTERACTION / FAIL.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_ON="${BASE_ON:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_ON="${R2_ON:-qwen3.8-flash-next-r2-rocm-topk:256k}"
BASE_OFF="${BASE_OFF:-qwen3.8-flash-next-r2-topk-base-nograph:256k}"
R2_OFF="${R2_OFF:-qwen3.8-flash-next-r2-topk-nograph:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
DEPTHS="${DEPTHS:-16384,32768,65536,131072}"
TG="${TG:-128}"
ROUNDS="${ROUNDS:-4}"
FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG=(--force)
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
ON_OUT="${ON_OUT:-$LOG_DIR/flashnext-stage14-topk-graph-on-$STAMP.json}"
OFF_OUT="${OFF_OUT:-$LOG_DIR/flashnext-stage14-topk-graph-off-$STAMP.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$LOG_DIR/flashnext-stage14-topk-analysis-$STAMP.json}"

echo "=== Stage 14 / pass 1: HIP graphs ON ==="
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASE_ON" \
  --r2 "$R2_ON" \
  --depths "$DEPTHS" \
  --tg "$TG" \
  --rounds "$ROUNDS" \
  --out "$ON_OUT" \
  "${FORCE_ARG[@]}"

echo
echo "=== Stage 14 / pass 2: HIP graphs OFF ==="
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
    echo "ROCm TOP_K is a candidate winner on the normal llama-swap production path."
    ;;
  3)
    echo "STAGE14=HIP_GRAPH_INTERACTION"
    echo "TOP_K wins graph-off but not cleanly graph-on. Do not discard it; profile graph interaction before promotion."
    ;;
  *)
    echo "STAGE14=FAIL"
    echo "Do not combine this TOP_K candidate into the final runtime."
    ;;
esac

exit "$RC"
