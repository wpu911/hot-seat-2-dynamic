#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-modern-pooled-off:256k}"
R2="${R2:-qwen3.8-flash-next-r2-modern-pooled-on:256k}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-qsa-pooled}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"

bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$RUNTIME"
REQUIRE_BOTH_GPUS=1 bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$RUNTIME"

if [[ "${FULL:-0}" == "1" ]]; then
  DEPTHS="${DEPTHS:-16384,32768,65536,114688,131072}"
  ROUNDS="${ROUNDS:-4}"
  MODE=full
else
  DEPTHS="${DEPTHS:-65536,114688}"
  ROUNDS="${ROUNDS:-2}"
  MODE=smoke
fi
TG="${TG:-128}"
OUT="${OUT:-$LOG_DIR/flashnext-r2-modern-pooled-${MODE}-$(date +%Y%m%d-%H%M%S).json}"
FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG=(--force)

echo "=== pooled-key cache $MODE A/B ==="
echo "baseline=$BASELINE"
echo "candidate=$R2"
echo "depths=$DEPTHS rounds=$ROUNDS tg=$TG"

python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASELINE" --r2 "$R2" --depths "$DEPTHS" \
  --rounds "$ROUNDS" --tg "$TG" --out "$OUT" "${FORCE_ARG[@]}"

set +e
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" \
  "$OUT" --baseline "$BASELINE" --r2 "$R2" \
  --deep-from "${DEEP_FROM:-65536}" \
  --min-deep-median-gain "${MIN_GAIN:-3.0}" \
  --max-deep-loss "${MAX_DEEP_LOSS:-3.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-5.0}"
RC=$?
set -e

echo
echo "RESULT=$OUT"
if [[ "$RC" -eq 0 && "$MODE" == smoke ]]; then
  echo "POOLED_SMOKE=PASS"
  echo "Confirm both orders with: FULL=1 bash $SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
  echo "After full PASS, rollback/checkpoint stress is mandatory."
elif [[ "$RC" -eq 0 ]]; then
  echo "POOLED_FULL=PASS"
  echo "Now run: bash $SCRIPT_DIR/run_stage8_rollback_stress.sh"
else
  echo "POOLED_${MODE^^}=FAIL"
fi
exit "$RC"
