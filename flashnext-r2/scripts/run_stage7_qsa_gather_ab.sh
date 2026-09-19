#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-modern-qsa-off:256k}"
R2="${R2:-qwen3.8-flash-next-r2-modern-qsa-on:256k}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-qsa-gather}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"

bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$RUNTIME"
REQUIRE_BOTH_GPUS=1 bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$RUNTIME"

if [[ "${FULL:-0}" == "1" ]]; then
  DEPTHS="${DEPTHS:-4096,16384,32768,65536,131072}"
  ROUNDS="${ROUNDS:-4}"
  MODE=full
else
  DEPTHS="${DEPTHS:-32768,65536}"
  ROUNDS="${ROUNDS:-2}"
  MODE=smoke
fi
TG="${TG:-128}"
OUT="${OUT:-$LOG_DIR/flashnext-r2-modern-qsa-${MODE}-$(date +%Y%m%d-%H%M%S).json}"
FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG=(--force)

echo "=== QSA gather $MODE A/B ==="
echo "baseline=$BASELINE"
echo "candidate=$R2"
echo "depths=$DEPTHS rounds=$ROUNDS tg=$TG"

python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASELINE" --r2 "$R2" --depths "$DEPTHS" \
  --rounds "$ROUNDS" --tg "$TG" --out "$OUT" "${FORCE_ARG[@]}"

set +e
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" \
  "$OUT" --baseline "$BASELINE" --r2 "$R2" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_GAIN:-5.0}" \
  --max-deep-loss "${MAX_LOSS:-3.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-5.0}"
RC=$?
set -e

echo
echo "RESULT=$OUT"
if [[ "$RC" -eq 0 && "$MODE" == smoke ]]; then
  echo "QSA_SMOKE=PASS"
  echo "Confirm with: FULL=1 bash $SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
elif [[ "$RC" -eq 0 ]]; then
  echo "QSA_FULL=PASS"
else
  echo "QSA_${MODE^^}=FAIL"
fi
exit "$RC"
