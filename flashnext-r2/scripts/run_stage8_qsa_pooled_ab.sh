#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-pooled-off:256k}"
R2="${R2:-qwen3.8-flash-next-r2-pooled-on:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-qsa-pooled-ladder.json}"
DEPTHS="${DEPTHS:-16384,32768,65536,114688,131072}"
ROUNDS="${ROUNDS:-2}"
TG="${TG:-128}"

python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --depths "$DEPTHS" \
  --rounds "$ROUNDS" \
  --tg "$TG" \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" \
  "$OUT" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_GAIN:-3.0}" \
  --max-deep-loss "${MAX_DEEP_LOSS:-3.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-5.0}"

echo
echo "Stage-8 exploratory A/B complete."
echo "If PASS, confirm in both orders with:"
echo "  ROUNDS=4 bash $SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
echo "Then stress MTP rollback/cache validity with:"
echo "  python3 $SCRIPT_DIR/bench_stage8_rollback_stress.py"
