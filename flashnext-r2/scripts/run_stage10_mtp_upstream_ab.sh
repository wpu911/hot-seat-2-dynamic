#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next:256k}"
R2="${R2:-qwen3.8-flash-next-r2-mtp-upstream:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-mtp-upstream-ab.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-384}" \
  --pp "${PP:-512,2048,8192}" \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage10_mtp.py" \
  "$OUT" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --min-median-tg-gain "${MIN_TG_GAIN:-3.0}" \
  --max-workload-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-acceptance-drop-pp "${MAX_ACC_DROP_PP:-3.0}" \
  --max-median-pp-loss "${MAX_PP_LOSS:-5.0}"

echo
echo "Stage-10 upstream-MTP A/B complete."
echo "Result: $OUT"
