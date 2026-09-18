#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROD="${PROD:-qwen3.8-flash-next:256k}"
FOUNDATION="${FOUNDATION:-qwen3.8-flash-next-r2-modern-foundation:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-modern-foundation-ab.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$PROD" --r2 "$FOUNDATION" \
  --rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" \
  --tg "${TG:-256}" --pp "${PP:-512,2048,8192}" --out "$OUT"

python3 "$SCRIPT_DIR/analyze_modern_foundation.py" \
  "$OUT" --baseline "$PROD" --foundation "$FOUNDATION" \
  --max-median-tg-loss "${MAX_MEDIAN_TG_LOSS:-2.0}" \
  --max-workload-tg-loss "${MAX_WORKLOAD_TG_LOSS:-3.0}" \
  --max-median-pp-loss "${MAX_MEDIAN_PP_LOSS:-3.0}" \
  --max-acceptance-drop-pp "${MAX_ACCEPTANCE_DROP_PP:-2.0}"

echo
echo "Modern foundation A/B complete: $OUT"
echo "Only a PASS foundation may become the base for Stage 10 MTP and Stage 12 FR-Spec."
