#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-graph-off:256k}"
R2="${R2:-qwen3.8-flash-next-r2-graph-on:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-graph-ab.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-256}" \
  --pp "${PP:-512,2048,8192}" \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_exact_ab.py" \
  "$OUT" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --label graph \
  --min-median-tg-gain "${MIN_TG_GAIN:-1.0}" \
  --max-workload-tg-loss "${MAX_TG_LOSS:-1.5}" \
  --max-median-pp-loss "${MAX_PP_LOSS:-2.0}"
