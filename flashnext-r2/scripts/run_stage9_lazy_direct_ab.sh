#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-lazy-mmap:256k}"
R2="${R2:-qwen3.8-flash-next-r2-lazy-direct:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-lazy-direct-ab.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$BASELINE" --r2 "$R2" \
  --rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" \
  --tg "${TG:-128}" --pp "${PP:-512,2048,8192}" --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage9_lazy_direct.py" \
  "$OUT" --baseline "$BASELINE" --r2 "$R2" \
  --min-median-pp-gain "${MIN_PP_GAIN:-5.0}" \
  --max-tg-loss "${MAX_TG_LOSS:-3.0}"
