#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-modern-lazy-mmap:256k}"
R2="${R2:-qwen3.8-flash-next-r2-modern-lazy-direct:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-modern-lazy-direct-ab.json}"

# Fresh-model alternation matters here because page-cache warmth can hide the
# direct-read advantage. The generic runner unloads each model between legs.
# The two aliases use one byte-identical real ELF; only --lazy-mode differs.
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-256}" \
  --pp "${PP:-512,2048,8192}" \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage11_lazy_direct.py" \
  "$OUT" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --min-median-pp-gain "${MIN_PP_GAIN:-5.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-3.0}" \
  --max-tg-loss "${MAX_TG_LOSS:-2.0}"

echo
echo "Stage-11 modern lazy direct-read A/B complete."
echo "OS page cache is intentionally not dropped. Compare both leg orders; warm-cache convergence is expected, not evidence of divine intervention."
