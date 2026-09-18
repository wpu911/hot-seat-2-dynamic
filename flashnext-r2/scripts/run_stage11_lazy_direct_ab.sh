#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-modern-lazy-mmap:256k}"
R2="${R2:-qwen3.8-flash-next-r2-modern-lazy-direct:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-modern-lazy-direct-realworld.json}"

# Do NOT use the generic repeated-seed PP benchmark here. PLE mmap behaviour is
# specifically sensitive to token/ngram diversity: repetitive prompts can keep
# touching the same tiny subset of the huge PLE table and make mmap look falsely
# healthy. This runner consumes recent local OpenClaw text, cuts disjoint windows,
# and enforces a minimum unique 4-gram ratio before timing.
python3 "$SCRIPT_DIR/bench_stage11_ple_realworld.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --sessions "${SESSIONS:-/app/share/openclaw_data/.openclaw/agents/main/sessions}" \
  --max-files "${MAX_FILES:-24}" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-128}" \
  --pp "${PP:-512,2048,8192}" \
  --min-ngram4-ratio "${MIN_NGRAM4_RATIO:-0.70}" \
  --out "$OUT" \
  ${FORCE:+--force}

python3 "$SCRIPT_DIR/analyze_stage11_lazy_direct.py" \
  "$OUT" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --min-median-pp-gain "${MIN_PP_GAIN:-5.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-3.0}" \
  --max-tg-loss "${MAX_TG_LOSS:-2.0}"

echo
echo "Stage-11 modern lazy direct-read A/B complete."
echo "PP prompts are disjoint diverse windows; the result log stores metrics/hashes only, not session text."
echo "OS page cache is intentionally not dropped, so both leg orders matter. Humans discovered caches and immediately turned benchmarks into a small branch of mythology."
