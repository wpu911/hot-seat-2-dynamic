#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-qsa-off:256k}"
R2="${R2:-qwen3.8-flash-next-r2-qsa-on:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-qsa-ladder.json}"
DEPTHS="${DEPTHS:-4096,16384,32768,65536,131072}"
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
  --r2 "$R2"

echo
echo "If the exploratory OFF/ON pass is promising, confirm in reverse order too:"
echo "  ROUNDS=4 bash $SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
