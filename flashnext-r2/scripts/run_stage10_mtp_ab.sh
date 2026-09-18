#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 10 must isolate MTP. The baseline is therefore the Stage-12 HC-only
# runtime, not production. Candidate = the same HC stack + PR #28243.
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-upstream-hc:256k}"
R2="${R2:-qwen3.8-flash-next-r2-hc-mtp:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-stage10-hc-vs-hc-mtp.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$BASELINE" \
  --r2 "$R2" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-512}" \
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
echo "Stage-10 MTP A/B complete."
echo "Baseline and candidate share the same upstream HC lineage; the intended variable is PR #28243 MTP."
echo "A PASS is still only a candidate result; cached Large-PP and long-context rollback/QSA regression must pass before production merge."
