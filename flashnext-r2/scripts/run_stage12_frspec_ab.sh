#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FULL="${FULL:-qwen3.8-flash-next-r2-frspec-full:256k}"
FR="${FR:-qwen3.8-flash-next-r2-frspec-65k:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-frspec-65k-ab.json}"

# Full -> FR -> Full -> FR through the real llama-swap :8090 path.
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$FULL" \
  --r2 "$FR" \
  --rounds "${ROUNDS:-4}" \
  --repeat "${REPEAT:-3}" \
  --tg "${TG:-256}" \
  --pp "${PP:-512,2048}" \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage12_frspec.py" \
  "$OUT" --baseline "$FULL" --frspec "$FR" \
  --min-median-tg-gain "${MIN_TG_GAIN:-3.0}" \
  --max-workload-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-acceptance-drop-pp "${MAX_ACCEPTANCE_DROP_PP:-1.5}" \
  --max-median-pp-loss "${MAX_PP_LOSS:-3.0}"

echo
echo "Stage-12 FR-Spec A/B complete: $OUT"
echo "If 65k passes, keep it as the first candidate. Do not descend to 32k automatically;"
echo "the reference workload already showed acceptance loss there."
