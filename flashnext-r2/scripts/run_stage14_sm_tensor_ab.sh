#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAYER="${LAYER:-qwen3.8-flash-next-r2-sm-layer:256k}"
TENSOR="${TENSOR:-qwen3.8-flash-next-r2-sm-tensor:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-sm-tensor-ab.json}"

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$LAYER" --r2 "$TENSOR" \
  --rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" \
  --tg "${TG:-256}" --pp "${PP:-512,2048,8192}" --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage14_sm_tensor.py" \
  "$OUT" --layer "$LAYER" --tensor "$TENSOR" \
  --min-median-tg-gain "${MIN_TG_GAIN:-3.0}" \
  --max-workload-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-median-pp-loss "${MAX_PP_LOSS:-5.0}" \
  --max-acceptance-drop-pp "${MAX_ACC_DROP_PP:-2.0}"

echo
echo "Stage14 layer-vs-tensor A/B complete: $OUT"
echo "This remains an optional branch. Do not merge it into the main pipeline merely because two GPUs exist."
