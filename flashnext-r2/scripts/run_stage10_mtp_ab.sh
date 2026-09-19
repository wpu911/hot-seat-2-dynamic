#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${BASELINE:-qwen3.8-flash-next-r2-modern-foundation:256k}"
R2="${R2:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_RUNTIME="${R2_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-mtp}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-stage10-foundation-vs-modern-mtp.json}"

bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$R2_RUNTIME"
REQUIRE_BOTH_GPUS=1 bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$R2_RUNTIME"

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

if [[ "${RUN_CACHED_STRESS:-1}" == "1" ]]; then
  python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
    --baseline "$BASELINE" \
    --r2 "$R2" \
    --prefix-tokens "${CACHED_PREFIX_TOKENS:-16384}" \
    --suffix-tokens "${CACHED_SUFFIX_TOKENS:-96}" \
    --n-predict "${CACHED_N_PREDICT:-256}" \
    --repeats "${CACHED_REPEATS:-2}" \
    --absolute-tg-floor "${CACHED_TG_FLOOR:-5.0}" \
    --min-self-retention "${CACHED_SELF_RETENTION:-0.50}" \
    --max-vs-baseline-loss "${CACHED_MAX_BASELINE_LOSS:-5.0}" \
    --max-acceptance-drop-pp "${CACHED_MAX_ACC_DROP_PP:-5.0}"
fi

echo
echo "Stage-10 modern-MTP A/B complete."
echo "Baseline and candidate share the full Sep18 upstream + exact production overlay lineage."
echo "The intended variable is PR #28243 only."
