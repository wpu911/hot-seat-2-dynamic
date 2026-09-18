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

# The steady-state A/B above uses cache_prompt=false by design. The previous
# Flash-Next 0.1 t/s incident lived specifically in cached Large-PP / rollback,
# so a throughput PASS is not enough. Run the high-LCP branch regression unless
# explicitly disabled for a quick smoke.
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
    --max-vs-baseline-loss "${CACHED_MAX_BASELINE_LOSS:-5.0}"
fi

echo
echo "Stage-10 MTP A/B complete."
echo "Baseline and candidate share the same upstream HC lineage; the intended variable is PR #28243 MTP."
echo "Promotion still requires the later long-context QSA/rollback combined-runtime regression before touching production."
