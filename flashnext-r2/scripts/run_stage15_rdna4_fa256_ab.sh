#!/usr/bin/env bash
set -euo pipefail

# Balanced BASE / patched-OFF / patched-ON long-context A/B through llama-swap.
# Sequence comes from bench_tensor_ratio_sweep.py:
#   BASE OFF ON ON OFF BASE
# which gives every arm two legs while damping simple thermal/order bias.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-rdna4-fa256-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-rdna4-fa256-on:256k}"
DEPTHS="${DEPTHS:-16384,65536,131072}"
TG="${TG:-128}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$LOG_DIR/flashnext-r2-stage15-fa256-$STAMP.json}"
ANALYSIS="${ANALYSIS:-$LOG_DIR/flashnext-r2-stage15-fa256-$STAMP.analysis.json}"

mkdir -p "$LOG_DIR"
python3 "$SCRIPT_DIR/lock_r2_environment.py" --check

printf '%s\n' '=== Stage 15 RDNA4 FA256 balanced A/B ==='
echo "BASE=$BASE_ALIAS"
echo "OFF=$OFF_ALIAS"
echo "ON=$ON_ALIAS"
echo "DEPTHS=$DEPTHS TG=$TG"
echo "OUT=$OUT"

python3 "$SCRIPT_DIR/bench_tensor_ratio_sweep.py" \
  --models "$BASE_ALIAS,$OFF_ALIAS,$ON_ALIAS" \
  --cycles 1 \
  --depths "$DEPTHS" \
  --tg "$TG" \
  --warmup-depth 4096 \
  --warmup-tg 64 \
  --out "$OUT"

python3 "$SCRIPT_DIR/analyze_stage15_rdna4_fa256.py" "$OUT" \
  --base "$BASE_ALIAS" \
  --off "$OFF_ALIAS" \
  --on "$ON_ALIAS" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-route-deep-pp-gain "${MIN_ROUTE_PP_GAIN:-3.0}" \
  --min-net-deep-pp-gain "${MIN_NET_PP_GAIN:-2.0}" \
  --max-carryover-pp-loss "${MAX_CARRYOVER_PP_LOSS:-4.0}" \
  --max-route-pp-loss "${MAX_ROUTE_PP_LOSS:-2.0}" \
  --max-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-acceptance-drop-pp "${MAX_ACCEPTANCE_DROP_PP:-3.0}" \
  --out "$ANALYSIS"

echo "STAGE15_FA256=PASS"
echo "WINNER_ALIAS=$ON_ALIAS"
echo "RESULT=$OUT"
echo "ANALYSIS=$ANALYSIS"
