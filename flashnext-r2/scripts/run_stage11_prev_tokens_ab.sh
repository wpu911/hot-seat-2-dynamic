#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROD="${PROD:-qwen3.8-flash-next:256k}"
OFF="${OFF:-qwen3.8-flash-next-r2-prev-off:256k}"
FAST="${FAST:-qwen3.8-flash-next-r2-prev-fast:256k}"

LOOKUP_OUT="${LOOKUP_OUT:-/app/share/openclaw_tools/logs/flashnext-r2-prev-index-off-fast.json}"
NET_OUT="${NET_OUT:-/app/share/openclaw_tools/logs/flashnext-r2-prev-index-prod-fast.json}"
DEPTHS="${DEPTHS:-4096,16384,32768,65536,131072}"
NET_DEPTHS="${NET_DEPTHS:-32768,65536,131072}"
ROUNDS="${ROUNDS:-4}"
TG="${TG:-128}"

# 1) Same patched binary: legacy scan vs indexed lookup.
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$OFF" --r2 "$FAST" \
  --depths "$DEPTHS" --rounds "$ROUNDS" --tg "$TG" \
  --out "$LOOKUP_OUT"

python3 "$SCRIPT_DIR/analyze_stage11_prev_tokens.py" \
  "$LOOKUP_OUT" --baseline "$OFF" --fast "$FAST" \
  --deep-from "${DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_LOOKUP_GAIN:-5.0}" \
  --max-any-tg-loss "${MAX_TG_LOSS:-2.0}" \
  --max-pp-loss "${MAX_PP_LOSS:-3.0}"

# 2) Net effect: original production binary vs patched FAST binary. This catches
#    any cost of maintaining seq_pos row sets that OFF-vs-FAST intentionally hides.
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$PROD" --r2 "$FAST" \
  --depths "$NET_DEPTHS" --rounds "$ROUNDS" --tg "$TG" \
  --out "$NET_OUT"

python3 "$SCRIPT_DIR/analyze_stage11_prev_tokens.py" \
  "$NET_OUT" --baseline "$PROD" --fast "$FAST" \
  --deep-from "${NET_DEEP_FROM:-32768}" \
  --min-deep-median-gain "${MIN_NET_GAIN:-3.0}" \
  --max-any-tg-loss "${MAX_NET_TG_LOSS:-2.0}" \
  --max-pp-loss "${MAX_NET_PP_LOSS:-3.0}"

echo
echo "Stage-11 A/B complete."
echo "lookup-only result : $LOOKUP_OUT"
echo "net production A/B : $NET_OUT"
echo
echo "Correctness stress should then run the VERIFY alias and inspect llama-server logs for any 'PLE-IDX MISMATCH'."
