#!/usr/bin/env bash
set -euo pipefail

# Mandatory correctness gate after pooled-key cache throughput PASS.
# Uses modern pooled aliases explicitly because the underlying Python stress
# tool intentionally keeps its historical defaults for reproducibility.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFF="${OFF:-qwen3.8-flash-next-r2-modern-pooled-off:256k}"
ON="${ON:-qwen3.8-flash-next-r2-modern-pooled-on:256k}"
OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-modern-pooled-rollback-$(date +%Y%m%d-%H%M%S).json}"
FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG=(--force)

python3 "$SCRIPT_DIR/bench_stage8_rollback_stress.py" \
  --off "$OFF" \
  --on "$ON" \
  --depth "${DEPTH:-65536}" \
  --n-predict "${N_PREDICT:-512}" \
  --compare-first "${COMPARE_FIRST:-256}" \
  --rounds "${ROUNDS:-4}" \
  --out "$OUT" \
  "${FORCE_ARG[@]}"

echo "ROLLBACK_REPORT=$OUT"
