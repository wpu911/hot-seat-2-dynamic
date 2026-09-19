#!/usr/bin/env bash
set -euo pipefail

# Stage16 balanced sweep through llama-swap :8090.
#
# Five arms:
#   untouched Phase3 base (normally n=2)
#   patched ELF + upstream dispatch n=2
#   patched ELF + narrow MMVF n=2
#   patched ELF + upstream dispatch n=4
#   patched ELF + narrow MMVF n=4
#
# cycles=2 gives A B C D E E D C B A. The strict generic MTP analyzer first
# enforces protected-prefix identity/completeness; the Stage16 analyzer then
# judges the n=2 control and the discriminating n=4 TG delta.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
OFF_N2="${OFF_N2:-qwen3.8-flash-next-r2-mmvf-off-n2:256k}"
ON_N2="${ON_N2:-qwen3.8-flash-next-r2-mmvf-on-n2:256k}"
OFF_N4="${OFF_N4:-qwen3.8-flash-next-r2-mmvf-off-n4:256k}"
ON_N4="${ON_N4:-qwen3.8-flash-next-r2-mmvf-on-n4:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RAW="${RAW:-$LOG_DIR/flashnext-r2-stage16-mmvf-$STAMP.json}"
STRICT="${STRICT:-$LOG_DIR/flashnext-r2-stage16-mmvf-$STAMP.strict.json}"
ANALYSIS="${ANALYSIS:-$LOG_DIR/flashnext-r2-stage16-mmvf-$STAMP.analysis.json}"

mkdir -p "$LOG_DIR"
python3 "$SCRIPT_DIR/lock_r2_environment.py" --check

MODELS="$BASE_ALIAS,$OFF_N2,$ON_N2,$OFF_N4,$ON_N4"
python3 "$SCRIPT_DIR/bench_mtp_depth_sweep.py" \
  --models "$MODELS" \
  --cycles "${CYCLES:-2}" \
  --repeat "${REPEAT:-2}" \
  --tg "${TG:-256}" \
  --pp "${PP:-512}" \
  --warmup-pp 256 \
  --warmup-tg 64 \
  --out "$RAW"

python3 "$SCRIPT_DIR/analyze_mtp_sweep.py" "$RAW" \
  --anchor "$BASE_ALIAS" \
  --min-gain 0 \
  --exact-prefix-tokens "${EXACT_PREFIX:-96}" \
  --out "$STRICT"

python3 "$SCRIPT_DIR/analyze_stage16_mmvf_narrow.py" "$STRICT" \
  --base "$BASE_ALIAS" \
  --off-n2 "$OFF_N2" \
  --on-n2 "$ON_N2" \
  --off-n4 "$OFF_N4" \
  --on-n4 "$ON_N4" \
  --min-n4-gain "${MIN_N4_GAIN:-3.0}" \
  --max-control-loss "${MAX_CONTROL_LOSS:-2.0}" \
  --max-workload-loss "${MAX_WORKLOAD_LOSS:-2.0}" \
  --max-acceptance-drop-pp "${MAX_ACCEPTANCE_DROP_PP:-3.0}" \
  --out "$ANALYSIS"

echo "STAGE16_MMVF=PASS"
echo "TUNING_BASE_ALIAS=$ON_N2"
echo "RAW=$RAW"
echo "STRICT=$STRICT"
echo "ANALYSIS=$ANALYSIS"
