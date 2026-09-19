#!/usr/bin/env bash
set -euo pipefail

# Phase-6b: empirical tensor-split ratio tuning after tensor mode itself wins.
# Same real ELF and same model/MTP/graph/GDN settings; only --tensor-split moves.
#
# Stage A: balanced five-arm 32K/64K sweep:
#   EVEN, MID, CAP, INV_MID, INV_CAP
#   then the exact reverse order to cancel first/last load bias.
# Stage B: choose the best correctness-preserving positive arm vs EVEN.
# Stage C: exact short + 32K/64K/128K + cached + rollback confirmation.
#
# This intentionally does not assume the 32 GiB GPU should receive more work.
# Heterogeneous cards can have capacity and kernel throughput pointing in opposite
# directions, because apparently one scalar ratio was too peaceful.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PHASE6_SRC="${PHASE6_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
EVEN="${EVEN:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
MID="${MID:-qwen3.8-flash-next-r2-split-tensor-mid:256k}"
CAP="${CAP:-qwen3.8-flash-next-r2-split-tensor-cap:256k}"
INV_MID="${INV_MID:-qwen3.8-flash-next-r2-split-tensor-inv-mid:256k}"
INV_CAP="${INV_CAP:-qwen3.8-flash-next-r2-split-tensor-inv-cap:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase6b-ratio-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase6b.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
for a in "$PROD_ALIAS" "$EVEN" "$MID" "$CAP" "$INV_MID" "$INV_CAP"; do
  require_alias "$a" || { echo "ERROR alias missing: $a" >&2; exit 2; }
done
[[ -f "$PHASE6_SRC/r2-meta/phase6b-ratios.env" ]] || {
  echo "ERROR ratio manifest missing; run prepare_phase6b_tensor_ratio_sweep.sh" >&2; exit 3;
}
# shellcheck disable=SC1090
source "$PHASE6_SRC/r2-meta/phase6b-ratios.env"
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 unhealthy" >&2; exit 4;
}

note "DEVICE0=$DEVICE0"
note "DEVICE1=$DEVICE1"
note "EVEN_RATIO=$EVEN_RATIO"
note "MID_RATIO=$MID_RATIO"
note "CAP_RATIO=$CAP_RATIO"
note "INV_MID_RATIO=$INV_MID_RATIO"
note "INV_CAP_RATIO=$INV_CAP_RATIO"

MODELS="$EVEN,$MID,$CAP,$INV_MID,$INV_CAP"
SWEEP="$RUN_DIR/ratio-sweep.json"
python3 "$SCRIPT_DIR/bench_tensor_ratio_sweep.py" \
  --models "$MODELS" \
  --cycles "${SMOKE_CYCLES:-1}" \
  --depths "${SMOKE_DEPTHS:-32768,65536}" \
  --tg "${SMOKE_TG:-192}" \
  --out "$SWEEP"

ANALYSIS="$RUN_DIR/ratio-sweep.analysis.json"
python3 "$SCRIPT_DIR/analyze_tensor_ratio_sweep.py" "$SWEEP" \
  --anchor "$EVEN" \
  --min-median-gain "${SMOKE_MIN_GAIN:-0.0}" \
  --max-depth-loss "${SMOKE_MAX_LOSS:-5.0}" \
  --max-median-pp-loss "${SMOKE_MAX_PP_LOSS:-12.0}" \
  --max-acceptance-drop-pp "${SMOKE_MAX_ACC_DROP_PP:-5.0}" \
  --out "$ANALYSIS"

readarray -t PICK < <(python3 - "$ANALYSIS" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
print('CAND='+str(j['winner']))
print('GAIN='+str(j.get('winner_gain_pct',0.0)))
PY
)
for kv in "${PICK[@]}"; do export "$kv"; done

ratio_for() {
  case "$1" in
    "$EVEN") printf '%s\n' "$EVEN_RATIO" ;;
    "$MID") printf '%s\n' "$MID_RATIO" ;;
    "$CAP") printf '%s\n' "$CAP_RATIO" ;;
    "$INV_MID") printf '%s\n' "$INV_MID_RATIO" ;;
    "$INV_CAP") printf '%s\n' "$INV_CAP_RATIO" ;;
    *) echo "ERROR unknown ratio alias: $1" >&2; return 2 ;;
  esac
}
CAND_RATIO="$(ratio_for "$CAND")"
note "SMOKE_RESULT=$SWEEP"
note "SMOKE_ANALYSIS=$ANALYSIS"
note "SMOKE_DECISION=$CAND"
note "SMOKE_GAIN_PCT=$GAIN"

if [[ "$CAND" == "$EVEN" ]]; then
  note "CONFIRM_SHORT=SKIP_EVEN"
  note "CONFIRM_LONG=SKIP_EVEN"
  note "CACHED_GATE=SKIP_EVEN"
  note "ROLLBACK_GATE=SKIP_EVEN"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  echo "No heterogeneous ratio beat 1:1 in the balanced smoke; keep EVEN."
  exit 0
fi

note "CONFIRM_CANDIDATE=$CAND"
note "CONFIRM_RATIO=$CAND_RATIO"

# Short path may regress slightly because the objective here is the long-context
# dual-GPU balance, but correctness/MTP must remain exact and loss is bounded.
SHORT="$RUN_DIR/confirm-short.json"
set +e
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" \
  --baseline "$EVEN" --r2 "$CAND" \
  --rounds "${SHORT_ROUNDS:-4}" --repeat "${SHORT_REPEAT:-3}" \
  --tg "${SHORT_TG:-512}" --pp "${SHORT_PP:-512,2048}" --out "$SHORT"
RC_SHORT_RUN=$?
set -e
if [[ "$RC_SHORT_RUN" != 0 ]]; then
  note "CONFIRM_SHORT=FAIL_LOAD_OR_BENCH"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  exit 0
fi
set +e
python3 "$SCRIPT_DIR/analyze_exact_ab.py" "$SHORT" \
  --baseline "$EVEN" --r2 "$CAND" --label phase6b-ratio-short \
  --min-median-tg-gain "${SHORT_MIN_GAIN:--2.0}" \
  --max-workload-tg-loss "${SHORT_MAX_LOSS:-3.0}" \
  --max-median-pp-loss "${SHORT_MAX_PP_LOSS:-5.0}"
RC_SHORT=$?
set -e
if [[ "$RC_SHORT" != 0 ]]; then
  note "CONFIRM_SHORT=REJECT"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  exit 0
fi
note "CONFIRM_SHORT=PASS"

# Now demand a repeatable >=1% long-context gain. A ratio that only wins the
# exploratory smoke does not get to become another permanent config collectible.
LONG="$RUN_DIR/confirm-long.json"
set +e
python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
  --baseline "$EVEN" --r2 "$CAND" \
  --depths "${CONFIRM_DEPTHS:-32768,65536,131072}" \
  --rounds "${CONFIRM_ROUNDS:-4}" --tg "${CONFIRM_TG:-256}" --out "$LONG"
RC_LONG_RUN=$?
set -e
if [[ "$RC_LONG_RUN" != 0 ]]; then
  note "CONFIRM_LONG=FAIL_LOAD_OR_BENCH"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  exit 0
fi
set +e
python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" "$LONG" \
  --baseline "$EVEN" --r2 "$CAND" --deep-from 32768 \
  --min-deep-median-gain "${CONFIRM_MIN_GAIN:-1.0}" \
  --max-deep-loss "${CONFIRM_MAX_LOSS:-3.0}" \
  --max-pp-loss "${CONFIRM_MAX_PP_LOSS:-7.0}"
RC_LONG=$?
set -e
if [[ "$RC_LONG" != 0 ]]; then
  note "CONFIRM_LONG=REJECT"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  exit 0
fi
note "CONFIRM_LONG=PASS"

CACHED="$RUN_DIR/confirm-cached.json"
set +e
python3 "$SCRIPT_DIR/bench_stage10_cached_largepp.py" \
  --baseline "$EVEN" --r2 "$CAND" \
  --prefix-tokens "${CACHED_PREFIX:-16384}" --suffix-tokens "${CACHED_SUFFIX:-96}" \
  --n-predict "${CACHED_TG:-512}" --repeats "${CACHED_REPEATS:-2}" \
  --max-vs-baseline-loss "${CACHED_MAX_LOSS:-3.0}" --out "$CACHED"
RC_CACHED=$?
set -e
if [[ "$RC_CACHED" != 0 ]]; then
  note "CACHED_GATE=REJECT"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  note "FINISHED=$(date -Is)"
  exit 0
fi
note "CACHED_GATE=PASS"

ROLL="$RUN_DIR/confirm-rollback-64k.json"
set +e
OFF="$EVEN" ON="$CAND" OUT="$ROLL" DEPTH="${ROLLBACK_DEPTH:-65536}" \
  N_PREDICT="${ROLLBACK_TG:-512}" COMPARE_FIRST="${ROLLBACK_COMPARE:-256}" \
  ROUNDS="${ROLLBACK_ROUNDS:-4}" bash "$SCRIPT_DIR/run_stage8_rollback_stress.sh"
RC_ROLL=$?
set -e
if [[ "$RC_ROLL" != 0 ]]; then
  note "ROLLBACK_GATE=REJECT"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
else
  note "ROLLBACK_GATE=PASS"
  note "PHASE6B_WINNER_ALIAS=$CAND"
  note "PHASE6B_WINNER_RATIO=$CAND_RATIO"
fi

note "SHORT_RESULT=$SHORT"
note "LONG_RESULT=$LONG"
note "CACHED_RESULT=$CACHED"
note "ROLLBACK_RESULT=$ROLL"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"
require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-6B BIDIRECTIONAL TENSOR RATIO SWEEP COMPLETE"
echo "================================================================"
cat "$SUMMARY"
