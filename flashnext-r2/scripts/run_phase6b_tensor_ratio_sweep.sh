#!/usr/bin/env bash
set -euo pipefail

# Phase-6b: tune only the tensor-split proportion after tensor mode itself has
# already beaten layer split. Same ELF, same model/MTP/graphs/GDN settings.
#
# Stage A: cheap 32K/64K smoke for EVEN vs MID and EVEN vs CAP.
# Stage B: choose the better positive candidate by measured deep TG.
# Stage C: exact short A/B + 32K/64K/128K confirmation + cached/rollback.
#
# No production promotion.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PHASE6_SRC="${PHASE6_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
EVEN="${EVEN:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
MID="${MID:-qwen3.8-flash-next-r2-split-tensor-mid:256k}"
CAP="${CAP:-qwen3.8-flash-next-r2-split-tensor-cap:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase6b-ratio-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/phase6b.log") 2>&1

note(){ printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias(){ grep -qE "^[[:space:]]*${1//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG"; }
for a in "$PROD_ALIAS" "$EVEN" "$MID" "$CAP"; do
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

smoke_one(){
  local cand="$1" tag="$2" out="$RUN_DIR/smoke-$tag.json"
  echo "=== ratio smoke EVEN vs $tag ==="
  set +e
  python3 "$SCRIPT_DIR/bench_qsa_context_ladder.py" \
    --baseline "$EVEN" --r2 "$cand" \
    --depths "${SMOKE_DEPTHS:-32768,65536}" \
    --rounds "${SMOKE_ROUNDS:-2}" --tg "${SMOKE_TG:-192}" --out "$out"
  local run_rc=$?
  set -e
  if [[ "$run_rc" != 0 ]]; then
    echo "LOAD_OR_BENCH_FAIL" > "$RUN_DIR/$tag.status"
    echo "$out"
    return 0
  fi
  set +e
  python3 "$SCRIPT_DIR/analyze_stage7_qsa.py" "$out" \
    --baseline "$EVEN" --r2 "$cand" --deep-from 32768 \
    --min-deep-median-gain "${SMOKE_MIN_GAIN:-0.0}" \
    --max-deep-loss "${SMOKE_MAX_LOSS:-5.0}" \
    --max-pp-loss "${SMOKE_MAX_PP_LOSS:-12.0}"
  local gate_rc=$?
  set -e
  echo "$gate_rc" > "$RUN_DIR/$tag.status"
  echo "$out"
}

MID_JSON="$(smoke_one "$MID" mid | tail -n1)"
CAP_JSON="$(smoke_one "$CAP" cap | tail -n1)"

# Pick the larger measured median deep TG gain, but only among smoke arms whose
# correctness/loss gate passed. A 0% threshold here is exploratory; the later
# confirmation requires a real >=1% gain.
CAND="$(python3 - "$RUN_DIR" "$MID_JSON" "$CAP_JSON" "$MID" "$CAP" <<'PY'
import json, pathlib, sys
run=pathlib.Path(sys.argv[1]); pairs=[('mid',sys.argv[2],sys.argv[4]),('cap',sys.argv[3],sys.argv[5])]
best=None
for tag,path,model in pairs:
    st=(run/f'{tag}.status').read_text().strip()
    if st != '0':
        continue
    ap=pathlib.Path(path).with_suffix('.qsa-analysis.json')
    if not ap.exists():
        continue
    j=json.loads(ap.read_text())
    gain=j.get('gate',{}).get('deep_median_tg_gain_pct')
    if isinstance(gain,(int,float)) and gain >= 0 and (best is None or gain > best[0]):
        best=(gain,model,tag)
if best:
    print(best[1])
PY
)"

if [[ -z "$CAND" ]]; then
  note "SMOKE_DECISION=KEEP_EVEN"
  note "PHASE6B_WINNER_ALIAS=$EVEN"
  note "PHASE6B_WINNER_RATIO=$EVEN_RATIO"
  note "PRODUCTION_PROMOTED=NO"
  echo "Neither heterogeneous ratio survived the exploratory long-context smoke."
  exit 0
fi

if [[ "$CAND" == "$MID" ]]; then CAND_RATIO="$MID_RATIO"; CAND_TAG=mid; else CAND_RATIO="$CAP_RATIO"; CAND_TAG=cap; fi
note "SMOKE_DECISION=$CAND_TAG"
note "CONFIRM_CANDIDATE=$CAND"
note "CONFIRM_RATIO=$CAND_RATIO"

# Exact short check. Ratio tuning is not allowed to buy deep-context speed by
# wrecking the everyday short path.
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
  exit 0
fi
note "CONFIRM_SHORT=PASS"

# Full long confirmation. 1% is intentionally modest but above the exploratory
# smoke threshold; if the gain cannot clear this over repeated 32K/64K/128K
# runs, extra configuration complexity is not worth keeping.
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

note "MID_SMOKE=$MID_JSON"
note "CAP_SMOKE=$CAP_JSON"
note "SHORT_RESULT=$SHORT"
note "LONG_RESULT=$LONG"
note "CACHED_RESULT=$CACHED"
note "ROLLBACK_RESULT=$ROLL"
note "PRODUCTION_PROMOTED=NO"
note "FINISHED=$(date -Is)"
require_alias "$PROD_ALIAS" || { echo "ERROR production alias disappeared" >&2; exit 20; }

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-6B TENSOR RATIO SWEEP COMPLETE"
echo "================================================================"
cat "$SUMMARY"
