#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 phase-2 real-machine runner.
#
# This starts only after Phase-1 produced a valid Modern MTP candidate. It treats
# candidate rejection as a result, not as a reason to stop unrelated experiments:
#   - TOP_K is confirmed/rejected first because QSA may layer on top of it.
#   - QSA gather is smoke-tested then fully confirmed.
#   - pooled-key cache runs only if QSA gather fully passes, then rollback stress.
#   - PLE direct-read and FR-Spec are independent branches off Modern MTP and are
#     still tested even when the long-context QSA branch is rejected.
#
# Nothing in this script promotes qwen3.8-flash-next:256k.
# All requests continue to go through llama-swap :8090.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BACKUP_ROOT="${BACKUP_ROOT:-/app/share/backup}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
MTP_SRC="${MTP_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
MTP_ALIAS="${MTP_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
TOPK_SRC="${TOPK_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-rocm-topk-20260918}"
TOPK_ALIAS="${TOPK_ALIAS:-qwen3.8-flash-next-r2-rocm-topk:256k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase2-$STAMP"
SUMMARY="$RUN_DIR/summary.env"
MASTER_LOG="$RUN_DIR/phase2.log"
BACKUP_DIR="$BACKUP_ROOT/flashnext-r2-phase2-$STAMP"
mkdir -p "$RUN_DIR" "$BACKUP_DIR"
exec > >(tee "$MASTER_LOG") 2>&1

note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
require_alias() {
  local alias="$1"
  grep -qE "^[[:space:]]*${alias//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR llama-swap alias missing: $alias" >&2
    return 1
  }
}
run_capture() {
  # Usage: run_capture VAR command...
  local __var="$1"; shift
  set +e
  "$@"
  local rc=$?
  set -e
  printf -v "$__var" '%s' "$rc"
}
latest_phase1_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase1-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || return 0
  awk -F= -v k="$key" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$file"
}

# ---------------- preflight ----------------
echo "=== Flash Next R2 Phase-2 preflight ==="
[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 10; }
[[ -f "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt" ]] || {
  echo "ERROR Modern MTP candidate is missing; finish Phase-1 first." >&2
  exit 11
}
require_alias "$PROD_ALIAS" || exit 12
require_alias "$MTP_ALIAS" || exit 13
curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 is not healthy" >&2
  exit 14
}

cp -a "$CONFIG" "$BACKUP_DIR/config-rocm714.yaml.before"
sha256sum "$CONFIG" "$BACKUP_DIR/config-rocm714.yaml.before" > "$BACKUP_DIR/config.sha256"
note "CONFIG_BACKUP=$BACKUP_DIR/config-rocm714.yaml.before"
note "MTP_SRC=$MTP_SRC"
note "MTP_ALIAS=$MTP_ALIAS"

PHASE1_SUMMARY="${PHASE1_SUMMARY:-$(latest_phase1_summary || true)}"
if [[ -n "$PHASE1_SUMMARY" && -f "$PHASE1_SUMMARY" ]]; then
  echo "phase1_summary=$PHASE1_SUMMARY"
  note "PHASE1_SUMMARY=$PHASE1_SUMMARY"
else
  echo "INFO no Phase-1 summary auto-discovered; current manifests/aliases will be authoritative."
fi

# ---------------- TOP_K decision ----------------
# A smoke PASS is not enough to become the QSA base. If Phase-1 did not run the
# full ladder, confirm it here. REJECT/HIP_GRAPH_INTERACTION fall back to MTP.
LONG_BASE_SRC="$MTP_SRC"
LONG_BASE_ALIAS="$MTP_ALIAS"
TOPK_SMOKE="$(summary_value "$PHASE1_SUMMARY" TOPK_SMOKE_RESULT 2>/dev/null || true)"
TOPK_FULL="$(summary_value "$PHASE1_SUMMARY" TOPK_FULL_RESULT 2>/dev/null || true)"

if [[ "$TOPK_SMOKE" == PASS ]]; then
  if [[ "$TOPK_FULL" != PASS ]]; then
    echo "=== TOP_K smoke passed earlier; running mandatory full confirmation ==="
    run_capture TOPK_FULL_RC env FULL=1 bash "$SCRIPT_DIR/run_stage14_rocm_topk_ab.sh"
    if [[ "$TOPK_FULL_RC" == 0 ]]; then
      TOPK_FULL=PASS
    else
      TOPK_FULL=REJECT
    fi
  fi
fi

if [[ "$TOPK_FULL" == PASS ]]; then
  [[ -f "$TOPK_SRC/r2-meta/stage14-rocm-topk-manifest.txt" ]] || {
    echo "ERROR Phase-1 says TOP_K full PASS but candidate manifest is missing." >&2
    exit 20
  }
  require_alias "$TOPK_ALIAS" || exit 21
  LONG_BASE_SRC="$TOPK_SRC"
  LONG_BASE_ALIAS="$TOPK_ALIAS"
  note "TOPK_FINAL=PASS"
else
  case "$TOPK_SMOKE" in
    HIP_GRAPH_INTERACTION) note "TOPK_FINAL=REJECT_GRAPH_INTERACTION" ;;
    PASS)                  note "TOPK_FINAL=REJECT_FULL_CONFIRMATION" ;;
    REJECT|FAIL)           note "TOPK_FINAL=REJECT_SMOKE" ;;
    *)                     note "TOPK_FINAL=NOT_SELECTED" ;;
  esac
fi
note "LONG_BASE_SRC=$LONG_BASE_SRC"
note "LONG_BASE_ALIAS=$LONG_BASE_ALIAS"

# ---------------- Stage 7: QSA gather ----------------
QSA_SRC="${QSA_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-qsa-gather-20260918}"
QSA_OFF="${QSA_OFF:-qwen3.8-flash-next-r2-modern-qsa-off:256k}"
QSA_ON="${QSA_ON:-qwen3.8-flash-next-r2-modern-qsa-on:256k}"
QSA_FULL_PASS=0

echo
echo "=== Stage 7 QSA gather ==="
if [[ ! -f "$QSA_SRC/r2-meta/stage7-modern-qsa-base.txt" ]]; then
  env BASE_SRC="$LONG_BASE_SRC" SOURCE_ALIAS="$LONG_BASE_ALIAS" \
    bash "$SCRIPT_DIR/prepare_stage7_qsa_gather.sh"
else
  echo "QSA_PREPARE_SKIP existing recorded candidate"
  # Guard against resuming a QSA tree prepared from a different lineage.
  REC_BASE="$(awk -F= '$1=="base_source"{print substr($0,index($0,"=")+1)}' "$QSA_SRC/r2-meta/stage7-modern-qsa-base.txt")"
  REC_ALIAS="$(awk -F= '$1=="source_alias"{print substr($0,index($0,"=")+1)}' "$QSA_SRC/r2-meta/stage7-modern-qsa-base.txt")"
  if [[ "$REC_BASE" != "$LONG_BASE_SRC" || "$REC_ALIAS" != "$LONG_BASE_ALIAS" ]]; then
    echo "ERROR existing QSA candidate was prepared from another base." >&2
    echo "recorded base=$REC_BASE alias=$REC_ALIAS" >&2
    echo "wanted   base=$LONG_BASE_SRC alias=$LONG_BASE_ALIAS" >&2
    exit 30
  fi
fi
require_alias "$QSA_OFF" || exit 31
require_alias "$QSA_ON" || exit 32

run_capture QSA_SMOKE_RC bash "$SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
if [[ "$QSA_SMOKE_RC" == 0 ]]; then
  note "QSA_SMOKE=PASS"
  run_capture QSA_FULL_RC env FULL=1 bash "$SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
  if [[ "$QSA_FULL_RC" == 0 ]]; then
    QSA_FULL_PASS=1
    note "QSA_FULL=PASS"
  else
    note "QSA_FULL=REJECT"
  fi
else
  note "QSA_SMOKE=REJECT"
  note "QSA_FULL=NOT_RUN"
fi

# ---------------- Stage 8: pooled-key cache ----------------
POOLED_SRC="${POOLED_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-qsa-pooled-20260918}"
POOLED_OFF="${POOLED_OFF:-qwen3.8-flash-next-r2-modern-pooled-off:256k}"
POOLED_ON="${POOLED_ON:-qwen3.8-flash-next-r2-modern-pooled-on:256k}"
POOLED_FULL_PASS=0
POOLED_ROLLBACK_PASS=0

if [[ "$QSA_FULL_PASS" == 1 ]]; then
  echo
echo "=== Stage 8 pooled-key cache ==="
  if [[ ! -f "$POOLED_SRC/r2-meta/stage8-modern-pooled-base.txt" ]]; then
    env BASE_SRC="$QSA_SRC" SOURCE_ALIAS="$QSA_ON" \
      bash "$SCRIPT_DIR/prepare_stage8_qsa_pooled.sh"
  else
    echo "POOLED_PREPARE_SKIP existing recorded candidate"
  fi
  require_alias "$POOLED_OFF" || exit 40
  require_alias "$POOLED_ON" || exit 41

  run_capture POOLED_SMOKE_RC bash "$SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
  if [[ "$POOLED_SMOKE_RC" == 0 ]]; then
    note "POOLED_SMOKE=PASS"
    run_capture POOLED_FULL_RC env FULL=1 bash "$SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
    if [[ "$POOLED_FULL_RC" == 0 ]]; then
      POOLED_FULL_PASS=1
      note "POOLED_FULL=PASS"
      run_capture POOLED_ROLLBACK_RC bash "$SCRIPT_DIR/run_stage8_rollback_stress.sh"
      if [[ "$POOLED_ROLLBACK_RC" == 0 ]]; then
        POOLED_ROLLBACK_PASS=1
        note "POOLED_ROLLBACK=PASS"
      else
        note "POOLED_ROLLBACK=REJECT"
      fi
    else
      note "POOLED_FULL=REJECT"
      note "POOLED_ROLLBACK=NOT_RUN"
    fi
  else
    note "POOLED_SMOKE=REJECT"
    note "POOLED_FULL=NOT_RUN"
    note "POOLED_ROLLBACK=NOT_RUN"
  fi
else
  note "POOLED_SMOKE=SKIP_QSA_REJECTED"
  note "POOLED_FULL=NOT_RUN"
  note "POOLED_ROLLBACK=NOT_RUN"
fi

# Select long-context winner for later composition, but do not promote anything.
if [[ "$POOLED_ROLLBACK_PASS" == 1 ]]; then
  LONG_WINNER_SRC="$POOLED_SRC"
  LONG_WINNER_ALIAS="$POOLED_ON"
elif [[ "$QSA_FULL_PASS" == 1 ]]; then
  LONG_WINNER_SRC="$QSA_SRC"
  LONG_WINNER_ALIAS="$QSA_ON"
else
  LONG_WINNER_SRC="$LONG_BASE_SRC"
  LONG_WINNER_ALIAS="$LONG_BASE_ALIAS"
fi
note "LONG_WINNER_SRC=$LONG_WINNER_SRC"
note "LONG_WINNER_ALIAS=$LONG_WINNER_ALIAS"

# ---------------- Stage 11: PLE direct-read ----------------
# Keep this isolated on Modern MTP. It is an I/O/prefill experiment and should
# not be credited with QSA/TOP_K long-context changes.
PLE_SRC="${PLE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-lazy-direct-20260918}"
PLE_MMAP="${PLE_MMAP:-qwen3.8-flash-next-r2-modern-lazy-mmap:256k}"
PLE_DIRECT="${PLE_DIRECT:-qwen3.8-flash-next-r2-modern-lazy-direct:256k}"
PLE_PASS=0

echo
echo "=== Stage 11 PLE direct-read ==="
if [[ ! -e "$PLE_SRC/.git" ]]; then
  env BASE_SRC="$MTP_SRC" SOURCE_ALIAS="$MTP_ALIAS" \
    bash "$SCRIPT_DIR/prepare_stage11_lazy_direct.sh"
else
  echo "PLE_PREPARE_SKIP existing candidate tree"
fi
require_alias "$PLE_MMAP" || exit 50
require_alias "$PLE_DIRECT" || exit 51
run_capture PLE_RC bash "$SCRIPT_DIR/run_stage11_lazy_direct_ab.sh"
if [[ "$PLE_RC" == 0 ]]; then
  PLE_PASS=1
  note "PLE_DIRECT=PASS"
else
  note "PLE_DIRECT=REJECT"
fi

# ---------------- Stage 13: FR-Spec ----------------
FR_SRC="${FR_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-frspec-20260918}"
FR_FULL="${FR_FULL:-qwen3.8-flash-next-r2-modern-frspec-full:256k}"
FR_65K="${FR_65K:-qwen3.8-flash-next-r2-modern-frspec-65k:256k}"
FR_PASS=0

echo
echo "=== Stage 13 FR-Spec ==="
if [[ ! -e "$FR_SRC/.git" ]]; then
  env BASE_SRC="$MTP_SRC" SOURCE_ALIAS="$MTP_ALIAS" \
    bash "$SCRIPT_DIR/prepare_stage13_frspec_modern.sh"
else
  echo "FRSPEC_PREPARE_SKIP existing candidate tree"
fi
require_alias "$FR_FULL" || exit 60
require_alias "$FR_65K" || exit 61
run_capture FR_RC bash "$SCRIPT_DIR/run_stage13_frspec_modern_ab.sh"
if [[ "$FR_RC" == 0 ]]; then
  FR_PASS=1
  note "FRSPEC=PASS"
else
  note "FRSPEC=REJECT"
fi

# ---------------- summary / integrity ----------------
require_alias "$PROD_ALIAS" || exit 70
cp -a "$CONFIG" "$RUN_DIR/config-after.yaml"
sha256sum "$CONFIG" > "$RUN_DIR/config-after.sha256"
note "PRODUCTION_ALIAS=$PROD_ALIAS"
note "PRODUCTION_PROMOTED=NO"
note "PLE_PASS=$PLE_PASS"
note "FRSPEC_PASS=$FR_PASS"
note "CONFIG_AFTER=$RUN_DIR/config-after.yaml"
note "MASTER_LOG=$MASTER_LOG"
note "FINISHED=$(date -Is)"

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-2 COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "No production promotion was performed."
echo "Next step is to compose ONLY the recorded winners into one clean runtime, then rerun short/32K/64K/128K + cached rollback + OpenClaw regression."
