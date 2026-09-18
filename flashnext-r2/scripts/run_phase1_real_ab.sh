#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 phase-1 real-machine runner.
#
# Scope is deliberately narrow:
#   1) freeze exact production source
#   2) build Sep-18 Modern Foundation with the exact production custom overlay
#   3) verify qwen4exp native recurrent rollback survived the forward-port
#   4) A/B production vs Modern Foundation through llama-swap :8090
#   5) add PR #28243 MTP on the same foundation and A/B it
#   6) prepare PR #28313 ROCm TOP_K and run the 32K/64K smoke ladder
#
# It NEVER replaces qwen3.8-flash-next:256k and never edits the production
# runtime path. Candidate aliases are additive only. Every config mutation is
# preceded by a timestamped backup under /app/share/backup.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
PROD_ALIAS="${PROD_ALIAS:-qwen3.8-flash-next:256k}"
PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BACKUP_ROOT="${BACKUP_ROOT:-/app/share/backup}"
START_AT="${START_AT:-snapshot}"
STOP_AFTER="${STOP_AFTER:-topk}"
TOPK_FULL="${TOPK_FULL:-0}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_DIR/flashnext-r2-phase1-$STAMP"
MASTER_LOG="$RUN_DIR/phase1.log"
SUMMARY="$RUN_DIR/summary.env"
BACKUP_DIR="$BACKUP_ROOT/flashnext-r2-phase1-$STAMP"
mkdir -p "$RUN_DIR" "$BACKUP_DIR"
exec > >(tee "$MASTER_LOG") 2>&1

PHASES=(snapshot foundation mtp topk)
phase_index() {
  local want="$1" i
  for i in "${!PHASES[@]}"; do
    [[ "${PHASES[$i]}" == "$want" ]] && { echo "$i"; return 0; }
  done
  return 1
}
START_I="$(phase_index "$START_AT")" || { echo "ERROR invalid START_AT=$START_AT" >&2; exit 2; }
STOP_I="$(phase_index "$STOP_AFTER")" || { echo "ERROR invalid STOP_AFTER=$STOP_AFTER" >&2; exit 3; }
(( START_I <= STOP_I )) || { echo "ERROR START_AT after STOP_AFTER" >&2; exit 4; }

note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
phase_enabled() {
  local i
  i="$(phase_index "$1")"
  (( i >= START_I && i <= STOP_I ))
}
run_phase() {
  local name="$1"; shift
  echo
  echo "================================================================"
  echo "PHASE=$name START $(date -Is)"
  printf 'CMD='; printf '%q ' "$@"; echo
  echo "================================================================"
  "$@"
  echo "PHASE=$name PASS $(date -Is)"
}
require_alias() {
  local alias="$1"
  grep -qE "^[[:space:]]*${alias//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
    echo "ERROR llama-swap alias missing: $alias" >&2
    exit 20
  }
}

# ---------- preflight ----------
echo "=== Flash Next R2 Phase-1 preflight ==="
echo "date=$(date -Is)"
echo "config=$CONFIG"
echo "production_alias=$PROD_ALIAS"
echo "production_src=$PROD_SRC"
echo "log=$MASTER_LOG"

[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 10; }
[[ -e "$PROD_SRC/.git" ]] || { echo "ERROR production source is not a git tree: $PROD_SRC" >&2; exit 11; }
require_alias "$PROD_ALIAS"

cp -a "$CONFIG" "$BACKUP_DIR/config-rocm714.yaml.before"
sha256sum "$CONFIG" "$BACKUP_DIR/config-rocm714.yaml.before" | tee "$BACKUP_DIR/config.sha256"
note "CONFIG_BACKUP=$BACKUP_DIR/config-rocm714.yaml.before"

curl -fsS --max-time 20 http://127.0.0.1:8090/v1/models > "$RUN_DIR/models-preflight.json" || {
  echo "ERROR llama-swap :8090 is not healthy" >&2
  exit 12
}
curl -fsS --max-time 20 http://127.0.0.1:8090/running > "$RUN_DIR/running-preflight.json" 2>/dev/null || true

{
  echo "=== disk ==="
  df -h /app/share 2>/dev/null || df -h .
  echo "=== memory ==="
  free -h 2>/dev/null || true
  echo "=== GPU ==="
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E 'Name:.*gfx|Marketing Name|gfx1100|gfx1201' | head -n 80 || true
  elif [[ -x /opt/host-rocm/core-10.0/bin/rocminfo ]]; then
    /opt/host-rocm/core-10.0/bin/rocminfo 2>/dev/null | grep -E 'Name:.*gfx|Marketing Name|gfx1100|gfx1201' | head -n 80 || true
  fi
} | tee "$RUN_DIR/machine-preflight.txt"

git -C "$PROD_SRC" rev-parse HEAD | tee "$RUN_DIR/production-head.txt"
git -C "$PROD_SRC" status --short | tee "$RUN_DIR/production-status.txt" || true

# ---------- snapshot ----------
CURRENT="/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current"
if phase_enabled snapshot; then
  run_phase snapshot env PROD_SRC="$PROD_SRC" bash "$SCRIPT_DIR/create_exact_prod_snapshot_repo.sh"
fi
[[ -e "$CURRENT/.git" ]] || {
  echo "ERROR exact-production snapshot pointer missing: $CURRENT" >&2
  exit 30
}
[[ -z "$(git -C "$CURRENT" status --porcelain)" ]] || {
  echo "ERROR exact-production snapshot is dirty: $CURRENT" >&2
  exit 31
}
note "EXACT_SNAPSHOT=$CURRENT"
note "EXACT_SNAPSHOT_HEAD=$(git -C "$CURRENT" rev-parse HEAD)"

# ---------- Modern Foundation ----------
FOUNDATION_SRC="/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918"
FOUNDATION_ALIAS="qwen3.8-flash-next-r2-modern-foundation:256k"
if phase_enabled foundation; then
  if [[ ! -f "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt" ]]; then
    run_phase foundation_prepare env EXACT_SRC="$CURRENT" bash "$SCRIPT_DIR/prepare_modern_foundation.sh"
  else
    echo "FOUNDATION_PREPARE_SKIP recorded candidate already exists"
  fi

  [[ -f "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt" ]] || {
    echo "ERROR foundation manifest missing" >&2; exit 40;
  }
  require_alias "$FOUNDATION_ALIAS"

  run_phase foundation_carryover_audit bash "$SCRIPT_DIR/verify_modern_foundation_carryover.sh"

  # Native recurrent rollback (#28123) is a hard prerequisite for modern MTP.
  # Without it, qwen4exp can fall back to whole-state speculative checkpoints,
  # which is exactly the old catastrophic TG path we do not want to benchmark.
  run_phase foundation_rs_rollback_audit env SRC="$FOUNDATION_SRC" CONFIG="$CONFIG" ALIAS="$FOUNDATION_ALIAS" \
    bash "$SCRIPT_DIR/verify_qwen4exp_native_rs_rollback.sh"
  note "NATIVE_RS_ROLLBACK=PASS"

  run_phase foundation_ab bash "$SCRIPT_DIR/run_modern_foundation_ab.sh"
  note "FOUNDATION_RESULT=PASS"
  note "FOUNDATION_SRC=$FOUNDATION_SRC"
  note "FOUNDATION_ALIAS=$FOUNDATION_ALIAS"
fi

if [[ "$STOP_AFTER" == foundation ]]; then
  note "PHASE1_STOP=foundation"
  echo "SUMMARY=$SUMMARY"
  exit 0
fi

# ---------- Modern MTP / PR #28243 ----------
MTP_SRC="/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918"
MTP_ALIAS="qwen3.8-flash-next-r2-modern-mtp:256k"
if phase_enabled mtp; then
  [[ -f "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt" ]] || {
    echo "ERROR foundation is required before MTP" >&2; exit 50;
  }
  require_alias "$FOUNDATION_ALIAS"

  # Recheck rollback plumbing when resuming directly at MTP. The source support
  # is useless if the selected llama-swap alias is no longer draft-mtp/n-max>0.
  run_phase mtp_rs_rollback_audit env SRC="$FOUNDATION_SRC" CONFIG="$CONFIG" ALIAS="$FOUNDATION_ALIAS" \
    bash "$SCRIPT_DIR/verify_qwen4exp_native_rs_rollback.sh"

  run_phase mtp_layout env CONFIG="$CONFIG" ALIAS="$FOUNDATION_ALIAS" bash "$SCRIPT_DIR/inspect_stage10_mtp_layout.sh"

  if [[ ! -f "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt" ]]; then
    run_phase mtp_prepare env BASE_SRC="$FOUNDATION_SRC" BASELINE_ALIAS="$FOUNDATION_ALIAS" \
      bash "$SCRIPT_DIR/prepare_stage10_upstream_mtp.sh"
  else
    echo "MTP_PREPARE_SKIP recorded candidate already exists"
  fi

  [[ -f "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt" ]] || {
    echo "ERROR MTP manifest missing" >&2; exit 51;
  }
  require_alias "$MTP_ALIAS"
  run_phase mtp_ab bash "$SCRIPT_DIR/run_stage10_mtp_ab.sh"
  note "MTP_RESULT=PASS"
  note "MTP_SRC=$MTP_SRC"
  note "MTP_ALIAS=$MTP_ALIAS"
fi

if [[ "$STOP_AFTER" == mtp ]]; then
  note "PHASE1_STOP=mtp"
  echo "SUMMARY=$SUMMARY"
  exit 0
fi

# ---------- ROCm TOP_K / PR #28313 ----------
TOPK_SRC="/app/share/llama_box/src/llama.cpp-flashnext-r2-rocm-topk-20260918"
TOPK_ALIAS="qwen3.8-flash-next-r2-rocm-topk:256k"
if phase_enabled topk; then
  [[ -f "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt" ]] || {
    echo "ERROR Modern MTP is required before TOP_K" >&2; exit 60;
  }
  require_alias "$MTP_ALIAS"

  if [[ ! -f "$TOPK_SRC/r2-meta/stage14-rocm-topk-manifest.txt" ]]; then
    run_phase topk_prepare env BASE_SRC="$MTP_SRC" BASE_ALIAS="$MTP_ALIAS" \
      bash "$SCRIPT_DIR/prepare_stage14_rocm_topk.sh"
  else
    echo "TOPK_PREPARE_SKIP recorded candidate already exists"
  fi
  [[ -f "$TOPK_SRC/r2-meta/stage14-rocm-topk-manifest.txt" ]] || {
    echo "ERROR TOP_K manifest missing" >&2; exit 61;
  }
  require_alias "$TOPK_ALIAS"

  echo
  echo "=== TOP_K 32K/64K smoke ==="
  set +e
  bash "$SCRIPT_DIR/run_stage14_rocm_topk_ab.sh"
  TOPK_RC=$?
  set -e

  case "$TOPK_RC" in
    0)
      note "TOPK_SMOKE_RESULT=PASS"
      note "TOPK_ALIAS=$TOPK_ALIAS"
      if [[ "$TOPK_FULL" == 1 ]]; then
        echo "=== TOP_K full 16K/32K/64K/128K confirmation ==="
        FULL=1 bash "$SCRIPT_DIR/run_stage14_rocm_topk_ab.sh"
        note "TOPK_FULL_RESULT=PASS"
      else
        note "TOPK_FULL_RESULT=NOT_RUN"
      fi
      ;;
    3)
      note "TOPK_SMOKE_RESULT=HIP_GRAPH_INTERACTION"
      note "TOPK_WINNER=UNRESOLVED"
      ;;
    *)
      note "TOPK_SMOKE_RESULT=REJECT"
      note "TOPK_WINNER=$MTP_ALIAS"
      echo "TOP_K candidate rejected by smoke gate; production remains untouched."
      ;;
  esac
fi

require_alias "$PROD_ALIAS"
sha256sum "$CONFIG" > "$RUN_DIR/config-after.sha256"
cp -a "$CONFIG" "$RUN_DIR/config-after.yaml"

note "PRODUCTION_ALIAS=$PROD_ALIAS"
note "PRODUCTION_PROMOTED=NO"
note "CONFIG_AFTER=$RUN_DIR/config-after.yaml"
note "MASTER_LOG=$MASTER_LOG"
note "FINISHED=$(date -Is)"

echo
echo "================================================================"
echo "FLASH NEXT R2 PHASE-1 COMPLETE"
echo "================================================================"
cat "$SUMMARY"
echo
echo "No production promotion was performed."
