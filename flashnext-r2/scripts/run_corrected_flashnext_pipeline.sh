#!/usr/bin/env bash
set -euo pipefail

# Guarded Flash Next R2 pipeline, corrected for the Sep-18 upstream foundation.
#
# Order:
#   0. freeze exact live production source (including local HotSeat edits)
#   1. forward-port ONLY the production custom overlay onto Sep-18 upstream
#   2. llama-swap A/B production vs modern foundation
#   3. add PR #28243 MTP delta on that SAME foundation
#   4. llama-swap A/B foundation vs modern MTP + cached Large-PP regression
#   5. add workload-ranked FR-Spec 65K draft sidecar
#   6. llama-swap A/B full-vocab draft vs FR-Spec 65K
#
# Every benchmark/analyzer exits non-zero on a failed gate. set -e means the
# pipeline stops immediately. It never promotes a candidate into the production
# alias automatically.
#
# Usage:
#   bash run_corrected_flashnext_pipeline.sh
#
# Resume after an already completed phase:
#   START_AT=mtp bash run_corrected_flashnext_pipeline.sh
#   START_AT=frspec bash run_corrected_flashnext_pipeline.sh
#
# Stop after a phase:
#   STOP_AFTER=foundation bash ...
#   STOP_AFTER=mtp bash ...
#
# Valid phases: snapshot | foundation | mtp | frspec

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIVE_PROD_SRC="${LIVE_PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
START_AT="${START_AT:-snapshot}"
STOP_AFTER="${STOP_AFTER:-frspec}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
MASTER_LOG="$LOG_DIR/flashnext-r2-corrected-pipeline-$STAMP.log"
mkdir -p "$LOG_DIR"
exec > >(tee "$MASTER_LOG") 2>&1

PHASES=(snapshot foundation mtp frspec)
phase_index() {
  local want="$1" i
  for i in "${!PHASES[@]}"; do
    [[ "${PHASES[$i]}" == "$want" ]] && { echo "$i"; return 0; }
  done
  return 1
}

START_I="$(phase_index "$START_AT")" || { echo "ERROR: invalid START_AT=$START_AT" >&2; exit 2; }
STOP_I="$(phase_index "$STOP_AFTER")" || { echo "ERROR: invalid STOP_AFTER=$STOP_AFTER" >&2; exit 3; }
(( START_I <= STOP_I )) || { echo "ERROR: START_AT occurs after STOP_AFTER" >&2; exit 4; }

run_phase() {
  local name="$1"; shift
  local idx
  idx="$(phase_index "$name")"
  (( idx >= START_I && idx <= STOP_I )) || return 0
  echo
  echo "================================================================"
  echo "PHASE=$name START $(date -Is)"
  echo "CMD=$*"
  echo "================================================================"
  "$@"
  echo "PHASE=$name PASS $(date -Is)"
}

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: required file missing: $1" >&2; exit 20; }
}

require_alias() {
  local alias="$1" cfg="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
  grep -qE "^[[:space:]]*${alias//./\.}:[[:space:]]*(#.*)?$" "$cfg" || {
    echo "ERROR: required llama-swap alias missing: $alias" >&2
    exit 21
  }
}

# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
if (( START_I <= 0 && STOP_I >= 0 )); then
  run_phase snapshot env PROD_SRC="$LIVE_PROD_SRC" bash "$SCRIPT_DIR/create_exact_prod_snapshot_repo.sh"
fi

CURRENT="/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current"
if (( STOP_I >= 1 )); then
  [[ -e "$CURRENT/.git" ]] || {
    echo "ERROR: exact-production current pointer unavailable: $CURRENT" >&2
    echo "If resuming, create/repoint the exact snapshot first." >&2
    exit 22
  }
  [[ -z "$(git -C "$CURRENT" status --porcelain)" ]] || {
    echo "ERROR: exact-production current snapshot is dirty" >&2
    exit 23
  }
fi

# ---------------------------------------------------------------------------
# modern foundation
# ---------------------------------------------------------------------------
FOUNDATION_SRC="/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918"
FOUNDATION_ALIAS="qwen3.8-flash-next-r2-modern-foundation:256k"
if (( START_I <= 1 && STOP_I >= 1 )); then
  if [[ ! -f "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt" ]]; then
    run_phase foundation env EXACT_SRC="$CURRENT" bash "$SCRIPT_DIR/prepare_modern_foundation.sh"
  else
    echo "PHASE=foundation PREPARE_SKIP existing recorded foundation: $FOUNDATION_SRC"
  fi
  require_file "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt"
  require_alias "$FOUNDATION_ALIAS"
  # This analyzer is a non-regression gate. It intentionally does not require a
  # speedup just for adopting a newer upstream foundation.
  bash "$SCRIPT_DIR/run_modern_foundation_ab.sh"
  echo "PHASE=foundation A_B_PASS $(date -Is)"
fi

(( STOP_I == 1 )) && {
  echo
  echo "PIPELINE_STOP_AFTER=foundation"
  echo "MASTER_LOG=$MASTER_LOG"
  exit 0
}

# ---------------------------------------------------------------------------
# modern MTP, PR #28243 only
# ---------------------------------------------------------------------------
MTP_SRC="/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918"
MTP_ALIAS="qwen3.8-flash-next-r2-modern-mtp:256k"
if (( START_I <= 2 && STOP_I >= 2 )); then
  require_file "$FOUNDATION_SRC/r2-meta/modern-foundation-manifest.txt"
  require_alias "$FOUNDATION_ALIAS"
  if [[ ! -f "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt" ]]; then
    run_phase mtp env BASE_SRC="$FOUNDATION_SRC" BASELINE_ALIAS="$FOUNDATION_ALIAS" \
      bash "$SCRIPT_DIR/prepare_stage10_upstream_mtp.sh"
  else
    echo "PHASE=mtp PREPARE_SKIP existing recorded candidate: $MTP_SRC"
  fi
  require_file "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt"
  require_alias "$MTP_ALIAS"
  # Includes fresh workload A/B and cached Large-PP/high-LCP regression by default.
  bash "$SCRIPT_DIR/run_stage10_mtp_ab.sh"
  echo "PHASE=mtp A_B_PASS $(date -Is)"
fi

(( STOP_I == 2 )) && {
  echo
  echo "PIPELINE_STOP_AFTER=mtp"
  echo "MASTER_LOG=$MASTER_LOG"
  exit 0
}

# ---------------------------------------------------------------------------
# FR-Spec 65K on the winning modern MTP lineage
# ---------------------------------------------------------------------------
FR_SRC="/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-frspec-20260918"
FR_ALIAS="qwen3.8-flash-next-r2-modern-frspec-65k:256k"
FULL_ALIAS="qwen3.8-flash-next-r2-modern-frspec-full:256k"
if (( START_I <= 3 && STOP_I >= 3 )); then
  require_file "$MTP_SRC/r2-meta/stage10-modern-mtp-base.txt"
  require_alias "$MTP_ALIAS"
  if [[ ! -e "$FR_SRC/.git" ]]; then
    run_phase frspec env BASE_SRC="$MTP_SRC" SOURCE_ALIAS="$MTP_ALIAS" \
      bash "$SCRIPT_DIR/prepare_stage13_frspec_modern.sh"
  else
    echo "PHASE=frspec PREPARE_SKIP existing candidate tree: $FR_SRC"
  fi
  require_alias "$FULL_ALIAS"
  require_alias "$FR_ALIAS"
  bash "$SCRIPT_DIR/run_stage13_frspec_modern_ab.sh"
  echo "PHASE=frspec A_B_PASS $(date -Is)"
fi

echo
echo "================================================================"
echo "CORRECTED FLASH NEXT R2 PIPELINE PASSED ALL REQUESTED PHASES"
echo "================================================================"
echo "production alias remains: qwen3.8-flash-next:256k"
echo "best final candidate : $FR_ALIAS"
echo "modern MTP fallback : $MTP_ALIAS"
echo "foundation fallback : $FOUNDATION_ALIAS"
echo "MASTER_LOG=$MASTER_LOG"
echo
echo "IMPORTANT: this script does NOT promote any candidate to production automatically."
echo "Promotion should happen only after reviewing the emitted A/B JSON and cached rollback logs."
