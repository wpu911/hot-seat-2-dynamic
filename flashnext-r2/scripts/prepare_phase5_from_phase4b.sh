#!/usr/bin/env bash
set -euo pipefail

# Prepare Phase-5 only from the deep-validated Phase-4b winner. This wrapper
# exists so the older Phase-5 builder can keep its SOURCE_ALIAS override without
# learning another generation of summary-file semantics.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"

latest_phase4b_summary(){
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase4b-param-validation-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value(){ awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"; }

P4B="${PHASE4B_SUMMARY:-$(latest_phase4b_summary || true)}"
[[ -n "$P4B" && -f "$P4B" ]] || {
  echo "ERROR Phase-4b summary missing; run run_phase4b_param_validation.sh first" >&2
  exit 2
}
[[ "$(summary_value "$P4B" VALIDATION)" == PASS ]] || {
  echo "ERROR Phase-4b has not recorded a safe winner" >&2
  exit 3
}
[[ "$(summary_value "$P4B" PRODUCTION_PROMOTED)" == NO ]] || {
  echo "ERROR Phase-4b summary does not prove production remained untouched" >&2
  exit 4
}
SOURCE_ALIAS="${SOURCE_ALIAS:-$(summary_value "$P4B" PHASE4B_WINNER_ALIAS)}"
[[ -n "$SOURCE_ALIAS" ]] || { echo "ERROR PHASE4B_WINNER_ALIAS missing" >&2; exit 5; }

echo "PHASE4B_SUMMARY=$P4B"
echo "PHASE5_SOURCE_ALIAS=$SOURCE_ALIAS"
SOURCE_ALIAS="$SOURCE_ALIAS" bash "$SCRIPT_DIR/prepare_phase5_gdn_microfusion.sh"
