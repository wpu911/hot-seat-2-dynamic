#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper.
#
# The old version of this script ran:
#   snapshot -> foundation -> MTP -> FR-Spec
#
# That order is now obsolete. The current R2 line must first establish the
# modern foundation, validate MTP, and run the ROCm TOP_K smoke before deciding
# which source/alias becomes the base for QSA gather, pooled-key cache, PLE and
# FR-Spec. Automatically jumping from MTP straight to FR-Spec can build the rest
# of the experiment on the wrong lineage.
#
# Keep this filename only so an old command does not silently execute the stale
# sequence. It now delegates to the guarded phase-1 runner:
#   snapshot -> foundation -> MTP -> TOP_K smoke
#
# After phase-1, continue according to NEXT_RUN_ORDER_20260918.md. Production is
# never promoted automatically.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${START_AT:-}" == "frspec" || "${STOP_AFTER:-}" == "frspec" ]]; then
  cat >&2 <<'EOF'
ERROR: the old 'frspec' pipeline phase was intentionally retired.

Current order after Modern MTP is:
  ROCm TOP_K smoke
  -> QSA gather
  -> pooled-key cache
  -> PLE direct-read
  -> FR-Spec

Run the phase-1 wrapper first, then select the actual winning lineage before
stacking later stages. This prevents a fast-looking but structurally wrong
candidate from becoming the base of every subsequent test.
EOF
  exit 64
fi

exec bash "$SCRIPT_DIR/run_phase1_real_ab.sh" "$@"
