#!/usr/bin/env bash
set -euo pipefail

# Final correctness tail after Phase-6/6b has selected a current winner.
# No production promotion happens here.
#
# Order matters:
#   1) frozen experiment environment is still identical
#   2) generated assistant turn is actually reusable with draft-MTP
#   3) multi-slot MTP does not cross-contaminate requests
#   4) real OpenClaw Gateway/session path works
#   5) read-only promotion evidence bundle is generated

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/lock_r2_environment.py" --check
python3 "$SCRIPT_DIR/run_mtp_turn_reuse_gate.py"
python3 "$SCRIPT_DIR/run_mtp_parallel_isolation.py"
python3 "$SCRIPT_DIR/run_final_openclaw_regression.py"
python3 "$SCRIPT_DIR/prepare_promotion_review.py"

echo
printf '%s\n' '============================================================'
printf '%s\n' 'FLASH NEXT R2 FINAL TAIL GATES PASSED'
printf '%s\n' '============================================================'
printf '%s\n' 'Environment lock: PASS'
printf '%s\n' 'MTP generated-turn reuse: PASS'
printf '%s\n' 'MTP multi-slot isolation: PASS'
printf '%s\n' 'OpenClaw Gateway regression: PASS'
printf '%s\n' 'Promotion review bundle: READY'
printf '%s\n' 'PRODUCTION_PROMOTED=NO'
