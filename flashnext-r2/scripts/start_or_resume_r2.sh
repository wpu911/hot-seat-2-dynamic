#!/usr/bin/env bash
set -euo pipefail

# Stable entrypoint for ChatGPT Work / host-controller sessions.
# It refuses to retroactively create an environment lock after real R2 benchmark
# summaries already exist, because that would certify only the *current* machine
# state, not the state those old measurements actually used.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
LOCK="${R2_ENV_LOCK:-$LOG_DIR/flashnext-r2-environment-lock.json}"

bash "$SCRIPT_DIR/selfcheck_r2_repo.sh"

have_measurements=0
if find "$LOG_DIR" -maxdepth 2 -type f \
    \( -path '*/flashnext-r2-phase1-*/summary.env' \
       -o -path '*/flashnext-r2-phase2-*/summary.env' \
       -o -path '*/flashnext-r2-phase3-validation-*/summary.env' \
       -o -path '*/flashnext-r2-phase4-*/summary.env' \
       -o -path '*/flashnext-r2-phase4b-param-validation-*/summary.env' \
       -o -path '*/flashnext-r2-phase5-gdn-*/summary.env' \
       -o -path '*/flashnext-r2-phase6-tensor-split-*/summary.env' \
       -o -path '*/flashnext-r2-phase6b-ratio-*/summary.env' \) \
    -print -quit 2>/dev/null | grep -q .; then
  have_measurements=1
fi

if [[ ! -f "$LOCK" ]]; then
  if [[ "$have_measurements" == 1 ]]; then
    echo "ERROR: R2 benchmark summaries already exist but the environment lock does not." >&2
    echo "Refusing to create a retroactive lock. Archive/inspect the old results and establish a new controlled baseline." >&2
    exit 20
  fi
  python3 "$SCRIPT_DIR/lock_r2_environment.py" --create --lock "$LOCK"
else
  python3 "$SCRIPT_DIR/lock_r2_environment.py" --check --lock "$LOCK"
fi

echo
python3 "$SCRIPT_DIR/report_r2_state_plus.py"

echo
echo "R2_RESUME_PREFLIGHT=PASS"
echo "ENVIRONMENT_LOCK=$LOCK"
echo "Execute only the NEXT_ACTION printed above; keep controlled benchmark commands behind r2_env_guard.sh."
