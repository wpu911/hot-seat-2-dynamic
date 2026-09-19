#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash r2_env_guard.sh <command> [args...]
#
# The lock must already exist. This wrapper is deliberately boring: verify the
# frozen llama-swap / production runtime / ROCm linkage, then exec the requested
# phase. It never rewrites a failed lock. Rewriting the control group until it
# agrees with the experiment is not reproducibility, despite the tempting UI.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -gt 0 ]] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }

python3 "$SCRIPT_DIR/lock_r2_environment.py" --check
exec "$@"
