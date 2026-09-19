#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/verify_phase6b_ratio_runtime.sh"
exec bash "$SCRIPT_DIR/run_phase6b_tensor_ratio_sweep.sh"
