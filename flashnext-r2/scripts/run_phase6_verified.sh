#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stable Phase-6 entrypoint. Audit the staged layer/tensor runtimes before model
# loading so a broken wrapper/RUNPATH cannot masquerade as a performance result.
bash "$SCRIPT_DIR/verify_phase6_split_runtime.sh"
exec bash "$SCRIPT_DIR/run_phase6_tensor_split_ab.sh"
