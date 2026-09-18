#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "NOTICE: selective-HC A/B is retired; running complete modern-foundation A/B." >&2
exec bash "$SCRIPT_DIR/run_modern_foundation_ab.sh" "$@"
