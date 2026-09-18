#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper.
# Canonical Stage-10 implementation lives in prepare_stage10_upstream_mtp.sh.
# Keep one implementation only: two scripts that differ in JMAX/base-patch logic
# are how benchmarks turn into folklore.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "NOTICE: prepare_stage10_mtp_upstream.sh is deprecated; forwarding to canonical Stage-10 script." >&2
exec bash "$SCRIPT_DIR/prepare_stage10_upstream_mtp.sh" "$@"
