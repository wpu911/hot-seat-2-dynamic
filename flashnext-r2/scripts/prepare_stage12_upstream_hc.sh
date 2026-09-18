#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED COMPATIBILITY ENTRY.
#
# The old implementation applied only #28896/#28901 to the Sep-11 production
# tree. That is no longer an acceptable foundation because PR #28243 currently
# bases on Sep-18 master 911f6cdc..., which is 130 upstream commits beyond the
# recorded Sep-11 base.
#
# Canonical replacement: forward-port the ENTIRE exact production custom overlay
# onto the complete Sep-18 upstream base with prepare_modern_foundation.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "NOTICE: selective-HC Stage 12 is retired; using complete modern foundation." >&2
exec bash "$SCRIPT_DIR/prepare_modern_foundation.sh" "$@"
