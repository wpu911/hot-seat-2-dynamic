#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FULL="${FULL:-qwen3.8-flash-next-r2-modern-frspec-full:256k}"
export FR="${FR:-qwen3.8-flash-next-r2-modern-frspec-65k:256k}"
export OUT="${OUT:-/app/share/openclaw_tools/logs/flashnext-r2-modern-frspec-65k-ab.json}"

exec bash "$SCRIPT_DIR/run_stage12_frspec_ab.sh" "$@"
