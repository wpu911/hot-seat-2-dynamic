#!/usr/bin/env bash
set -euo pipefail

# Verify that an experimental llama-server runtime remains usable independently
# of its temporary CMake build directory. System/ROCm libraries may stay external;
# llama.cpp-local shared libraries must not resolve back into build-r2-* trees.
#
# Usage:
#   bash verify_runtime_bundle.sh /path/to/runtime/bin
#   SERVER=/path/to/runtime/bin/llama-server.real bash verify_runtime_bundle.sh /path/to/runtime/bin
#
# SERVER override exists for wrapper runtimes. ldd/readelf must inspect the real
# ELF, not the shell wrapper that injects split/lazy-mode arguments.

RUNTIME="${1:-${RUNTIME:-}}"
[[ -n "$RUNTIME" ]] || { echo "ERROR: runtime bin directory required" >&2; exit 2; }
RUNTIME="$(readlink -f "$RUNTIME")"
SERVER="${SERVER:-$RUNTIME/llama-server}"
SERVER="$(readlink -f "$SERVER")"
[[ -x "$SERVER" ]] || { echo "ERROR: llama-server not executable: $SERVER" >&2; exit 3; }
case "$SERVER" in
  "$RUNTIME"/*) ;;
  *) echo "ERROR: audited server is outside runtime dir: $SERVER" >&2; exit 4 ;;
esac

OUT="${OUT:-}"
if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  exec > >(tee "$OUT") 2>&1
fi

echo "=== Flash Next R2 runtime verification ==="
echo "runtime=$RUNTIME"
echo "server=$SERVER"
echo "sha256=$(sha256sum "$SERVER" | awk '{print $1}')"

if command -v readelf >/dev/null 2>&1; then
  echo "--- RUNPATH/RPATH ---"
  readelf -d "$SERVER" 2>/dev/null | grep -E 'RPATH|RUNPATH|NEEDED' || true
fi

echo "--- ldd ---"
LDD="$(mktemp)"
trap 'rm -f "$LDD"' EXIT
ldd "$SERVER" | tee "$LDD"

if grep -q 'not found' "$LDD"; then
  echo "ERROR: unresolved shared library dependency" >&2
  exit 10
fi

if grep -E '/app/share/llama_box/src/.*/build[^/]*/' "$LDD" >/dev/null; then
  echo "ERROR: runtime still resolves llama.cpp libraries from a build directory:" >&2
  grep -E '/app/share/llama_box/src/.*/build[^/]*/' "$LDD" >&2 || true
  exit 11
fi

BAD_LOCAL=0
while IFS= read -r line; do
  case "$line" in
    *libllama*.so*|*libggml*.so*)
      path="$(sed -n 's/.*=> \([^ ]*\).*/\1/p' <<<"$line")"
      [[ -z "$path" ]] && continue
      case "$path" in
        "$RUNTIME"/*) ;;
        /usr/lib/*|/lib/*|/opt/rocm/*|/opt/host-rocm/*) ;;
        *) echo "ERROR: llama/ggml library resolves outside staged runtime: $line" >&2; BAD_LOCAL=1 ;;
      esac
      ;;
  esac
done < "$LDD"
[[ "$BAD_LOCAL" == 0 ]] || exit 12

echo "--- version ---"
"$SERVER" --version

echo "--- devices ---"
DEV="$($SERVER --list-devices 2>&1)" || {
  echo "$DEV"
  echo "ERROR: --list-devices failed" >&2
  exit 13
}
printf '%s\n' "$DEV"

HAS_7900=0; HAS_9700=0
if grep -Eiq '7900[[:space:]_-]*XTX|gfx1100' <<<"$DEV"; then HAS_7900=1; fi
if grep -Eiq 'R9700|AI PRO R9700|gfx1201' <<<"$DEV"; then HAS_9700=1; fi
echo "detected_7900_or_gfx1100=$HAS_7900"
echo "detected_r9700_or_gfx1201=$HAS_9700"
if [[ "${REQUIRE_BOTH_GPUS:-0}" == 1 && ( "$HAS_7900" != 1 || "$HAS_9700" != 1 ) ]]; then
  echo "ERROR: expected both gfx1100/7900 XTX and gfx1201/R9700 in --list-devices" >&2
  exit 14
fi

echo "RUNTIME_BUNDLE=PASS"
