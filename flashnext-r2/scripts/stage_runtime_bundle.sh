#!/usr/bin/env bash
set -euo pipefail

# Stage one experimental llama.cpp build into a relocatable model-private runtime.
#
# Why this exists: `cp build/bin/* runtime/` is not enough when CMake emitted
# RUNPATH entries back into a temporary build tree. The binary can look healthy
# until that build directory is cleaned, at which point production archaeology
# begins. This helper copies the complete bin payload, rewrites only build-tree
# RUNPATH entries to $ORIGIN, verifies ldd resolution, and records a manifest.
#
# Usage:
#   bash stage_runtime_bundle.sh BUILD_BIN RUNTIME [META_DIR]
#
# Environment:
#   REQUIRE_BOTH_GPUS=1  also require gfx1100 + gfx1201 in --list-devices

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="${1:-}"
RUNTIME="${2:-}"
META_DIR="${3:-}"

[[ -n "$BUILD_BIN" && -n "$RUNTIME" ]] || {
  echo "ERROR usage: $0 BUILD_BIN RUNTIME [META_DIR]" >&2
  exit 2
}
BUILD_BIN="$(readlink -f "$BUILD_BIN")"
[[ -d "$BUILD_BIN" ]] || { echo "ERROR build bin missing: $BUILD_BIN" >&2; exit 3; }
[[ -x "$BUILD_BIN/llama-server" ]] || { echo "ERROR llama-server missing: $BUILD_BIN/llama-server" >&2; exit 4; }

if [[ -e "$RUNTIME" ]]; then
  echo "ERROR runtime already exists; refusing to merge bundles: $RUNTIME" >&2
  exit 5
fi
mkdir -p "$RUNTIME"
RUNTIME="$(readlink -f "$RUNTIME")"

if [[ -z "$META_DIR" ]]; then
  META_DIR="$RUNTIME/../r2-runtime-meta"
fi
mkdir -p "$META_DIR"
META_DIR="$(readlink -f "$META_DIR")"

printf '%s\n' '=== stage R2 runtime bundle ==='
echo "build_bin=$BUILD_BIN"
echo "runtime=$RUNTIME"
echo "meta=$META_DIR"

cp -a "$BUILD_BIN/." "$RUNTIME/"

# Normalize before computing final hashes, because patchelf changes ELF bytes.
OUT="$META_DIR/rpath-normalize.txt" \
  bash "$SCRIPT_DIR/normalize_runtime_rpath.sh" "$RUNTIME"

OUT="$META_DIR/runtime-verify.txt" REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-0}" \
  bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$RUNTIME"

SERVER="$RUNTIME/llama-server"
[[ -x "$RUNTIME/llama-server.real" ]] && SERVER="$RUNTIME/llama-server.real"

{
  echo "created=$(date -Is)"
  echo "build_bin=$BUILD_BIN"
  echo "runtime=$RUNTIME"
  echo "server=$SERVER"
  echo "server_sha256=$(sha256sum "$SERVER" | awk '{print $1}')"
  echo "require_both_gpus=${REQUIRE_BOTH_GPUS:-0}"
  echo "rpath_normalize_log=$META_DIR/rpath-normalize.txt"
  echo "runtime_verify_log=$META_DIR/runtime-verify.txt"
} > "$META_DIR/runtime-manifest.env"

# Hash real files and record symlink topology separately. Hashing after RUNPATH
# normalization makes this manifest useful for rollback and later drift checks.
find "$RUNTIME" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 -r sha256sum > "$META_DIR/SHA256SUMS.txt"
find "$RUNTIME" -maxdepth 1 -type l -printf '%f -> %l\n' \
  | sort > "$META_DIR/symlinks.txt"

"$SERVER" --version > "$META_DIR/llama-server-version.txt" 2>&1 || true

printf '%s\n' 'RUNTIME_STAGE=PASS'
echo "RUNTIME=$RUNTIME"
echo "SERVER=$SERVER"
echo "SERVER_SHA256=$(sha256sum "$SERVER" | awk '{print $1}')"
echo "MANIFEST=$META_DIR/runtime-manifest.env"
