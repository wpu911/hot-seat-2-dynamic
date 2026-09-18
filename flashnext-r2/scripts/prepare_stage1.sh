#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-1 preparation.
# Safety rule: NEVER edits the production source tree in place.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-20260918}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${PATCH:-${SCRIPT_DIR}/../patches/0001-mmq-id-jmax.patch}"

if [[ ! -d "$PROD_SRC/.git" && ! -f "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: R2 target already exists: $R2_SRC" >&2
  echo "Refusing to overwrite anything." >&2
  exit 3
fi
if [[ ! -f "$PATCH" ]]; then
  echo "ERROR: patch missing: $PATCH" >&2
  exit 4
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Production source : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "R2 worktree       : $R2_SRC"
echo "Patch             : $PATCH"

# Create a detached worktree from the exact production HEAD.
git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"

# Record exact starting state before any R2 change.
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "patch=$PATCH"
} > "$R2_SRC/r2-meta/base.txt"
git -C "$PROD_SRC" status --short > "$R2_SRC/r2-meta/production-status-at-create.txt" || true

cd "$R2_SRC"

# Apply as a 3-way patch. If upstream/current HotSeat moved mmq.cuh too far,
# fail here rather than forcing a bad merge into a model runtime.
if ! git apply --3way "$PATCH"; then
  echo "ERROR: J-cap patch did not apply cleanly. R2 worktree is kept for manual functional migration." >&2
  exit 10
fi

git diff --check
git diff > r2-meta/stage1-jcap.diff

# Prefer host ROCm 10 mounted into llama_box_714; fall back to /opt/rocm.
if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/bin/hipcc || -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then
    ROCM_PATH=/opt/host-rocm/core-10.0
  else
    ROCM_PATH=/opt/rocm
  fi
fi
export ROCM_PATH

HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-flashnext-r2}"

echo "ROCm path         : $ROCM_PATH"
echo "HIP compiler      : $HIP_CXX"
echo "AMDGPU targets    : $AMDGPU_TARGETS"
echo "Build directory   : $BUILD"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"

cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

BIN="$BUILD/bin/llama-server"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: build finished but llama-server not found: $BIN" >&2
  exit 20
fi

sha256sum "$BIN" | tee r2-meta/llama-server.sha256
"$BIN" --version | tee r2-meta/llama-server.version.txt || true

echo
echo "Stage-1 build complete."
echo "NO production config/runtime was changed."
echo "R2 binary: $BIN"
echo "Next: run bit-exact gate and JMAX off/64/32/16 PP A/B on an isolated test port."
