#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-1 preparation.
# Safety rule: NEVER edits the production source tree/runtime in place.
# Stage-1 now uses llama-swap itself for A/B; no private llama-server test port.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-20260918}"
R2_RUNTIME="${R2_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-stage1}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${PATCH:-${SCRIPT_DIR}/../patches/0001-mmq-id-jmax.patch}"
ALIAS_TOOL="${SCRIPT_DIR}/install_llamaswap_r2_alias.py"
BENCH_TOOL="${SCRIPT_DIR}/bench_llamaswap_ab.py"

if [[ ! -d "$PROD_SRC/.git" && ! -f "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: R2 target already exists: $R2_SRC" >&2
  echo "Refusing to overwrite anything. Remove only after confirming it is an abandoned R2 worktree." >&2
  exit 3
fi
if [[ -e "$R2_RUNTIME" ]]; then
  echo "ERROR: R2 runtime already exists: $R2_RUNTIME" >&2
  echo "Refusing to overwrite an existing candidate runtime." >&2
  exit 4
fi
if [[ ! -f "$PATCH" || ! -f "$ALIAS_TOOL" || ! -f "$BENCH_TOOL" ]]; then
  echo "ERROR: R2 patch/tool bundle incomplete under $SCRIPT_DIR" >&2
  exit 5
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: llama-swap config not found: $CONFIG" >&2
  exit 6
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Production source : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "R2 worktree       : $R2_SRC"
echo "R2 runtime        : $R2_RUNTIME"
echo "llama-swap config : $CONFIG"
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
  echo "base_alias=$BASE_ALIAS"
  echo "r2_alias=$R2_ALIAS"
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

# Prefer host ROCm 10 mounted into llama_box_714. R9700/gfx1201 requires this path.
if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/bin/hipcc || -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then
    ROCM_PATH=/opt/host-rocm/core-10.0
  elif [[ -x /opt/rocm/core-10.0/bin/hipcc || -x /opt/rocm/core-10.0/lib/llvm/bin/clang++ ]]; then
    ROCM_PATH=/opt/rocm/core-10.0
  else
    echo "ERROR: ROCm10 compiler not found. Refusing to build gfx1201 with the old ROCm7.14 toolchain." >&2
    exit 11
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

unset HSA_OVERRIDE_GFX_VERSION || true

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

# Create an independent runtime directory. Copy the complete build/bin set so the
# candidate never borrows production .so files by accident.
mkdir -p "$R2_RUNTIME/bin"
cp -a "$BUILD/bin/." "$R2_RUNTIME/bin/"
cp -a r2-meta "$R2_RUNTIME/"
{
  echo "created=$(date -Is)"
  echo "source=$R2_SRC"
  echo "production_head=$PROD_HEAD"
  echo "targets=$AMDGPU_TARGETS"
  echo "rocm_path=$ROCM_PATH"
  sha256sum "$R2_RUNTIME/bin/llama-server"
} > "$R2_RUNTIME/manifest.txt"

# Register a second llama-swap alias by cloning the CURRENT production Flash Next
# block. This preserves every model path, MTP draft path, HotSeat variable, GPU
# split and KV setting. Only binary path + JMAX differ.
python3 "$ALIAS_TOOL" \
  --config "$CONFIG" \
  --source-alias "$BASE_ALIAS" \
  --alias "$R2_ALIAS" \
  --r2-bin "$R2_RUNTIME/bin" \
  --jmax "${JMAX:-32}" \
  --validate

# llama-swap production was started with -watch-config, so no container restart is
# required merely to register the alias. Verify the control plane is still alive.
curl -fsS http://127.0.0.1:8090/health >/dev/null
curl -fsS http://127.0.0.1:8090/v1/models > "$R2_RUNTIME/models-after-alias.json" || true

echo
echo "Stage-1 build + llama-swap alias registration complete."
echo "Production alias untouched : $BASE_ALIAS"
echo "R2 alias                  : $R2_ALIAS"
echo "R2 binary                 : $R2_RUNTIME/bin/llama-server"
echo "JMAX                       : ${JMAX:-32}"
echo
echo "Run the real-path A/B through llama-swap:"
echo "  python3 $BENCH_TOOL --baseline '$BASE_ALIAS' --r2 '$R2_ALIAS'"
