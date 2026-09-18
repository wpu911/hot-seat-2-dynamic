#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-2: HyperConnection glue fusions.
# This stage is intentionally independent from Stage-1 J-cap so TG effects can be
# measured without mixing PP-only changes. It starts from the exact production
# source HEAD and creates a separate worktree/runtime.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-hc-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-hc}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/0004-hc-combine-hc-mix.patch"
PATCH_URL="https://raw.githubusercontent.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4/185252d1edb27fde6b332908eb7c89a20cadc4bb/johnv8/patches/seria-fuzje/0004-fuzje-hc-combine-hc-mix-z-forka-c689018e4-f76552838.patch"
PATCH_GIT_BLOB="17776143967955297e5a547e04cf209ed0d3d8e2"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-hc-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-hc-on:256k}"

if [[ ! -d "$PROD_SRC/.git" && ! -f "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: Stage-2 worktree already exists: $R2_SRC" >&2
  echo "Refusing to overwrite. Remove it explicitly only after checking it." >&2
  exit 3
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Production source : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "Stage-2 worktree  : $R2_SRC"
echo "Stage-2 runtime   : $RUNTIME"

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
ACTUAL_BLOB="$(git hash-object "$PATCH")"
if [[ "$ACTUAL_BLOB" != "$PATCH_GIT_BLOB" ]]; then
  echo "ERROR: pinned upstream HC patch hash mismatch" >&2
  echo "expected git blob: $PATCH_GIT_BLOB" >&2
  echo "actual   git blob: $ACTUAL_BLOB" >&2
  exit 4
fi
echo "HC patch verified : $ACTUAL_BLOB"

# Separate detached worktree. Production tree is never patched in place.
git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "patch_url=$PATCH_URL"
  echo "patch_git_blob=$PATCH_GIT_BLOB"
} > "$R2_SRC/r2-meta/stage2-hc-base.txt"
git -C "$PROD_SRC" status --short > "$R2_SRC/r2-meta/production-status-at-create.txt" || true

cd "$R2_SRC"
if ! git apply --3way "$PATCH"; then
  echo "ERROR: HC patch did not apply cleanly to the production HEAD." >&2
  echo "The worktree is intentionally kept for functional/manual migration." >&2
  exit 10
fi

git diff --check
git diff > r2-meta/stage2-hc.diff

# ROCm10 toolchain mounted in llama_box_714 is preferred.
if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then
    ROCM_PATH=/opt/host-rocm/core-10.0
  else
    ROCM_PATH=/opt/rocm
  fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-hc}"

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

if [[ ! -x "$BUILD/bin/llama-server" ]]; then
  echo "ERROR: build completed without llama-server" >&2
  exit 20
fi

# Stage a self-contained bin directory. Never overwrite production runtime.
if [[ -e "$RUNTIME" ]]; then
  echo "ERROR: runtime already exists: $RUNTIME" >&2
  exit 21
fi
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage2-hc-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage2-hc-version.txt" || true

# Register two aliases using the SAME binary. Only the fusion env differs.
# JMAX=0 on both sides, deliberately keeping Stage-1 out of this TG experiment.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax 0 \
  --env GGML_JOHNV8_HC_FUSE=0 \
  --env GGML_JOHNV8_MIX_FUSE=0 \
  --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax 0 \
  --env GGML_JOHNV8_HC_FUSE=1 \
  --env GGML_JOHNV8_MIX_FUSE=1 \
  --replace \
  --validate

echo
echo "Stage-2 HC build and llama-swap aliases are ready."
echo "OFF alias: $OFF_ALIAS"
echo " ON alias: $ON_ALIAS"
echo "Binary   : $RUNTIME/llama-server"
echo "Production alias/runtime remain untouched."
echo
echo "Next command:"
echo "  bash $SCRIPT_DIR/run_stage2_hc_ab.sh"
