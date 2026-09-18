#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 8: incremental pooled-key cache for the QSA indexer.
# Based on ggml-org/llama.cpp PR #28699, pinned to head commit 141f3f56...
# This stage is tested independently from QSA gather so its contribution is clear.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-qsa-pool-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-qsa-pool}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr28699-qsa-pooled-141f3f56.patch"
PATCH_URL="https://github.com/Rhonstin/llama.cpp/commit/141f3f5646aa15e88d53198610a7540f4f4b0d71.patch"
PATCH_HEAD="141f3f5646aa15e88d53198610a7540f4f4b0d71"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-pool-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-pool-on:256k}"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-8 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 3
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: pooled-cache patch does not identify pinned commit $PATCH_HEAD" >&2
  exit 4
fi

echo "Production HEAD   : $PROD_HEAD"
echo "Stage-8 worktree  : $R2_SRC"
echo "Stage-8 runtime   : $RUNTIME"
echo "PR28699 head      : $PATCH_HEAD"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28699"
  echo "patch_head=$PATCH_HEAD"
  echo "patch_url=$PATCH_URL"
  echo "patch_sha256=$(sha256sum "$PATCH" | awk '{print $1}')"
} > "$R2_SRC/r2-meta/stage8-qsa-pool-base.txt"
git -C "$PROD_SRC" status --short > "$R2_SRC/r2-meta/production-status-at-create.txt" || true

cd "$R2_SRC"
if ! git apply --3way "$PATCH"; then
  echo "ERROR: PR #28699 did not apply cleanly to production HotSeat tree." >&2
  echo "The worktree is kept for manual/functional migration; production is untouched." >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check
git diff > r2-meta/stage8-qsa-pooled.diff

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
BUILD="${BUILD:-$R2_SRC/build-r2-qsa-pool}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing"; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage8-qsa-pool-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage8-qsa-pool-version.txt" || true

# OFF: presence of LLAMA_QSA_NO_POOLED_CACHE disables the cache.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax 0 \
  --env LLAMA_QSA_NO_POOLED_CACHE=1 \
  --replace

# ON: make absolutely sure the presence-based kill switch is absent in the clone.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax 0 \
  --unset-env LLAMA_QSA_NO_POOLED_CACHE \
  --replace \
  --validate

echo
echo "Stage-8 QSA pooled-key cache runtime ready."
echo "OFF alias: $OFF_ALIAS"
echo " ON alias: $ON_ALIAS"
echo "Runtime  : $RUNTIME/llama-server"
echo "Next     : bash $SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
