#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 8: incremental pooled-key cache for qwen4exp QSA.
# Stack: exact production snapshot + PR #28213 QSA gather + PR #28699 pooled cache.
# Same binary on both aliases; QSA gather ON on both. Only pooled cache differs.
# Run via with_exact_prod.sh so live uncommitted HotSeat edits are preserved.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-qsa-pooled-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-qsa-pooled}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

GATHER_HEAD="beed2f78ac42cf16710b763e6f3ba20665c6d233"
GATHER_PATCH="$PATCH_DIR/pr28213-qsa-gather-${GATHER_HEAD:0:8}.patch"
GATHER_URL="https://github.com/abdel-darwish-27/llama.cpp/commit/${GATHER_HEAD}.patch"
POOLED_HEAD="141f3f5646aa15e88d53198610a7540f4f4b0d71"
POOLED_PATCH="$PATCH_DIR/pr28699-qsa-pooled-${POOLED_HEAD:0:8}.patch"
POOLED_URL="https://github.com/Rhonstin/llama.cpp/commit/${POOLED_HEAD}.patch"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-pooled-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-pooled-on:256k}"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty; Stage 8 refuses to lose live HotSeat changes." >&2
  echo "Use: bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage8_qsa_pooled.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: Stage-8 worktree already exists: $R2_SRC" >&2
  exit 4
fi
if [[ -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-8 runtime already exists: $RUNTIME" >&2
  exit 5
fi

if ! grep -RqiE 'radix.*top.?k|top.?k.*radix|radix_select|top_k_radix' "$PROD_SRC/ggml/src/ggml-cuda"; then
  echo "ERROR: ROCm long-row radix TOP_K not confirmed in exact production source." >&2
  echo "Run verify_stage7_rocm_topk.sh first." >&2
  exit 6
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Exact production  : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "Stage-8 worktree  : $R2_SRC"
echo "Stage-8 runtime   : $RUNTIME"
echo "QSA gather head   : $GATHER_HEAD"
echo "Pooled-cache head : $POOLED_HEAD"

mkdir -p "$PATCH_DIR"
fetch_patch() {
  local url="$1" dst="$2" head="$3"
  if [[ ! -f "$dst" ]]; then curl -fL --retry 3 --connect-timeout 20 "$url" -o "$dst"; fi
  if ! head -n 1 "$dst" | grep -qi "${head:0:12}"; then
    echo "ERROR: patch does not identify pinned commit $head: $dst" >&2
    exit 7
  fi
  echo "patch=$(basename "$dst") sha256=$(sha256sum "$dst" | awk '{print $1}')"
}
fetch_patch "$GATHER_URL" "$GATHER_PATCH" "$GATHER_HEAD"
fetch_patch "$POOLED_URL" "$POOLED_PATCH" "$POOLED_HEAD"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "gather_pr=ggml-org/llama.cpp#28213"
  echo "gather_head=$GATHER_HEAD"
  echo "gather_sha256=$(sha256sum "$GATHER_PATCH" | awk '{print $1}')"
  echo "pooled_pr=ggml-org/llama.cpp#28699"
  echo "pooled_head=$POOLED_HEAD"
  echo "pooled_sha256=$(sha256sum "$POOLED_PATCH" | awk '{print $1}')"
} > "$R2_SRC/r2-meta/stage8-qsa-pooled-base.txt"

cd "$R2_SRC"
apply_and_commit() {
  local patch="$1" msg="$2"
  if ! git apply --3way "$patch"; then
    echo "ERROR: failed to apply $patch" >&2
    echo "Candidate tree kept for manual semantic migration." >&2
    git status --short >&2 || true
    exit 10
  fi
  git diff --check
  git -c user.name='FlashNext R2 Experiment' -c user.email='flashnext-r2@local.invalid' commit -am "$msg" >/dev/null
}
apply_and_commit "$GATHER_PATCH" "r2 stage8 base: qsa gather pr28213"
apply_and_commit "$POOLED_PATCH" "r2 stage8: qsa pooled-key cache pr28699"

git diff HEAD~2..HEAD > r2-meta/stage8-qsa-pooled-stack.diff

grep -Rni 'LLAMA_QSA_NO_POOLED_CACHE' src | tee r2-meta/stage8-killswitch.txt
if ! grep -Rq 'LLAMA_QSA_NO_POOLED_CACHE' src; then echo "ERROR: pooled-cache kill switch missing" >&2; exit 11; fi
if ! grep -Rq 'QWEN4EXP_QSA_GATHER' src; then echo "ERROR: QSA gather kill switch missing" >&2; exit 12; fi

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-qsa-pooled}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage8-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage8-version.txt" || true

# Preserve exact production JMAX/HotSeat/MTP env. Both arms force QSA gather ON.
# OFF explicitly disables pooled cache.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=1 \
  --env LLAMA_QSA_NO_POOLED_CACHE=1 \
  --replace

# ON explicitly removes any inherited presence-based kill switch before validate.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --unset-env LLAMA_QSA_NO_POOLED_CACHE \
  --env QWEN4EXP_QSA_GATHER=1 \
  --replace --validate

echo
echo "Stage-8 QSA pooled-key runtime ready."
echo "OFF alias: $OFF_ALIAS"
echo " ON alias: $ON_ALIAS"
echo "Runtime  : $RUNTIME/llama-server"
echo "Production alias/runtime remain untouched."
echo "Next: bash $SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
