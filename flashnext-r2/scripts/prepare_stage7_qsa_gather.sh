#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 7B: QSA gather-based sparse attention for decode.
# Based on ggml-org/llama.cpp PR #28213, pinned to its current single-commit head.
# The SAME patched binary is exposed through OFF/ON llama-swap aliases so the
# only variable is QWEN4EXP_QSA_GATHER.
#
# Run through with_exact_prod.sh. A raw live production tree may contain
# intentional uncommitted HotSeat edits and must not be reduced to HEAD.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-qsa-gather-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-qsa-gather}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr28213-qsa-gather-beed2f78.patch"
PATCH_URL="https://github.com/abdel-darwish-27/llama.cpp/commit/beed2f78ac42cf16710b763e6f3ba20665c6d233.patch"
PATCH_HEAD="beed2f78ac42cf16710b763e6f3ba20665c6d233"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-qsa-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-qsa-on:256k}"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty; Stage 7 refuses to benchmark HEAD while dropping live HotSeat edits." >&2
  echo "Use: bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage7_qsa_gather.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: Stage-7 QSA worktree already exists: $R2_SRC" >&2
  exit 4
fi
if [[ -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-7 QSA runtime already exists: $RUNTIME" >&2
  exit 5
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Exact production  : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "Stage-7 worktree  : $R2_SRC"
echo "Stage-7 runtime   : $RUNTIME"
echo "PR28213 head      : $PATCH_HEAD"

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: downloaded patch does not identify pinned commit $PATCH_HEAD" >&2
  head -n 3 "$PATCH" >&2 || true
  exit 6
fi

# Long-context QSA numbers are meaningless if TOP_K falls back to CPU.
if ! grep -RqiE 'radix.*top.?k|top.?k.*radix|radix_select|top_k_radix' "$PROD_SRC/ggml/src/ggml-cuda"; then
  echo "ERROR: ROCm long-row radix TOP_K not confirmed in exact production source." >&2
  echo "Run verify_stage7_rocm_topk.sh first." >&2
  exit 7
fi

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28213"
  echo "patch_head=$PATCH_HEAD"
  echo "patch_url=$PATCH_URL"
  echo "patch_sha256=$(sha256sum "$PATCH" | awk '{print $1}')"
} > "$R2_SRC/r2-meta/stage7-qsa-base.txt"

cd "$R2_SRC"
if ! git apply --3way "$PATCH"; then
  echo "ERROR: PR #28213 did not apply cleanly to the exact production HotSeat tree." >&2
  echo "Worktree is kept for functional migration. Production remains untouched." >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check
git diff > r2-meta/stage7-qsa-gather.diff

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-qsa-gather}"

echo "ROCm path         : $ROCM_PATH"
echo "HIP compiler      : $HIP_CXX"
echo "AMDGPU targets    : $AMDGPU_TARGETS"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing"; exit 20; }
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage7-qsa-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage7-qsa-version.txt" || true

# Same binary, exact inherited production arguments/env. Only QSA gather differs.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=0 --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=1 --replace --validate

echo
echo "Stage-7 QSA gather runtime ready."
echo "OFF alias: $OFF_ALIAS"
echo " ON alias: $ON_ALIAS"
echo "Runtime  : $RUNTIME/llama-server"
echo "Next     : bash $SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
