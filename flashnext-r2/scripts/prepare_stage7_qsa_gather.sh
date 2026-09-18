#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 QSA gather stage, rebased onto the corrected modern line.
# Based on ggml-org/llama.cpp PR #28213, pinned to its current one-commit head.
#
# The SAME patched binary is exposed through OFF/ON llama-swap aliases so the
# only variable is QWEN4EXP_QSA_GATHER.
#
# Default base is modern MTP. If Stage-14 ROCm TOP_K is confirmed as a winner,
# layer it underneath this stage by overriding both variables together:
#
#   BASE_SRC=/app/share/llama_box/src/llama.cpp-flashnext-r2-rocm-topk-20260918 \
#   SOURCE_ALIAS=qwen3.8-flash-next-r2-rocm-topk:256k \
#   bash prepare_stage7_qsa_gather.sh
#
# Never mix a source from one lineage with a llama-swap alias from another.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-qsa-gather-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-qsa-gather}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr28213-qsa-gather-beed2f78.patch"
PATCH_URL="https://github.com/abdel-darwish-27/llama.cpp/commit/beed2f78ac42cf16710b763e6f3ba20665c6d233.patch"
PATCH_HEAD="beed2f78ac42cf16710b763e6f3ba20665c6d233"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-modern-qsa-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-modern-qsa-on:256k}"

if [[ ! -e "$BASE_SRC/.git" ]]; then
  echo "ERROR: modern base source is not a git worktree/repo: $BASE_SRC" >&2
  echo "Prepare Modern Foundation + Stage10 MTP first, or override BASE_SRC/SOURCE_ALIAS together." >&2
  exit 2
fi
if [[ -n "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: BASE_SRC has tracked modifications; QSA A/B requires a committed base." >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 3
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: llama-swap config missing: $CONFIG" >&2
  exit 4
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: QSA source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
fi

grep -qE "^[[:space:]]*${SOURCE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: SOURCE_ALIAS missing from llama-swap config: $SOURCE_ALIAS" >&2
  exit 6
}

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
echo "Modern QSA base : $BASE_SRC"
echo "Base HEAD       : $BASE_HEAD"
echo "Source alias    : $SOURCE_ALIAS"
echo "QSA worktree    : $R2_SRC"
echo "QSA runtime     : $RUNTIME"
echo "PR28213 head    : $PATCH_HEAD"

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: downloaded patch does not identify pinned commit $PATCH_HEAD" >&2
  head -n 3 "$PATCH" >&2 || true
  exit 7
fi
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

# Long-context QSA numbers are meaningless if TOP_K is not GPU-resident.
if ! grep -RqiE 'radix.*top.?k|top.?k.*radix|radix_select|top_k_.*radix' "$BASE_SRC/ggml/src/ggml-cuda"; then
  echo "ERROR: ROCm long-row radix TOP_K not confirmed in selected modern base." >&2
  exit 8
fi

git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

# Keep this as a real commit so pooled-cache and final-combination stages can use
# the QSA-gather winner as a clean base instead of inheriting a dirty tree.
if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo "ERROR: PR #28213 did not apply cleanly to the selected modern base." >&2
  echo "Candidate tree retained for semantic merge. Production remains untouched." >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check "$BASE_HEAD"..HEAD

# Make sure the runtime switch and gather graph really landed.
if ! grep -Rq 'QWEN4EXP_QSA_GATHER' src; then
  echo "ERROR: QSA gather runtime switch missing after patch." >&2
  exit 11
fi
if ! grep -RqiE 'gather.*qsa|qsa.*gather|ggml_get_rows' src/models/qwen4exp.cpp; then
  echo "ERROR: QSA gather graph markers missing after patch." >&2
  exit 12
fi

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "source_alias=$SOURCE_ALIAS"
  echo "upstream_pr=ggml-org/llama.cpp#28213"
  echo "patch_head=$PATCH_HEAD"
  echo "patch_sha256=$PATCH_SHA"
  echo "candidate_head=$(git rev-parse HEAD)"
} > r2-meta/stage7-modern-qsa-base.txt

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
BUILD="${BUILD:-$R2_SRC/build-r2-modern-qsa-gather}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage7-modern-qsa-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage7-modern-qsa-version.txt || true

# Same candidate binary on both aliases. Only QWEN4EXP_QSA_GATHER changes.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=0 --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=1 --replace --validate

echo
echo "Modern QSA gather candidate ready."
echo "Base source: $BASE_SRC"
echo "Source alias: $SOURCE_ALIAS"
echo "OFF alias   : $OFF_ALIAS"
echo "ON alias    : $ON_ALIAS"
echo "Runtime     : $RUNTIME/llama-server"
echo "Next        : bash $SCRIPT_DIR/run_stage7_qsa_gather_ab.sh"
