#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 11: O(log n) n-gram predecessor lookup for qwen4exp PLE.
# Source: ggml-org/llama.cpp PR #27992 (closed/unmerged), head 211e29d1...
#
# This stage deliberately uses three aliases over ONE patched binary:
#   idx-off    LLAMA_KV_PREV_TOKENS=off
#   idx-fast   LLAMA_KV_PREV_TOKENS=fast
#   idx-verify LLAMA_KV_PREV_TOKENS=verify
#
# Production remains untouched. The exact-production snapshot should be passed as
# PROD_SRC so existing HotSeat/MTP/custom source edits are retained.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-prev-index-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-prev-index}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next:256k}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-prev-off:256k}"
FAST_ALIAS="${FAST_ALIAS:-qwen3.8-flash-next-r2-prev-fast:256k}"
VERIFY_ALIAS="${VERIFY_ALIAS:-qwen3.8-flash-next-r2-prev-verify:256k}"

PR_BASE="a7cc83bbae43e548df42c0af0df68f391315aa77"
PR_HEAD="211e29d1f6d866f63e71dd4f54380864064523a6"
PR_REMOTE="https://github.com/tvanderka/llama.cpp.git"

[[ -e "$PROD_SRC/.git" ]] || {
  echo "ERROR: clean exact-production snapshot not found: $PROD_SRC" >&2
  echo "Create one first with create_exact_prod_snapshot_repo.sh and point PROD_SRC at it." >&2
  exit 2
}
[[ -z "$(git -C "$PROD_SRC" status --porcelain)" ]] || {
  echo "ERROR: Stage 11 requires a clean committed exact-production snapshot." >&2
  exit 3
}
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: Stage-11 worktree/runtime already exists; refusing overwrite" >&2
  exit 4
}

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Exact production HEAD : $PROD_HEAD"
echo "PR27992 base/head     : $PR_BASE / $PR_HEAD"
echo "Stage-11 worktree    : $R2_SRC"
echo "Stage-11 runtime     : $RUNTIME"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

# Fetch the immutable closed PR head and build one cumulative diff from the PR's
# original base. This carries both commits without depending on a mutable .patch URL.
git fetch --no-tags "$PR_REMOTE" "$PR_HEAD"
git cat-file -e "$PR_BASE^{commit}"
git cat-file -e "$PR_HEAD^{commit}"
git diff --binary "$PR_BASE" "$PR_HEAD" > r2-meta/pr27992-prev-index.diff

if ! git apply --3way r2-meta/pr27992-prev-index.diff; then
  echo "ERROR: PR #27992 does not apply cleanly to this exact production snapshot." >&2
  echo "Worktree retained for functional porting; production is untouched." >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$PROD_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#27992"
  echo "pr_base=$PR_BASE"
  echo "pr_head=$PR_HEAD"
  echo "diff_sha256=$(sha256sum r2-meta/pr27992-prev-index.diff | awk '{print $1}')"
} > r2-meta/stage11-prev-index-base.txt

# Sanity: the experiment must expose all three modes and the indexed lookup.
grep -Rni "LLAMA_KV_PREV_TOKENS" src | tee r2-meta/stage11-mode-probe.txt
grep -Rni "get_prev_tokens_indexed\|seq_pos_token_le" src | tee r2-meta/stage11-index-probe.txt

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
BUILD="${BUILD:-$R2_SRC/build-r2-prev-index}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage11-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage11-version.txt || true

# Same binary and same production command. Only LLAMA_KV_PREV_TOKENS differs.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax 0 \
  --env LLAMA_KV_PREV_TOKENS=off --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$FAST_ALIAS" --r2-bin "$RUNTIME" --jmax 0 \
  --env LLAMA_KV_PREV_TOKENS=fast --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$VERIFY_ALIAS" --r2-bin "$RUNTIME" --jmax 0 \
  --env LLAMA_KV_PREV_TOKENS=verify --replace --validate

echo
echo "Stage-11 aliases ready:"
echo "  OFF    : $OFF_ALIAS"
echo "  FAST   : $FAST_ALIAS"
echo "  VERIFY : $VERIFY_ALIAS"
echo "  runtime: $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage11_prev_tokens_ab.sh"
