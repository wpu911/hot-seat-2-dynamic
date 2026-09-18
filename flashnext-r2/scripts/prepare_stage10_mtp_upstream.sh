#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 10B: test the newer Qwen3.8-Flash-Next MTP implementation
# from ggml-org/llama.cpp PR #28243 against the exact production engine.
#
# This is deliberately a separate binary comparison. The production engine
# already carries a custom MTP path, so pretending this can be toggled by one env
# variable would be scientifically decorative rather than useful.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-mtp-upstream-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-mtp-upstream}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-mtp-upstream:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr28243-qwen4exp-mtp.patch"
PATCH_URL="https://github.com/ggml-org/llama.cpp/pull/28243.patch"
PATCH_HEAD="53b1389d0bf98fa367e2a0ce0475008e762ebf28"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git repo/worktree: $PROD_SRC" >&2
  echo "Create an exact snapshot first with create_exact_prod_snapshot_repo.sh" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: Stage 10 requires a CLEAN committed exact-production snapshot." >&2
  echo "Refusing to build from a dirty live tree." >&2
  exit 3
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-10 experiment source/runtime already exists." >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 4
fi

mkdir -p "$PATCH_DIR"
# PR patch URL is mutable, so pin it by verifying the final commit in the mail
# series. If upstream changes the PR after this script was written, stop rather
# than silently benchmark a different implementation.
curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH.tmp"
LAST_COMMIT="$(grep -E '^From [0-9a-f]{40} ' "$PATCH.tmp" | tail -n 1 | awk '{print $2}')"
if [[ "$LAST_COMMIT" != "$PATCH_HEAD" ]]; then
  echo "ERROR: PR #28243 moved." >&2
  echo "expected head=$PATCH_HEAD" >&2
  echo "downloaded head=$LAST_COMMIT" >&2
  rm -f "$PATCH.tmp"
  exit 5
fi
mv "$PATCH.tmp" "$PATCH"
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Exact production HEAD : $PROD_HEAD"
echo "PR #28243 head        : $PATCH_HEAD"
echo "PR patch sha256       : $PATCH_SHA"
echo "Stage-10 worktree     : $R2_SRC"
echo "Stage-10 runtime      : $RUNTIME"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$PROD_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28243"
  echo "upstream_head=$PATCH_HEAD"
  echo "patch_sha256=$PATCH_SHA"
} > "$R2_SRC/r2-meta/stage10-mtp-base.txt"

cd "$R2_SRC"
# The production engine already has custom MTP code, so conflicts are expected
# to be possible. Never resolve them by choosing 'theirs' wholesale: that could
# delete HotSeat/checkpoint work. If git am conflicts, leave the experiment tree
# isolated and stop for a functional merge.
if ! git am --3way "$PATCH"; then
  echo >&2
  echo "ERROR: PR #28243 conflicts with the exact production engine." >&2
  echo "Production is untouched. Experiment worktree kept at:" >&2
  echo "  $R2_SRC" >&2
  echo "Inspect with:" >&2
  echo "  git -C '$R2_SRC' status" >&2
  echo "  git -C '$R2_SRC' diff --cc" >&2
  echo "Do NOT run git checkout --theirs over qwen4exp/llama-graph/model-loader." >&2
  exit 10
fi

git diff --check HEAD~12..HEAD || true
git diff "$PROD_HEAD"..HEAD > r2-meta/stage10-mtp-upstream.diff

# Confirm the features that make this experiment worth doing actually landed.
grep -RqE 'graph_mtp|QWEN4EXP MTP' src/models/qwen4exp.cpp \
  || { echo "ERROR: Qwen4Exp MTP graph not found after PR stack" >&2; exit 11; }
grep -RqE 'mtp-shared-embd|mtp_shared_embd' convert_hf_to_gguf.py conversion \
  || { echo "ERROR: shared-MTP conversion support missing" >&2; exit 12; }

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
BUILD="${BUILD:-$R2_SRC/build-r2-mtp-upstream}"

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
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage10-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage10-version.txt" || true

# First test compatibility with the EXISTING production draft artifact and all
# existing llama-swap arguments. Shared-embedding draft conversion is Stage 10C
# and is only attempted after this runtime path proves correct.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" \
  --source-alias "$SOURCE_ALIAS" \
  --alias "$R2_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax 0 \
  --replace \
  --validate

echo
echo "Stage-10 upstream-MTP runtime prepared."
echo "baseline alias : $SOURCE_ALIAS"
echo "R2 alias       : $R2_ALIAS"
echo "R2 binary      : $RUNTIME/llama-server"
echo "Existing production draft is intentionally reused for the first compatibility A/B."
echo "Next: bash $SCRIPT_DIR/run_stage10_mtp_upstream_ab.sh"
