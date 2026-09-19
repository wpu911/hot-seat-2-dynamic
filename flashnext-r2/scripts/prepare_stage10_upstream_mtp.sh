#!/usr/bin/env bash
set -euo pipefail

# Corrected Stage 10: evaluate PR #28243 on the COMPLETE Sep-18 upstream base,
# with the exact production HotSeat/custom overlay already forward-ported by
# prepare_modern_foundation.sh.
#
# The earlier selective-HC stack is intentionally retired. PR #28243 currently
# bases on 911f6cdc..., 130 upstream commits after the Sep-11 production base.
# Testing it on "Sep11 + a few chosen PRs" is not the same engine.
#
# A/B variable:
#   modern-foundation = Sep18 upstream + exact production custom overlay
#   modern-MTP        = same foundation + PR #28243 delta only

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-mtp}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASELINE_ALIAS="${BASELINE_ALIAS:-qwen3.8-flash-next-r2-modern-foundation:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

MTP_REPO="https://github.com/danielhanchen/llama.cpp.git"
MTP_BASE="911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"
MTP_HEAD="53b1389d0bf98fa367e2a0ce0475008e762ebf28"
MTP_COMMITS_EXPECTED=12
MTP_DIFF="$PATCH_DIR/pr28243-delta-${MTP_BASE:0:8}-${MTP_HEAD:0:8}.diff"

EXPECTED_MTP_FILES=(
  common/speculative.cpp
  conversion/bailingmoe3.py
  conversion/base.py
  conversion/command_r.py
  conversion/dots3.py
  conversion/glm.py
  conversion/qwen.py
  conversion/qwen4exp.py
  convert_hf_to_gguf.py
  gguf-py/gguf/constants.py
  gguf-py/gguf/tensor_mapping.py
  src/llama-arch.cpp
  src/llama-arch.h
  src/llama-context.cpp
  src/llama-model-loader.h
  src/llama-model.cpp
  src/llama-model.h
  src/models/models.h
  src/models/qwen4exp.cpp
)

[[ -e "$BASE_SRC/.git" ]] || {
  echo "ERROR: modern foundation missing: $BASE_SRC" >&2
  echo "Run prepare_modern_foundation.sh first." >&2
  exit 2
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain)" ]] || {
  echo "ERROR: modern foundation must be clean/committed." >&2
  exit 3
}
[[ -f "$BASE_SRC/r2-meta/modern-foundation-manifest.txt" ]] || {
  echo "ERROR: source is not a recorded modern foundation: $BASE_SRC" >&2
  exit 4
}
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: Stage-10 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
}
[[ -f "$CONFIG" ]] || { echo "ERROR: llama-swap config missing: $CONFIG" >&2; exit 6; }

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
if ! git -C "$BASE_SRC" merge-base --is-ancestor "$MTP_BASE" "$BASE_HEAD"; then
  echo "ERROR: modern foundation does not descend from PR28243 base $MTP_BASE" >&2
  exit 7
fi
if ! grep -qE "^[[:space:]]*${BASELINE_ALIAS//./\.}:[[:space:]]*(#.*)?$" "$CONFIG"; then
  echo "ERROR: modern-foundation alias missing: $BASELINE_ALIAS" >&2
  echo "Prepare and benchmark the foundation first." >&2
  exit 8
fi

if [[ "${SKIP_MTP_LAYOUT_CHECK:-0}" != "1" ]]; then
  CONFIG="$CONFIG" ALIAS="$BASELINE_ALIAS" bash "$SCRIPT_DIR/inspect_stage10_mtp_layout.sh" || {
    echo "ERROR: inherited draft layout not confirmed compatible with PR #28243" >&2
    exit 9
  }
fi

mkdir -p "$PATCH_DIR"

echo "=== Corrected Stage 10 MTP ==="
echo "foundation source : $BASE_SRC"
echo "foundation head   : $BASE_HEAD"
echo "baseline alias    : $BASELINE_ALIAS"
echo "PR base/head      : $MTP_BASE .. $MTP_HEAD"
echo "candidate source  : $R2_SRC"
echo "candidate runtime : $RUNTIME"

git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

# Fetch pinned PR objects and derive its final audited base..head delta locally.
git fetch --no-tags "$MTP_REPO" "$MTP_HEAD"
git cat-file -e "$MTP_HEAD^{commit}"
git cat-file -e "$MTP_BASE^{commit}" || {
  echo "ERROR: pinned PR base not available after fetch" >&2; exit 10;
}
MERGE_BASE="$(git merge-base "$MTP_BASE" "$MTP_HEAD")"
[[ "$MERGE_BASE" == "$MTP_BASE" ]] || {
  echo "ERROR: pinned PR base is not ancestor of head: merge_base=$MERGE_BASE" >&2; exit 11;
}
MTP_COMMIT_COUNT="$(git rev-list --count "$MTP_BASE..$MTP_HEAD")"
[[ "$MTP_COMMIT_COUNT" == "$MTP_COMMITS_EXPECTED" ]] || {
  echo "ERROR: PR commit count mismatch got=$MTP_COMMIT_COUNT expected=$MTP_COMMITS_EXPECTED" >&2; exit 12;
}

git diff --binary --full-index "$MTP_BASE" "$MTP_HEAD" > "$MTP_DIFF.tmp"
mapfile -t ACTUAL_MTP_FILES < <(git diff --name-only "$MTP_BASE" "$MTP_HEAD")
printf '%s\n' "${EXPECTED_MTP_FILES[@]}" | sort > r2-meta/stage10-mtp-files-expected.txt
printf '%s\n' "${ACTUAL_MTP_FILES[@]}" | sort > r2-meta/stage10-mtp-files-actual.txt
if ! cmp -s r2-meta/stage10-mtp-files-expected.txt r2-meta/stage10-mtp-files-actual.txt; then
  echo "ERROR: PR28243 file set differs from audited 19-file delta" >&2
  diff -u r2-meta/stage10-mtp-files-expected.txt r2-meta/stage10-mtp-files-actual.txt >&2 || true
  exit 13
fi
mv "$MTP_DIFF.tmp" "$MTP_DIFF"
MTP_DIFF_SHA="$(sha256sum "$MTP_DIFF" | awk '{print $1}')"

# Apply ONLY the PR delta to the forward-ported modern foundation.
if ! git apply --3way --index "$MTP_DIFF"; then
  echo >&2
  echo "ERROR: PR #28243 conflicts with the forward-ported production overlay." >&2
  echo "Candidate retained for semantic merge: $R2_SRC" >&2
  echo "Do not blanket --theirs over HotSeat/checkpoint/MTP paths." >&2
  git status --short >&2 || true
  exit 20
fi

git diff --cached --check
git -c user.name='FlashNext R2 Experiment' \
    -c user.email='flashnext-r2@local.invalid' \
    commit -m "r2 stage10: qwen4exp MTP PR28243 delta $MTP_HEAD" >/dev/null
CANDIDATE_HEAD="$(git rev-parse HEAD)"
git diff "$BASE_HEAD"..HEAD > r2-meta/stage10-mtp-only.diff

{
  echo "created=$(date -Is)"
  echo "modern_foundation_source=$BASE_SRC"
  echo "modern_foundation_head=$BASE_HEAD"
  echo "modern_upstream_base=$MTP_BASE"
  echo "upstream_pr=ggml-org/llama.cpp#28243"
  echo "pr_repo=$MTP_REPO"
  echo "pr_head=$MTP_HEAD"
  echo "pr_commit_count=$MTP_COMMIT_COUNT"
  echo "mtp_diff_sha256=$MTP_DIFF_SHA"
  echo "candidate_head=$CANDIDATE_HEAD"
  echo "baseline_alias=$BASELINE_ALIAS"
} > r2-meta/stage10-modern-mtp-base.txt

# Required PR features and the late memory-sharing correctness guard.
grep -RniE 'qwen4exp_shared_model|LLM_GRAPH_TYPE_DECODER_MTP|n_layer_nextn|QWEN4EXP MTP' src common \
  | head -n 260 | tee r2-meta/stage10-mtp-probes.txt

grep -q 'qwen4exp_shared_model' src/models/qwen4exp.cpp || {
  echo "ERROR: shared target-module MTP path missing" >&2; exit 21;
}
grep -q 'LLM_GRAPH_TYPE_DECODER_MTP' src/models/qwen4exp.cpp || {
  echo "ERROR: qwen4exp graph_mtp missing" >&2; exit 22;
}
grep -q 'gemma4-assistant' common/speculative.cpp || {
  echo "ERROR: MTP memory-sharing correctness guard missing" >&2; exit 23;
}

# Foundation features and custom engine must survive the apply.
grep -Rq 'ggml_dsv4_hc_pre_gated' ggml src || {
  echo "ERROR: modern HC op disappeared while applying MTP" >&2; exit 24;
}
if ! grep -Rqs 'HOTSEAT\|Q122_DYNKV' ggml src tools/server 2>/dev/null; then
  echo "ERROR: custom HotSeat/DynamicKV markers missing from candidate" >&2
  exit 25
fi

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-modern-mtp}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 30; }
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
fi

# Keep the candidate independent from its CMake tree. Hashes are recorded only
# after RUNPATH normalization, so a later build cleanup cannot silently change
# which libllama/libggml this alias loads.
REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
  bash "$SCRIPT_DIR/stage_runtime_bundle.sh" \
    "$BUILD/bin" "$RUNTIME" "$R2_SRC/r2-meta/runtime-bundle"

sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage10-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage10-version.txt || true

# Clone the modern-foundation alias. All model/draft/HotSeat parameters stay the
# same; only the llama-server binary changes to the MTP candidate.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASELINE_ALIAS" \
  --alias "$R2_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --replace --validate

echo
echo "Corrected Stage-10 modern-MTP candidate ready."
echo "baseline : $BASELINE_ALIAS"
echo "candidate: $R2_ALIAS"
echo "runtime  : $RUNTIME/llama-server"
echo "Next     : bash $SCRIPT_DIR/run_stage10_mtp_ab.sh"
