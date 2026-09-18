#!/usr/bin/env bash
set -euo pipefail

# Stage 10: evaluate ggml-org/llama.cpp PR #28243
# "models: Qwen3.8-Flash-Next MTP" on top of the SAME upstream HC baseline used
# by Stage 12.
#
# Why stack Stage 12 first?
# PR #28243 was refreshed against current master after #28896/#28901 landed.
# Comparing production directly against "HC + new MTP" would mix two variables.
# This script therefore builds:
#   exact production + upstream HC (#28896/#28901) + PR #28243 delta
# and run_stage10_mtp_ab.sh compares it against the Stage-12 HC-only alias.
#
# Run this through with_exact_prod.sh. Production files/runtime are never edited.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-hc-mtp-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-hc-mtp}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
CONFIG_SOURCE_ALIAS="${CONFIG_SOURCE_ALIAS:-qwen3.8-flash-next:256k}"
HC_BASELINE_ALIAS="${HC_BASELINE_ALIAS:-qwen3.8-flash-next-r2-upstream-hc:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-hc-mtp:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

# Stage-12 upstream HC stack. These are deliberately identical to
# prepare_stage12_upstream_hc.sh so the only difference in the A/B is MTP.
HC_COMMITS=(
  2e9561b94fbbe98198dc14f0dbaf7dd04e5a567f
  41fd638228b570bb843c66828c58fb55a2f1b292
  26276bc99e23c17031a3cf10368c751cb8eb7599
)

# PR #28243 current GitHub base/head as of 2026-09-18 21:47 CST.
# GitHub compare confirms: status=ahead, behind_by=0, total_commits=12.
# Use this exact base/head delta. Do NOT diff from the first historical parent:
# the PR branch later merged newer master, which would drag unrelated upstream
# changes into the benchmark.
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

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git tree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty. Use:" >&2
  echo "  bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage10_upstream_mtp.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-10 candidate source/runtime already exists" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 4
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: llama-swap config missing: $CONFIG" >&2
  exit 5
fi

# Stage 10 is an MTP-only comparison against Stage 12. Refuse to run the A/B if
# the HC-only alias has not been prepared first.
if ! grep -qE "^[[:space:]]*${HC_BASELINE_ALIAS//./\.}:[[:space:]]*(#.*)?$" "$CONFIG"; then
  echo "ERROR: Stage-12 HC baseline alias is missing: $HC_BASELINE_ALIAS" >&2
  echo "Prepare and validate Stage 12 first." >&2
  exit 6
fi

# Existing draft layout preflight. A newly converted draft can deliberately
# bypass this with SKIP_MTP_LAYOUT_CHECK=1.
if [[ "${SKIP_MTP_LAYOUT_CHECK:-0}" != "1" ]]; then
  CONFIG="$CONFIG" ALIAS="$CONFIG_SOURCE_ALIAS" bash "$SCRIPT_DIR/inspect_stage10_mtp_layout.sh" || {
    echo "ERROR: current production draft is not confirmed compatible with PR #28243" >&2
    exit 7
  }
fi

mkdir -p "$PATCH_DIR"
HC_PATCHES=()
for c in "${HC_COMMITS[@]}"; do
  p="$PATCH_DIR/upstream-qwen4exp-hc-${c:0:8}.patch"
  if [[ ! -f "$p" ]]; then
    curl -fL --retry 3 --connect-timeout 20 \
      "https://github.com/ggml-org/llama.cpp/commit/$c.patch" -o "$p"
  fi
  head -n 1 "$p" | grep -qi "${c:0:12}" || {
    echo "ERROR: HC patch identity mismatch: $c" >&2; exit 8;
  }
  HC_PATCHES+=("$p")
done

echo "Exact production : $PROD_SRC"
echo "Exact HEAD       : $(git -C "$PROD_SRC" rev-parse HEAD)"
echo "HC baseline alias: $HC_BASELINE_ALIAS"
echo "MTP PR delta     : $MTP_BASE..$MTP_HEAD"
echo "Candidate tree   : $R2_SRC"
echo "Candidate runtime: $RUNTIME"
for p in "${HC_PATCHES[@]}"; do sha256sum "$p"; done

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" HEAD
mkdir -p "$R2_SRC/r2-meta"
R2_BASE_HEAD="$(git -C "$R2_SRC" rev-parse HEAD)"

cd "$R2_SRC"
apply_mail_patch() {
  local p="$1" label="$2"
  echo "=== applying $label: $(basename "$p") ==="
  if ! git -c user.name='FlashNext R2 Experiment' \
           -c user.email='flashnext-r2@local.invalid' \
           am --3way "$p"; then
    echo >&2
    echo "ERROR: conflict while applying $label" >&2
    echo "Production untouched; candidate retained at $R2_SRC" >&2
    git status --short >&2 || true
    echo "Resolve semantically. Do not blanket checkout --theirs over HotSeat paths." >&2
    exit 10
  fi
}

for p in "${HC_PATCHES[@]}"; do apply_mail_patch "$p" "Stage-12 upstream HC"; done
HC_HEAD="$(git rev-parse HEAD)"
printf '%s\n' "$HC_HEAD" > r2-meta/stage10-hc-baseline-head.txt

# Fetch the pinned PR head as git objects and derive the final base..head delta
# locally. This is safer than a mutable /pull/28243.patch URL and also handles
# the branch's merge commit without asking git-am to replay unrelated master.
echo "=== fetching pinned PR #28243 objects ==="
git fetch --no-tags "$MTP_REPO" "$MTP_HEAD"
git cat-file -e "$MTP_HEAD^{commit}"
git cat-file -e "$MTP_BASE^{commit}" || {
  echo "ERROR: pinned PR base is not available after fetching head" >&2
  exit 11
}

MERGE_BASE="$(git merge-base "$MTP_BASE" "$MTP_HEAD")"
if [[ "$MERGE_BASE" != "$MTP_BASE" ]]; then
  echo "ERROR: pinned PR base is not an ancestor of head" >&2
  echo "merge_base=$MERGE_BASE expected=$MTP_BASE" >&2
  exit 12
fi
MTP_COMMIT_COUNT="$(git rev-list --count "$MTP_BASE..$MTP_HEAD")"
if [[ "$MTP_COMMIT_COUNT" != "$MTP_COMMITS_EXPECTED" ]]; then
  echo "ERROR: PR commit count mismatch: got=$MTP_COMMIT_COUNT expected=$MTP_COMMITS_EXPECTED" >&2
  exit 13
fi

git diff --binary --full-index "$MTP_BASE" "$MTP_HEAD" > "$MTP_DIFF.tmp"
mapfile -t ACTUAL_MTP_FILES < <(git diff --name-only "$MTP_BASE" "$MTP_HEAD")
printf '%s\n' "${ACTUAL_MTP_FILES[@]}" > r2-meta/stage10-mtp-files.txt

# Exact file-set gate. If the PR changes later, this pinned head should not, but
# this also catches a bad fetch/range before it touches the experiment tree.
printf '%s\n' "${EXPECTED_MTP_FILES[@]}" | sort > r2-meta/stage10-mtp-files-expected.txt
printf '%s\n' "${ACTUAL_MTP_FILES[@]}" | sort > r2-meta/stage10-mtp-files-actual.txt
if ! cmp -s r2-meta/stage10-mtp-files-expected.txt r2-meta/stage10-mtp-files-actual.txt; then
  echo "ERROR: PR28243 file set differs from the audited 19-file delta" >&2
  diff -u r2-meta/stage10-mtp-files-expected.txt r2-meta/stage10-mtp-files-actual.txt >&2 || true
  exit 14
fi
mv "$MTP_DIFF.tmp" "$MTP_DIFF"
MTP_DIFF_SHA="$(sha256sum "$MTP_DIFF" | awk '{print $1}')"
echo "MTP delta sha256: $MTP_DIFF_SHA"

# Apply only the audited PR delta on top of the same HC baseline. --3way uses the
# fetched base blobs when our custom HotSeat source has moved nearby code.
if ! git apply --3way --index "$MTP_DIFF"; then
  echo >&2
  echo "ERROR: PR #28243 delta conflicts with exact-production + upstream-HC tree." >&2
  echo "Production untouched; candidate retained at $R2_SRC" >&2
  git status --short >&2 || true
  echo "Resolve by function, preserving HotSeat ownership/Transit/spec gates." >&2
  exit 15
fi

git diff --cached --check
git -c user.name='FlashNext R2 Experiment' \
    -c user.email='flashnext-r2@local.invalid' \
    commit -m "r2 stage10: qwen4exp MTP PR28243 delta $MTP_HEAD" >/dev/null

CANDIDATE_HEAD="$(git rev-parse HEAD)"
printf '%s\n' "$CANDIDATE_HEAD" > r2-meta/stage10-candidate-head.txt

git diff --check "$R2_BASE_HEAD"..HEAD
git diff "$R2_BASE_HEAD".."$HC_HEAD" > r2-meta/stage10-hc-only.diff
git diff "$HC_HEAD"..HEAD > r2-meta/stage10-mtp-only.diff
git diff "$R2_BASE_HEAD"..HEAD > r2-meta/stage10-full-stack.diff

{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$R2_BASE_HEAD"
  echo "hc_commits=${HC_COMMITS[*]}"
  echo "hc_head=$HC_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28243"
  echo "pr_repo=$MTP_REPO"
  echo "pr_base=$MTP_BASE"
  echo "pr_head=$MTP_HEAD"
  echo "pr_commit_count=$MTP_COMMIT_COUNT"
  echo "mtp_diff_sha256=$MTP_DIFF_SHA"
  echo "candidate_head=$CANDIDATE_HEAD"
  for p in "${HC_PATCHES[@]}"; do
    echo "hc_patch_sha256=$(sha256sum "$p" | awk '{print $1}') $(basename "$p")"
  done
} > r2-meta/stage10-hc-mtp-base.txt

# Required MTP features.
grep -Rni 'qwen4exp_shared_model' src/models/qwen4exp.cpp | tee r2-meta/mtp-shared-model-probe.txt
grep -RniE 'n_layer_nextn|LLM_GRAPH_TYPE_DECODER_MTP|QWEN4EXP MTP' src common \
  | head -n 220 | tee r2-meta/mtp-runtime-probes.txt

grep -q 'qwen4exp_shared_model' src/models/qwen4exp.cpp || {
  echo "ERROR: shared target-module path missing" >&2; exit 16;
}
grep -q 'LLM_GRAPH_TYPE_DECODER_MTP' src/models/qwen4exp.cpp || {
  echo "ERROR: qwen4exp MTP graph missing" >&2; exit 17;
}

# Critical correctness commit d1a92352: qwen4exp borrows embeddings through
# ctx_other but keeps its own memory. Only gemma4-assistant is memory-shared.
# Without this, draft catch-up/rollback is skipped and M-RoPE positions can repeat.
if ! grep -q 'gemma4-assistant' common/speculative.cpp; then
  echo "ERROR: qwen4exp MTP memory-sharing correctness guard missing" >&2
  exit 18
fi

# Confirm Stage-12 HC features are present too, so baseline/candidate lineage is
# genuinely the same below the MTP delta.
if ! grep -Rq 'ggml_dsv4_hc_pre_gated' ggml src; then
  echo "ERROR: Stage-12 HC baseline feature missing in Stage-10 stack" >&2
  exit 19
fi

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-hc-mtp}"

echo "ROCm path      : $ROCM_PATH"
echo "HIP compiler   : $HIP_CXX"
echo "AMDGPU targets : $AMDGPU_TARGETS"

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
  "$BUILD/bin/test-backend-ops" test -o DSV4_HC -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o DSV4_HC -b HIP0 \
    || true
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage10-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage10-version.txt" || true

# Candidate receives the exact production runtime arguments/env. We do not change
# JMAX, HotSeat knobs or MTP depth in this stage.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" \
  --source-alias "$CONFIG_SOURCE_ALIAS" \
  --alias "$R2_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax keep \
  --replace \
  --validate

echo
echo "Stage-10 HC+upstream-MTP candidate ready."
echo "A/B baseline   : $HC_BASELINE_ALIAS"
echo "Candidate alias: $R2_ALIAS"
echo "Runtime        : $RUNTIME/llama-server"
echo "Only the MTP delta differs from the Stage-12 HC baseline."
echo "Next: bash $SCRIPT_DIR/run_stage10_mtp_ab.sh"
