#!/usr/bin/env bash
set -euo pipefail

# Stage 10: isolated candidate for ggml-org/llama.cpp PR #28243
# "models: Qwen3.8-Flash-Next MTP".
#
# IMPORTANT:
#   Run through with_exact_prod.sh so PROD_SRC is a clean exact snapshot of the
#   live production tree, including uncommitted HotSeat edits.
#
# This stage preserves the production llama-swap model block verbatim except for
# the candidate runtime path. JMAX and --spec-draft-n-max are inherited.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-mtp-upstream-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-mtp-upstream}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-mtp-upstream:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

# Pin the PR delta to the exact base/head reported by GitHub on 2026-09-18.
# Do NOT diff from the first historical parent: the branch has merged newer
# upstream master, and that older range would drag unrelated post-Sep11 master
# changes into the experiment.
MTP_BASE="911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"
MTP_HEAD="53b1389d0bf98fa367e2a0ce0475008e762ebf28"
PATCH="$PATCH_DIR/pr28243-qwen4exp-mtp-${MTP_BASE:0:8}-${MTP_HEAD:0:8}.patch"
PATCH_URL="https://github.com/danielhanchen/llama.cpp/compare/${MTP_BASE}...${MTP_HEAD}.patch"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git tree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty." >&2
  echo "Use the exact-production wrapper, otherwise local HotSeat edits can disappear:" >&2
  echo "  bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage10_upstream_mtp.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: candidate worktree already exists: $R2_SRC" >&2
  exit 4
fi
if [[ -e "$RUNTIME" ]]; then
  echo "ERROR: candidate runtime already exists: $RUNTIME" >&2
  exit 5
fi

# Do not spend an hour compiling a runtime that cannot load the current draft.
# The check is conservative; bypass only for deliberate converter migration work.
if [[ "${SKIP_MTP_LAYOUT_CHECK:-0}" != "1" ]]; then
  if ! CONFIG="$CONFIG" ALIAS="$BASE_ALIAS" bash "$SCRIPT_DIR/inspect_stage10_mtp_layout.sh"; then
    echo >&2
    echo "ERROR: current production MTP draft is not confirmed compatible with PR #28243." >&2
    echo "Set SKIP_MTP_LAYOUT_CHECK=1 only when intentionally testing a newly converted draft." >&2
    exit 6
  fi
fi

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi

# A compare patch is a mail patch series. Pin both ends: if upstream rebases or
# the compare unexpectedly resolves a different range, fail rather than benchmark
# mystery code wearing a familiar filename.
if ! grep -qi "${MTP_HEAD:0:12}" "$PATCH"; then
  echo "ERROR: downloaded MTP patch does not contain pinned head $MTP_HEAD" >&2
  exit 7
fi
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

echo "Production exact base: $PROD_SRC"
echo "Production HEAD      : $(git -C "$PROD_SRC" rev-parse HEAD)"
echo "MTP PR delta         : $MTP_BASE..$MTP_HEAD"
echo "Patch SHA256         : $PATCH_SHA"
echo "Candidate worktree   : $R2_SRC"
echo "Candidate runtime    : $RUNTIME"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" HEAD
mkdir -p "$R2_SRC/r2-meta"
R2_BASE_HEAD="$(git -C "$R2_SRC" rev-parse HEAD)"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$R2_BASE_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28243"
  echo "pr_base=$MTP_BASE"
  echo "pr_head=$MTP_HEAD"
  echo "patch_sha256=$PATCH_SHA"
} > "$R2_SRC/r2-meta/stage10-mtp-base.txt"

cd "$R2_SRC"

# Preserve the PR's commit series. If custom HotSeat MTP code overlaps, stop at
# the first semantic conflict instead of accepting a suspicious fuzzy apply.
if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo >&2
  echo "ERROR: PR #28243 conflicts with the exact production HotSeat tree." >&2
  echo "No production file was changed. Candidate tree is kept for functional migration:" >&2
  echo "  $R2_SRC" >&2
  git status --short >&2 || true
  echo "Resolve there deliberately, then continue with: git am --continue" >&2
  exit 10
fi

git diff --check "$R2_BASE_HEAD"..HEAD
git diff "$R2_BASE_HEAD"..HEAD > r2-meta/stage10-mtp-stack.diff
printf '%s\n' "$R2_BASE_HEAD" > r2-meta/stage10-exact-production-head.txt
printf '%s\n' "$(git rev-parse HEAD)" > r2-meta/stage10-candidate-head.txt

# Feature probes. These are more useful than trusting a successful patch exit.
grep -Rni 'qwen4exp_shared_model' src/models/qwen4exp.cpp | tee r2-meta/mtp-shared-model-probe.txt
grep -RniE 'n_layer_nextn|LLM_GRAPH_TYPE_DECODER_MTP|QWEN4EXP MTP' src common \
  | head -n 200 | tee r2-meta/mtp-runtime-probes.txt

if ! grep -q 'qwen4exp_shared_model' src/models/qwen4exp.cpp; then
  echo "ERROR: shared target-module path missing after PR port" >&2
  exit 11
fi
if ! grep -q 'LLM_GRAPH_TYPE_DECODER_MTP' src/models/qwen4exp.cpp; then
  echo "ERROR: qwen4exp MTP graph missing after PR port" >&2
  exit 12
fi

# Critical correctness fix in the current PR series: qwen4exp may borrow target
# embeddings through ctx_other, but it does NOT share the target KV/recurrent
# memory. If this guard is missing, draft catch-up/rollback can be skipped and
# M-RoPE positions repeat. That is not a speed optimization; it is a correctness
# landmine wearing a stopwatch.
if ! grep -q 'gemma4-assistant' common/speculative.cpp; then
  echo "ERROR: PR28243 qwen4exp memory-sharing correctness guard is missing" >&2
  exit 13
fi

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

cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage10-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage10-version.txt" || true

# Clone production exactly. No JMAX or MTP-depth change is allowed in this stage.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" \
  --source-alias "$BASE_ALIAS" \
  --alias "$R2_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax keep \
  --replace \
  --validate

echo
echo "Stage-10 upstream MTP candidate ready."
echo "Baseline alias : $BASE_ALIAS"
echo "Candidate alias: $R2_ALIAS"
echo "Runtime        : $RUNTIME/llama-server"
echo "Production runtime/config block remains intact."
echo
echo "Next: bash $SCRIPT_DIR/run_stage10_mtp_ab.sh"
