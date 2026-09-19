#!/usr/bin/env bash
set -euo pipefail

# Stage 15: RDNA4 head-dim-256 FlashAttention MMA route.
#
# Upstream candidate: ggml-org/llama.cpp PR #26419
#   ggml-cuda: enable MMA FlashAttention for head dim 256 on AMD RDNA
#
# This matters specifically to the R9700/gfx1201 half of the mixed gfx1100 +
# gfx1201 box. The upstream dispatch is guarded by GGML_CUDA_CC_IS_RDNA4, so the
# new route is never selected on the 7900 XTX/gfx1100.
#
# PR #26419 changes the DKQ>128 WMMA kernel implementation as well as dispatch.
# Therefore we expose THREE arms:
#   BASE : untouched selected lineage
#   OFF  : PR binary, but new RDNA4 DKQ256 dispatch disabled
#   ON   : exact same PR binary, new dispatch enabled
#
# BASE->OFF catches regressions from the kernel changes themselves. OFF->ON then
# isolates the new R9700 dispatch without pretending a two-binary A/B has only
# one variable. Production alias/runtime are never modified.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-rdna4-fa256-20260919}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-rdna4-fa256}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"

PR=26419
PR_HEAD="9ce55841c8e1ed074ce43ae8beb22636e6ef6aee"
PATCH="$PATCH_DIR/pr${PR}-rdna4-fa256-${PR_HEAD:0:8}.patch"
PATCH_URL="https://github.com/ggml-org/llama.cpp/pull/${PR}.patch"
EXPECTED_FILES=(
  "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
  "ggml/src/ggml-cuda/fattn.cu"
  "tests/test-backend-ops.cpp"
)

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-rdna4-fa256-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-rdna4-fa256-on:256k}"
GATE_ENV="GGML_RDNA4_FA256_MMA"

[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR base source missing: $BASE_SRC" >&2; exit 2; }
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR base source has tracked modifications: $BASE_SRC" >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 3
}
[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 4; }
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR Stage-15 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
}
grep -qE "^[[:space:]]*${BASE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR base llama-swap alias missing: $BASE_ALIAS" >&2
  exit 6
}

mkdir -p "$PATCH_DIR" "$LOG_DIR"
TMP_PATCH="$PATCH.tmp"
curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$TMP_PATCH"
LAST_COMMIT="$(grep -E '^From [0-9a-f]{40} ' "$TMP_PATCH" | tail -n1 | awk '{print $2}')"
if [[ "$LAST_COMMIT" != "$PR_HEAD" ]]; then
  echo "ERROR PR #$PR moved; refusing unaudited patch" >&2
  echo "expected=$PR_HEAD downloaded=$LAST_COMMIT" >&2
  rm -f "$TMP_PATCH"
  exit 7
fi
mv "$TMP_PATCH" "$PATCH"
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
echo "=== Flash Next Stage 15 RDNA4 FA256 ==="
echo "base source      : $BASE_SRC"
echo "base head        : $BASE_HEAD"
echo "base alias       : $BASE_ALIAS"
echo "PR/head          : #$PR / $PR_HEAD"
echo "patch sha256     : $PATCH_SHA"
echo "candidate source : $R2_SRC"
echo "candidate runtime: $RUNTIME"

git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo >&2
  echo "ERROR PR #$PR conflicts with selected R2 lineage." >&2
  echo "Candidate retained for semantic merge: $R2_SRC" >&2
  git status --short >&2 || true
  exit 10
fi

mapfile -t CHANGED < <(git diff --name-only "$BASE_HEAD"..HEAD | sort -u)
printf '%s\n' "${EXPECTED_FILES[@]}" | sort > r2-meta/stage15-expected-files.txt
printf '%s\n' "${CHANGED[@]}" | sort > r2-meta/stage15-actual-files.txt
if ! cmp -s r2-meta/stage15-expected-files.txt r2-meta/stage15-actual-files.txt; then
  echo "ERROR PR #$PR file set changed from the audited three-file delta" >&2
  diff -u r2-meta/stage15-expected-files.txt r2-meta/stage15-actual-files.txt >&2 || true
  exit 11
fi

git diff --check "$BASE_HEAD"..HEAD

# Add a host-side process-local route gate. Both OFF/ON aliases use this exact
# binary, so the OFF->ON A/B changes only the RDNA4 DKQ256 dispatcher. The PR's
# kernel implementation remains present in both arms, and BASE->OFF separately
# checks whether those non-dispatch changes regress the mixed-GPU workload.
python3 - <<'PY'
from pathlib import Path
p = Path("ggml/src/ggml-cuda/fattn.cu")
s = p.read_text(encoding="utf-8")
old = '''        if (GGML_CUDA_CC_IS_RDNA4(cc) && Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
'''
new = '''        const char * rdna4_fa256_env = getenv("GGML_RDNA4_FA256_MMA");
        const bool rdna4_fa256_enabled = rdna4_fa256_env == nullptr || rdna4_fa256_env[0] != '0';
        if (rdna4_fa256_enabled && GGML_CUDA_CC_IS_RDNA4(cc) && Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
'''
if s.count(old) != 1:
    raise SystemExit(f"ERROR expected exactly one PR26419 RDNA4 dispatch hunk, found {s.count(old)}")
p.write_text(s.replace(old, new), encoding="utf-8")
PY

git diff --check
git add ggml/src/ggml-cuda/fattn.cu
git -c user.name='FlashNext R2 Experiment' \
    -c user.email='flashnext-r2@local.invalid' \
    commit -m "r2 stage15: add runtime gate for RDNA4 FA256 route" >/dev/null
CANDIDATE_HEAD="$(git rev-parse HEAD)"

grep -n 'GGML_RDNA4_FA256_MMA' ggml/src/ggml-cuda/fattn.cu | tee r2-meta/stage15-route-gate.txt
grep -n 'GGML_CUDA_CC_IS_RDNA4.*Q->ne\[0\].*256' ggml/src/ggml-cuda/fattn.cu | tee -a r2-meta/stage15-route-gate.txt

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "base_alias=$BASE_ALIAS"
  echo "upstream_pr=ggml-org/llama.cpp#$PR"
  echo "pr_head=$PR_HEAD"
  echo "patch_sha256=$PATCH_SHA"
  echo "candidate_head=$CANDIDATE_HEAD"
  echo "route_env=$GATE_ENV"
  echo "off_alias=$OFF_ALIAS"
  echo "on_alias=$ON_ALIAS"
} > r2-meta/stage15-rdna4-fa256-manifest.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-rdna4-fa256}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR llama-server missing" >&2; exit 20; }

# PR26419 changes a numerical kernel, so backend correctness is not optional.
# Running the unfiltered FLASH_ATTN_EXT suite lets every visible backend/device
# participate instead of accidentally proving only the R9700 half of this box.
FA_TEST_LOG="$LOG_DIR/flashnext-stage15-fa256-backend-$(date +%Y%m%d-%H%M%S).log"
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  if ! "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT >"$FA_TEST_LOG" 2>&1; then
    cat "$FA_TEST_LOG" >&2 || true
    echo "ERROR FLASH_ATTN_EXT backend suite failed" >&2
    exit 21
  fi
  cat "$FA_TEST_LOG"
fi

REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
  bash "$SCRIPT_DIR/stage_runtime_bundle.sh" \
    "$BUILD/bin" "$RUNTIME" "$R2_SRC/r2-meta/runtime-bundle"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage15-llama-server.sha256

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env "$GATE_ENV=0" --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env "$GATE_ENV=1" --replace --validate

echo
echo "Stage-15 RDNA4 FA256 candidate ready."
echo "BASE: $BASE_ALIAS"
echo "OFF : $OFF_ALIAS"
echo "ON  : $ON_ALIAS"
echo "runtime: $RUNTIME/llama-server"
echo "production alias/runtime untouched"
echo "Next: bash $SCRIPT_DIR/run_stage15_rdna4_fa256_ab.sh"
