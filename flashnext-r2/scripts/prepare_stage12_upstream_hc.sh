#!/usr/bin/env bash
set -euo pipefail

# Stage 12: evaluate the now-merged upstream qwen4exp HC improvements that landed
# after our 2026-09-11 production base:
#   PR #28896  qwen4exp: enable rms_norm + mul fusion
#   PR #28901  qwen4exp: add HC fused ops
#
# This intentionally does NOT stack the experimental JohnTDI HC patch. The goal
# is to compare the clean upstream implementation against the current production
# engine, then decide which lineage should survive.
#
# Run through with_exact_prod.sh so production's live HotSeat edits are included.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-upstream-hc-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-upstream-hc}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-upstream-hc:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

COMMITS=(
  2e9561b94fbbe98198dc14f0dbaf7dd04e5a567f
  41fd638228b570bb843c66828c58fb55a2f1b292
  26276bc99e23c17031a3cf10368c751cb8eb7599
)

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git tree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty. Run through exact snapshot wrapper:" >&2
  echo "  bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage12_upstream_hc.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-12 worktree/runtime already exists" >&2
  exit 4
fi

mkdir -p "$PATCH_DIR"
PATCHES=()
for c in "${COMMITS[@]}"; do
  p="$PATCH_DIR/upstream-qwen4exp-hc-${c:0:8}.patch"
  if [[ ! -f "$p" ]]; then
    curl -fL --retry 3 --connect-timeout 20 \
      "https://github.com/ggml-org/llama.cpp/commit/$c.patch" -o "$p"
  fi
  if ! head -n 1 "$p" | grep -qi "${c:0:12}"; then
    echo "ERROR: patch identity mismatch for $c" >&2
    exit 5
  fi
  PATCHES+=("$p")
done

echo "Exact production : $PROD_SRC"
echo "Exact HEAD       : $(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Candidate tree   : $R2_SRC"
echo "Candidate runtime: $RUNTIME"
for p in "${PATCHES[@]}"; do sha256sum "$p"; done

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" HEAD
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$(git -C "$PROD_SRC" rev-parse HEAD)"
  echo "pr28896_commits=${COMMITS[0]},${COMMITS[1]}"
  echo "pr28901_commit=${COMMITS[2]}"
  for p in "${PATCHES[@]}"; do echo "patch_sha256=$(sha256sum "$p" | awk '{print $1}') $(basename "$p")"; done
} > "$R2_SRC/r2-meta/stage12-upstream-hc-base.txt"

cd "$R2_SRC"
for p in "${PATCHES[@]}"; do
  echo "=== applying $(basename "$p") ==="
  if ! git -c user.name='FlashNext R2 Experiment' \
           -c user.email='flashnext-r2@local.invalid' \
           am --3way "$p"; then
    echo >&2
    echo "ERROR: upstream HC commit conflicts with exact production HotSeat tree." >&2
    echo "Candidate tree retained for semantic merge: $R2_SRC" >&2
    git status --short >&2 || true
    exit 10
  fi
done

git diff --check HEAD~3..HEAD || true

grep -RniE 'ggml_dsv4_hc_pre_gated|fused_dsv4_hc_pre|TENSOR_ALLOW_RESHAPE' src ggml/include ggml/src \
  | head -n 180 | tee r2-meta/stage12-feature-probes.txt
if ! grep -Rq 'ggml_dsv4_hc_pre_gated' ggml src; then
  echo "ERROR: upstream qwen4exp gated HC op not present" >&2
  exit 11
fi

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-upstream-hc}"

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

# Run targeted HC backend tests when the local test tool supports the filter.
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o DSV4_HC -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o DSV4_HC -b HIP0 \
    || true
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage12-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage12-version.txt" || true

# Preserve every production argument/env including MTP, HotSeat and JMAX.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$R2_ALIAS" --r2-bin "$RUNTIME" --jmax keep --replace --validate

echo
echo "Stage-12 merged-upstream HC candidate ready."
echo "Baseline : $BASE_ALIAS"
echo "Candidate: $R2_ALIAS"
echo "Runtime  : $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage12_upstream_hc_ab.sh"
