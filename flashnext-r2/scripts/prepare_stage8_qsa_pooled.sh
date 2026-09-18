#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 pooled-key cache stage, modern lineage.
#
# Base is an already-committed QSA-gather candidate. This stage applies ONLY
# ggml-org/llama.cpp PR #28699, so OFF/ON isolates the incremental pooled-key
# cache instead of re-applying both gather + cache on top of the old Sep11 tree.
#
# Default stack:
#   modern foundation
#     + PR28243 MTP
#     + PR28213 QSA gather
#     + PR28699 pooled-key cache   <- this stage
#
# If Stage14 TOP_K won and was used underneath Stage7, pass the resulting Stage7
# source/alias here with BASE_SRC/SOURCE_ALIAS. The invariant is simple: source
# tree and source alias must describe the same engine.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-qsa-gather-20260918}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next-r2-modern-qsa-on:256k}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-qsa-pooled-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-qsa-pooled}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

POOLED_HEAD="141f3f5646aa15e88d53198610a7540f4f4b0d71"
POOLED_PATCH="$PATCH_DIR/pr28699-qsa-pooled-${POOLED_HEAD:0:8}.patch"
POOLED_URL="https://github.com/Rhonstin/llama.cpp/commit/${POOLED_HEAD}.patch"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-modern-pooled-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-modern-pooled-on:256k}"

[[ -e "$BASE_SRC/.git" ]] || {
  echo "ERROR: QSA-gather base source missing: $BASE_SRC" >&2
  echo "Run prepare_stage7_qsa_gather.sh and its A/B first." >&2
  exit 2
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: BASE_SRC has tracked modifications; pooled-cache A/B requires a committed base." >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 3
}
[[ -f "$CONFIG" ]] || { echo "ERROR: llama-swap config missing: $CONFIG" >&2; exit 4; }
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: pooled-cache source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
}

grep -qE "^[[:space:]]*${SOURCE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: SOURCE_ALIAS missing from llama-swap config: $SOURCE_ALIAS" >&2
  exit 6
}

if ! grep -Rq 'QWEN4EXP_QSA_GATHER' "$BASE_SRC/src"; then
  echo "ERROR: selected base does not contain QSA gather support." >&2
  exit 7
fi

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
echo "Pooled-cache base : $BASE_SRC"
echo "Base HEAD         : $BASE_HEAD"
echo "Source alias      : $SOURCE_ALIAS"
echo "Candidate tree    : $R2_SRC"
echo "Candidate runtime : $RUNTIME"
echo "PR28699 head      : $POOLED_HEAD"

mkdir -p "$PATCH_DIR"
if [[ ! -f "$POOLED_PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$POOLED_URL" -o "$POOLED_PATCH"
fi
if ! head -n 1 "$POOLED_PATCH" | grep -qi "${POOLED_HEAD:0:12}"; then
  echo "ERROR: pooled-cache patch does not identify pinned commit $POOLED_HEAD" >&2
  exit 8
fi
PATCH_SHA="$(sha256sum "$POOLED_PATCH" | awk '{print $1}')"

git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$POOLED_PATCH"; then
  echo "ERROR: PR #28699 conflicts with the selected modern QSA-gather base." >&2
  echo "Candidate retained for semantic merge; production untouched." >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check "$BASE_HEAD"..HEAD

grep -Rni 'LLAMA_QSA_NO_POOLED_CACHE' src | tee r2-meta/stage8-killswitch.txt
if ! grep -Rq 'LLAMA_QSA_NO_POOLED_CACHE' src; then
  echo "ERROR: pooled-cache kill switch missing" >&2
  exit 11
fi
if ! grep -Rq 'QWEN4EXP_QSA_GATHER' src; then
  echo "ERROR: QSA gather support disappeared while applying pooled cache" >&2
  exit 12
fi

# PR28699's important multi-GPU property: cache buffers are allocated per
# indexer-cache buffer type/device, rather than making all layer splits bounce
# pooled rows over inter-GPU links. Keep a probe in the experiment manifest.
grep -RniE 'pooled|watermark|PARTIAL_ONLY|seq_rm|NO_POOLED_CACHE' src \
  | head -n 220 | tee r2-meta/stage8-pooled-probes.txt

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "source_alias=$SOURCE_ALIAS"
  echo "pooled_pr=ggml-org/llama.cpp#28699"
  echo "pooled_head=$POOLED_HEAD"
  echo "pooled_sha256=$PATCH_SHA"
  echo "candidate_head=$(git rev-parse HEAD)"
} > r2-meta/stage8-modern-pooled-base.txt

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
BUILD="${BUILD:-$R2_SRC/build-r2-modern-qsa-pooled}"

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
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage8-modern-pooled-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage8-modern-pooled-version.txt || true

# Same candidate binary in both arms. QSA gather stays ON in both. Only the
# pooled-cache presence-based kill switch differs.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env QWEN4EXP_QSA_GATHER=1 \
  --env LLAMA_QSA_NO_POOLED_CACHE=1 \
  --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --unset-env LLAMA_QSA_NO_POOLED_CACHE \
  --env QWEN4EXP_QSA_GATHER=1 \
  --replace --validate

echo
echo "Modern pooled-key cache candidate ready."
echo "Base source : $BASE_SRC"
echo "Source alias: $SOURCE_ALIAS"
echo "OFF alias   : $OFF_ALIAS"
echo "ON alias    : $ON_ALIAS"
echo "Runtime     : $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage8_qsa_pooled_ab.sh"
echo "Then run rollback/checkpoint stress before treating ON as a winner."
