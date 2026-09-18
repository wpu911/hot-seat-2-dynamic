#!/usr/bin/env bash
set -euo pipefail

# Stage 14: qwen4exp --split-mode tensor experiment for the heterogeneous
# RX 7900 XTX + R9700 setup. Based on ggml-org/llama.cpp PR #28569.
#
# This is intentionally OUTSIDE the main optimization pipeline. Tensor parallel
# can reduce per-layer serial work but adds cross-device synchronization, and on
# asymmetric PCIe GPUs that can be either useful or a very expensive group chat.
#
# A/B:
#   layer alias  -> same patched binary, --split-mode layer
#   tensor alias -> same patched binary, --split-mode tensor
# Every other argument, including --tensor-split/device order, is inherited from
# the chosen modern MTP alias.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-sm-tensor-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-sm-tensor}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
LAYER_ALIAS="${LAYER_ALIAS:-qwen3.8-flash-next-r2-sm-layer:256k}"
TENSOR_ALIAS="${TENSOR_ALIAS:-qwen3.8-flash-next-r2-sm-tensor:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr28569-sm-tensor-53c2a4c9.patch"
PATCH_URL="https://github.com/kh0pper/llama.cpp/commit/53c2a4c9fd411ec1cbb0f08edcb6071fa8224818.patch"
PATCH_HEAD="53c2a4c9fd411ec1cbb0f08edcb6071fa8224818"

[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR: modern MTP base missing: $BASE_SRC" >&2; exit 2; }
[[ -z "$(git -C "$BASE_SRC" status --porcelain)" ]] || { echo "ERROR: base must be clean" >&2; exit 3; }
[[ -f "$BASE_SRC/r2-meta/stage10-modern-mtp-base.txt" ]] || {
  echo "ERROR: Stage14 should be based on corrected modern MTP candidate" >&2; exit 4;
}
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || { echo "ERROR: Stage14 source/runtime exists" >&2; exit 5; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 6; }

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: pinned PR28569 patch mismatch" >&2; exit 7
fi

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"
if ! git apply --3way --index "$PATCH"; then
  echo "ERROR: PR28569 does not apply cleanly; tree retained: $R2_SRC" >&2
  exit 10
fi
git diff --cached --check
git -c user.name='FlashNext R2 Experiment' -c user.email='flashnext-r2@local.invalid' \
  commit -m "r2 stage14: qwen4exp split-mode tensor PR28569" >/dev/null

# The PR is tiny. Verify both semantic changes exist.
if grep -A25 -n 'llm_arch_supports_sm_tensor' src/llama-arch.cpp | grep -q 'QWEN4EXP.*return false'; then
  echo "ERROR: QWEN4EXP still explicitly disabled for sm tensor" >&2; exit 11
fi
grep -n 'ggml_build_forward_expand(gf, res_hc)' src/models/qwen4exp.cpp | tee r2-meta/stage14-hc-init-expand.txt

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#28569"
  echo "pr_head=$PATCH_HEAD"
  echo "patch_sha256=$(sha256sum "$PATCH" | awk '{print $1}')"
  echo "candidate_head=$(git rev-parse HEAD)"
} > r2-meta/stage14-sm-tensor-base.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-sm-tensor}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-llama-archs

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
# Architecture regression if available. Do not pretend a failed qwen4exp tensor
# scheduler test is "probably fine" merely because compilation succeeded.
if [[ -x "$BUILD/bin/test-llama-archs" ]]; then
  "$BUILD/bin/test-llama-archs" -a qwen4exp || {
    echo "ERROR: qwen4exp architecture test failed" >&2; exit 21;
  }
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage14-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage14-version.txt || true

# Same binary. First clone twice, then change only split mode.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$LAYER_ALIAS" \
  --r2-bin "$RUNTIME" --jmax keep --replace
python3 "$SCRIPT_DIR/set_alias_split_mode.py" --config "$CONFIG" --alias "$LAYER_ALIAS" --mode layer

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$TENSOR_ALIAS" \
  --r2-bin "$RUNTIME" --jmax keep --replace
python3 "$SCRIPT_DIR/set_alias_split_mode.py" --config "$CONFIG" --alias "$TENSOR_ALIAS" --mode tensor

/app/share/llama_box/bin/llama-swap -config "$CONFIG" -validate

echo
echo "Stage14 split-mode aliases ready."
echo "layer : $LAYER_ALIAS"
echo "tensor: $TENSOR_ALIAS"
echo "same binary: $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage14_sm_tensor_ab.sh"
