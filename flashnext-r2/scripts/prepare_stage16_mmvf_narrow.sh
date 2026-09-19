#!/usr/bin/env bash
set -euo pipefail

# Stage 16: keep narrow AMD weights on MMVF for multi-row decode.
#
# Source candidate: TheTom/llama-cpp-turboquant PR #363
#   "cuda: recover batched decode on AMD by keeping narrow weights off the GEMM"
#
# The patch is not TurboQuant-specific: it touches only ggml-cuda/mmvf.cu and
# changes dispatch for F32/F16/BF16 narrow weights once decode has >3 rows.
# Qwen4exp has many narrow structural matmuls on the hot path, and draft-MTP
# n-max 3/4 can create exactly the multi-row verify widths where upstream falls
# from vector kernels into a high-overhead GEMM.
#
# Conveniently the patch contains its own exact rollback knob:
#   GGML_MMVF_NARROW_MAX=0  -> restore upstream dispatch
# so OFF/ON use the same ELF. We additionally keep n-max=2 controls because the
# change should not buy benchmark points by regressing the current shallow path.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-mmvf-narrow-20260919}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-mmvf-narrow/bin}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

PR=363
PR_HEAD="43653697242d0cb7968364dd2ee6abfefef4f126"
PATCH="$PATCH_DIR/turboquant-pr${PR}-mmvf-narrow-${PR_HEAD:0:8}.patch"
PATCH_URL="https://github.com/TheTom/llama-cpp-turboquant/pull/${PR}.patch"
EXPECTED_FILE="ggml/src/ggml-cuda/mmvf.cu"

CTRL_OFF_N2="${CTRL_OFF_N2:-qwen3.8-flash-next-r2-mmvf-off-n2:256k}"
CTRL_ON_N2="${CTRL_ON_N2:-qwen3.8-flash-next-r2-mmvf-on-n2:256k}"
OFF_N4="${OFF_N4:-qwen3.8-flash-next-r2-mmvf-off-n4:256k}"
ON_N4="${ON_N4:-qwen3.8-flash-next-r2-mmvf-on-n4:256k}"

[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR Phase-3 source missing: $BASE_SRC" >&2; exit 2; }
[[ -f "$BASE_SRC/r2-meta/phase3-compose-manifest.txt" ]] || {
  echo "ERROR Stage16 base is not a Phase-3 composed source: $BASE_SRC" >&2
  exit 3
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR Stage16 base source has tracked changes" >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 4
}
[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 5; }
[[ ! -e "$R2_SRC" && ! -e "$(dirname "$RUNTIME")" ]] || {
  echo "ERROR Stage16 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 6
}
grep -qE "^[[:space:]]*${BASE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR base alias missing: $BASE_ALIAS" >&2
  exit 7
}

mkdir -p "$PATCH_DIR"
TMP="$PATCH.tmp"
curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$TMP"
LAST_COMMIT="$(grep -E '^From [0-9a-f]{40} ' "$TMP" | tail -n1 | awk '{print $2}')"
if [[ "$LAST_COMMIT" != "$PR_HEAD" ]]; then
  echo "ERROR TurboQuant PR #$PR moved; refusing unaudited patch" >&2
  echo "expected=$PR_HEAD downloaded=$LAST_COMMIT" >&2
  rm -f "$TMP"
  exit 8
fi
mv "$TMP" "$PATCH"
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git clone --quiet --no-hardlinks "$BASE_SRC" "$R2_SRC"
git -C "$R2_SRC" checkout --quiet --detach "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo "ERROR TurboQuant PR #$PR does not apply cleanly to current R2 source." >&2
  echo "Candidate retained for semantic port: $R2_SRC" >&2
  git status --short >&2 || true
  exit 10
fi

mapfile -t CHANGED < <(git diff --name-only "$BASE_HEAD"..HEAD | sort -u)
if [[ ${#CHANGED[@]} -ne 1 || "${CHANGED[0]}" != "$EXPECTED_FILE" ]]; then
  echo "ERROR Stage16 delta is no longer the audited one-file MMVF change" >&2
  printf 'changed: %s\n' "${CHANGED[@]}" >&2
  exit 11
fi
git diff --check "$BASE_HEAD"..HEAD

for marker in GGML_MMVF_NARROW_MIN GGML_MMVF_NARROW_MAX ggml_cuda_mmvf_weight_is_narrow; do
  grep -q "$marker" "$EXPECTED_FILE" || {
    echo "ERROR expected MMVF marker missing: $marker" >&2
    exit 12
  }
done

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "base_alias=$BASE_ALIAS"
  echo "source_pr=TheTom/llama-cpp-turboquant#$PR"
  echo "pr_head=$PR_HEAD"
  echo "patch_sha256=$PATCH_SHA"
  echo "candidate_head=$(git rev-parse HEAD)"
  echo "changed_file=$EXPECTED_FILE"
  echo "rollback_env=GGML_MMVF_NARROW_MAX=0"
  echo "ctrl_off_n2=$CTRL_OFF_N2"
  echo "ctrl_on_n2=$CTRL_ON_N2"
  echo "off_n4=$OFF_N4"
  echo "on_n4=$ON_N4"
} > r2-meta/stage16-mmvf-narrow-manifest.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-mmvf-narrow}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops
[[ -x "$BUILD/bin/llama-server" ]] || { echo "ERROR llama-server missing" >&2; exit 20; }

# MMVF is a backend dispatch change. Run MUL_MAT coverage before any end-to-end
# timing so a fast wrong narrow matmul never reaches the benchmark leaderboard.
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o MUL_MAT
fi

REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
  bash "$SCRIPT_DIR/stage_runtime_bundle.sh" \
    "$BUILD/bin" "$RUNTIME" "$R2_SRC/r2-meta/runtime-bundle"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage16-llama-server.sha256

# n-max=2 control, same patched ELF. OFF restores upstream dispatch exactly;
# ON uses per-architecture narrow bands from the candidate.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$CTRL_OFF_N2" --r2-bin "$RUNTIME" --jmax keep \
  --spec-draft-n-max 2 \
  --env GGML_MMVF_NARROW_MAX=0 \
  --unset-env GGML_MMVF_NARROW_MIN \
  --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$CTRL_ON_N2" --r2-bin "$RUNTIME" --jmax keep \
  --spec-draft-n-max 2 \
  --unset-env GGML_MMVF_NARROW_MIN \
  --unset-env GGML_MMVF_NARROW_MAX \
  --replace

# n-max=4 is the discriminating width. Both arms use the same model/runtime and
# speculative depth; only the narrow-MMVF dispatch is toggled.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$OFF_N4" --r2-bin "$RUNTIME" --jmax keep \
  --spec-draft-n-max 4 \
  --env GGML_MMVF_NARROW_MAX=0 \
  --unset-env GGML_MMVF_NARROW_MIN \
  --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$ON_N4" --r2-bin "$RUNTIME" --jmax keep \
  --spec-draft-n-max 4 \
  --unset-env GGML_MMVF_NARROW_MIN \
  --unset-env GGML_MMVF_NARROW_MAX \
  --replace --validate

cat <<EOF
STAGE16_MMVF_READY=1
BASE_ALIAS=$BASE_ALIAS
CTRL_OFF_N2=$CTRL_OFF_N2
CTRL_ON_N2=$CTRL_ON_N2
OFF_N4=$OFF_N4
ON_N4=$ON_N4
R2_SRC=$R2_SRC
RUNTIME=$RUNTIME
PRODUCTION_PROMOTED=NO
Next: bash $SCRIPT_DIR/run_stage16_mmvf_narrow_ab.sh
EOF
