#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-3: Gated DeltaNet fusion A/B.
#
# Goal: isolate GDN prolog + in-kernel L2 normalization while holding HC/JMAX
# constant on both aliases. Production source/runtime are never overwritten.
#
# The upstream GDN patch series is pinned by commit URL + git blob hash. We use:
#   E7   : prolog (sigmoid beta + softplus(alpha+dt)*A) inside GDN kernel
#   E7b  : L2 q/k normalization inside GDN kernel
#   E7b2 : FMA switch to match stock norm accumulation
#
# On HIP gfx1201 upstream later found __fmul_rn/__fadd_rn can differ from normal
# a*b/a+b. For the tested FMA=1 path we therefore normalize the remaining pure
# scale/add operations to ordinary operators after applying the upstream patches.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-gdn-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-gdn}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-gdn-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-gdn-on:256k}"
HC_MODE="${HC_MODE:-1}"
JMAX="${JMAX:-0}"

PIN="185252d1edb27fde6b332908eb7c89a20cadc4bb"
BASE_URL="https://raw.githubusercontent.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4/$PIN/johnv8/patches/seria-fuzje"

HC_PATCH="$PATCH_DIR/0004-hc-combine-hc-mix.patch"
HC_URL="$BASE_URL/0004-fuzje-hc-combine-hc-mix-z-forka-c689018e4-f76552838.patch"
HC_BLOB="17776143967955297e5a547e04cf209ed0d3d8e2"

GDN_PROLOG_PATCH="$PATCH_DIR/0008-gdn-prolog.patch"
GDN_PROLOG_URL="$BASE_URL/0008-E7-prolog-GDN-sigmoid-beta-softplus-alpha-dt-A-liczo.patch"
GDN_PROLOG_BLOB="2717c0af96ea14612ccf85835b1933ed6277d77c"

GDN_L2_PATCH="$PATCH_DIR/0012-gdn-l2.patch"
GDN_L2_URL="$BASE_URL/0012-E7b-l2norm-q-k-liczony-w-jadrze-gated_delta_net-op_p.patch"
GDN_L2_BLOB="d9f7ba405caedd67d9b57b8e1d23e997d3292212"

GDN_FMA_PATCH="$PATCH_DIR/0016-gdn-l2-fma.patch"
GDN_FMA_URL="$BASE_URL/0016-E7b-przelacznik-GGML_JOHNV8_GDN_L2_FMA-fmaf-vs-rn-do.patch"
GDN_FMA_BLOB="b7417dc4fb105ec0665562b8b37d1c55e0a9bd61"

fetch_verify() {
  local out="$1" url="$2" blob="$3" label="$4"
  if [[ ! -f "$out" ]]; then
    curl -fL --retry 3 --connect-timeout 20 "$url" -o "$out"
  fi
  local actual
  actual="$(git hash-object "$out")"
  if [[ "$actual" != "$blob" ]]; then
    echo "ERROR: $label patch hash mismatch" >&2
    echo "expected: $blob" >&2
    echo "actual  : $actual" >&2
    exit 4
  fi
  echo "$label verified: $actual"
}

if [[ ! -d "$PROD_SRC/.git" && ! -f "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$R2_SRC" ]]; then
  echo "ERROR: Stage-3 worktree already exists: $R2_SRC" >&2
  exit 3
fi
if [[ -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-3 runtime already exists: $RUNTIME" >&2
  exit 3
fi

mkdir -p "$PATCH_DIR"
fetch_verify "$HC_PATCH" "$HC_URL" "$HC_BLOB" "HC"
fetch_verify "$GDN_PROLOG_PATCH" "$GDN_PROLOG_URL" "$GDN_PROLOG_BLOB" "GDN prolog"
fetch_verify "$GDN_L2_PATCH" "$GDN_L2_URL" "$GDN_L2_BLOB" "GDN L2"
fetch_verify "$GDN_FMA_PATCH" "$GDN_FMA_URL" "$GDN_FMA_BLOB" "GDN FMA"

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Production source : $PROD_SRC"
echo "Production HEAD   : $PROD_HEAD"
echo "Stage-3 worktree  : $R2_SRC"
echo "Stage-3 runtime   : $RUNTIME"
echo "HC fixed mode     : $HC_MODE"
echo "JMAX fixed value  : $JMAX"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "upstream_pin=$PIN"
  echo "hc_mode=$HC_MODE"
  echo "jmax=$JMAX"
  echo "hc_blob=$HC_BLOB"
  echo "gdn_prolog_blob=$GDN_PROLOG_BLOB"
  echo "gdn_l2_blob=$GDN_L2_BLOB"
  echo "gdn_fma_blob=$GDN_FMA_BLOB"
} > "$R2_SRC/r2-meta/stage3-gdn-base.txt"
git -C "$PROD_SRC" status --short > "$R2_SRC/r2-meta/production-status-at-create.txt" || true

cd "$R2_SRC"

apply_or_stop() {
  local patch="$1" label="$2"
  echo "Applying $label ..."
  if ! git apply --3way "$patch"; then
    echo "ERROR: $label did not apply cleanly to production HEAD." >&2
    echo "The worktree is kept at: $R2_SRC" >&2
    echo "Do functional migration there; DO NOT force a context-blind patch." >&2
    exit 10
  fi
}

# Keep HC code in the binary so Stage-3 can hold HC at the Stage-2 winner value.
apply_or_stop "$HC_PATCH" "HC fusion"
apply_or_stop "$GDN_PROLOG_PATCH" "GDN prolog"
apply_or_stop "$GDN_L2_PATCH" "GDN L2"
apply_or_stop "$GDN_FMA_PATCH" "GDN L2 FMA selector"

# HIP numeric compatibility fix for the exact Stage-3 path.
# Upstream patch 0017 also touches unrelated shared-expert code and depends on
# preceding series patches, so we intentionally migrate only the GDN-relevant
# arithmetic here. FMA accumulation remains selected with GDN_L2_FMA=1.
python3 - <<'PY'
from pathlib import Path
p = Path("ggml/src/ggml-cuda/gated_delta_net.cu")
s = p.read_text()
repls = {
    "const float ab = __fadd_rn(g_raw, pro_dt[h_idx]);": "const float ab = g_raw + pro_dt[h_idx];",
    "g_raw = __fmul_rn(sp, pro_a[h_idx]);": "g_raw = sp * pro_a[h_idx];",
    "k_reg[r] = __fmul_rn(sk, k_reg[r]);": "k_reg[r] = sk * k_reg[r];",
    "q_reg[r] = __fmul_rn(sq, q_reg[r]);": "q_reg[r] = sq * q_reg[r];",
}
for old, new in repls.items():
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"numeric compatibility migration expected one match, got {n}: {old}")
    s = s.replace(old, new)
p.write_text(s)
PY

git diff --check
git diff > r2-meta/stage3-gdn.diff

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
BUILD="${BUILD:-$R2_SRC/build-r2-gdn}"

echo "ROCm path         : $ROCM_PATH"
echo "HIP compiler      : $HIP_CXX"
echo "AMDGPU targets    : $AMDGPU_TARGETS"
echo "Build directory   : $BUILD"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

if [[ ! -x "$BUILD/bin/llama-server" ]]; then
  echo "ERROR: build completed without llama-server" >&2
  exit 20
fi
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage3-gdn-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage3-gdn-version.txt" || true

COMMON_ENV=(
  --env "GGML_JOHNV8_HC_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_MIX_FUSE=$HC_MODE"
)

# SAME binary, SAME HC setting, SAME JMAX. Only GDN env differs.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax "$JMAX" \
  "${COMMON_ENV[@]}" \
  --env GGML_JOHNV8_GDN_PROLOG=0 \
  --env GGML_JOHNV8_GDN_L2=0 \
  --env GGML_JOHNV8_GDN_L2_FMA=1 \
  --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" \
  --r2-bin "$RUNTIME" \
  --jmax "$JMAX" \
  "${COMMON_ENV[@]}" \
  --env GGML_JOHNV8_GDN_PROLOG=1 \
  --env GGML_JOHNV8_GDN_L2=1 \
  --env GGML_JOHNV8_GDN_L2_FMA=1 \
  --replace \
  --validate

echo
echo "Stage-3 GDN build and llama-swap aliases are ready."
echo "OFF alias: $OFF_ALIAS"
echo " ON alias: $ON_ALIAS"
echo "Binary   : $RUNTIME/llama-server"
echo "HC mode  : $HC_MODE (held constant in both aliases)"
echo "JMAX     : $JMAX (held constant in both aliases)"
echo "Production alias/runtime remain untouched."
echo
echo "Next command:"
echo "  bash $SCRIPT_DIR/run_stage3_gdn_ab.sh"
