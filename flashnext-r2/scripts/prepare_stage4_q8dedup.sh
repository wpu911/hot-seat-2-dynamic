#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage-4: Q8_1 activation quantization dedup A/B.
# Builds a binary containing HC + GDN + Q8 dedup code, then holds HC/GDN/JMAX
# constant and toggles only GGML_JOHNV8_Q8_DEDUP between two llama-swap aliases.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-q8dedup-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-q8dedup}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"

OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-q8-off:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-q8-on:256k}"
HC_MODE="${HC_MODE:-1}"
GDN_MODE="${GDN_MODE:-1}"
JMAX="${JMAX:-0}"

PIN="185252d1edb27fde6b332908eb7c89a20cadc4bb"
BASE_URL="https://raw.githubusercontent.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4/$PIN/johnv8/patches/seria-fuzje"

P_HC="$PATCH_DIR/0004-hc-combine-hc-mix.patch"
U_HC="$BASE_URL/0004-fuzje-hc-combine-hc-mix-z-forka-c689018e4-f76552838.patch"
B_HC="17776143967955297e5a547e04cf209ed0d3d8e2"

P_GDNP="$PATCH_DIR/0008-gdn-prolog.patch"
U_GDNP="$BASE_URL/0008-E7-prolog-GDN-sigmoid-beta-softplus-alpha-dt-A-liczo.patch"
B_GDNP="2717c0af96ea14612ccf85835b1933ed6277d77c"

P_Q8="$PATCH_DIR/0009-q8-dedup.patch"
U_Q8="$BASE_URL/0009-E6d-pamiec-podreczna-kwantyzacji-Q8_1-aktywacji-dla-.patch"
B_Q8="5da75737593bb93de3932407249eff999bbb3f6f"

P_GDNL2="$PATCH_DIR/0012-gdn-l2.patch"
U_GDNL2="$BASE_URL/0012-E7b-l2norm-q-k-liczony-w-jadrze-gated_delta_net-op_p.patch"
B_GDNL2="d9f7ba405caedd67d9b57b8e1d23e997d3292212"

P_GDNFMA="$PATCH_DIR/0016-gdn-l2-fma.patch"
U_GDNFMA="$BASE_URL/0016-E7b-przelacznik-GGML_JOHNV8_GDN_L2_FMA-fmaf-vs-rn-do.patch"
B_GDNFMA="b7417dc4fb105ec0665562b8b37d1c55e0a9bd61"

fetch_verify() {
  local out="$1" url="$2" blob="$3" label="$4"
  if [[ ! -f "$out" ]]; then curl -fL --retry 3 --connect-timeout 20 "$url" -o "$out"; fi
  local actual="$(git hash-object "$out")"
  if [[ "$actual" != "$blob" ]]; then
    echo "ERROR: $label hash mismatch expected=$blob actual=$actual" >&2
    exit 4
  fi
  echo "$label verified: $actual"
}

if [[ ! -d "$PROD_SRC/.git" && ! -f "$PROD_SRC/.git" ]]; then
  echo "ERROR: production source is not a git worktree: $PROD_SRC" >&2; exit 2
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-4 source/runtime already exists; refusing overwrite" >&2; exit 3
fi

mkdir -p "$PATCH_DIR"
fetch_verify "$P_HC" "$U_HC" "$B_HC" "HC"
fetch_verify "$P_GDNP" "$U_GDNP" "$B_GDNP" "GDN prolog"
fetch_verify "$P_Q8" "$U_Q8" "$B_Q8" "Q8 dedup"
fetch_verify "$P_GDNL2" "$U_GDNL2" "$B_GDNL2" "GDN L2"
fetch_verify "$P_GDNFMA" "$U_GDNFMA" "$B_GDNFMA" "GDN FMA"

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$PROD_HEAD"
  echo "upstream_pin=$PIN"
  echo "hc_mode=$HC_MODE"
  echo "gdn_mode=$GDN_MODE"
  echo "jmax=$JMAX"
  echo "q8_blob=$B_Q8"
} > "$R2_SRC/r2-meta/stage4-q8-base.txt"

cd "$R2_SRC"
apply_or_stop() {
  local patch="$1" label="$2"
  echo "Applying $label ..."
  if ! git apply --3way "$patch"; then
    echo "ERROR: $label did not apply cleanly. Worktree kept: $R2_SRC" >&2
    exit 10
  fi
}

# Preserve upstream series ordering for files touched by these selected features.
apply_or_stop "$P_HC" "HC fusion"
apply_or_stop "$P_GDNP" "GDN prolog"
apply_or_stop "$P_Q8" "Q8_1 activation dedup"
apply_or_stop "$P_GDNL2" "GDN L2"
apply_or_stop "$P_GDNFMA" "GDN FMA selector"

# Same minimal HIP numeric compatibility migration as Stage 3.
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
        raise SystemExit(f"expected one GDN numeric migration match, got {n}: {old}")
    s = s.replace(old, new)
p.write_text(s)
PY

git diff --check
git diff > r2-meta/stage4-q8.diff

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-q8dedup}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

[[ -x "$BUILD/bin/llama-server" ]] || { echo "ERROR: no llama-server" >&2; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage4-q8-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage4-q8-version.txt" || true

COMMON_ENV=(
  --env "GGML_JOHNV8_HC_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_MIX_FUSE=$HC_MODE"
  --env "GGML_JOHNV8_GDN_PROLOG=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2=$GDN_MODE"
  --env "GGML_JOHNV8_GDN_L2_FMA=1"
)

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax "$JMAX" \
  "${COMMON_ENV[@]}" --env GGML_JOHNV8_Q8_DEDUP=0 --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax "$JMAX" \
  "${COMMON_ENV[@]}" --env GGML_JOHNV8_Q8_DEDUP=1 --replace --validate

echo
echo "Stage-4 Q8 dedup aliases ready."
echo "OFF: $OFF_ALIAS"
echo " ON: $ON_ALIAS"
echo "HC=$HC_MODE GDN=$GDN_MODE JMAX=$JMAX held constant."
echo "Next: bash $SCRIPT_DIR/run_stage4_q8dedup_ab.sh"
