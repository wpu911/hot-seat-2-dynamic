#!/usr/bin/env bash
set -euo pipefail

# Phase-5: isolate the remaining JohnTDI GDN micro-fusions on top of the fully
# validated modern R2 lineage.  Modern upstream already carries HC fusion,
# fused GDN execution, RDNA3/RDNA4 ncols_opt and Q8 broadcast dedup, so this
# experiment ports ONLY:
#   E7   - sigmoid(beta) + softplus(alpha+dt)*A in the GDN kernel
#   E7b  - q/k L2 normalization in the GDN kernel
#   E7b2 - selectable FMA accumulation used to match stock norm numerics
#
# No HC patch, no old Q8 dedup patch, no manual JMAX patch.
# Production is never overwritten.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-gdn-microfusion-20260919}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-gdn-microfusion/bin}"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-gdn-modern-off:256k}"
PROLOG_ALIAS="${PROLOG_ALIAS:-qwen3.8-flash-next-r2-gdn-prolog:256k}"
FULL_ALIAS="${FULL_ALIAS:-qwen3.8-flash-next-r2-gdn-prolog-l2:256k}"

latest_phase4_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase4-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value() {
  awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"
}
PHASE4_SUMMARY="${PHASE4_SUMMARY:-$(latest_phase4_summary || true)}"
[[ -n "$PHASE4_SUMMARY" && -f "$PHASE4_SUMMARY" ]] || {
  echo "ERROR: Phase-4 summary not found; run run_phase4_mtp_graph_sweep.sh first." >&2
  exit 2
}
SOURCE_ALIAS="${SOURCE_ALIAS:-$(summary_value "$PHASE4_SUMMARY" PARAM_WINNER_ALIAS)}"
[[ -n "$SOURCE_ALIAS" ]] || { echo "ERROR: PARAM_WINNER_ALIAS missing in Phase-4 summary" >&2; exit 3; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 4; }
[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR: final source missing: $BASE_SRC" >&2; exit 5; }
[[ -f "$BASE_SRC/r2-meta/phase3-compose-manifest.txt" ]] || {
  echo "ERROR: base is not the recorded Phase-3 composed source: $BASE_SRC" >&2
  exit 6
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: base source has tracked modifications" >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 7
}
[[ ! -e "$R2_SRC" && ! -e "$(dirname "$RUNTIME")" ]] || {
  echo "ERROR: Phase-5 source/runtime already exists; refusing overwrite" >&2
  echo "R2_SRC=$R2_SRC RUNTIME=$RUNTIME" >&2
  exit 8
}

grep -qE "^[[:space:]]*${SOURCE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: Phase-4 winner alias is not present in llama-swap config: $SOURCE_ALIAS" >&2
  exit 9
}

# Pinned public patch series. These are the same audited blobs used by the older
# Stage-3 experiment, but Stage-5 intentionally omits the old HC/JMAX/Q8 patches.
PIN="185252d1edb27fde6b332908eb7c89a20cadc4bb"
BASE_URL="https://raw.githubusercontent.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4/$PIN/johnv8/patches/seria-fuzje"
P7="$PATCH_DIR/0008-gdn-prolog.patch"
U7="$BASE_URL/0008-E7-prolog-GDN-sigmoid-beta-softplus-alpha-dt-A-liczo.patch"
H7="2717c0af96ea14612ccf85835b1933ed6277d77c"
P7B="$PATCH_DIR/0012-gdn-l2.patch"
U7B="$BASE_URL/0012-E7b-l2norm-q-k-liczony-w-jadrze-gated_delta_net-op_p.patch"
H7B="d9f7ba405caedd67d9b57b8e1d23e997d3292212"
P7F="$PATCH_DIR/0016-gdn-l2-fma.patch"
U7F="$BASE_URL/0016-E7b-przelacznik-GGML_JOHNV8_GDN_L2_FMA-fmaf-vs-rn-do.patch"
H7F="b7417dc4fb105ec0665562b8b37d1c55e0a9bd61"

mkdir -p "$PATCH_DIR"
fetch_verify() {
  local out="$1" url="$2" hash="$3" label="$4"
  [[ -f "$out" ]] || curl -fL --retry 3 --connect-timeout 20 "$url" -o "$out"
  local got
  got="$(git hash-object "$out")"
  [[ "$got" == "$hash" ]] || {
    echo "ERROR: $label patch identity mismatch: expected=$hash got=$got" >&2
    exit 10
  }
  echo "$label patch verified: $got"
}
fetch_verify "$P7" "$U7" "$H7" "GDN E7"
fetch_verify "$P7B" "$U7B" "$H7B" "GDN E7b"
fetch_verify "$P7F" "$U7F" "$H7F" "GDN E7b-FMA"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git clone --quiet --no-hardlinks "$BASE_SRC" "$R2_SRC"
git -C "$R2_SRC" checkout --quiet --detach "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

apply_or_stop() {
  local p="$1" label="$2"
  echo "Applying $label"
  if ! git apply --3way --whitespace=nowarn "$p"; then
    echo "ERROR: $label does not apply cleanly to the modern final source." >&2
    echo "Tree retained for semantic inspection: $R2_SRC" >&2
    echo "Do not use blanket --theirs; modern upstream already owns adjacent GDN code." >&2
    exit 20
  fi
}
apply_or_stop "$P7"  "E7 GDN prolog"
apply_or_stop "$P7B" "E7b GDN L2"
apply_or_stop "$P7F" "E7b FMA selector"

# JohnTDI's final HIP compatibility audit found explicit rn intrinsics can change
# gfx1201 bits relative to the stock expressions. Keep FMA accumulation selectable,
# but use ordinary operators for the pure add/scale sites, matching the previously
# audited migration rather than dragging unrelated later-series patches in.
python3 - <<'PY'
from pathlib import Path
p=Path('ggml/src/ggml-cuda/gated_delta_net.cu')
s=p.read_text()
repls={
 'const float ab = __fadd_rn(g_raw, pro_dt[h_idx]);':'const float ab = g_raw + pro_dt[h_idx];',
 'g_raw = __fmul_rn(sp, pro_a[h_idx]);':'g_raw = sp * pro_a[h_idx];',
 'k_reg[r] = __fmul_rn(sk, k_reg[r]);':'k_reg[r] = sk * k_reg[r];',
 'q_reg[r] = __fmul_rn(sq, q_reg[r]);':'q_reg[r] = sq * q_reg[r];',
}
for old,new in repls.items():
    n=s.count(old)
    if n != 1:
        raise SystemExit(f'ERROR compatibility migration expected one match, got {n}: {old}')
    s=s.replace(old,new)
p.write_text(s)
PY

git diff --check
git add -A
git -c user.name='FlashNext R2 Phase5' -c user.email='flashnext-r2@local.invalid' \
  commit --quiet -m 'r2 phase5: port isolated GDN prolog and L2 microfusions'
CAND_HEAD="$(git rev-parse HEAD)"

# Hard feature probes. Modern upstream HC/fused-GDN must remain, while the new
# runtime switches must be present exactly as the experiment expects.
for n in GGML_JOHNV8_GDN_PROLOG GGML_JOHNV8_GDN_L2 GGML_JOHNV8_GDN_L2_FMA; do
  grep -Rqs "$n" ggml src || { echo "ERROR: missing Phase-5 switch: $n" >&2; exit 21; }
done
grep -Rqs 'fused_gdn_ar' src || { echo "ERROR: modern fused GDN support disappeared" >&2; exit 22; }
grep -Rqs 'fused_dsv4_hc_pre\|ggml_dsv4_hc_pre_gated' src ggml || {
  echo "ERROR: modern HC fusion marker disappeared" >&2; exit 23;
}

{
  echo "created=$(date -Is)"
  echo "phase4_summary=$PHASE4_SUMMARY"
  echo "source_alias=$SOURCE_ALIAS"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "candidate_head=$CAND_HEAD"
  echo "johnv8_pin=$PIN"
  echo "e7_blob=$H7"
  echo "e7b_blob=$H7B"
  echo "e7b_fma_blob=$H7F"
} > r2-meta/phase5-gdn-manifest.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-phase5-gdn}"
cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops
[[ -x "$BUILD/bin/llama-server" ]] || { echo "ERROR: llama-server missing after build" >&2; exit 30; }

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"

# Preserve the validated Phase-3 PLE winner. Its direct-read selection lived in a
# runtime wrapper, not in the cloned alias text, so rebuilding must recreate it.
PLE_PASS="$(awk -F= '$1=="ple_pass"{print $2}' "$BASE_SRC/r2-meta/phase3-compose-manifest.txt" | tail -n1)"
if [[ "$PLE_PASS" == 1 ]]; then
  mv "$RUNTIME/llama-server" "$RUNTIME/llama-server.real"
  cat > "$RUNTIME/llama-server" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -lzm|--lazy-mode) shift; [[ $# -gt 0 ]] && shift ;;
    -lzm=*|--lazy-mode=*) shift ;;
    *) out+=("$1"); shift ;;
  esac
done
exec "$SELF_DIR/llama-server.real" --lazy-mode on-direct "${out[@]}"
EOF
  chmod +x "$RUNTIME/llama-server"
fi

REAL="$RUNTIME/llama-server"
[[ -x "$RUNTIME/llama-server.real" ]] && REAL="$RUNTIME/llama-server.real"
sha256sum "$REAL" | tee r2-meta/phase5-gdn-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/phase5-gdn-version.txt || true

# Same binary and same Phase-4-selected MTP/Graph/model settings on all arms.
# Only the three GDN experiment env variables differ.
common=(--config "$CONFIG" --source-alias "$SOURCE_ALIAS" --r2-bin "$RUNTIME" --jmax keep --replace)
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" "${common[@]}" --alias "$OFF_ALIAS" \
  --env GGML_JOHNV8_GDN_PROLOG=0 --env GGML_JOHNV8_GDN_L2=0 --env GGML_JOHNV8_GDN_L2_FMA=1
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" "${common[@]}" --alias "$PROLOG_ALIAS" \
  --env GGML_JOHNV8_GDN_PROLOG=1 --env GGML_JOHNV8_GDN_L2=0 --env GGML_JOHNV8_GDN_L2_FMA=1
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" "${common[@]}" --alias "$FULL_ALIAS" \
  --env GGML_JOHNV8_GDN_PROLOG=1 --env GGML_JOHNV8_GDN_L2=1 --env GGML_JOHNV8_GDN_L2_FMA=1 --validate

cat <<EOF
PHASE5_GDN_READY=1
SOURCE_ALIAS=$SOURCE_ALIAS
OFF_ALIAS=$OFF_ALIAS
PROLOG_ALIAS=$PROLOG_ALIAS
FULL_ALIAS=$FULL_ALIAS
R2_SRC=$R2_SRC
RUNTIME=$RUNTIME
PLE_DIRECT_PRESERVED=$PLE_PASS
PRODUCTION_PROMOTED=NO
Next: bash $SCRIPT_DIR/run_phase5_gdn_microfusion_ab.sh
EOF
