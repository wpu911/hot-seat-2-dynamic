#!/usr/bin/env bash
set -euo pipefail

# Phase-6: test upstream qwen4exp tensor-split enablement (#28569) on the
# already validated Flash Next R2 lineage.
#
# Why this exists:
#   * the current deployment is a heterogeneous 2-GPU ROCm box
#   * layer split has not recovered the old single-GPU Flash Next throughput
#   * upstream #28569 re-enables -sm tensor for qwen4exp with a tiny scheduler fix
#   * ROCm users in the PR report roughly flat short-context speed but better
#     long-context retention versus layer split
#
# Important constraints from upstream:
#   * --fit is not implemented for SPLIT_MODE_TENSOR, so tensor candidates MUST
#     run with --fit off
#   * this first arm deliberately uses tensor-split 1,1 because it is the only
#     ratio with published ROCm qwen4exp reports. Heterogeneous ratio tuning is
#     a later phase after device order / VRAM headroom are measured live.
#
# Production alias/runtime are never replaced.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-gdn-microfusion-20260919}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
LAYER_ALIAS="${LAYER_ALIAS:-qwen3.8-flash-next-r2-split-layer:256k}"
TENSOR_ALIAS="${TENSOR_ALIAS:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
PR_HEAD="53c2a4c9fd411ec1cbb0f08edcb6071fa8224818"
PR_BASE="e71b80510c848c00175924ecf3c40333ccae8eb5"
PATCH="$PATCH_DIR/pr28569-qwen4exp-sm-tensor-${PR_HEAD:0:8}.patch"
PATCH_URL="https://github.com/kh0pper/llama.cpp/commit/${PR_HEAD}.patch"

latest_phase5_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase5-gdn-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value() {
  awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"
}

P5="${PHASE5_SUMMARY:-$(latest_phase5_summary || true)}"
[[ -n "$P5" && -f "$P5" ]] || {
  echo "ERROR: Phase-5 summary missing; run run_phase5_gdn_microfusion_ab.sh first." >&2
  exit 2
}
SOURCE_ALIAS="${SOURCE_ALIAS:-$(summary_value "$P5" PHASE5_WINNER_ALIAS)}"
[[ -n "$SOURCE_ALIAS" ]] || { echo "ERROR: PHASE5_WINNER_ALIAS missing" >&2; exit 3; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 4; }
[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR: Phase-5 source missing: $BASE_SRC" >&2; exit 5; }
[[ -f "$BASE_SRC/r2-meta/phase5-gdn-manifest.txt" ]] || {
  echo "ERROR: BASE_SRC is not the recorded Phase-5 source" >&2
  exit 6
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: Phase-5 source has tracked modifications" >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 7
}
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME_ROOT" ]] || {
  echo "ERROR: Phase-6 source/runtime already exists; refusing overwrite" >&2
  exit 8
}
grep -qE "^[[:space:]]*${SOURCE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: Phase-5 winner alias missing in llama-swap config: $SOURCE_ALIAS" >&2
  exit 9
}

mkdir -p "$PATCH_DIR"
[[ -f "$PATCH" ]] || curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
head -n1 "$PATCH" | grep -qi "${PR_HEAD:0:12}" || {
  echo "ERROR: PR #28569 patch identity mismatch" >&2
  exit 10
}
PATCH_SHA256="$(sha256sum "$PATCH" | awk '{print $1}')"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git clone --quiet --no-hardlinks "$BASE_SRC" "$R2_SRC"
git -C "$R2_SRC" checkout --quiet --detach "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git apply --3way --whitespace=nowarn "$PATCH"; then
  echo "ERROR: PR #28569 did not apply cleanly to Phase-5 source." >&2
  echo "Candidate retained for semantic inspection: $R2_SRC" >&2
  exit 11
fi

git diff --check

# Hard semantic probes. Do not trust a successful 3-way apply if adjacent
# upstream refactors changed the actual meaning.
python3 - <<'PY'
from pathlib import Path
arch = Path('src/llama-arch.cpp').read_text()
start = arch.find('bool llm_arch_supports_sm_tensor')
if start < 0:
    raise SystemExit('ERROR llm_arch_supports_sm_tensor missing')
end = arch.find('\n}', start)
body = arch[start:end]
if 'case LLM_ARCH_QWEN4EXP' in body:
    raise SystemExit('ERROR qwen4exp is still disabled for tensor split')
q = Path('src/models/qwen4exp.cpp').read_text()
needle = 'ggml_build_forward_expand(gf, res_hc);'
if q.count(needle) < 1:
    raise SystemExit('ERROR qwen4exp hc_init split-anchor missing')
PY

git add -A
git -c user.name='FlashNext R2 Phase6' -c user.email='flashnext-r2@local.invalid' \
  commit --quiet -m 'r2 phase6: enable qwen4exp tensor split for isolated A/B'
CAND_HEAD="$(git rev-parse HEAD)"

{
  echo "created=$(date -Is)"
  echo "phase5_summary=$P5"
  echo "source_alias=$SOURCE_ALIAS"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "candidate_head=$CAND_HEAD"
  echo "pr=ggml-org/llama.cpp#28569"
  echo "pr_base=$PR_BASE"
  echo "pr_head=$PR_HEAD"
  echo "patch_sha256=$PATCH_SHA256"
  echo "tensor_split_first_arm=1,1"
  echo "tensor_fit=off"
} > r2-meta/phase6-tensor-split-manifest.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-phase6-tensor-split}"
cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-llama-archs
[[ -x "$BUILD/bin/llama-server" ]] || { echo "ERROR: llama-server missing" >&2; exit 20; }

# Architecture smoke before touching llama-swap aliases.
if [[ -x "$BUILD/bin/test-llama-archs" ]]; then
  "$BUILD/bin/test-llama-archs" -a qwen4exp | tee r2-meta/phase6-test-llama-archs.txt
fi
"$BUILD/bin/llama-server" --list-devices > r2-meta/phase6-device-list.txt 2>&1 || true

# Rebuilding from source loses any runtime wrapper chosen by Phase-3 (notably
# PLE on-direct), so fold that behavior into the split wrappers when needed.
PLE_PASS=0
if [[ -f "$R2_SRC/r2-meta/phase3-compose-manifest.txt" ]]; then
  PLE_PASS="$(awk -F= '$1=="ple_pass"{print $2}' "$R2_SRC/r2-meta/phase3-compose-manifest.txt" | tail -n1)"
  [[ -n "$PLE_PASS" ]] || PLE_PASS=0
fi

make_wrapper() {
  local dst="$1" mode="$2" ratio="$3"
  mkdir -p "$dst"
  cp -a "$BUILD/bin/." "$dst/"
  mv "$dst/llama-server" "$dst/llama-server.real"
  cat > "$dst/llama-server" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
out=()
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    -sm|--split-mode)
      shift; [[ \$# -gt 0 ]] && shift ;;
    -sm=*|--split-mode=*) shift ;;
EOF
  if [[ "$mode" == tensor ]]; then
    cat >> "$dst/llama-server" <<'EOF'
    -ts|--tensor-split)
      shift; [[ $# -gt 0 ]] && shift ;;
    -ts=*|--tensor-split=*) shift ;;
    -fit|--fit)
      shift
      if [[ $# -gt 0 && ( "$1" == "on" || "$1" == "off" ) ]]; then shift; fi ;;
    -fit=*|--fit=*) shift ;;
EOF
  fi
  if [[ "$PLE_PASS" == 1 ]]; then
    cat >> "$dst/llama-server" <<'EOF'
    -lzm|--lazy-mode)
      shift; [[ $# -gt 0 ]] && shift ;;
    -lzm=*|--lazy-mode=*) shift ;;
EOF
  fi
  cat >> "$dst/llama-server" <<'EOF'
    *) out+=("$1"); shift ;;
  esac
done
EOF
  if [[ "$mode" == tensor ]]; then
    cat >> "$dst/llama-server" <<EOF
extra=(--split-mode tensor --tensor-split "$ratio" --fit off)
EOF
  else
    cat >> "$dst/llama-server" <<'EOF'
extra=(--split-mode layer)
EOF
  fi
  if [[ "$PLE_PASS" == 1 ]]; then
    cat >> "$dst/llama-server" <<'EOF'
extra+=(--lazy-mode on-direct)
EOF
  fi
  cat >> "$dst/llama-server" <<'EOF'
exec "$SELF_DIR/llama-server.real" "${extra[@]}" "${out[@]}"
EOF
  chmod +x "$dst/llama-server"
}

LAYER_BIN="$RUNTIME_ROOT/layer/bin"
TENSOR_BIN="$RUNTIME_ROOT/tensor-1x1/bin"
make_wrapper "$LAYER_BIN" layer ""
make_wrapper "$TENSOR_BIN" tensor "1,1"

S_LAYER="$(sha256sum "$LAYER_BIN/llama-server.real" | awk '{print $1}')"
S_TENSOR="$(sha256sum "$TENSOR_BIN/llama-server.real" | awk '{print $1}')"
[[ "$S_LAYER" == "$S_TENSOR" ]] || { echo "ERROR: layer/tensor real binaries differ" >&2; exit 21; }
echo "$S_LAYER" > r2-meta/phase6-real-binary.sha256

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$LAYER_ALIAS" \
  --r2-bin "$LAYER_BIN" --jmax keep --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$TENSOR_ALIAS" \
  --r2-bin "$TENSOR_BIN" --jmax keep --replace --validate

cat <<EOF
PHASE6_TENSOR_SPLIT_READY=1
SOURCE_ALIAS=$SOURCE_ALIAS
LAYER_ALIAS=$LAYER_ALIAS
TENSOR_ALIAS=$TENSOR_ALIAS
TENSOR_SPLIT=1,1
TENSOR_FIT=off
REAL_BINARY_SHA256=$S_LAYER
PLE_DIRECT_PRESERVED=$PLE_PASS
PRODUCTION_PROMOTED=NO
Next: bash $SCRIPT_DIR/run_phase6_tensor_split_ab.sh
EOF
