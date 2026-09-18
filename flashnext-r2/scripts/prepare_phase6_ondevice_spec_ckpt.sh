#!/usr/bin/env bash
set -euo pipefail

# Phase-6 experimental A/B for ggml-org/llama.cpp#28118.
#
# The PR keeps transient speculative recurrent-state checkpoints on device. It is
# potentially a large TG win for qwen4exp, but upstream documents a hard-abort
# caveat when the recurrent state occupies more than one cell range. Therefore:
#   * this phase is A/B only, never promotion
#   * both aliases are forced to --parallel 1
#   * cached-prefix and rollback stress are mandatory
#   * even a PASS is recorded as EXPERIMENTAL until a safe multi-range fallback
#     exists or the exact production workload proves it never fragments.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-ondevice-ckpt-20260919}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-ondevice-ckpt}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-ckpt-host:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-ckpt-device:256k}"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PR_HEAD="82bacc5475f08bbe7c879ffe54d26e1fe01c8eb0"
PR_BASE="85c55223caf0a2ad0d1d88e5a73ab3fe36107867"
PATCH="$PATCH_DIR/pr28118-ondevice-ckpt-${PR_HEAD:0:8}.patch"
PATCH_URL="https://github.com/vahpetr/llama.cpp/commit/${PR_HEAD}.patch"

latest_phase4_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase4-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value(){ awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"; }
P4="${PHASE4_SUMMARY:-$(latest_phase4_summary || true)}"
[[ -n "$P4" && -f "$P4" ]] || { echo "ERROR Phase-4 summary missing" >&2; exit 2; }
SOURCE_ALIAS="${SOURCE_ALIAS:-$(summary_value "$P4" PARAM_WINNER_ALIAS)}"
[[ -n "$SOURCE_ALIAS" ]] || { echo "ERROR PARAM_WINNER_ALIAS missing" >&2; exit 3; }
[[ -f "$CONFIG" ]] || { echo "ERROR config missing" >&2; exit 4; }
[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR base source missing: $BASE_SRC" >&2; exit 5; }
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || { echo "ERROR base source dirty" >&2; exit 6; }
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME_ROOT" ]] || { echo "ERROR Phase-6 target exists; refusing overwrite" >&2; exit 7; }
grep -qE "^[[:space:]]*${SOURCE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || { echo "ERROR source alias missing: $SOURCE_ALIAS" >&2; exit 8; }

mkdir -p "$PATCH_DIR"
[[ -f "$PATCH" ]] || curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
head -n1 "$PATCH" | grep -qi "${PR_HEAD:0:12}" || { echo "ERROR PR28118 patch identity mismatch" >&2; exit 9; }
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git clone --quiet --no-hardlinks "$BASE_SRC" "$R2_SRC"
git -C "$R2_SRC" checkout --quiet --detach "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git apply --3way --whitespace=nowarn "$PATCH"; then
  echo "ERROR PR28118 did not apply cleanly to the selected final source." >&2
  echo "Candidate retained for semantic inspection: $R2_SRC" >&2
  exit 10
fi

# Convert the hard-wired PR into a same-binary runtime switch so OFF/ON are a
# legitimate A/B. OFF exactly restores PARTIAL_ONLY. ON adds ON_DEVICE.
python3 - <<'PY'
from pathlib import Path
p=Path('tools/server/server-context.cpp')
s=p.read_text()
old='LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE'
count=s.count(old)
if count != 8:
    raise SystemExit(f'ERROR expected 8 PR28118 checkpoint sites, got {count}')
s=s.replace(old, 'server_spec_ckpt_flags()')
marker='struct server_context_impl {'
if marker not in s:
    raise SystemExit('ERROR server_context_impl marker missing')
helper=r'''static llama_state_seq_flags server_spec_ckpt_flags() {
    static const bool on_device = []() {
        const char * e = std::getenv("QWEN4EXP_SPEC_CKPT_ON_DEVICE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
        (on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : 0);
}

'''
s=s.replace(marker, helper+marker, 1)
if '#include <cstdlib>' not in s:
    lines=s.splitlines(True)
    # Put the standard header before the first non-include block; harmless even
    # if the file's project headers precede it.
    at=0
    while at < len(lines) and (lines[at].startswith('#include') or not lines[at].strip()):
        at += 1
    lines.insert(at, '#include <cstdlib>\n')
    s=''.join(lines)
p.write_text(s)
PY

git diff --check
git add -A
git -c user.name='FlashNext R2 Phase6' -c user.email='flashnext-r2@local.invalid' \
  commit --quiet -m 'r2 phase6: switchable on-device speculative recurrent checkpoints'
CAND_HEAD="$(git rev-parse HEAD)"

grep -q 'QWEN4EXP_SPEC_CKPT_ON_DEVICE' tools/server/server-context.cpp || { echo "ERROR runtime switch missing" >&2; exit 11; }
[[ "$(grep -o 'server_spec_ckpt_flags()' tools/server/server-context.cpp | wc -l)" -ge 9 ]] || {
  echo "ERROR checkpoint call-site rewrite incomplete" >&2; exit 12;
}

{
  echo "created=$(date -Is)"
  echo "phase4_summary=$P4"
  echo "source_alias=$SOURCE_ALIAS"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "candidate_head=$CAND_HEAD"
  echo "pr=ggml-org/llama.cpp#28118"
  echo "pr_base=$PR_BASE"
  echo "pr_head=$PR_HEAD"
  echo "patch_sha256=$PATCH_SHA"
  echo "promotion_eligible=no_known_multirange_abort"
} > r2-meta/phase6-ondevice-ckpt-manifest.txt

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then ROCM_PATH=/opt/host-rocm/core-10.0; else ROCM_PATH=/opt/rocm; fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-phase6-ondevice-ckpt}"
cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server
[[ -x "$BUILD/bin/llama-server" ]] || { echo "ERROR llama-server missing" >&2; exit 20; }

# Both arms use byte-identical real binaries and are forced to -np 1, the only
# topology explicitly validated by PR28118. This is an experiment, not a silent
# production config mutation.
make_runtime(){
  local dst="$1" envval="$2"
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
    -np|--parallel) shift; [[ \$# -gt 0 ]] && shift ;;
    -np=*|--parallel=*) shift ;;
    *) out+=("\$1"); shift ;;
  esac
done
export QWEN4EXP_SPEC_CKPT_ON_DEVICE="$envval"
exec "\$SELF_DIR/llama-server.real" --parallel 1 "\${out[@]}"
EOF
  chmod +x "$dst/llama-server"
}
OFF_BIN="$RUNTIME_ROOT/host/bin"
ON_BIN="$RUNTIME_ROOT/device/bin"
make_runtime "$OFF_BIN" 0
make_runtime "$ON_BIN" 1
S0="$(sha256sum "$OFF_BIN/llama-server.real" | awk '{print $1}')"
S1="$(sha256sum "$ON_BIN/llama-server.real" | awk '{print $1}')"
[[ "$S0" == "$S1" ]] || { echo "ERROR real binaries differ" >&2; exit 21; }
echo "$S0" > r2-meta/phase6-real-binary.sha256

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$OFF_ALIAS" \
  --r2-bin "$OFF_BIN" --jmax keep --unset-env QWEN4EXP_SPEC_CKPT_ON_DEVICE --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" --alias "$ON_ALIAS" \
  --r2-bin "$ON_BIN" --jmax keep --unset-env QWEN4EXP_SPEC_CKPT_ON_DEVICE --replace --validate

cat <<EOF
PHASE6_ONDEVICE_CKPT_READY=1
SOURCE_ALIAS=$SOURCE_ALIAS
OFF_ALIAS=$OFF_ALIAS
ON_ALIAS=$ON_ALIAS
REAL_BINARY_SHA256=$S0
FORCED_PARALLEL=1
PROMOTION_ELIGIBLE=NO
KNOWN_RISK=ON_DEVICE recurrent state aborts if cell_ranges.size()>1
Next: bash $SCRIPT_DIR/run_phase6_ondevice_spec_ckpt_ab.sh
EOF
