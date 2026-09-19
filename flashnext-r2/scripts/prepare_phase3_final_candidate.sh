#!/usr/bin/env bash
set -euo pipefail

# Compose only Phase-2 winners into one clean Flash Next R2 candidate.
#
# This does NOT promote production. It builds a new source/runtime/llama-swap
# alias from the recorded long-context winner, then optionally adds:
#   - PR #29030 PLE direct-read, only if Phase-2 recorded PLE_PASS=1
#   - FR-Spec qwen4exp port + winning trimmed draft, only if FRSPEC_PASS=1
#
# Remaining parameter tuning (MTP depth/JMAX/graphs) happens after this
# composition. Long-context kernel winners such as ROCm TOP_K, RDNA4 FA256,
# QSA gather and pooled cache are already part of the recorded long winner.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
BACKUP_ROOT="${BACKUP_ROOT:-/app/share/backup}"
FINAL_SRC="${FINAL_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919}"
FINAL_RUNTIME="${FINAL_RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-final-pre-sweep/bin}"
FINAL_ALIAS="${FINAL_ALIAS:-qwen3.8-flash-next-r2-final-pre-sweep:256k}"
FR_ALIAS="${FR_ALIAS:-qwen3.8-flash-next-r2-modern-frspec-65k:256k}"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PLE_HEAD="5fd6ce3d053f9477681be81a1be08cb3f5958523"
PLE_PATCH="$PATCH_DIR/pr29030-lazy-direct-${PLE_HEAD:0:8}.patch"
PLE_URL="https://github.com/pwilkin/llama.cpp/commit/${PLE_HEAD}.patch"
STAMP="$(date +%Y%m%d-%H%M%S)"
META_DIR="$FINAL_SRC/r2-meta"

latest_phase2_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase2-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value() {
  local file="$1" key="$2"
  awk -F= -v k="$key" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$file"
}
find_alias_block() {
  python3 - "$CONFIG" "$1" <<'PY'
import re, sys
p, alias = sys.argv[1:]
lines = open(p, encoding='utf-8').read().splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i,line in enumerate(lines):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); out=[line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    print('\n'.join(out)); raise SystemExit(0)
raise SystemExit(3)
PY
}
extract_draft() {
  python3 - "$CONFIG" "$1" <<'PY'
import re, sys
p, alias = sys.argv[1:]
lines=open(p,encoding='utf-8').read().splitlines()
pat=re.compile(r'^(\s*)'+re.escape(alias)+r':\s*(?:#.*)?$')
block=None
for i,line in enumerate(lines):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); out=[line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind: break
        out.append(s)
    block='\n'.join(out); break
if block is None: raise SystemExit('ERROR alias not found: '+alias)
opts=r'(?:--spec-draft-model|--model-draft|--draft-model|-md)'
m=list(re.finditer(rf'(?<!\S){opts}\s+("[^"]*"|\'[^\']*\'|\S+)',block))
if len(m)!=1: raise SystemExit(f'ERROR expected one draft option, found {len(m)}')
print(m[0].group(1).strip('"\''))
PY
}

PHASE2_SUMMARY="${PHASE2_SUMMARY:-$(latest_phase2_summary || true)}"
[[ -n "$PHASE2_SUMMARY" && -f "$PHASE2_SUMMARY" ]] || {
  echo "ERROR: Phase-2 summary not found. Run run_phase2_real_ab.sh first or set PHASE2_SUMMARY." >&2
  exit 2
}
[[ -f "$CONFIG" ]] || { echo "ERROR config missing: $CONFIG" >&2; exit 3; }
[[ ! -e "$FINAL_SRC" && ! -e "$FINAL_RUNTIME" ]] || {
  echo "ERROR final source/runtime already exists; refusing overwrite" >&2
  echo "source=$FINAL_SRC runtime=$FINAL_RUNTIME" >&2
  exit 4
}

LONG_SRC="$(summary_value "$PHASE2_SUMMARY" LONG_WINNER_SRC)"
LONG_ALIAS="$(summary_value "$PHASE2_SUMMARY" LONG_WINNER_ALIAS)"
PLE_PASS="$(summary_value "$PHASE2_SUMMARY" PLE_PASS)"
FR_PASS="$(summary_value "$PHASE2_SUMMARY" FRSPEC_PASS)"
PROD_PROMOTED="$(summary_value "$PHASE2_SUMMARY" PRODUCTION_PROMOTED)"

[[ "$PROD_PROMOTED" == NO ]] || {
  echo "ERROR: Phase-2 summary does not prove production remained untouched." >&2
  exit 5
}
[[ -n "$LONG_SRC" && -e "$LONG_SRC/.git" ]] || {
  echo "ERROR: long-context winner source missing: $LONG_SRC" >&2
  exit 6
}
[[ -z "$(git -C "$LONG_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: long-context winner source has tracked modifications" >&2
  git -C "$LONG_SRC" status --short --untracked-files=no >&2 || true
  exit 7
}
find_alias_block "$LONG_ALIAS" >/dev/null || {
  echo "ERROR: long-context winner alias missing: $LONG_ALIAS" >&2
  exit 8
}

BASE_HEAD="$(git -C "$LONG_SRC" rev-parse HEAD)"
echo "=== Flash Next R2 Phase-3 composition ==="
echo "phase2 summary : $PHASE2_SUMMARY"
echo "long source    : $LONG_SRC"
echo "long head      : $BASE_HEAD"
echo "long alias     : $LONG_ALIAS"
echo "PLE winner     : $PLE_PASS"
echo "FR-Spec winner : $FR_PASS"
echo "final source   : $FINAL_SRC"
echo "final runtime  : $FINAL_RUNTIME"
echo "final alias    : $FINAL_ALIAS"

# Independent clone, not a linked worktree. This keeps the final candidate easy
# to archive/move and prevents later worktree cleanup from invalidating it.
git clone --quiet --no-hardlinks "$LONG_SRC" "$FINAL_SRC"
git -C "$FINAL_SRC" checkout --quiet --detach "$BASE_HEAD"
mkdir -p "$META_DIR" "$PATCH_DIR"

{
  echo "created=$(date -Is)"
  echo "phase2_summary=$PHASE2_SUMMARY"
  echo "base_source=$LONG_SRC"
  echo "base_head=$BASE_HEAD"
  echo "base_alias=$LONG_ALIAS"
  echo "ple_pass=$PLE_PASS"
  echo "frspec_pass=$FR_PASS"
} > "$META_DIR/phase3-compose-manifest.txt"

# ----- optional PLE direct-read winner -----
PLE_ENABLED=0
if [[ "$PLE_PASS" == 1 ]]; then
  if [[ ! -f "$PLE_PATCH" ]]; then
    curl -fL --retry 3 --connect-timeout 20 "$PLE_URL" -o "$PLE_PATCH"
  fi
  head -n1 "$PLE_PATCH" | grep -qi "${PLE_HEAD:0:12}" || {
    echo "ERROR: pinned PLE patch identity mismatch" >&2
    exit 10
  }
  if ! git -C "$FINAL_SRC" -c user.name='FlashNext R2 Final' \
      -c user.email='flashnext-r2@local.invalid' am --3way "$PLE_PATCH"; then
    echo "ERROR: PLE direct-read winner conflicts with selected long-context winner." >&2
    echo "Final candidate retained for semantic merge: $FINAL_SRC" >&2
    exit 11
  fi
  grep -Rq 'LLAMA_LAZY_MODE_DIRECT' "$FINAL_SRC/common" "$FINAL_SRC/include" "$FINAL_SRC/src" || {
    echo "ERROR: PLE direct-read marker missing after composition" >&2
    exit 12
  }
  PLE_ENABLED=1
  echo "phase3_ple_head=$(git -C "$FINAL_SRC" rev-parse HEAD)" >> "$META_DIR/phase3-compose-manifest.txt"
fi

# ----- optional FR-Spec winner -----
FR_DRAFT=""
if [[ "$FR_PASS" == 1 ]]; then
  FR_DRAFT="$(extract_draft "$FR_ALIAS")"
  [[ -f "$FR_DRAFT" ]] || {
    echo "ERROR: winning FR-Spec draft sidecar missing: $FR_DRAFT" >&2
    exit 20
  }

  python3 "$SCRIPT_DIR/port_stage12_frspec_qwen4exp.py" "$FINAL_SRC/src/models/qwen4exp.cpp"
  git -C "$FINAL_SRC" diff --check
  if ! git -C "$FINAL_SRC" diff --quiet; then
    git -C "$FINAL_SRC" add src/models/qwen4exp.cpp
    git -C "$FINAL_SRC" -c user.name='FlashNext R2 Final' \
      -c user.email='flashnext-r2@local.invalid' \
      commit --quiet -m 'r2 final: apply validated qwen4exp FR-Spec port'
  fi
  echo "frspec_draft=$FR_DRAFT" >> "$META_DIR/phase3-compose-manifest.txt"
  sha256sum "$FR_DRAFT" > "$META_DIR/phase3-frspec-draft.sha256"
fi

[[ -z "$(git -C "$FINAL_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: final source has uncommitted tracked changes before build" >&2
  git -C "$FINAL_SRC" status --short --untracked-files=no >&2 || true
  exit 30
}

git -C "$FINAL_SRC" diff --check "$BASE_HEAD"..HEAD
FINAL_HEAD="$(git -C "$FINAL_SRC" rev-parse HEAD)"
echo "final_head=$FINAL_HEAD" >> "$META_DIR/phase3-compose-manifest.txt"
git -C "$FINAL_SRC" diff --name-status "$BASE_HEAD"..HEAD > "$META_DIR/phase3-changed-files.txt"

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
BUILD="${BUILD:-$FINAL_SRC/build-r2-final-pre-sweep}"

cmake -S "$FINAL_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 40; }
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
  if grep -Rq 'GGML_RDNA4_FA256_MMA' "$FINAL_SRC/ggml/src/ggml-cuda"; then
    "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT
  fi
fi

# Stage before adding an optional launcher wrapper so the real ELF and all local
# libllama/libggml dependencies are normalized and verified independently from
# the temporary CMake tree.
REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
  bash "$SCRIPT_DIR/stage_runtime_bundle.sh" \
    "$BUILD/bin" "$FINAL_RUNTIME" "$META_DIR/runtime-bundle"

# When PLE direct-read won, force the validated mode with a wrapper so a stale
# inherited --lazy-mode cannot silently change the final candidate.
if [[ "$PLE_ENABLED" == 1 ]]; then
  mv "$FINAL_RUNTIME/llama-server" "$FINAL_RUNTIME/llama-server.real"
  cat > "$FINAL_RUNTIME/llama-server" <<'EOF'
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
  chmod +x "$FINAL_RUNTIME/llama-server"

  OUT="$META_DIR/runtime-bundle/runtime-verify-post-wrapper.txt" REQUIRE_BOTH_GPUS="${REQUIRE_BOTH_GPUS:-1}" \
    bash "$SCRIPT_DIR/verify_runtime_bundle.sh" "$FINAL_RUNTIME"
fi

REAL_BIN="$FINAL_RUNTIME/llama-server"
[[ -x "$REAL_BIN" ]] || { echo "ERROR final launcher missing" >&2; exit 41; }
if [[ -x "$FINAL_RUNTIME/llama-server.real" ]]; then
  sha256sum "$FINAL_RUNTIME/llama-server.real" > "$META_DIR/phase3-llama-server.sha256"
else
  sha256sum "$FINAL_RUNTIME/llama-server" > "$META_DIR/phase3-llama-server.sha256"
fi
"$FINAL_RUNTIME/llama-server" --version | tee "$META_DIR/phase3-version.txt" || true

# Clone the long-context winner's exact args/env/split settings. Only runtime and,
# if validated, the FR-Spec draft path change here.
INSTALL=(python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py"
  --config "$CONFIG"
  --source-alias "$LONG_ALIAS"
  --alias "$FINAL_ALIAS"
  --r2-bin "$FINAL_RUNTIME"
  --jmax keep
  --replace
  --validate)
if [[ "$FR_PASS" == 1 ]]; then
  INSTALL+=(--draft-model "$FR_DRAFT")
fi
"${INSTALL[@]}"

# Keep an exact post-compose config backup. The installer already makes its own
# timestamped backup; this one is colocated with the final manifest for recovery.
mkdir -p "$BACKUP_ROOT/flashnext-r2-final-$STAMP"
cp -a "$CONFIG" "$BACKUP_ROOT/flashnext-r2-final-$STAMP/config-rocm714.yaml.after-compose"

cat <<EOF

PHASE3_COMPOSE=READY
FINAL_SRC=$FINAL_SRC
FINAL_HEAD=$FINAL_HEAD
FINAL_RUNTIME=$FINAL_RUNTIME
FINAL_ALIAS=$FINAL_ALIAS
BASE_ALIAS=$LONG_ALIAS
PLE_ENABLED=$PLE_ENABLED
FRSPEC_ENABLED=$FR_PASS
FRSPEC_DRAFT=$FR_DRAFT
PRODUCTION_PROMOTED=NO

Next: bash $SCRIPT_DIR/run_phase3_final_validation.sh
EOF
