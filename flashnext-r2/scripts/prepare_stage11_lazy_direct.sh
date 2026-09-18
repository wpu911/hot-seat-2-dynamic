#!/usr/bin/env bash
set -euo pipefail

# Stage 11: PR #29030 direct row reads for qwen4exp lazy PLE tensors.
# Compare --lazy-mode on (mmap demand paging) vs on-direct (explicit sorted,
# deduplicated, parallel positional reads) using the SAME candidate binary.
#
# IMPORTANT: run via with_exact_prod.sh so uncommitted production HotSeat edits
# are frozen into the baseline before this PR is applied.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-lazy-direct-20260918}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-lazy-direct}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next:256k}"
MMAP_ALIAS="${MMAP_ALIAS:-qwen3.8-flash-next-r2-lazy-mmap:256k}"
DIRECT_ALIAS="${DIRECT_ALIAS:-qwen3.8-flash-next-r2-lazy-direct:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH_HEAD="5fd6ce3d053f9477681be81a1be08cb3f5958523"
PATCH="$PATCH_DIR/pr29030-lazy-direct-${PATCH_HEAD:0:8}.patch"
PATCH_URL="https://github.com/pwilkin/llama.cpp/commit/${PATCH_HEAD}.patch"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git tree: $PROD_SRC" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: PROD_SRC is dirty." >&2
  echo "Run through exact snapshot wrapper:" >&2
  echo "  bash $SCRIPT_DIR/with_exact_prod.sh $SCRIPT_DIR/prepare_stage11_lazy_direct.sh" >&2
  exit 3
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME_ROOT" ]]; then
  echo "ERROR: Stage-11 candidate tree/runtime already exists" >&2
  echo "R2_SRC=$R2_SRC" >&2
  echo "RUNTIME_ROOT=$RUNTIME_ROOT" >&2
  exit 4
fi

# Inspect the production alias before building. PR #29030 compares lazy mmap with
# direct reads. If production forces --no-mmap, do not silently pretend this is
# the same experiment.
BLOCK="$(python3 - "$CONFIG" "$BASE_ALIAS" <<'PY'
import re, sys
p, alias = sys.argv[1:]
lines = open(p, encoding='utf-8').read().splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m: continue
    ind = len(m.group(1)); out=[line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    print('\n'.join(out)); raise SystemExit(0)
raise SystemExit(3)
PY
)" || { echo "ERROR: production alias not found: $BASE_ALIAS" >&2; exit 5; }

if grep -qE '(^|[[:space:]])--no-mmap([[:space:]]|$)' <<<"$BLOCK"; then
  echo "ERROR: production Flash Next alias explicitly contains --no-mmap." >&2
  echo "PR #29030 lazy mmap/direct A/B requires the lazy mmap path; inspect config before proceeding." >&2
  exit 6
fi

echo "=== production lazy-mode hints ==="
printf '%s\n' "$BLOCK" | grep -E -- '-lzm|--lazy-mode|mmap' || echo '(no explicit lazy-mode; candidate wrappers will force modes)'

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: pinned PR #29030 patch identity mismatch" >&2
  exit 7
fi
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

echo "Exact production : $PROD_SRC"
echo "Exact HEAD       : $(git -C "$PROD_SRC" rev-parse HEAD)"
echo "PR29030 head     : $PATCH_HEAD"
echo "Patch SHA256     : $PATCH_SHA"
echo "Stage-11 tree    : $R2_SRC"
echo "Runtime root     : $RUNTIME_ROOT"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" HEAD
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$(git -C "$PROD_SRC" rev-parse HEAD)"
  echo "upstream_pr=ggml-org/llama.cpp#29030"
  echo "patch_head=$PATCH_HEAD"
  echo "patch_sha256=$PATCH_SHA"
} > "$R2_SRC/r2-meta/stage11-lazy-direct-base.txt"

cd "$R2_SRC"
if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo >&2
  echo "ERROR: PR #29030 conflicts with exact production HotSeat tree." >&2
  echo "This is plausible because production modifies llama-model-loader.cpp." >&2
  echo "Candidate tree kept for a deliberate functional merge: $R2_SRC" >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check HEAD~1..HEAD

grep -Rni 'LLAMA_LAZY_MODE_DIRECT' common include src | head -n 100 | tee r2-meta/lazy-direct-probe.txt
if ! grep -Rq 'LLAMA_LAZY_MODE_DIRECT' common include src; then
  echo "ERROR: direct lazy mode missing after patch" >&2
  exit 11
fi
if ! grep -Rq 'llama_lazy_reader' src; then
  echo "ERROR: lazy reader implementation missing after patch" >&2
  exit 12
fi

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
BUILD="${BUILD:-$R2_SRC/build-r2-lazy-direct}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }

# Two full runtime copies, same real binary and libraries. Only their wrapper
# strips inherited lazy-mode flags and injects one explicit mode.
make_runtime() {
  local dst="$1" mode="$2"
  mkdir -p "$dst"
  cp -a "$BUILD/bin/." "$dst/"
  mv "$dst/llama-server" "$dst/llama-server.real"
  cat > "$dst/llama-server" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
MODE="$mode"
out=()
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    -lzm|--lazy-mode)
      shift
      [[ \$# -gt 0 ]] && shift
      ;;
    -lzm=*|--lazy-mode=*)
      shift
      ;;
    *)
      out+=("\$1")
      shift
      ;;
  esac
done
exec "\$SELF_DIR/llama-server.real" --lazy-mode "\$MODE" "\${out[@]}"
EOF
  chmod +x "$dst/llama-server"
}

MMAP_BIN="$RUNTIME_ROOT/mmap/bin"
DIRECT_BIN="$RUNTIME_ROOT/direct/bin"
make_runtime "$MMAP_BIN" on
make_runtime "$DIRECT_BIN" on-direct

sha256sum "$MMAP_BIN/llama-server.real" "$DIRECT_BIN/llama-server.real" | tee "$R2_SRC/r2-meta/stage11-real-binaries.sha256"
MMAP_SHA="$(sha256sum "$MMAP_BIN/llama-server.real" | awk '{print $1}')"
DIRECT_SHA="$(sha256sum "$DIRECT_BIN/llama-server.real" | awk '{print $1}')"
[[ "$MMAP_SHA" == "$DIRECT_SHA" ]] || { echo "ERROR: OFF/ON real binaries differ" >&2; exit 21; }

# Clone production block verbatim apart from runtime path. Keep JMAX and all HotSeat/MTP env.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$MMAP_ALIAS" --r2-bin "$MMAP_BIN" --jmax keep --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$DIRECT_ALIAS" --r2-bin "$DIRECT_BIN" --jmax keep --replace --validate

echo
echo "Stage-11 lazy direct-read A/B is ready."
echo "MMAP alias  : $MMAP_ALIAS"
echo "DIRECT alias: $DIRECT_ALIAS"
echo "same_real_binary_sha256=$MMAP_SHA"
echo "Production alias/runtime remain untouched."
echo
echo "Next: bash $SCRIPT_DIR/run_stage11_lazy_direct_ab.sh"
