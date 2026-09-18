#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 9: direct row reads for qwen4exp lazy PLE table.
# Based on ggml-org/llama.cpp PR #29030, pinned to current head.
# Compare --lazy-mode on vs --lazy-mode on-direct using the SAME patched binary.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-lazy-direct-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-lazy-direct}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next:256k}"
OFF_ALIAS="${OFF_ALIAS:-qwen3.8-flash-next-r2-lazy-mmap:256k}"
ON_ALIAS="${ON_ALIAS:-qwen3.8-flash-next-r2-lazy-direct:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
PATCH="$PATCH_DIR/pr29030-lazy-direct-5fd6ce3d.patch"
PATCH_URL="https://github.com/pwilkin/llama.cpp/commit/5fd6ce3d053f9477681be81a1be08cb3f5958523.patch"
PATCH_HEAD="5fd6ce3d053f9477681be81a1be08cb3f5958523"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: PROD_SRC is not a git worktree/repo: $PROD_SRC" >&2
  exit 2
fi
# New stages require a clean exact snapshot, not a dirty live production tree.
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "ERROR: Stage 9 requires a clean committed exact-production snapshot." >&2
  echo "Use: bash $SCRIPT_DIR/prepare_from_exact_snapshot.sh 9" >&2
  exit 3
fi
if [[ -e "$R2_SRC" || -e "$RUNTIME" ]]; then
  echo "ERROR: Stage-9 worktree/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 4
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: llama-swap config missing: $CONFIG" >&2
  exit 5
fi

# Lazy mode is designed around lazy/mmap-backed tensors. If the production alias
# explicitly uses --no-mmap, do not silently change that load policy in this A/B.
python3 - "$CONFIG" "$SOURCE_ALIAS" <<'PY'
from pathlib import Path
import re, sys
p, alias = Path(sys.argv[1]), sys.argv[2]
lines = p.read_text().splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m: continue
    ind = len(m.group(1)); out=[]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    blob='\n'.join(out)
    if re.search(r'(^|\s)--no-mmap(?:\s|$)', blob):
        raise SystemExit('ERROR: production alias contains --no-mmap; Stage 9 will not change load policy implicitly')
    sys.exit(0)
raise SystemExit('ERROR: source alias not found: ' + alias)
PY

mkdir -p "$PATCH_DIR"
if [[ ! -f "$PATCH" ]]; then
  curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$PATCH"
fi
if ! head -n 1 "$PATCH" | grep -qi "${PATCH_HEAD:0:12}"; then
  echo "ERROR: downloaded PR29030 patch does not match pinned head $PATCH_HEAD" >&2
  exit 6
fi

PROD_HEAD="$(git -C "$PROD_SRC" rev-parse HEAD)"
echo "Exact production HEAD : $PROD_HEAD"
echo "Stage-9 worktree      : $R2_SRC"
echo "Stage-9 runtime       : $RUNTIME"
echo "PR #29030 head        : $PATCH_HEAD"

git -C "$PROD_SRC" worktree add --detach "$R2_SRC" "$PROD_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "exact_production_source=$PROD_SRC"
  echo "exact_production_head=$PROD_HEAD"
  echo "upstream_pr=ggml-org/llama.cpp#29030"
  echo "patch_head=$PATCH_HEAD"
  echo "patch_url=$PATCH_URL"
  echo "patch_sha256=$(sha256sum "$PATCH" | awk '{print $1}')"
} > "$R2_SRC/r2-meta/stage9-lazy-direct-base.txt"

cd "$R2_SRC"
if ! git apply --3way "$PATCH"; then
  echo "ERROR: PR #29030 did not apply cleanly to the exact production snapshot." >&2
  echo "Worktree kept for functional migration: $R2_SRC" >&2
  exit 10
fi
git diff --check
git diff > r2-meta/stage9-lazy-direct.diff

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

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing"; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage9-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage9-version.txt" || true

# Clone the exact current production command twice. JMAX is held at 0 to avoid a
# PP-only confound. Then modify lazy mode inside only the test aliases.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$OFF_ALIAS" --r2-bin "$RUNTIME" --jmax 0 --replace
python3 "$SCRIPT_DIR/set_alias_lazy_mode.py" \
  --config "$CONFIG" --alias "$OFF_ALIAS" --mode on

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$ON_ALIAS" --r2-bin "$RUNTIME" --jmax 0 --replace
python3 "$SCRIPT_DIR/set_alias_lazy_mode.py" \
  --config "$CONFIG" --alias "$ON_ALIAS" --mode on-direct

/app/share/llama_box/bin/llama-swap -config "$CONFIG" -validate

echo
echo "Stage-9 lazy direct aliases ready."
echo "mmap lazy : $OFF_ALIAS"
echo "direct    : $ON_ALIAS"
echo "same binary: $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage9_lazy_direct_ab.sh"
