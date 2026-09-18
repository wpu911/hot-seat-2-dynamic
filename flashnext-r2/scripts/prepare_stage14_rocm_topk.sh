#!/usr/bin/env bash
set -euo pipefail

# Stage 14: ROCm TOP_K optimization for Qwen3.8 Flash Next long-context QSA.
#
# Candidate: ggml-org/llama.cpp PR #28313 "ROCm: resolve TOP_K kernels".
# The PR changes exactly one file: ggml/src/ggml-cuda/top-k.cu.
#
# Default base is the corrected modern-MTP candidate. This stage is deliberately
# isolated from QSA gather/pooled-cache changes: first determine whether the HIP
# TOP_K kernel itself is a win on gfx1100 + gfx1201, then combine winners later.
#
# The upstream PR measurements explicitly disabled HIP graphs because of an
# upstream ROCm graph-update issue. Therefore this prepare step registers four
# llama-swap aliases so the end-to-end test can distinguish TOP_K gains from a
# HIP-graph interaction:
#   base graphs ON
#   candidate graphs ON
#   base graphs OFF
#   candidate graphs OFF

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-rocm-topk-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-rocm-topk}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
BASE_ALIAS="${BASE_ALIAS:-qwen3.8-flash-next-r2-modern-mtp:256k}"
R2_ALIAS="${R2_ALIAS:-qwen3.8-flash-next-r2-rocm-topk:256k}"
BASE_NOGRAPH_ALIAS="${BASE_NOGRAPH_ALIAS:-qwen3.8-flash-next-r2-topk-base-nograph:256k}"
R2_NOGRAPH_ALIAS="${R2_NOGRAPH_ALIAS:-qwen3.8-flash-next-r2-topk-nograph:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-/app/share/llama_box/src/flashnext-r2-vendor}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"

PR=28313
PR_HEAD="93ceb53397b8885c55533bd680f6ff430418317e"
PATCH="$PATCH_DIR/pr${PR}-rocm-topk-${PR_HEAD:0:8}.patch"
PATCH_URL="https://github.com/ggml-org/llama.cpp/pull/${PR}.patch"
EXPECTED_FILE="ggml/src/ggml-cuda/top-k.cu"

[[ -e "$BASE_SRC/.git" ]] || {
  echo "ERROR: base source missing: $BASE_SRC" >&2
  echo "Prepare the modern foundation/MTP stage first, or override BASE_SRC/BASE_ALIAS." >&2
  exit 2
}
# Untracked build/r2-meta files are harmless. Tracked source edits are not.
[[ -z "$(git -C "$BASE_SRC" status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: base source has tracked modifications; refusing ambiguous A/B." >&2
  git -C "$BASE_SRC" status --short --untracked-files=no >&2 || true
  exit 3
}
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 4; }
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: Stage-14 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
}

grep -qE "^[[:space:]]*${BASE_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: base llama-swap alias missing: $BASE_ALIAS" >&2
  exit 6
}

mkdir -p "$PATCH_DIR" "$LOG_DIR"
TMP_PATCH="$PATCH.tmp"
curl -fL --retry 3 --connect-timeout 20 "$PATCH_URL" -o "$TMP_PATCH"
LAST_COMMIT="$(grep -E '^From [0-9a-f]{40} ' "$TMP_PATCH" | tail -n1 | awk '{print $2}')"
if [[ "$LAST_COMMIT" != "$PR_HEAD" ]]; then
  echo "ERROR: PR #$PR moved; refusing to benchmark an unaudited patch." >&2
  echo "expected head=$PR_HEAD" >&2
  echo "downloaded head=$LAST_COMMIT" >&2
  rm -f "$TMP_PATCH"
  exit 7
fi
mv "$TMP_PATCH" "$PATCH"
PATCH_SHA="$(sha256sum "$PATCH" | awk '{print $1}')"

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
echo "=== Flash Next Stage 14 ROCm TOP_K ==="
echo "base source      : $BASE_SRC"
echo "base head        : $BASE_HEAD"
echo "base alias       : $BASE_ALIAS"
echo "PR/head          : #$PR / $PR_HEAD"
echo "patch sha256     : $PATCH_SHA"
echo "candidate source : $R2_SRC"
echo "candidate runtime: $RUNTIME"

git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
cd "$R2_SRC"

if ! git -c user.name='FlashNext R2 Experiment' \
         -c user.email='flashnext-r2@local.invalid' \
         am --3way "$PATCH"; then
  echo >&2
  echo "ERROR: PR #$PR conflicts with the selected modern base." >&2
  echo "Production is untouched. Candidate retained for semantic merge: $R2_SRC" >&2
  git status --short >&2 || true
  exit 10
fi

# The PR is intentionally one-file-only. If that ceases to be true, stop.
mapfile -t CHANGED < <(git diff --name-only "$BASE_HEAD"..HEAD | sort -u)
printf '%s\n' "${CHANGED[@]}" | tee r2-meta/stage14-changed-files.txt
if [[ ${#CHANGED[@]} -ne 1 || "${CHANGED[0]}" != "$EXPECTED_FILE" ]]; then
  echo "ERROR: Stage-14 delta is no longer the audited one-file TOP_K change." >&2
  exit 11
fi

git diff --check "$BASE_HEAD"..HEAD
if ! grep -qE 'top_k_nary_search_cuda|HIP_VERSION|top_k_one_' "$EXPECTED_FILE"; then
  echo "ERROR: expected PR #28313 ROCm TOP_K kernels not found after apply." >&2
  exit 12
fi

{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "base_alias=$BASE_ALIAS"
  echo "upstream_pr=ggml-org/llama.cpp#$PR"
  echo "pr_head=$PR_HEAD"
  echo "patch_sha256=$PATCH_SHA"
  echo "candidate_head=$(git rev-parse HEAD)"
  echo "changed_file=$EXPECTED_FILE"
} > r2-meta/stage14-rocm-topk-manifest.txt

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
BUILD="${BUILD:-$R2_SRC/build-r2-rocm-topk}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }

# Correctness first. Do not waive this failure: TOP_K changes QSA cell selection.
TOPK_TEST_LOG="$LOG_DIR/flashnext-stage14-topk-tests-$(date +%Y%m%d-%H%M%S).log"
if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  if "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 >"$TOPK_TEST_LOG" 2>&1; then
    :
  elif "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 >"$TOPK_TEST_LOG" 2>&1; then
    :
  elif "$BUILD/bin/test-backend-ops" test -o TOP_K >"$TOPK_TEST_LOG" 2>&1; then
    :
  else
    cat "$TOPK_TEST_LOG" >&2 || true
    echo "ERROR: TOP_K backend correctness tests failed." >&2
    exit 21
  fi
  cat "$TOPK_TEST_LOG"

  # Perf output is advisory; CLI syntax differs slightly across llama.cpp eras.
  TOPK_PERF_LOG="$LOG_DIR/flashnext-stage14-topk-perf-$(date +%Y%m%d-%H%M%S).log"
  "$BUILD/bin/test-backend-ops" perf -o TOP_K -b ROCm0 >"$TOPK_PERF_LOG" 2>&1 \
    || "$BUILD/bin/test-backend-ops" perf -o TOP_K -b HIP0 >"$TOPK_PERF_LOG" 2>&1 \
    || "$BUILD/bin/test-backend-ops" perf -o TOP_K >"$TOPK_PERF_LOG" 2>&1 \
    || true
  echo "TOPK_PERF_LOG=$TOPK_PERF_LOG"
fi

mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee r2-meta/stage14-llama-server.sha256
"$RUNTIME/llama-server" --version | tee r2-meta/stage14-version.txt || true

# Discover the base runtime from the live alias. Needed only to build a graph-OFF
# baseline alias with the exact same binary as BASE_ALIAS.
BASE_RUNTIME="${BASE_RUNTIME:-$(python3 - "$CONFIG" "$BASE_ALIAS" <<'PY'
from pathlib import Path
import re, sys
p, alias = sys.argv[1:]
lines = Path(p).read_text(encoding='utf-8').splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i,line in enumerate(lines):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); block=[line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind: break
        block.append(s)
    text='\n'.join(block)
    mm=re.search(r'(?m)^\s*(/\S*/llama-server)\s*$', text)
    if not mm:
        mm=re.search(r'(/app/share/llm/Qwen3\.8-Flash-Next-GGUF/runtime-text/[^\s\"\x27]+/bin)/llama-server', text)
        if mm: print(mm.group(1)); raise SystemExit(0)
    if mm:
        print(str(Path(mm.group(1)).parent)); raise SystemExit(0)
    raise SystemExit('ERROR: could not discover base llama-server runtime')
raise SystemExit('ERROR: base alias not found')
PY
)}"

test -x "$BASE_RUNTIME/llama-server" || {
  echo "ERROR: discovered BASE_RUNTIME has no llama-server: $BASE_RUNTIME" >&2
  exit 22
}

# Candidate with normal graph policy.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$R2_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --replace --validate

# Same binaries, but graphs disabled. This separates TOP_K speed from the known
# ROCm graph-update interaction mentioned in the upstream PR measurements.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$BASE_ALIAS" \
  --alias "$BASE_NOGRAPH_ALIAS" --r2-bin "$BASE_RUNTIME" --jmax keep \
  --env GGML_CUDA_DISABLE_GRAPHS=1 --replace --validate
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$R2_ALIAS" \
  --alias "$R2_NOGRAPH_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --env GGML_CUDA_DISABLE_GRAPHS=1 --replace --validate

echo
echo "Stage-14 ROCm TOP_K candidate ready."
echo "graphs ON : $BASE_ALIAS  vs  $R2_ALIAS"
echo "graphs OFF: $BASE_NOGRAPH_ALIAS  vs  $R2_NOGRAPH_ALIAS"
echo "runtime   : $RUNTIME/llama-server"
echo "production alias/runtime remain untouched."
echo "Next: bash $SCRIPT_DIR/run_stage14_rocm_topk_ab.sh"
