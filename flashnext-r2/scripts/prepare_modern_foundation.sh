#!/usr/bin/env bash
set -euo pipefail

# Build a modern Flash Next foundation correctly:
#
#   official Sep-11 base (b0dcb819...)
#       + exact current production custom overlay
#       -> extract overlay as one binary diff
#
#   official Sep-18 PR28243 base (911f6cdc...)
#       + the exact custom overlay above
#       -> modern foundation
#
# This is preferable to cherry-picking a handful of recent qwen4exp PRs because
# 130 upstream commits separate the two official bases. We keep ALL upstream
# changes and forward-port only our own production delta. Any semantic conflict
# stops the script; production is never modified.

OLD_UPSTREAM="${OLD_UPSTREAM:-b0dcb8192b201e402ec3eff524e55450f8070e3e}"
NEW_UPSTREAM="${NEW_UPSTREAM:-911f6cdc8ab8a530b2bee09ee61471a6f3178eeb}"
EXACT_SRC="${EXACT_SRC:-/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current}"
FOUNDATION="${FOUNDATION:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-foundation}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next:256k}"
FOUNDATION_ALIAS="${FOUNDATION_ALIAS:-qwen3.8-flash-next-r2-modern-foundation:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[[ -e "$EXACT_SRC/.git" ]] || {
  echo "ERROR: exact production snapshot not found: $EXACT_SRC" >&2
  echo "Create it with create_exact_prod_snapshot_repo.sh, then point EXACT_SRC at it." >&2
  exit 2
}
[[ -z "$(git -C "$EXACT_SRC" status --porcelain)" ]] || {
  echo "ERROR: exact production snapshot must be clean/committed." >&2
  git -C "$EXACT_SRC" status --short >&2 || true
  exit 3
}
[[ ! -e "$FOUNDATION" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: modern foundation source/runtime already exists; refusing overwrite" >&2
  echo "source=$FOUNDATION runtime=$RUNTIME" >&2
  exit 4
}
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 5; }

EXACT_HEAD="$(git -C "$EXACT_SRC" rev-parse HEAD)"

echo "=== Flash Next modern foundation ==="
echo "old official base : $OLD_UPSTREAM"
echo "exact prod head   : $EXACT_HEAD"
echo "new official base : $NEW_UPSTREAM"
echo "foundation        : $FOUNDATION"
echo "runtime           : $RUNTIME"

if ! git -C "$EXACT_SRC" cat-file -e "$OLD_UPSTREAM^{commit}" 2>/dev/null; then
  git -C "$EXACT_SRC" fetch --no-tags https://github.com/ggml-org/llama.cpp.git "$OLD_UPSTREAM"
fi
if ! git -C "$EXACT_SRC" cat-file -e "$NEW_UPSTREAM^{commit}" 2>/dev/null; then
  git -C "$EXACT_SRC" fetch --no-tags https://github.com/ggml-org/llama.cpp.git "$NEW_UPSTREAM"
fi

if ! git -C "$EXACT_SRC" merge-base --is-ancestor "$OLD_UPSTREAM" "$EXACT_HEAD"; then
  echo "ERROR: recorded Sep-11 base is not an ancestor of exact production snapshot." >&2
  echo "Refusing to manufacture a misleading custom overlay." >&2
  exit 6
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OVERLAY="$TMP/exact-production-overlay.patch"
FILES="$TMP/overlay-files.txt"

# Snapshot metadata is intentionally committed so the snapshot itself is clean, but
# it is NOT production source and must never be replayed onto upstream. The previous
# version diffed the entire snapshot commit and accidentally treated the audit patch
# and usage note as migration inputs.
DIFF_PATHS=(
  .
  ':(exclude)r2-snapshot-meta/**'
  ':(exclude)USE_AS_PROD_SRC.txt'
)

git -C "$EXACT_SRC" diff --binary "$OLD_UPSTREAM" "$EXACT_HEAD" -- "${DIFF_PATHS[@]}" > "$OVERLAY"
git -C "$EXACT_SRC" diff --check "$OLD_UPSTREAM" "$EXACT_HEAD" -- "${DIFF_PATHS[@]}"
git -C "$EXACT_SRC" diff --name-status "$OLD_UPSTREAM" "$EXACT_HEAD" -- "${DIFF_PATHS[@]}" > "$FILES"

if grep -Eq '(^|[[:space:]])(r2-snapshot-meta/|USE_AS_PROD_SRC\.txt$)' "$FILES"; then
  echo "ERROR: snapshot-only metadata leaked into production overlay file list" >&2
  cat "$FILES" >&2
  exit 7
fi

OVERLAY_SHA="$(sha256sum "$OVERLAY" | awk '{print $1}')"
OVERLAY_BYTES="$(wc -c < "$OVERLAY")"
OVERLAY_FILES="$(wc -l < "$FILES")"

echo "custom overlay    : $OVERLAY_FILES files / $OVERLAY_BYTES bytes"
echo "overlay sha256    : $OVERLAY_SHA"

# Create a standalone clone because we want a clean experimental repository with
# its own refs/provenance.
git clone --quiet "$EXACT_SRC" "$FOUNDATION"
git -C "$FOUNDATION" checkout --quiet --detach "$NEW_UPSTREAM"
mkdir -p "$FOUNDATION/r2-meta"
cp "$OVERLAY" "$FOUNDATION/r2-meta/exact-production-overlay.patch"
cp "$FILES" "$FOUNDATION/r2-meta/exact-production-overlay-files.txt"

{
  echo "created=$(date -Is)"
  echo "old_upstream=$OLD_UPSTREAM"
  echo "new_upstream=$NEW_UPSTREAM"
  echo "exact_production_source=$EXACT_SRC"
  echo "exact_production_head=$EXACT_HEAD"
  echo "overlay_sha256=$OVERLAY_SHA"
  echo "overlay_bytes=$OVERLAY_BYTES"
  echo "overlay_file_count=$OVERLAY_FILES"
  echo "snapshot_metadata_excluded=1"
} > "$FOUNDATION/r2-meta/modern-foundation-manifest.txt"

cd "$FOUNDATION"

if ! git apply --3way --whitespace=nowarn r2-meta/exact-production-overlay.patch; then
  echo >&2
  echo "ERROR: exact production overlay conflicts with Sep-18 upstream." >&2
  echo "This is the expected place for a functional merge, not a reason to use --theirs." >&2
  echo "Foundation tree is retained at: $FOUNDATION" >&2
  git status --short >&2 || true
  exit 10
fi

git diff --check

git add -A
git -c user.name='FlashNext R2 Foundation' \
    -c user.email='flashnext-r2@local.invalid' \
    commit --quiet -m "r2 foundation: forward-port exact production overlay onto 911f6cdc"
FOUNDATION_HEAD="$(git rev-parse HEAD)"

echo "foundation head   : $FOUNDATION_HEAD"
echo "foundation dirty  : $(git status --porcelain | wc -l)"

for needle in \
  'Q122 V3 PREFILL_EPOCH_RESET' \
  'Q122_SPEC_DISABLE_AFTER_CACHED_LARGE_PP' \
  'Q122_DYNKV' \
  'HOTSEAT'; do
  if grep -Rqs "$needle" ggml src tools/server 2>/dev/null; then
    echo "PROBE_PRESENT $needle"
  else
    echo "PROBE_WARNING missing textual marker: $needle"
  fi
done

if ! grep -Rq 'fused_dsv4_hc_pre\|ggml_dsv4_hc_pre_gated' src ggml 2>/dev/null; then
  echo "ERROR: modern qwen4exp/HC foundation marker not found after forward-port." >&2
  exit 11
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
BUILD="${BUILD:-$FOUNDATION/build-r2-modern-foundation}"

cmake -S "$FOUNDATION" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server test-backend-ops

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$FOUNDATION/r2-meta/modern-foundation-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$FOUNDATION/r2-meta/modern-foundation-version.txt" || true

if [[ -x "$BUILD/bin/test-backend-ops" ]]; then
  "$BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K -b HIP0 \
    || "$BUILD/bin/test-backend-ops" test -o TOP_K
fi

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$FOUNDATION_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --replace --validate

echo
echo "Modern foundation ready."
echo "alias     : $FOUNDATION_ALIAS"
echo "source    : $FOUNDATION"
echo "head      : $FOUNDATION_HEAD"
echo "runtime   : $RUNTIME/llama-server"
echo "production: untouched"
echo "Next      : bash $SCRIPT_DIR/run_modern_foundation_ab.sh"
