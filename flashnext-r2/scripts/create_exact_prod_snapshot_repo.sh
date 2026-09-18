#!/usr/bin/env bash
set -euo pipefail

# Freeze the LIVE production llama.cpp working tree, including intentional
# uncommitted HotSeat edits, into a private local git snapshot repository.
#
# Why this exists:
#   git worktree add <dst> HEAD reproduces commits, not uncommitted edits.
#   The production tree is known to carry intentional local HotSeat changes.
#   Benchmarking a clean HEAD would therefore benchmark the wrong engine.
#
# The snapshot repo is never production. It exists only as an immutable base for
# R2 experiment worktrees.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
SNAP_ROOT="${SNAP_ROOT:-/app/share/llama_box/src/llama.cpp-prod-exact-snapshots}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SNAP="${SNAP:-$SNAP_ROOT/prod-exact-$STAMP}"

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: not a git worktree: $PROD_SRC" >&2
  exit 2
fi
if [[ -e "$SNAP" ]]; then
  echo "ERROR: snapshot destination exists: $SNAP" >&2
  exit 3
fi
if git -C "$PROD_SRC" ls-files -u | grep -q .; then
  echo "ERROR: production tree has unresolved merges" >&2
  git -C "$PROD_SRC" status --short >&2 || true
  exit 4
fi

mkdir -p "$SNAP_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PATCH="$TMP/working.patch"
UNTRACKED="$TMP/untracked.txt"

HEAD_SHA="$(git -C "$PROD_SRC" rev-parse HEAD)"
git -C "$PROD_SRC" diff --binary HEAD > "$PATCH"
git -C "$PROD_SRC" diff --check HEAD
# Restrict untracked copy to source/build-system trees. This catches custom .cu/.cpp
# additions without dragging build products or 190 GB model files into git.
git -C "$PROD_SRC" ls-files --others --exclude-standard -- \
  CMakeLists.txt cmake common examples ggml include src tests tools \
  > "$UNTRACKED"

echo "Production source : $PROD_SRC"
echo "Production HEAD   : $HEAD_SHA"
echo "Tracked diff      : $(wc -c < "$PATCH") bytes"
echo "Untracked source  : $(wc -l < "$UNTRACKED") files"
echo "Snapshot repo     : $SNAP"

# Clone commits only, then replay the exact live working-tree state.
git clone --quiet --no-checkout "$PROD_SRC" "$SNAP"
git -C "$SNAP" checkout --quiet --detach "$HEAD_SHA"

if [[ -s "$PATCH" ]]; then
  git -C "$SNAP" apply --binary --whitespace=nowarn "$PATCH"
fi
while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  mkdir -p "$SNAP/$(dirname "$rel")"
  cp -a "$PROD_SRC/$rel" "$SNAP/$rel"
done < "$UNTRACKED"

mkdir -p "$SNAP/r2-snapshot-meta"
cp "$PATCH" "$SNAP/r2-snapshot-meta/production-working.patch"
cp "$UNTRACKED" "$SNAP/r2-snapshot-meta/untracked-source.txt"
git -C "$PROD_SRC" status --short > "$SNAP/r2-snapshot-meta/production-status.txt" || true
{
  echo "created=$(date -Is)"
  echo "production_source=$PROD_SRC"
  echo "production_head=$HEAD_SHA"
  echo "production_patch_sha256=$(sha256sum "$PATCH" | awk '{print $1}')"
  echo "untracked_source_count=$(wc -l < "$UNTRACKED")"
} > "$SNAP/r2-snapshot-meta/manifest.txt"

# Commit the reproduced state so every downstream `worktree add ... HEAD` includes
# the local HotSeat modifications. The commit is local to this snapshot repo only.
git -C "$SNAP" add -A
git -C "$SNAP" \
  -c user.name='FlashNext R2 Snapshot' \
  -c user.email='local-snapshot@invalid' \
  commit --quiet --allow-empty -m "local snapshot: exact production working tree $STAMP"

SNAP_HEAD="$(git -C "$SNAP" rev-parse HEAD)"

# Verify every path modified relative to the original HEAD byte-for-byte against
# the live production tree. This includes newly added source files.
mapfile -t CHANGED < <(git -C "$SNAP" diff-tree --no-commit-id --name-only -r "$SNAP_HEAD")
# The snapshot metadata is intentionally new and does not exist in production.
for rel in "${CHANGED[@]}"; do
  [[ "$rel" == r2-snapshot-meta/* ]] && continue
  if [[ -e "$PROD_SRC/$rel" && -e "$SNAP/$rel" ]]; then
    if ! cmp -s "$PROD_SRC/$rel" "$SNAP/$rel"; then
      echo "ERROR: snapshot differs from production: $rel" >&2
      exit 10
    fi
  elif [[ ! -e "$PROD_SRC/$rel" && ! -e "$SNAP/$rel" ]]; then
    : # deleted in both logical states
  else
    echo "ERROR: snapshot path existence differs: $rel" >&2
    exit 11
  fi
done

cat > "$SNAP/USE_AS_PROD_SRC.txt" <<EOF
SNAPSHOT=$SNAP
SNAPSHOT_HEAD=$SNAP_HEAD
ORIGINAL_PRODUCTION=$PROD_SRC
ORIGINAL_HEAD=$HEAD_SHA

Use this snapshot as PROD_SRC for Flash Next R2 prepare scripts.
It contains the original committed HEAD plus the live tracked/untracked source edits.
EOF

echo
echo "SNAPSHOT=$SNAP"
echo "SNAPSHOT_HEAD=$SNAP_HEAD"
echo "Snapshot is clean: $(git -C "$SNAP" status --porcelain | wc -l) dirty entries"
