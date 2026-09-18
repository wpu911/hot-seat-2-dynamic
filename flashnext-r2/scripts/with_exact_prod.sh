#!/usr/bin/env bash
set -euo pipefail

# Run any Flash Next R2 prepare script against an exact, CLEAN git snapshot of
# the live production working tree, including intentional uncommitted HotSeat
# edits. This prevents `git worktree add HEAD` inside older prepare scripts from
# silently benchmarking a different engine.
#
# Usage:
#   bash flashnext-r2/scripts/with_exact_prod.sh \
#     flashnext-r2/scripts/prepare_stage8_qsa_pooled.sh
#
# Optional reuse:
#   EXACT_PROD_SRC=/path/to/prod-exact-... bash .../with_exact_prod.sh <prepare>
#
# All other environment overrides (R2_SRC, RUNTIME, ROCM_PATH, aliases, etc.)
# are passed through unchanged.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIVE_PROD_SRC="${LIVE_PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <prepare-script> [args...]" >&2
  exit 2
fi

TARGET="$1"
shift
if [[ ! -f "$TARGET" ]]; then
  # Allow a basename relative to this scripts directory.
  if [[ -f "$SCRIPT_DIR/$TARGET" ]]; then
    TARGET="$SCRIPT_DIR/$TARGET"
  else
    echo "ERROR: prepare script not found: $TARGET" >&2
    exit 3
  fi
fi

if [[ -n "${EXACT_PROD_SRC:-}" ]]; then
  SNAP="$EXACT_PROD_SRC"
  test -e "$SNAP/.git" || { echo "ERROR: EXACT_PROD_SRC is not a git tree: $SNAP" >&2; exit 4; }
  if [[ -n "$(git -C "$SNAP" status --porcelain)" ]]; then
    echo "ERROR: EXACT_PROD_SRC is dirty; expected a frozen clean snapshot: $SNAP" >&2
    git -C "$SNAP" status --short >&2 || true
    exit 5
  fi
  echo "Reusing exact production snapshot: $SNAP"
else
  SNAP_TOOL="$SCRIPT_DIR/create_exact_prod_snapshot_repo.sh"
  test -x "$SNAP_TOOL" || test -f "$SNAP_TOOL" || {
    echo "ERROR: exact snapshot tool missing: $SNAP_TOOL" >&2
    exit 6
  }

  TMP="$(mktemp)"
  trap 'rm -f "$TMP"' EXIT
  echo "Creating exact production snapshot from: $LIVE_PROD_SRC"
  PROD_SRC="$LIVE_PROD_SRC" bash "$SNAP_TOOL" | tee "$TMP"
  SNAP="$(grep '^SNAPSHOT=' "$TMP" | tail -n1 | cut -d= -f2-)"
  if [[ -z "$SNAP" || ! -e "$SNAP/.git" ]]; then
    echo "ERROR: snapshot tool did not return a valid SNAPSHOT path" >&2
    exit 7
  fi
  if [[ -n "$(git -C "$SNAP" status --porcelain)" ]]; then
    echo "ERROR: newly created exact snapshot is unexpectedly dirty" >&2
    exit 8
  fi
fi

echo
echo "=== exact production base ==="
echo "LIVE_PROD_SRC=$LIVE_PROD_SRC"
echo "EXACT_PROD_SRC=$SNAP"
echo "EXACT_HEAD=$(git -C "$SNAP" rev-parse HEAD)"
echo "TARGET=$TARGET"
echo

# The target script already knows how to create its isolated worktree/runtime.
# We only replace its notion of PROD_SRC with the frozen exact snapshot.
PROD_SRC="$SNAP" bash "$TARGET" "$@"
