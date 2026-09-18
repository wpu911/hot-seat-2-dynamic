#!/usr/bin/env bash
set -euo pipefail

# Canonical entry point for any R2 stage that compiles llama.cpp.
# It first freezes the LIVE production working tree, including uncommitted HotSeat
# edits, into a local clean commit and then passes that exact snapshot as PROD_SRC.
#
# Usage:
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 1
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 2
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 3
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 4
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 7
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 8
#   bash flashnext-r2/scripts/prepare_from_exact_snapshot.sh 9
#
# Reuse a previously frozen snapshot:
#   EXACT_PROD_SRC=/app/share/.../prod-exact-... bash ... 7

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
LIVE_PROD_SRC="${LIVE_PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"

case "$STAGE" in
  1) PREP="$SCRIPT_DIR/prepare_stage1.sh" ;;
  2) PREP="$SCRIPT_DIR/prepare_stage2_hc.sh" ;;
  3) PREP="$SCRIPT_DIR/prepare_stage3_gdn.sh" ;;
  4) PREP="$SCRIPT_DIR/prepare_stage4_q8dedup.sh" ;;
  7) PREP="$SCRIPT_DIR/prepare_stage7_qsa_gather.sh" ;;
  8) PREP="$SCRIPT_DIR/prepare_stage8_qsa_pooled_cache.sh" ;;
  9) PREP="$SCRIPT_DIR/prepare_stage9_lazy_direct.sh" ;;
  *)
    echo "Usage: $0 {1|2|3|4|7|8|9}" >&2
    echo "Stages 5/6 are alias/runtime sweeps and do not create a new source worktree." >&2
    exit 2
    ;;
esac

test -f "$PREP" || { echo "ERROR: stage prepare script missing: $PREP" >&2; exit 3; }

if [[ -n "${EXACT_PROD_SRC:-}" ]]; then
  SNAP="$EXACT_PROD_SRC"
  test -e "$SNAP/.git" || { echo "ERROR: EXACT_PROD_SRC is not a git repo/worktree: $SNAP" >&2; exit 4; }
  if [[ -n "$(git -C "$SNAP" status --porcelain)" ]]; then
    echo "ERROR: EXACT_PROD_SRC is dirty; an exact snapshot must be clean/committed" >&2
    git -C "$SNAP" status --short >&2
    exit 5
  fi
else
  OUT="$(PROD_SRC="$LIVE_PROD_SRC" bash "$SCRIPT_DIR/create_exact_prod_snapshot_repo.sh")"
  printf '%s\n' "$OUT"
  SNAP="$(printf '%s\n' "$OUT" | sed -n 's/^SNAPSHOT=//p' | tail -n 1)"
  test -n "$SNAP" || { echo "ERROR: snapshot helper did not return SNAPSHOT=" >&2; exit 6; }
fi

echo
echo "=== exact production snapshot selected ==="
echo "stage=$STAGE"
echo "snapshot=$SNAP"
echo "snapshot_head=$(git -C "$SNAP" rev-parse HEAD)"
echo "prepare=$PREP"
echo

# The stage script now sees a clean HEAD whose commit already contains the live
# HotSeat working-tree modifications, so its own detached worktree logic is exact.
PROD_SRC="$SNAP" bash "$PREP"
