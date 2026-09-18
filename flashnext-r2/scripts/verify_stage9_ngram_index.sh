#!/usr/bin/env bash
set -euo pipefail

# Stage 9: verify that production already contains upstream #28040
# (O(log n) n-gram history lookup via seq_pos_tok_le / per-sequence position index).
# Do not re-port the older closed #27992 implementation if #28040 is present.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/flashnext-stage9-ngram-index-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "=== Flash Next Stage 9 n-gram index verification ==="
echo "date=$(date -Is)"
echo "source=$PROD_SRC"

test -e "$PROD_SRC/.git" || { echo "ERROR: not a git worktree: $PROD_SRC"; exit 2; }
printf 'HEAD='; git -C "$PROD_SRC" rev-parse HEAD

CELLS="$PROD_SRC/src/llama-kv-cells.h"
CACHE="$PROD_SRC/src/llama-kv-cache.cpp"

if [[ ! -f "$CELLS" || ! -f "$CACHE" ]]; then
  echo "ERROR: expected KV source files missing" >&2
  exit 3
fi

echo
echo "=== seq_pos index ==="
grep -nE 'seq_pos_tok_le|seq_pos_add|seq_pos_rm|std::set|seq_pos' "$CELLS" | head -n 120 || true

echo
echo "=== get_prev_tokens implementation ==="
grep -nA80 -B10 'void llama_kv_cache::get_prev_tokens' "$CACHE" | head -n 120 || true

PASS=1
if ! grep -q 'seq_pos_tok_le' "$CELLS"; then
  echo "MISSING: llama_kv_cells::seq_pos_tok_le" >&2
  PASS=0
fi
if ! grep -q 'seq_pos_tok_le' "$CACHE"; then
  echo "MISSING: get_prev_tokens does not call seq_pos_tok_le" >&2
  PASS=0
fi

# The old implementation built/scanned a temporary history map from all used cells.
# We do not key on exact old variable names alone; flag obvious full-cell scans inside
# the get_prev_tokens body for manual inspection.
BODY="$(sed -n '/void llama_kv_cache::get_prev_tokens/,/^}/p' "$CACHE")"
if grep -qE 'for_each_token_in|v_cells\[[^]]+\]\.used|for \([^;]*used' <<<"$BODY"; then
  echo "WARNING: get_prev_tokens still appears to scan used KV cells" >&2
  PASS=0
fi

echo
if [[ "$PASS" == 1 ]]; then
  echo "NGRAM_POSITION_INDEX=PRESENT"
  echo "ACTION=SKIP_OLD_27992_PATCH"
  echo "The production source already has the refactored logarithmic lookup path."
else
  echo "NGRAM_POSITION_INDEX=NOT_CONFIRMED"
  echo "ACTION=INSPECT_BEFORE_PORTING"
  exit 10
fi

echo "LOG=$LOG"
