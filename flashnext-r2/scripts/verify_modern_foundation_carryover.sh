#!/usr/bin/env bash
set -euo pipefail

# Verify that optimizations from the old, closed #27977 line which are already
# represented in the Sep-18 qwen4exp foundation are not accidentally re-ported.
#
# #27977 bundled several unrelated ideas. In the modern foundation:
#   - predecessor lookup is superseded by per-sequence position index
#     (seq_pos_tok_le), so no O(cache * 256 seqs) scan is needed;
#   - qwen4exp block pooling already sums small-r slices directly;
#   - qwen4exp indexer-head reduction already sums strided head slices directly.
#
# QSA gather itself is NOT considered carried over here. It remains Stage 7.

SRC="${SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"

[[ -e "$SRC/.git" ]] || { echo "ERROR: source missing: $SRC" >&2; exit 2; }
[[ -f "$SRC/r2-meta/modern-foundation-manifest.txt" ]] || {
  echo "ERROR: not a recorded modern foundation: $SRC" >&2
  exit 3
}

fail=0
probe() {
  local name="$1"; shift
  if "$@"; then
    echo "PASS  $name"
  else
    echo "FAIL  $name" >&2
    fail=1
  fi
}

probe "per-sequence position lookup exists" \
  grep -q 'llama_token seq_pos_tok_le' "$SRC/src/llama-kv-cells.h"

probe "get_prev_tokens uses seq_pos_tok_le" \
  grep -q 'seq_pos_tok_le(seq_id, p)' "$SRC/src/llama-kv-cache.cpp"

# The obsolete implementation walked all used cells and all LLAMA_MAX_SEQ ids.
# Do not fail merely because helper names still exist elsewhere; require the new
# direct lookup at the actual predecessor call site above.

probe "qwen4exp pooled block reduction uses slice adds" \
  bash -lc "grep -q 'r is small, so summing slices beats a transpose plus sum_rows' '$SRC/src/models/qwen4exp.cpp'"

probe "qwen4exp indexer-head reduction uses slice adds" \
  bash -lc "grep -q 'heads sit side by side on ne\\[1\\]' '$SRC/src/models/qwen4exp.cpp' && grep -q 'summed = summed ? ggml_add' '$SRC/src/models/qwen4exp.cpp'"

# QSA gather is intentionally absent from this carry-over gate. If upstream later
# merges it, Stage 7 should be retired rather than stacked a second time.
if grep -Rq 'QWEN4EXP_QSA_GATHER' "$SRC/src" 2>/dev/null; then
  echo "NOTICE QSA gather kill-switch already exists in foundation; retire Stage 7 patch before testing."
else
  echo "INFO   QSA gather is not present in foundation; Stage 7 remains a valid independent experiment."
fi

if [[ "$fail" != 0 ]]; then
  echo >&2
  echo "CARRYOVER_AUDIT=FAIL"
  echo "Do not apply old #27977 wholesale. Inspect the missing invariant first." >&2
  exit 10
fi

echo
echo "CARRYOVER_AUDIT=PASS"
echo "Do not port the old #27977 predecessor/head-reduction bundle again."
echo "Keep only still-independent experiments such as QSA gather."
