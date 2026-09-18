#!/usr/bin/env bash
set -euo pipefail

# Verify that optimizations from older qwen4exp lines which are already
# represented in the Sep-18 foundation are not accidentally re-ported.
#
# Already expected in the modern foundation:
#   - predecessor lookup uses per-sequence position index (seq_pos_tok_le);
#   - qwen4exp block pooling sums small-r slices directly;
#   - qwen4exp indexer-head reduction sums strided head slices directly;
#   - indexer KV cache does not allocate a useless V cache (#28330 equivalent).
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
# Require the new direct lookup at the actual predecessor call site above.

probe "qwen4exp pooled block reduction uses slice adds" \
  bash -lc "grep -q 'r is small, so summing slices beats a transpose plus sum_rows' '$SRC/src/models/qwen4exp.cpp'"

probe "qwen4exp indexer-head reduction uses slice adds" \
  bash -lc "grep -q 'heads sit side by side on ne\\[1\\]' '$SRC/src/models/qwen4exp.cpp' && grep -q 'summed = summed ? ggml_add' '$SRC/src/models/qwen4exp.cpp'"

# #28330's intent is already present in the Sep18 line: shape the indexer cache
# like MLA so llama_kv_cache skips V allocation. Test both assignments because
# a half-applied version would still allocate the unwanted side.
probe "indexer cache suppresses unused V allocation (#28330 equivalent)" \
  bash -lc "grep -q 'hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size' '$SRC/src/llama-memory-hybrid-idx.cpp' && grep -q 'hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size' '$SRC/src/llama-memory-hybrid-idx.cpp'"

# QSA gather is intentionally absent from this carry-over gate. If upstream later
# merges it, Stage 7 should be retired rather than stacked a second time.
if grep -Rq 'QWEN4EXP_QSA_GATHER' "$SRC/src" 2>/dev/null; then
  echo "NOTICE QSA gather kill-switch already exists in foundation; retire Stage 7 patch before testing."
else
  echo "INFO   QSA gather is not present in foundation; Stage 7 remains a valid independent experiment."
fi

# #28569 re-enables -sm tensor. It is deliberately informational here, not a
# carry-over requirement: heterogeneous gfx1100+gfx1201 tensor splitting can add
# inter-device traffic and must be benchmarked separately if we ever enable it.
if grep -A8 -B8 'bool llm_arch_supports_sm_tensor' "$SRC/src/llama-arch.cpp" | grep -q 'LLM_ARCH_QWEN4EXP'; then
  echo "INFO   qwen4exp -sm tensor is disabled in this foundation; #28569 remains an optional experiment, not a missing fix."
else
  echo "INFO   qwen4exp -sm tensor appears enabled; verify scheduler placement before any tensor-split A/B."
fi

if [[ "$fail" != 0 ]]; then
  echo >&2
  echo "CARRYOVER_AUDIT=FAIL"
  echo "Do not apply old optimization bundles wholesale. Inspect the missing invariant first." >&2
  exit 10
fi

echo
echo "CARRYOVER_AUDIT=PASS"
echo "Do not port the old #27977 predecessor/head-reduction bundle or #28330 again."
echo "Keep only still-independent experiments such as QSA gather."
