#!/usr/bin/env bash
set -euo pipefail

# Verify which early R2/JohnTDI optimizations are already represented in the
# corrected Sep-18 Modern Foundation and therefore must NOT be stacked again.
#
# This is intentionally source-based. A benchmark cannot tell you that two
# implementations are the same idea; it merely gives humans another number to
# argue about.

SRC="${SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"
[[ -e "$SRC/.git" ]] || { echo "ERROR source missing: $SRC" >&2; exit 2; }

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

echo "=== Modern Flash Next kernel carry-over audit ==="
echo "source=$SRC"
echo "head=$(git -C "$SRC" rev-parse HEAD)"

# Old Stage-1 / GGML_JOHNV8_MMQ_ID_JMAX manually capped the tile-width search
# against expected rows per expert. Modern upstream has a first-class ncols_opt
# field: for RDNA3/RDNA4, mul_mat_id uses average per-expert occupancy while the
# launch grid still keeps ncols_max. That is the same core optimization with a
# cleaner automatic heuristic, so the old env/JMAX patch must stay retired.
probe "mmq_args carries ncols_opt" \
  grep -q 'int64_t ncols_opt' "$SRC/ggml/src/ggml-cuda/mmq.cuh"
probe "RDNA3/RDNA4 MoE uses average per-expert ncols_opt" \
  bash -lc "grep -q 'GGML_CUDA_CC_IS_RDNA3(cc).*GGML_CUDA_CC_IS_RDNA4(cc)' '$SRC/ggml/src/ggml-cuda/mmq.cu' && grep -q 'ncols_opt = (ne12.n_expert_used + ne02 - 1) / ne02' '$SRC/ggml/src/ggml-cuda/mmq.cu'"
probe "tile search uses ncols_opt" \
  grep -q 'args.ncols_opt + config.J - 1' "$SRC/ggml/src/ggml-cuda/mmq.cuh"

# Old Stage-4 Q8 activation dedup avoided quantizing the broadcast activation once
# per expert. Modern upstream performs the equivalent directly in mul_mat_id:
# build an inverse token map and quantize/scatter each broadcast token once.
probe "MoE broadcast activation dedup exists" \
  grep -q 'const bool dedup_bcast = ne11 == 1 && n_expert_used > 1' "$SRC/ggml/src/ggml-cuda/mmq.cu"
probe "Q8_1 quantize-scatter dedup path exists" \
  grep -q 'quantize_scatter_mmq_q8_1_cuda' "$SRC/ggml/src/ggml-cuda/mmq.cu"

# Modern qwen4exp already carries upstream HyperConnection fused ops. The old
# Stage-12 upstream-HC experiment is therefore obsolete. JohnTDI HC remains only
# an optional gfx1201-specific comparison if it implements something beyond this.
probe "modern HyperConnection fused op exists" \
  bash -lc "grep -Rq 'fused_dsv4_hc_pre\|ggml_dsv4_hc_pre_gated' '$SRC/src' '$SRC/ggml'"

# GDN is subtler. Modern upstream already has fused autoregressive/chunked GDN,
# but qwen4exp still performs q/k L2 normalization and some prolog math outside
# that fused operator. Therefore the old blanket 'GDN fusion' stage is retired;
# only a narrowly scoped extra prolog/L2 fusion may remain worth an RDNA4 A/B.
probe "fused GDN controls exist" \
  bash -lc "grep -Rq 'fused_gdn_ar' '$SRC/src' && grep -Rq 'fused_gdn_ch' '$SRC/src'"
probe "qwen4exp still has external GDN L2 normalization" \
  grep -q 'build_gdn_l2_norm(ctx0, q_conv' "$SRC/src/models/qwen4exp.cpp"

if [[ "$fail" != 0 ]]; then
  echo
  echo "MODERN_KERNEL_AUDIT=FAIL"
  echo "Do not assume the modern foundation supersedes the old stages until the missing invariant is inspected." >&2
  exit 10
fi

echo
echo "MODERN_KERNEL_AUDIT=PASS"
echo "RETIRE Stage-1 manual JMAX patch/sweep: modern ncols_opt already handles RDNA3/RDNA4 MoE tile width."
echo "RETIRE Stage-4 Q8 activation dedup: modern quantize_scatter_mmq_q8_1_cuda already deduplicates broadcast activations."
echo "RETIRE old upstream-HC stage: Modern Foundation already contains HC fused ops."
echo "KEEP ONLY AS OPTIONAL RDNA4 EXPERIMENT: extra GDN prolog/L2 fusion and any JohnTDI HC kernel that is demonstrably beyond upstream."
