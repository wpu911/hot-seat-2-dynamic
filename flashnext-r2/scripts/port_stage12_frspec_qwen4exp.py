#!/usr/bin/env python3
"""Semantically port reduced-vocab d2t support into modern qwen4exp MTP.

This targets the graph_mtp layout used by the newer Qwen3.8-Flash-Next MTP work
(PR #28243), not the older drluoto fork line numbers. It makes three narrow edits:
  1. load d2t and size draft-only LM-head tensors to K rows;
  2. scatter K draft logits back into the full target vocabulary in graph_mtp;
  3. preserve graph_mtp continuation inputs as graph outputs (CIRU H121 fix).

Every anchor must match exactly once. If the source has drifted, the script stops
rather than performing interpretive surgery on a 200 GB model runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

MARK = "[TAG_FLASHNEXT_R2_FRSPEC]"


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {n}")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="path to src/models/qwen4exp.cpp")
    args = ap.parse_args()
    p = Path(args.source)
    text = p.read_text(encoding="utf-8")
    if MARK in text:
        print("FR-Spec semantic port already present")
        return

    # 1) Draft-only vocab width. Real target/trunk models stay full-vocabulary.
    anchor = '''    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);\n    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;\n\n    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);\n'''
    replacement = '''    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);\n    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;\n\n    // [TAG_FLASHNEXT_R2_FRSPEC] A draft-only sidecar may carry a reduced LM head plus\n    // d2t[draft_row] = target_token_id. The target/trunk itself always remains full vocab.\n    int64_t n_vocab_out = n_vocab;\n    const ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t");\n    if (mtp_only && d2t_meta != nullptr) {\n        n_vocab_out = d2t_meta->ne[0];\n        GGML_ASSERT(n_vocab_out > 0 && n_vocab_out < n_vocab);\n        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_out }, 0);\n        LLAMA_LOG_INFO("%s: QWEN4EXP MTP FR-Spec d2t enabled, draft vocab = %lld / %lld\\n",\n                __func__, (long long) n_vocab_out, (long long) n_vocab);\n    }\n\n    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);\n'''
    text = once(text, anchor, replacement, "draft-vocab loader anchor")

    old = '''    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);\n'''
    new = '''    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"),\n                           { n_embd, mtp_only ? n_vocab_out : n_vocab }, TENSOR_NOT_REQUIRED);\n'''
    text = once(text, old, new, "output.weight width")

    # A self-contained modern sidecar may store the draft head on the NextN block.
    old = '''        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);\n'''
    new = '''        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il),\n                { n_embd, mtp_only ? n_vocab_out : n_vocab }, flags | TENSOR_NOT_REQUIRED);\n'''
    text = once(text, old, new, "NextN shared head width")

    # 2) Preserve MTP continuation inputs. Newer upstream may eventually acquire
    # this itself, so tolerate an already-applied equivalent rather than duplicate.
    input_anchor = '''    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);\n    ggml_set_input(inp->h);\n    ggml_set_name(inp->h, "mtp_h_input");\n'''
    input_replacement = input_anchor + '''\n    // [TAG_FLASHNEXT_R2_FRSPEC] keep continuation-step inputs alive when graph memory is recycled.\n    ggml_set_output(inp->tokens);\n    ggml_set_output(inp->h);\n'''
    if "ggml_set_output(inp->tokens)" not in text:
        text = once(text, input_anchor, input_replacement, "MTP input lifetime")

    # 3) Expand reduced draft logits to the full vocabulary *inside graph_mtp*.
    # Unselected ids are -inf, so speculative sampling cannot emit an id the draft
    # never scored; target verification remains full-vocabulary and unchanged.
    logits_anchor = '''    cur = build_lora_mm(head_w, cur, head_s);\n    cb(cur, "result_output", -1);\n    res->t_logits = cur;\n'''
    logits_replacement = '''    cur = build_lora_mm(head_w, cur, head_s);\n\n    // [TAG_FLASHNEXT_R2_FRSPEC] scatter reduced draft logits back to target-vocab shape.\n    if (model.d2t != nullptr) {\n        const int64_t n_draft_vocab = cur->ne[0];\n        const int64_t n_outputs     = cur->ne[1];\n        const int64_t n_vocab_full  = (int64_t) model.vocab.n_tokens();\n        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64 || model.d2t->type == GGML_TYPE_I32);\n        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);\n        GGML_ASSERT(n_draft_vocab < n_vocab_full);\n\n        ggml_tensor * full = ggml_fill(ctx0,\n                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_outputs), -INFINITY);\n        cur = ggml_set_rows(ctx0, full,\n                ggml_reshape_3d(ctx0, cur,       1, n_draft_vocab, n_outputs),\n                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));\n        cur = ggml_reshape_2d(ctx0, cur, n_vocab_full, n_outputs);\n        cb(cur, "result_output_d2t", -1);\n    }\n\n    cb(cur, "result_output", -1);\n    res->t_logits = cur;\n'''
    text = once(text, logits_anchor, logits_replacement, "graph_mtp logits expansion")

    # Require the actual modern MTP graph, otherwise this script matched some
    # accidental fragments in an incompatible source.
    if "llama_model_qwen4exp::graph_mtp::graph_mtp" not in text:
        raise RuntimeError("modern qwen4exp graph_mtp not found")

    p.write_text(text, encoding="utf-8")
    print(f"OK semantic FR-Spec port applied: {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
