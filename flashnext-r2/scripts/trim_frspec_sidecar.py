#!/usr/bin/env python3
"""Create a reduced-vocabulary Qwen3.8 Flash Next MTP sidecar.

Adapted from drluoto's FR-Spec experiment, but with no hard-coded source tree,
atomic output, explicit ranking input, and sanity checks. The original draft is
never modified.

Usage:
  PYTHONPATH=/path/to/llama.cpp/gguf-py \
    python3 trim_frspec_sidecar.py full-draft.gguf draft-frspec-65536.gguf \
      --k 65536 --rank frequency.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except Exception as e:
    raise SystemExit(
        "cannot import gguf; set PYTHONPATH to the chosen llama.cpp/gguf-py directory: " + str(e)
    )


def field_value(f):
    return f.contents()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--k", type=int, default=65536)
    ap.add_argument("--rank", required=True, help="JSON with a 'rank' token-id array")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    rank_path = Path(args.rank).resolve()
    if not src.is_file():
        raise SystemExit(f"source draft missing: {src}")
    if dst.exists():
        raise SystemExit(f"destination already exists; refusing overwrite: {dst}")
    if src == dst:
        raise SystemExit("source and destination must differ")
    if not rank_path.is_file():
        raise SystemExit(f"ranking file missing: {rank_path}")

    rank_obj = json.loads(rank_path.read_text(encoding="utf-8"))
    rank = rank_obj.get("rank")
    if not isinstance(rank, list) or not rank:
        raise SystemExit("ranking JSON has no non-empty 'rank' array")

    reader = GGUFReader(str(src))
    outs = [t for t in reader.tensors if t.name == "output.weight"]
    if len(outs) != 1:
        raise SystemExit(f"expected exactly one output.weight, found {len(outs)}")
    out_t = outs[0]

    # GGUF tensor shape is reported in logical ggml order while reader.data is
    # laid out row-major for indexed access. drluoto's tested implementation uses
    # shape[1] as vocabulary and data[keep] to copy quantized rows byte-for-byte.
    if len(out_t.shape) != 2:
        raise SystemExit(f"unexpected output.weight rank: shape={list(out_t.shape)}")
    n_vocab = int(out_t.shape[1])
    if not (1 < args.k < n_vocab):
        raise SystemExit(f"--k must be in 2..{n_vocab-1}; got {args.k}")

    keep_ordered: list[int] = []
    seen: set[int] = set()
    for x in rank:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < n_vocab and i not in seen:
            keep_ordered.append(i)
            seen.add(i)
            if len(keep_ordered) >= args.k:
                break
    # Ensure K rows even when the local history has not exercised the whole vocab.
    if len(keep_ordered) < args.k:
        for i in range(n_vocab):
            if i not in seen:
                keep_ordered.append(i)
                seen.add(i)
                if len(keep_ordered) >= args.k:
                    break

    # Sort selected ids so d2t is monotonic and deterministic. Selection priority
    # came from frequency; row order does not need to preserve that ranking.
    keep = np.asarray(sorted(keep_ordered[:args.k]), dtype=np.int64)
    if len(np.unique(keep)) != args.k:
        raise SystemExit("internal error: duplicate kept token ids")

    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise SystemExit("GGUF lacks general.architecture")
    arch = field_value(arch_field)
    if isinstance(arch, (list, tuple)) and len(arch) == 1:
        arch = arch[0]
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")
    if not isinstance(arch, str):
        arch = str(arch)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"source={src}")
    print(f"destination={dst}")
    print(f"architecture={arch}")
    print(f"output.weight type={out_t.tensor_type.name} n_vocab={n_vocab} -> K={args.k}")
    print(f"ranking_source={rank_path}")

    writer = GGUFWriter(str(tmp), arch=arch, endianess=reader.endianess)
    # Copy all metadata. Writer creates general.architecture itself.
    for f in reader.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, field_value(f), vt, sub_type=st)
    writer.add_string(
        "general.frspec.note",
        f"output.weight trimmed to {args.k} workload-ranked rows; d2t maps draft row to target token id",
    )
    writer.add_string("general.frspec.rank_source", rank_path.name)

    plan = []
    for t in reader.tensors:
        if t.name == "d2t":
            raise SystemExit("source sidecar already contains d2t; refusing to trim a previously trimmed draft")
        if t.name == "output.weight":
            data = np.ascontiguousarray(t.data[keep])
            print(f"output.weight data {tuple(t.data.shape)} -> {tuple(data.shape)}")
        else:
            data = t.data
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
        plan.append(data)

    d2t = keep.copy()
    writer.add_tensor_info(
        "d2t", d2t.shape, d2t.dtype, d2t.nbytes, gguf.GGMLQuantizationType.I64
    )
    plan.append(d2t)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for data in plan:
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        writer.close()

        check = GGUFReader(str(tmp))
        chk_out = [t for t in check.tensors if t.name == "output.weight"]
        chk_d2t = [t for t in check.tensors if t.name == "d2t"]
        if len(chk_out) != 1 or len(chk_d2t) != 1:
            raise RuntimeError("output verification failed: output.weight/d2t missing")
        if int(chk_out[0].shape[1]) != args.k:
            raise RuntimeError(
                f"output verification failed: expected vocab {args.k}, got {int(chk_out[0].shape[1])}"
            )
        if int(chk_d2t[0].shape[0]) != args.k:
            raise RuntimeError(
                f"d2t verification failed: expected {args.k}, got {int(chk_d2t[0].shape[0])}"
            )
        tmp.replace(dst)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"OK={dst}")
    print(f"size_bytes={dst.stat().st_size}")


if __name__ == "__main__":
    main()
