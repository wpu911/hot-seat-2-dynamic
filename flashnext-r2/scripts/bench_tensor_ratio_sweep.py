#!/usr/bin/env python3
"""Balanced N-arm tensor-split ratio sweep through llama-swap :8090.

Phase-6b used to compare only EVEN -> MID/CAP, which quietly assumes VRAM
capacity is a decent proxy for the best work split. On heterogeneous GPUs that
can be exactly backwards: capacity, memory bandwidth and kernel throughput are
three different things. This runner therefore treats the ratio as an empirical
parameter and tests both directions around 1:1.

Sequence for one mirror cycle with models A..E:
  A B C D E E D C B A

Each leg starts from a model-specific cold state, gets one unmeasured warm-up,
then runs fixed-length long-context retrieval. Load/compute failure invalidates
only that arm; unrelated llama-swap models are never globally unloaded. A busy
experiment alias is different: the sweep stops instead of benchmarking through
somebody else's request and calling the resulting soup a measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bench_qsa_context_ladder import URL, get_optional, run_one, unload

DEFAULT_MODELS = [
    "qwen3.8-flash-next-r2-split-tensor-1x1:256k",
    "qwen3.8-flash-next-r2-split-tensor-mid:256k",
    "qwen3.8-flash-next-r2-split-tensor-cap:256k",
    "qwen3.8-flash-next-r2-split-tensor-inv-mid:256k",
    "qwen3.8-flash-next-r2-split-tensor-inv-cap:256k",
]


def mirrored(models: list[str], cycles: int) -> list[str]:
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    pair = models + list(reversed(models))
    return pair * cycles


def clean(models: list[str], url: str, force: bool, *, best_effort: bool = False) -> None:
    errors = []
    for model in models:
        try:
            unload(url, model, force)
        except Exception as e:
            if best_effort:
                print(f"WARNING cleanup unload {model}: {e}", flush=True)
            else:
                errors.append(f"{model}: {e}")
    if errors:
        raise RuntimeError("cannot establish a cold experiment state; " + " | ".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--cycles", type=int, default=1,
                    help="one cycle means forward+reverse, so every arm gets two legs")
    ap.add_argument("--depths", default="32768,65536")
    ap.add_argument("--tg", type=int, default=192)
    ap.add_argument("--warmup-depth", type=int, default=4096)
    ap.add_argument("--warmup-tg", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-tensor-ratio-sweep.json")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if len(models) < 2 or len(set(models)) != len(models):
        raise SystemExit("ERROR --models must contain at least two unique aliases")
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    if not depths:
        raise SystemExit("ERROR no depths")
    sequence = mirrored(models, args.cycles)

    result = {
        "schema_version": 1,
        "benchmark": "tensor_ratio_sweep",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": args.url,
        "models": models,
        "sequence": sequence,
        "cycles": args.cycles,
        "depths": depths,
        "requested_predict": args.tg,
        "ignore_eos": True,
        "warmup": {"depth": args.warmup_depth, "tg": args.warmup_tg},
        "legs": [],
    }

    clean(models, args.url, args.force)
    time.sleep(2)
    try:
        for idx, model in enumerate(sequence, 1):
            print(f"\n=== ratio leg {idx}/{len(sequence)}: {model} ===", flush=True)
            clean(models, args.url, args.force)
            time.sleep(2)
            leg = {
                "model": model,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "warmup": None,
                "rows": [],
                "error": None,
            }
            try:
                warm = run_one(args.url, model, args.warmup_depth, args.warmup_tg)
                leg["warmup"] = warm
                if not warm.get("full_generation") or not warm.get("marker_hit"):
                    raise RuntimeError(
                        f"warmup failed full={warm.get('full_generation')} marker={warm.get('marker_hit')}"
                    )
                wd, wa = warm.get("drafted"), warm.get("accepted")
                if not isinstance(wd, (int, float)) or wd <= 0 or not isinstance(wa, (int, float)):
                    raise RuntimeError("warmup missing MTP draft/accept counters")

                for depth in depths:
                    print(f"depth={depth}", flush=True)
                    row = run_one(args.url, model, depth, args.tg)
                    leg["rows"].append(row)
                    d, a = row.get("drafted"), row.get("accepted")
                    mtp_ok = isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float))
                    acc = a / d if mtp_ok else None
                    print(json.dumps({
                        "depth": depth,
                        "prompt_n": row.get("prompt_n"),
                        "pp": row.get("pp"),
                        "tg": row.get("tg"),
                        "acceptance": acc,
                        "marker_hit": row.get("marker_hit"),
                        "full_generation": row.get("full_generation"),
                    }, ensure_ascii=False), flush=True)
                    if not row.get("full_generation"):
                        raise RuntimeError(f"short generation at depth {depth}: {row.get('predicted_n')}")
                    if not row.get("marker_hit"):
                        raise RuntimeError(f"needle retrieval failed at depth {depth}")
                    if not mtp_ok:
                        raise RuntimeError(f"missing MTP counters at depth {depth}")
            except Exception as e:
                leg["error"] = f"{type(e).__name__}: {e}"
                print(f"ARM_FAIL {model}: {leg['error']}", flush=True)
            leg["running_after"] = get_optional(args.url, "/running")
            leg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            result["legs"].append(leg)
    finally:
        clean(models, args.url, args.force, best_effort=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT={out}")


if __name__ == "__main__":
    main()
