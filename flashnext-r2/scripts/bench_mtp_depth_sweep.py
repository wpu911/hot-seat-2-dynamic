#!/usr/bin/env python3
"""Balanced Flash Next MTP draft-depth sweep through llama-swap :8090.

The ordinary A/B helper is ideal for two candidates but becomes wasteful and
order-biased when comparing four speculative depths. This runner keeps the same
real production path and benchmark semantics while using mirrored 4-arm cycles:

  1 -> 2 -> 3 -> 4 -> 4 -> 3 -> 2 -> 1

for the default two cycles. Every measured leg gets a fresh model load, one
unmeasured warm-up, fixed-length TG with ignore_eos=true, and model-specific
unload only. Unrelated llama-swap models are never globally unloaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bench_llamaswap_ab import (
    URL,
    SCHEMA_VERSION,
    unload_model,
    run_leg,
    summarize,
)

DEFAULT_MODELS = [
    "qwen3.8-flash-next-r2-final-mtp1:256k",
    "qwen3.8-flash-next-r2-final-mtp2:256k",
    "qwen3.8-flash-next-r2-final-mtp3:256k",
    "qwen3.8-flash-next-r2-final-mtp4:256k",
]


def build_sequence(models: list[str], cycles: int) -> list[str]:
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    seq: list[str] = []
    for i in range(cycles):
        seq.extend(models if i % 2 == 0 else list(reversed(models)))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated aliases in increasing draft depth order")
    ap.add_argument("--cycles", type=int, default=2,
                    help="number of mirrored four-arm cycles; default gives each arm two legs")
    ap.add_argument("--repeat", type=int, default=3,
                    help="samples per PP/TG workload inside each leg")
    ap.add_argument("--tg", type=int, default=512)
    ap.add_argument("--pp", default="512")
    ap.add_argument("--warmup-pp", type=int, default=256)
    ap.add_argument("--warmup-tg", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-mtp-depth-sweep.json")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if len(models) < 2 or len(set(models)) != len(models):
        raise SystemExit("ERROR --models must contain at least two unique aliases")
    pp_tokens = [int(x) for x in args.pp.split(",") if x.strip()]
    sequence = build_sequence(models, args.cycles)

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "mtp_depth_sweep",
        "url": args.url,
        "models": models,
        "sequence": sequence,
        "cycles": args.cycles,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixed_tg": True,
        "requested_tg": args.tg,
        "warmup": {"pp": args.warmup_pp, "tg": args.warmup_tg},
        "legs": [],
    }

    # Start from a clean state for only the experiment aliases. This avoids a
    # previous interrupted sweep leaving one arm resident and receiving a warm
    # cache advantage.
    for model in models:
        unload_model(args.url, model, args.force)
    time.sleep(2)

    previous = None
    try:
        for idx, model in enumerate(sequence, 1):
            print(f"\n=== MTP sweep leg {idx}/{len(sequence)}: {model} ===", flush=True)
            if previous is not None:
                unload_model(args.url, previous, args.force)
                time.sleep(3)
            unload_model(args.url, model, args.force)
            time.sleep(2)

            leg = run_leg(
                args.url,
                model,
                pp_tokens,
                args.tg,
                args.repeat,
                args.warmup_pp,
                args.warmup_tg,
            )
            result["legs"].append(leg)
            print(json.dumps(summarize(leg), ensure_ascii=False, indent=2), flush=True)
            previous = model
    finally:
        # Best effort cleanup of experiment arms only. A busy unrelated model is
        # never touched.
        for model in models:
            try:
                unload_model(args.url, model, args.force)
            except Exception as e:
                print(f"WARNING cleanup unload failed for {model}: {e}", flush=True)

    result["summaries"] = [summarize(x) for x in result["legs"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT={out}")


if __name__ == "__main__":
    main()
