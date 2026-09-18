#!/usr/bin/env python3
"""Summarize bench_llamaswap_ab.py output and apply a conservative Stage-1 gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def collect(data, model):
    legs = [x for x in data.get("legs", []) if x.get("model") == model]
    pp_keys = sorted({k for leg in legs for k in leg.get("pp", {})}, key=lambda x: int(x))
    pp = {}
    for k in pp_keys:
        vals = []
        for leg in legs:
            for row in leg.get("pp", {}).get(k, []):
                vals.append(row.get("pp"))
        pp[k] = med(vals)

    tg = {}
    acc = {}
    for workload in ("zh", "code", "tool"):
        tvals, avals = [], []
        for leg in legs:
            for row in leg.get("tg", []):
                if row.get("workload") != workload:
                    continue
                tvals.append(row.get("tg"))
                d, a = row.get("drafted"), row.get("accepted")
                if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                    avals.append(a / d)
        tg[workload] = med(tvals)
        acc[workload] = med(avals)
    return {"pp": pp, "tg": tg, "acceptance": acc, "legs": len(legs)}


def fmt(x, suffix=""):
    return "n/a" if x is None else f"{x:.2f}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", default="qwen3.8-flash-next:256k")
    ap.add_argument("--r2", default="qwen3.8-flash-next-r2:256k")
    ap.add_argument("--min-pp-gain", type=float, default=3.0)
    ap.add_argument("--max-tg-loss", type=float, default=2.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    report = {
        "baseline": b,
        "r2": r,
        "delta_pct": {"pp": {}, "tg": {}, "acceptance": {}},
    }

    print(f"baseline legs={b['legs']}  r2 legs={r['legs']}")
    print("\nPP")
    pp_gains = []
    for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x)):
        d = pct(r["pp"].get(k), b["pp"].get(k))
        report["delta_pct"]["pp"][k] = d
        if d is not None:
            pp_gains.append(d)
        print(f"  {k:>6}: {fmt(b['pp'].get(k))} -> {fmt(r['pp'].get(k))}  {fmt(d, '%')}")

    print("\nTG")
    tg_losses = []
    for w in ("zh", "code", "tool"):
        d = pct(r["tg"].get(w), b["tg"].get(w))
        report["delta_pct"]["tg"][w] = d
        if d is not None:
            tg_losses.append(d)
        print(f"  {w:>6}: {fmt(b['tg'].get(w))} -> {fmt(r['tg'].get(w))}  {fmt(d, '%')}")

    print("\nMTP acceptance")
    for w in ("zh", "code", "tool"):
        old, new = b["acceptance"].get(w), r["acceptance"].get(w)
        d = pct(new, old)
        report["delta_pct"]["acceptance"][w] = d
        print(f"  {w:>6}: {fmt(None if old is None else old*100, '%')} -> {fmt(None if new is None else new*100, '%')}")

    pp_med = med(pp_gains)
    worst_tg = min(tg_losses) if tg_losses else None
    pass_pp = pp_med is not None and pp_med >= args.min_pp_gain
    pass_tg = worst_tg is None or worst_tg >= -args.max_tg_loss
    gate = "PASS" if pass_pp and pass_tg else "FAIL"
    report["gate"] = {
        "result": gate,
        "median_pp_gain_pct": pp_med,
        "worst_tg_delta_pct": worst_tg,
        "min_pp_gain_pct": args.min_pp_gain,
        "max_tg_loss_pct": args.max_tg_loss,
    }

    print("\nGATE")
    print(f"  median PP gain : {fmt(pp_med, '%')}")
    print(f"  worst TG delta : {fmt(worst_tg, '%')}")
    print(f"  result         : {gate}")

    out = Path(args.result).with_suffix(".analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nANALYSIS={out}")


if __name__ == "__main__":
    main()
