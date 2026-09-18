#!/usr/bin/env python3
"""Analyze Stage-2 HC fusion A/B results.

Gate philosophy:
- output must remain identical at temperature=0 for all TG workloads;
- median TG gain must be material (default >= 3%);
- no workload may regress more than 2%;
- PP is not the target, but median PP must not regress more than 3%.
"""
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
            vals += [row.get("pp") for row in leg.get("pp", {}).get(k, [])]
        pp[k] = med(vals)

    tg, outputs, acceptance = {}, {}, {}
    for workload in ("zh", "code", "tool"):
        tvals, outs, avals = [], [], []
        for leg in legs:
            for row in leg.get("tg", []):
                if row.get("workload") != workload:
                    continue
                tvals.append(row.get("tg"))
                outs.append(row.get("content", ""))
                d, a = row.get("drafted"), row.get("accepted")
                if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                    avals.append(a / d)
        tg[workload] = med(tvals)
        outputs[workload] = outs
        acceptance[workload] = med(avals)
    return {"legs": len(legs), "pp": pp, "tg": tg, "outputs": outputs, "acceptance": acceptance}


def all_same(xs):
    return bool(xs) and len(set(xs)) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--min-median-tg-gain", type=float, default=3.0)
    ap.add_argument("--max-workload-tg-loss", type=float, default=2.0)
    ap.add_argument("--max-median-pp-loss", type=float, default=3.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    pp_delta = {}
    for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x)):
        pp_delta[k] = pct(r["pp"].get(k), b["pp"].get(k))

    tg_delta = {}
    for w in ("zh", "code", "tool"):
        tg_delta[w] = pct(r["tg"].get(w), b["tg"].get(w))

    exact = {}
    for w in ("zh", "code", "tool"):
        bo = b["outputs"].get(w, [])
        ro = r["outputs"].get(w, [])
        exact[w] = all_same(bo) and all_same(ro) and bo[0] == ro[0]

    pp_med = med(list(pp_delta.values()))
    tg_med = med(list(tg_delta.values()))
    tg_worst = min([x for x in tg_delta.values() if isinstance(x, (int, float))], default=None)

    pass_exact = all(exact.values())
    pass_tg_gain = tg_med is not None and tg_med >= args.min_median_tg_gain
    pass_tg_worst = tg_worst is None or tg_worst >= -args.max_workload_tg_loss
    pass_pp = pp_med is None or pp_med >= -args.max_median_pp_loss
    result = "PASS" if pass_exact and pass_tg_gain and pass_tg_worst and pass_pp else "FAIL"

    report = {
        "baseline": args.baseline,
        "r2": args.r2,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"pp": pp_delta, "tg": tg_delta},
        "bit_exact": exact,
        "gate": {
            "result": result,
            "median_tg_gain_pct": tg_med,
            "worst_tg_delta_pct": tg_worst,
            "median_pp_delta_pct": pp_med,
            "min_median_tg_gain_pct": args.min_median_tg_gain,
            "max_workload_tg_loss_pct": args.max_workload_tg_loss,
            "max_median_pp_loss_pct": args.max_median_pp_loss,
        },
    }

    print(f"baseline legs={b['legs']}  r2 legs={r['legs']}")
    print("\nTG")
    for w in ("zh", "code", "tool"):
        print(f"  {w:>6}: {b['tg'].get(w)} -> {r['tg'].get(w)}   delta={tg_delta.get(w)}%   exact={exact[w]}")
    print("\nPP")
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"  {k:>6}: {b['pp'].get(k)} -> {r['pp'].get(k)}   delta={pp_delta[k]}%")
    print("\nGATE")
    print(f"  median TG gain : {tg_med}")
    print(f"  worst TG delta : {tg_worst}")
    print(f"  median PP delta: {pp_med}")
    print(f"  bit exact      : {pass_exact}")
    print(f"  result         : {result}")

    out = Path(args.result).with_suffix(".hc-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
