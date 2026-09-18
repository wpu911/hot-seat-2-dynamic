#!/usr/bin/env python3
"""Analyze --lazy-mode on vs on-direct for Flash Next.

Stage 11 targets prompt processing / PLE row I/O. Decode must stay neutral and
deterministic. The same candidate binary is used for both arms.
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

    tg, outputs = {}, {}
    for workload in ("zh", "code", "tool"):
        vals, outs = [], []
        for leg in legs:
            for row in leg.get("tg", []):
                if row.get("workload") == workload:
                    vals.append(row.get("tg"))
                    outs.append(row.get("content", ""))
        tg[workload] = med(vals)
        outputs[workload] = outs
    return {"legs": len(legs), "pp": pp, "tg": tg, "outputs": outputs}


def same(xs):
    return bool(xs) and len(set(xs)) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--min-median-pp-gain", type=float, default=5.0)
    ap.add_argument("--max-pp-loss", type=float, default=3.0)
    ap.add_argument("--max-tg-loss", type=float, default=2.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    pp_delta = {k: pct(r["pp"].get(k), b["pp"].get(k))
                for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))}
    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in ("zh", "code", "tool")}

    exact = {}
    for w in ("zh", "code", "tool"):
        bo, ro = b["outputs"].get(w, []), r["outputs"].get(w, [])
        exact[w] = same(bo) and same(ro) and bo[0] == ro[0]

    pp_vals = [x for x in pp_delta.values() if isinstance(x, (int, float))]
    tg_vals = [x for x in tg_delta.values() if isinstance(x, (int, float))]
    pp_med = med(pp_vals)
    pp_worst = min(pp_vals) if pp_vals else None
    tg_worst = min(tg_vals) if tg_vals else None

    pass_exact = all(exact.values())
    pass_pp_gain = pp_med is not None and pp_med >= args.min_median_pp_gain
    pass_pp_worst = pp_worst is None or pp_worst >= -args.max_pp_loss
    pass_tg = tg_worst is None or tg_worst >= -args.max_tg_loss
    result = "PASS" if pass_exact and pass_pp_gain and pass_pp_worst and pass_tg else "FAIL"

    report = {
        "baseline": args.baseline,
        "r2": args.r2,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"pp": pp_delta, "tg": tg_delta},
        "bit_exact": exact,
        "gate": {
            "result": result,
            "median_pp_gain_pct": pp_med,
            "worst_pp_delta_pct": pp_worst,
            "worst_tg_delta_pct": tg_worst,
            "min_median_pp_gain_pct": args.min_median_pp_gain,
            "max_pp_loss_pct": args.max_pp_loss,
            "max_tg_loss_pct": args.max_tg_loss,
        },
    }

    print("PP")
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"  {k:>6}: {b['pp'].get(k)} -> {r['pp'].get(k)}  delta={pp_delta[k]}%")
    print("TG")
    for w in ("zh", "code", "tool"):
        print(f"  {w:>6}: {b['tg'].get(w)} -> {r['tg'].get(w)}  delta={tg_delta[w]}%  exact={exact[w]}")
    print("GATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.result).with_suffix(".lazy-direct-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
