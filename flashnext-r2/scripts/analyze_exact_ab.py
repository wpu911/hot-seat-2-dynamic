#!/usr/bin/env python3
"""Generic exact-output A/B analyzer for Flash Next R2 stages."""
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
            vals.extend(row.get("pp") for row in leg.get("pp", {}).get(k, []))
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
    ap.add_argument("--label", default="ab")
    ap.add_argument("--min-median-tg-gain", type=float, default=0.5)
    ap.add_argument("--max-workload-tg-loss", type=float, default=1.5)
    ap.add_argument("--max-median-pp-loss", type=float, default=2.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b, r = collect(data, args.baseline), collect(data, args.r2)
    pp_delta = {k: pct(r["pp"].get(k), b["pp"].get(k)) for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))}
    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in ("zh", "code", "tool")}

    exact = {}
    for w in ("zh", "code", "tool"):
        bo, ro = b["outputs"].get(w, []), r["outputs"].get(w, [])
        exact[w] = all_same(bo) and all_same(ro) and bo[0] == ro[0]

    pp_med = med(list(pp_delta.values()))
    tg_med = med(list(tg_delta.values()))
    tg_worst = min([x for x in tg_delta.values() if isinstance(x, (int, float))], default=None)
    passed = (
        all(exact.values()) and
        tg_med is not None and tg_med >= args.min_median_tg_gain and
        (tg_worst is None or tg_worst >= -args.max_workload_tg_loss) and
        (pp_med is None or pp_med >= -args.max_median_pp_loss)
    )

    report = {
        "label": args.label,
        "baseline": args.baseline,
        "r2": args.r2,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"pp": pp_delta, "tg": tg_delta},
        "bit_exact": exact,
        "gate": {
            "result": "PASS" if passed else "FAIL",
            "median_tg_gain_pct": tg_med,
            "worst_tg_delta_pct": tg_worst,
            "median_pp_delta_pct": pp_med,
            "min_median_tg_gain_pct": args.min_median_tg_gain,
            "max_workload_tg_loss_pct": args.max_workload_tg_loss,
            "max_median_pp_loss_pct": args.max_median_pp_loss,
        },
    }

    print(f"label={args.label} baseline_legs={b['legs']} r2_legs={r['legs']}")
    for w in ("zh", "code", "tool"):
        print(f"TG {w:>4}: {b['tg'].get(w)} -> {r['tg'].get(w)} delta={tg_delta[w]}% exact={exact[w]}")
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"PP {k:>5}: {b['pp'].get(k)} -> {r['pp'].get(k)} delta={pp_delta[k]}%")
    print(f"GATE median_tg={tg_med}% worst_tg={tg_worst}% median_pp={pp_med}% exact={all(exact.values())} result={report['gate']['result']}")

    out = Path(args.result).with_suffix(f".{args.label}-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
