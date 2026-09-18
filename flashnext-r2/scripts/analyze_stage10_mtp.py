#!/usr/bin/env python3
"""Analyze production MTP vs PR #28243 candidate through llama-swap.

Promotion gate:
- deterministic workload outputs must match exactly;
- median TG gain must be material;
- no workload may regress beyond the allowed loss;
- MTP acceptance must not materially deteriorate;
- PP is secondary but must not collapse.
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

    return {
        "legs": len(legs),
        "pp": pp,
        "tg": tg,
        "outputs": outputs,
        "acceptance": acceptance,
    }


def all_same(xs):
    return bool(xs) and len(set(xs)) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--min-median-tg-gain", type=float, default=3.0)
    ap.add_argument("--max-workload-tg-loss", type=float, default=2.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0,
                    help="maximum allowed acceptance-rate drop in percentage points")
    ap.add_argument("--max-median-pp-loss", type=float, default=5.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in ("zh", "code", "tool")}
    pp_delta = {
        k: pct(r["pp"].get(k), b["pp"].get(k))
        for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))
    }
    exact = {}
    acc_drop_pp = {}
    for w in ("zh", "code", "tool"):
        bo, ro = b["outputs"].get(w, []), r["outputs"].get(w, [])
        exact[w] = all_same(bo) and all_same(ro) and bo[0] == ro[0]
        ba, ra = b["acceptance"].get(w), r["acceptance"].get(w)
        acc_drop_pp[w] = None if ba is None or ra is None else (ra - ba) * 100.0

    tg_vals = [x for x in tg_delta.values() if isinstance(x, (int, float))]
    pp_vals = [x for x in pp_delta.values() if isinstance(x, (int, float))]
    acc_vals = [x for x in acc_drop_pp.values() if isinstance(x, (int, float))]

    tg_med = med(tg_vals)
    tg_worst = min(tg_vals) if tg_vals else None
    pp_med = med(pp_vals)
    acc_worst = min(acc_vals) if acc_vals else None

    pass_exact = all(exact.values())
    pass_tg = tg_med is not None and tg_med >= args.min_median_tg_gain
    pass_tg_worst = tg_worst is None or tg_worst >= -args.max_workload_tg_loss
    pass_pp = pp_med is None or pp_med >= -args.max_median_pp_loss
    pass_acc = acc_worst is not None and acc_worst >= -args.max_acceptance_drop_pp
    result = "PASS" if pass_exact and pass_tg and pass_tg_worst and pass_pp and pass_acc else "FAIL"

    report = {
        "baseline": args.baseline,
        "r2": args.r2,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"tg": tg_delta, "pp": pp_delta},
        "acceptance_delta_percentage_points": acc_drop_pp,
        "bit_exact": exact,
        "gate": {
            "result": result,
            "median_tg_gain_pct": tg_med,
            "worst_tg_delta_pct": tg_worst,
            "median_pp_delta_pct": pp_med,
            "worst_acceptance_delta_pp": acc_worst,
            "min_median_tg_gain_pct": args.min_median_tg_gain,
            "max_workload_tg_loss_pct": args.max_workload_tg_loss,
            "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
            "max_median_pp_loss_pct": args.max_median_pp_loss,
        },
    }

    print("TG")
    for w in ("zh", "code", "tool"):
        print(f"  {w:>6}: {b['tg'].get(w)} -> {r['tg'].get(w)}  delta={tg_delta[w]}%  exact={exact[w]}")
    print("ACCEPTANCE")
    for w in ("zh", "code", "tool"):
        print(f"  {w:>6}: {b['acceptance'].get(w)} -> {r['acceptance'].get(w)}  delta_pp={acc_drop_pp[w]}")
    print("PP")
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"  {k:>6}: {b['pp'].get(k)} -> {r['pp'].get(k)}  delta={pp_delta[k]}%")
    print("GATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.result).with_suffix(".mtp-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
