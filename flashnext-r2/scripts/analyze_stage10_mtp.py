#!/usr/bin/env python3
"""Analyze modern foundation MTP vs PR #28243 candidate through llama-swap.

Promotion gate:
- benchmark must be the fixed-length TG schema;
- deterministic workload outputs must match exactly;
- median TG gain must be material;
- no workload may regress beyond the allowed loss;
- MTP acceptance must be present for every workload and not materially deteriorate;
- PP is secondary but must not collapse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

WORKLOADS = ("zh", "code", "tool")


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

    tg, outputs, acceptance, full, samples = {}, {}, {}, {}, {}
    requested = data.get("requested_tg")
    for workload in WORKLOADS:
        rows = [
            row
            for leg in legs
            for row in leg.get("tg", [])
            if row.get("workload") == workload
        ]
        samples[workload] = len(rows)
        tg[workload] = med([row.get("tg") for row in rows])
        outputs[workload] = [row.get("content", "") for row in rows]
        avals = []
        for row in rows:
            d, a = row.get("drafted"), row.get("accepted")
            if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                avals.append(a / d)
        acceptance[workload] = med(avals)
        full[workload] = bool(rows) and all(row.get("full_generation") is True for row in rows)
        if isinstance(requested, (int, float)):
            full[workload] = full[workload] and all(
                isinstance(row.get("predicted_n"), (int, float)) and row.get("predicted_n") >= requested
                for row in rows
            )

    return {
        "legs": len(legs),
        "pp": pp,
        "tg": tg,
        "outputs": outputs,
        "acceptance": acceptance,
        "full_generation": full,
        "samples": samples,
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
    requested = data.get("requested_tg")
    fixed_schema = data.get("fixed_tg") is True and isinstance(requested, (int, float)) and requested > 0

    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in WORKLOADS}
    pp_delta = {
        k: pct(r["pp"].get(k), b["pp"].get(k))
        for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))
    }
    exact = {}
    acc_drop_pp = {}
    for w in WORKLOADS:
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

    fixed_complete = fixed_schema and all(b["full_generation"].values()) and all(r["full_generation"].values())
    samples_complete = all(b["samples"].get(w, 0) > 0 and r["samples"].get(w, 0) > 0 for w in WORKLOADS)
    acceptance_complete = all(
        isinstance(b["acceptance"].get(w), (int, float)) and
        isinstance(r["acceptance"].get(w), (int, float))
        for w in WORKLOADS
    )

    checks = {
        "fixed_tg_schema": fixed_schema,
        "fixed_generation_complete": fixed_complete,
        "samples_complete": samples_complete,
        "bit_exact": all(exact.values()),
        "median_tg_gain": tg_med is not None and tg_med >= args.min_median_tg_gain,
        "worst_tg": tg_worst is not None and tg_worst >= -args.max_workload_tg_loss,
        "median_pp": pp_med is not None and pp_med >= -args.max_median_pp_loss,
        "acceptance_present": acceptance_complete,
        "acceptance": acceptance_complete and acc_worst is not None and acc_worst >= -args.max_acceptance_drop_pp,
    }
    result = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "baseline": args.baseline,
        "r2": args.r2,
        "benchmark_schema": {"fixed_tg": data.get("fixed_tg"), "requested_tg": requested},
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"tg": tg_delta, "pp": pp_delta},
        "acceptance_delta_percentage_points": acc_drop_pp,
        "bit_exact": exact,
        "checks": checks,
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
    for w in WORKLOADS:
        print(f"  {w:>6}: {b['tg'].get(w)} -> {r['tg'].get(w)}  delta={tg_delta[w]}%  exact={exact[w]} fixed={b['full_generation'].get(w) and r['full_generation'].get(w)}")
    print("ACCEPTANCE")
    for w in WORKLOADS:
        print(f"  {w:>6}: {b['acceptance'].get(w)} -> {r['acceptance'].get(w)}  delta_pp={acc_drop_pp[w]}")
    print("PP")
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"  {k:>6}: {b['pp'].get(k)} -> {r['pp'].get(k)}  delta={pp_delta[k]}%")
    print("GATE")
    print(json.dumps({"checks": checks, "gate": report["gate"]}, ensure_ascii=False, indent=2))

    out = Path(args.result).with_suffix(".mtp-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
