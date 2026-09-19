#!/usr/bin/env python3
"""Generic exact-output A/B analyzer for Flash Next R2 stages.

This gate is used by short-context comparison stages such as Phase-6 tensor
split. It therefore rejects stale pre-schema results, truncated generations and
missing MTP counters instead of silently treating absent data as acceptable.
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
            vals.extend(row.get("pp") for row in leg.get("pp", {}).get(k, []))
        pp[k] = med(vals)

    tg, outputs, acceptance = {}, {}, {}
    tg_samples, acceptance_samples, full_generation = {}, {}, {}
    requested = data.get("requested_tg")
    for workload in WORKLOADS:
        rows = []
        for leg in legs:
            rows.extend(r for r in leg.get("tg", []) if r.get("workload") == workload)
        tvals = [r.get("tg") for r in rows]
        outs = [r.get("content", "") for r in rows]
        avals = []
        for row in rows:
            d, a = row.get("drafted"), row.get("accepted")
            if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                avals.append(a / d)
        tg[workload] = med(tvals)
        outputs[workload] = outs
        acceptance[workload] = med(avals)
        tg_samples[workload] = len(rows)
        acceptance_samples[workload] = len(avals)
        full = bool(rows) and all(r.get("full_generation") is True for r in rows)
        if isinstance(requested, (int, float)):
            full = full and all(
                isinstance(r.get("predicted_n"), (int, float)) and r.get("predicted_n") >= requested
                for r in rows
            )
        full_generation[workload] = full

    return {
        "legs": len(legs),
        "pp": pp,
        "tg": tg,
        "outputs": outputs,
        "acceptance": acceptance,
        "tg_samples": tg_samples,
        "acceptance_samples": acceptance_samples,
        "full_generation": full_generation,
    }


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
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    schema_ok = (
        isinstance(data.get("schema_version"), int)
        and data.get("schema_version") >= 3
        and data.get("fixed_tg") is True
        and isinstance(data.get("requested_tg"), (int, float))
        and data.get("requested_tg") > 0
    )

    b, r = collect(data, args.baseline), collect(data, args.r2)
    pp_delta = {
        k: pct(r["pp"].get(k), b["pp"].get(k))
        for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))
    }
    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in WORKLOADS}

    exact = {}
    acc_delta = {}
    sample_ok = {}
    full_ok = {}
    acceptance_ok = {}
    for w in WORKLOADS:
        bo, ro = b["outputs"].get(w, []), r["outputs"].get(w, [])
        exact[w] = all_same(bo) and all_same(ro) and bo[0] == ro[0]
        ba, ra = b["acceptance"].get(w), r["acceptance"].get(w)
        acc_delta[w] = None if ba is None or ra is None else (ra - ba) * 100.0
        sample_ok[w] = b["tg_samples"].get(w, 0) > 0 and r["tg_samples"].get(w, 0) > 0
        full_ok[w] = bool(b["full_generation"].get(w)) and bool(r["full_generation"].get(w))
        acceptance_ok[w] = (
            sample_ok[w]
            and b["acceptance_samples"].get(w, 0) == b["tg_samples"].get(w, 0)
            and r["acceptance_samples"].get(w, 0) == r["tg_samples"].get(w, 0)
            and isinstance(ba, (int, float)) and isinstance(ra, (int, float))
        )

    pp_vals = [x for x in pp_delta.values() if isinstance(x, (int, float))]
    tg_vals = [x for x in tg_delta.values() if isinstance(x, (int, float))]
    acc_vals = [x for x in acc_delta.values() if isinstance(x, (int, float))]
    pp_med = med(pp_vals)
    tg_med = med(tg_vals)
    tg_worst = min(tg_vals) if tg_vals else None
    acc_worst = min(acc_vals) if acc_vals else None

    checks = {
        "schema": schema_ok,
        "samples": all(sample_ok.values()),
        "fixed_generation": all(full_ok.values()),
        "bit_exact": all(exact.values()),
        "acceptance_present": all(acceptance_ok.values()),
        "acceptance": acc_worst is not None and acc_worst >= -args.max_acceptance_drop_pp,
        "median_tg": tg_med is not None and tg_med >= args.min_median_tg_gain,
        "worst_tg": tg_worst is not None and tg_worst >= -args.max_workload_tg_loss,
        "median_pp": pp_med is not None and pp_med >= -args.max_median_pp_loss,
    }
    passed = all(checks.values())

    report = {
        "label": args.label,
        "baseline": args.baseline,
        "r2": args.r2,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"pp": pp_delta, "tg": tg_delta},
        "acceptance_delta_percentage_points": acc_delta,
        "bit_exact": exact,
        "gate": {
            "result": "PASS" if passed else "FAIL",
            "checks": checks,
            "schema_version": data.get("schema_version"),
            "median_tg_gain_pct": tg_med,
            "worst_tg_delta_pct": tg_worst,
            "median_pp_delta_pct": pp_med,
            "worst_acceptance_delta_pp": acc_worst,
            "min_median_tg_gain_pct": args.min_median_tg_gain,
            "max_workload_tg_loss_pct": args.max_workload_tg_loss,
            "max_median_pp_loss_pct": args.max_median_pp_loss,
            "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
        },
    }

    print(f"label={args.label} baseline_legs={b['legs']} r2_legs={r['legs']}")
    for w in WORKLOADS:
        print(
            f"TG {w:>4}: {b['tg'].get(w)} -> {r['tg'].get(w)} "
            f"delta={tg_delta[w]}% exact={exact[w]} full={full_ok[w]} "
            f"accΔ={acc_delta[w]}pp"
        )
    for k in sorted(pp_delta, key=lambda x: int(x)):
        print(f"PP {k:>5}: {b['pp'].get(k)} -> {r['pp'].get(k)} delta={pp_delta[k]}%")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.result).with_suffix(f".{args.label}-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
