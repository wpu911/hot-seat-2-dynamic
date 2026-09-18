#!/usr/bin/env python3
"""Analyze Stage-11 long-context OFF vs FAST results.

The patched binary maintains the index in both modes. OFF still uses the legacy
scan, FAST queries the (seq,pos) index. This isolates lookup speed. A separate
production-vs-FAST pass measures net benefit including index-maintenance cost.
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
    by_depth = {}
    for leg in data.get("legs", []):
        if leg.get("model") != model:
            continue
        for row in leg.get("rows", []):
            by_depth.setdefault(int(row["target_depth"]), []).append(row)

    out = {}
    for d, rows in by_depth.items():
        acc = []
        for r in rows:
            dr, ac = r.get("drafted"), r.get("accepted")
            if isinstance(dr, (int, float)) and dr > 0 and isinstance(ac, (int, float)):
                acc.append(ac / dr)
        out[d] = {
            "tg": med([r.get("tg") for r in rows]),
            "pp": med([r.get("pp") for r in rows]),
            "marker_ok": all(bool(r.get("marker_hit")) for r in rows),
            "acceptance": med(acc),
            "samples": len(rows),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--deep-from", type=int, default=32768)
    ap.add_argument("--min-deep-median-gain", type=float, default=5.0)
    ap.add_argument("--max-any-tg-loss", type=float, default=2.0)
    ap.add_argument("--max-pp-loss", type=float, default=3.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    f = collect(data, args.fast)
    depths = sorted(set(b) | set(f))

    deltas = {}
    deep = []
    all_tg = []
    pp = []
    marker_ok = True

    print("depth    TG off -> fast        delta       PP delta    marker")
    for d in depths:
        bd, fd = b.get(d, {}), f.get(d, {})
        td = pct(fd.get("tg"), bd.get("tg"))
        pd = pct(fd.get("pp"), bd.get("pp"))
        mok = bool(bd.get("marker_ok")) and bool(fd.get("marker_ok"))
        marker_ok &= mok
        deltas[str(d)] = {"tg_pct": td, "pp_pct": pd, "marker_ok": mok}
        if isinstance(td, (int, float)):
            all_tg.append(td)
            if d >= args.deep_from:
                deep.append(td)
        if isinstance(pd, (int, float)):
            pp.append(pd)
        print(f"{d:>6}  {str(bd.get('tg')):>8} -> {str(fd.get('tg')):<8}  {str(td):>9}  {str(pd):>9}  {mok}")

    deep_med = med(deep)
    worst_tg = min(all_tg) if all_tg else None
    pp_med = med(pp)

    pass_marker = marker_ok
    pass_gain = deep_med is not None and deep_med >= args.min_deep_median_gain
    pass_tg = worst_tg is None or worst_tg >= -args.max_any_tg_loss
    pass_pp = pp_med is None or pp_med >= -args.max_pp_loss
    result = "PASS" if pass_marker and pass_gain and pass_tg and pass_pp else "FAIL"

    report = {
        "baseline": b,
        "fast": f,
        "delta": deltas,
        "gate": {
            "result": result,
            "marker_ok": marker_ok,
            "deep_from": args.deep_from,
            "deep_median_tg_gain_pct": deep_med,
            "worst_tg_delta_pct": worst_tg,
            "median_pp_delta_pct": pp_med,
            "min_deep_median_gain_pct": args.min_deep_median_gain,
            "max_any_tg_loss_pct": args.max_any_tg_loss,
            "max_pp_loss_pct": args.max_pp_loss,
        },
    }

    print("\nGATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    out = Path(args.result).with_suffix(".prev-index-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
