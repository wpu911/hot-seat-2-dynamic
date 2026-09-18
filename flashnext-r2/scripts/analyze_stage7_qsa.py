#!/usr/bin/env python3
"""Analyze Stage-7 QSA gather long-context ladder results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def med(xs):
    ys = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(ys) if ys else None


def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def collect(data, model):
    rows = {}
    for leg in data.get("legs", []):
        if leg.get("model") != model:
            continue
        for r in leg.get("rows", []):
            rows.setdefault(int(r["target_depth"]), []).append(r)
    out = {}
    for d, rr in rows.items():
        acc = []
        for x in rr:
            dn, an = x.get("drafted"), x.get("accepted")
            if isinstance(dn, (int, float)) and dn > 0 and isinstance(an, (int, float)):
                acc.append(an / dn)
        out[d] = {
            "prompt_n": med([x.get("prompt_n") for x in rr]),
            "pp": med([x.get("pp") for x in rr]),
            "tg": med([x.get("tg") for x in rr]),
            "acceptance": med(acc),
            "marker_ok": all(bool(x.get("marker_hit")) for x in rr),
            "samples": len(rr),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--deep-from", type=int, default=32768)
    ap.add_argument("--min-deep-median-gain", type=float, default=5.0)
    ap.add_argument("--max-deep-loss", type=float, default=3.0)
    ap.add_argument("--max-pp-loss", type=float, default=5.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)
    depths = sorted(set(b) | set(r))

    report = {"baseline": b, "r2": r, "delta": {}, "gate": {}}
    deep = []
    deep_worst = None
    pp_losses = []
    marker_ok = True

    print("depth       prompt_n        PP base -> on       TG base -> on       delta      marker")
    for d in depths:
        bd, rd = b.get(d, {}), r.get(d, {})
        td = pct(rd.get("tg"), bd.get("tg"))
        pd = pct(rd.get("pp"), bd.get("pp"))
        mok = bool(bd.get("marker_ok")) and bool(rd.get("marker_ok"))
        marker_ok &= mok
        report["delta"][str(d)] = {"tg_pct": td, "pp_pct": pd, "marker_ok": mok}
        if d >= args.deep_from and td is not None:
            deep.append(td)
            deep_worst = td if deep_worst is None else min(deep_worst, td)
        if pd is not None:
            pp_losses.append(pd)
        print(f"{d:>7}  {str(rd.get('prompt_n')):>10}  {str(bd.get('pp')):>10} -> {str(rd.get('pp')):<10}  "
              f"{str(bd.get('tg')):>8} -> {str(rd.get('tg')):<8}  {str(td):>8}  {mok}")

    deep_med = med(deep)
    pp_med = med(pp_losses)
    pass_gain = deep_med is not None and deep_med >= args.min_deep_median_gain
    pass_worst = deep_worst is None or deep_worst >= -args.max_deep_loss
    pass_pp = pp_med is None or pp_med >= -args.max_pp_loss
    gate = "PASS" if marker_ok and pass_gain and pass_worst and pass_pp else "FAIL"

    report["gate"] = {
        "result": gate,
        "marker_ok": marker_ok,
        "deep_from": args.deep_from,
        "deep_median_tg_gain_pct": deep_med,
        "deep_worst_tg_delta_pct": deep_worst,
        "median_pp_delta_pct": pp_med,
        "min_deep_median_gain_pct": args.min_deep_median_gain,
        "max_deep_loss_pct": args.max_deep_loss,
        "max_pp_loss_pct": args.max_pp_loss,
    }

    print("\nGATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    out = Path(args.result).with_suffix(".qsa-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if gate == "PASS" else 2)


if __name__ == "__main__":
    main()
