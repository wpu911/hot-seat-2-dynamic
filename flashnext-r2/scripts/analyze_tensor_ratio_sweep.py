#!/usr/bin/env python3
"""Analyze the balanced Phase-6b tensor-ratio sweep.

The 1:1 arm is the anchor. A heterogeneous ratio may advance only when it keeps
retrieval/fixed-length/MTP correctness, has no severe per-depth regression, and
shows a real median long-context TG gain. This is an exploratory gate; the
selected arm still goes through short, 32K/64K/128K, cached and rollback gates.
"""
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


def collect(data: dict, model: str):
    depths = [int(x) for x in data.get("depths", [])]
    legs = [x for x in data.get("legs", []) if x.get("model") == model]
    successful = [x for x in legs if not x.get("error")]
    rows = {d: [] for d in depths}
    for leg in successful:
        for row in leg.get("rows", []):
            d = int(row.get("target_depth", -1))
            if d in rows:
                rows[d].append(row)

    per_depth = {}
    complete = bool(successful) and len(successful) == len(legs)
    for d in depths:
        rr = rows[d]
        acc = []
        marker = bool(rr)
        full = bool(rr)
        mtp = bool(rr)
        for r in rr:
            marker &= bool(r.get("marker_hit"))
            full &= bool(r.get("full_generation"))
            dn, an = r.get("drafted"), r.get("accepted")
            ok = isinstance(dn, (int, float)) and dn > 0 and isinstance(an, (int, float))
            mtp &= ok
            if ok:
                acc.append(an / dn)
        complete &= bool(rr) and marker and full and mtp
        per_depth[d] = {
            "samples": len(rr),
            "pp": med([r.get("pp") for r in rr]),
            "tg": med([r.get("tg") for r in rr]),
            "acceptance": med(acc),
            "marker": marker,
            "full_generation": full,
            "mtp": mtp,
        }
    return {
        "legs": len(legs),
        "successful_legs": len(successful),
        "complete": complete,
        "errors": [x.get("error") for x in legs if x.get("error")],
        "depths": per_depth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--anchor", default="qwen3.8-flash-next-r2-split-tensor-1x1:256k")
    ap.add_argument("--min-median-gain", type=float, default=0.0,
                    help="exploratory smoke threshold; full confirmation should require >=1%")
    ap.add_argument("--max-depth-loss", type=float, default=5.0)
    ap.add_argument("--max-median-pp-loss", type=float, default=12.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    if data.get("benchmark") != "tensor_ratio_sweep" or data.get("ignore_eos") is not True:
        raise SystemExit("ERROR unsupported/incomplete ratio sweep schema")
    models = data.get("models") or []
    if args.anchor not in models:
        raise SystemExit(f"ERROR anchor missing from sweep: {args.anchor}")

    stats = {m: collect(data, m) for m in models}
    anchor = stats[args.anchor]
    if not anchor["complete"]:
        raise SystemExit("ERROR 1:1 anchor did not complete cleanly; ratio selection is meaningless")

    report = {
        "anchor": args.anchor,
        "models": {},
        "winner": args.anchor,
        "winner_gain_pct": 0.0,
    }
    best = None
    for m in models:
        s = stats[m]
        if m == args.anchor:
            report["models"][m] = {
                **s,
                "median_tg_gain_pct": 0.0,
                "worst_tg_delta_pct": 0.0,
                "median_pp_delta_pct": 0.0,
                "worst_acceptance_delta_pp": 0.0,
                "qualifies": s["complete"],
            }
            continue

        tg_delta = []
        pp_delta = []
        acc_delta = []
        for d, a in anchor["depths"].items():
            b = s["depths"].get(d, {})
            td = pct(b.get("tg"), a.get("tg"))
            pd = pct(b.get("pp"), a.get("pp"))
            aa, ba = a.get("acceptance"), b.get("acceptance")
            ad = (ba - aa) * 100.0 if isinstance(aa, (int, float)) and isinstance(ba, (int, float)) else None
            if td is not None: tg_delta.append(td)
            if pd is not None: pp_delta.append(pd)
            if ad is not None: acc_delta.append(ad)

        med_tg = med(tg_delta)
        worst_tg = min(tg_delta) if tg_delta else None
        med_pp = med(pp_delta)
        worst_acc = min(acc_delta) if acc_delta else None
        qualifies = (
            s["complete"]
            and isinstance(med_tg, (int, float)) and med_tg >= args.min_median_gain
            and isinstance(worst_tg, (int, float)) and worst_tg >= -args.max_depth_loss
            and isinstance(med_pp, (int, float)) and med_pp >= -args.max_median_pp_loss
            and isinstance(worst_acc, (int, float)) and worst_acc >= -args.max_acceptance_drop_pp
        )
        report["models"][m] = {
            **s,
            "median_tg_gain_pct": med_tg,
            "worst_tg_delta_pct": worst_tg,
            "median_pp_delta_pct": med_pp,
            "worst_acceptance_delta_pp": worst_acc,
            "qualifies": qualifies,
        }
        if qualifies and (best is None or med_tg > best[0]):
            best = (med_tg, m)

    if best is not None and best[0] > 0:
        report["winner_gain_pct"], report["winner"] = best

    out = Path(args.out or Path(args.result).with_suffix(".ratio-analysis.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TENSOR RATIO SWEEP")
    for m in models:
        r = report["models"][m]
        print(
            f"{m}: complete={r['complete']} gain={r.get('median_tg_gain_pct')} "
            f"worstTG={r.get('worst_tg_delta_pct')} PP={r.get('median_pp_delta_pct')} "
            f"acc={r.get('worst_acceptance_delta_pp')} qualifies={r.get('qualifies')}"
        )
    print(f"WINNER={report['winner']}")
    print(f"WINNER_GAIN_PCT={report['winner_gain_pct']}")
    print(f"ANALYSIS={out}")


if __name__ == "__main__":
    main()
