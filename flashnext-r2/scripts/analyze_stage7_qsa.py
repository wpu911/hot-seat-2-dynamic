#!/usr/bin/env python3
"""Analyze Stage-7/8 QSA long-context ladder results.

The analyzer is shared by QSA gather and pooled-key cache A/B. It requires the
current fixed-length ladder schema, successful needle retrieval, complete MTP
acceptance counters and bounded PP/TG regressions.
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


def collect(data, model):
    rows = {}
    requested = data.get("requested_predict")
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
        full = bool(rr) and all(x.get("full_generation") is True for x in rr)
        if isinstance(requested, (int, float)):
            full = full and all(
                isinstance(x.get("predicted_n"), (int, float)) and x.get("predicted_n") >= requested
                for x in rr
            )
        out[d] = {
            "prompt_n": med([x.get("prompt_n") for x in rr]),
            "pp": med([x.get("pp") for x in rr]),
            "tg": med([x.get("tg") for x in rr]),
            "acceptance": med(acc),
            "acceptance_samples": len(acc),
            "marker_ok": all(bool(x.get("marker_hit")) for x in rr),
            "full_generation": full,
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
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    schema_ok = (
        isinstance(data.get("schema_version"), int)
        and data.get("schema_version") >= 3
        and data.get("ignore_eos") is True
        and isinstance(data.get("requested_predict"), (int, float))
        and data.get("requested_predict") > 0
    )

    b = collect(data, args.baseline)
    r = collect(data, args.r2)
    depths = sorted(set(b) | set(r))

    report = {"baseline": b, "r2": r, "delta": {}, "gate": {}}
    deep = []
    deep_worst = None
    deep_acc = []
    pp_losses = []
    marker_ok = bool(depths)
    full_ok = bool(depths)
    acceptance_ok = bool(depths)
    samples_ok = bool(depths)

    print("depth       prompt_n        PP base -> on       TG base -> on       delta      acc dPP   marker full")
    for d in depths:
        bd, rd = b.get(d, {}), r.get(d, {})
        td = pct(rd.get("tg"), bd.get("tg"))
        pd = pct(rd.get("pp"), bd.get("pp"))
        ba, ra = bd.get("acceptance"), rd.get("acceptance")
        ad = None if ba is None or ra is None else (ra - ba) * 100.0
        mok = bool(bd.get("marker_ok")) and bool(rd.get("marker_ok"))
        fok = bool(bd.get("full_generation")) and bool(rd.get("full_generation"))
        aok = (
            isinstance(ba, (int, float)) and isinstance(ra, (int, float))
            and bd.get("acceptance_samples", 0) == bd.get("samples", 0)
            and rd.get("acceptance_samples", 0) == rd.get("samples", 0)
        )
        sok = bd.get("samples", 0) > 0 and rd.get("samples", 0) > 0
        marker_ok &= mok
        full_ok &= fok
        acceptance_ok &= aok
        samples_ok &= sok
        report["delta"][str(d)] = {
            "tg_pct": td,
            "pp_pct": pd,
            "acceptance_delta_pp": ad,
            "marker_ok": mok,
            "full_generation": fok,
            "acceptance_complete": aok,
        }
        if d >= args.deep_from:
            if td is not None:
                deep.append(td)
                deep_worst = td if deep_worst is None else min(deep_worst, td)
            if ad is not None:
                deep_acc.append(ad)
        if pd is not None:
            pp_losses.append(pd)
        print(f"{d:>7}  {str(rd.get('prompt_n')):>10}  {str(bd.get('pp')):>10} -> {str(rd.get('pp')):<10}  "
              f"{str(bd.get('tg')):>8} -> {str(rd.get('tg')):<8}  {str(td):>8}  {str(ad):>8}  {mok} {fok}")

    deep_med = med(deep)
    pp_med = med(pp_losses)
    acc_worst = min(deep_acc) if deep_acc else None
    pass_gain = deep_med is not None and deep_med >= args.min_deep_median_gain
    pass_worst = deep_worst is not None and deep_worst >= -args.max_deep_loss
    pass_pp = pp_med is not None and pp_med >= -args.max_pp_loss
    pass_acc = acceptance_ok and acc_worst is not None and acc_worst >= -args.max_acceptance_drop_pp

    checks = {
        "schema": schema_ok,
        "samples": samples_ok,
        "marker": marker_ok,
        "fixed_generation": full_ok,
        "acceptance_present": acceptance_ok,
        "deep_gain": pass_gain,
        "deep_worst": pass_worst,
        "pp": pass_pp,
        "acceptance": pass_acc,
    }
    gate = "PASS" if all(checks.values()) else "FAIL"

    report["gate"] = {
        "result": gate,
        "checks": checks,
        "schema_version": data.get("schema_version"),
        "marker_ok": marker_ok,
        "fixed_generation_ok": full_ok,
        "acceptance_complete": acceptance_ok,
        "deep_from": args.deep_from,
        "deep_median_tg_gain_pct": deep_med,
        "deep_worst_tg_delta_pct": deep_worst,
        "deep_worst_acceptance_delta_pp": acc_worst,
        "median_pp_delta_pct": pp_med,
        "min_deep_median_gain_pct": args.min_deep_median_gain,
        "max_deep_loss_pct": args.max_deep_loss,
        "max_pp_loss_pct": args.max_pp_loss,
        "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
    }

    print("\nGATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    out = Path(args.result).with_suffix(".qsa-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if gate == "PASS" else 2)


if __name__ == "__main__":
    main()
