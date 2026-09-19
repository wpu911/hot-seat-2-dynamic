#!/usr/bin/env python3
"""Analyze Stage-15 BASE -> patched-OFF -> patched-ON long-context A/B.

The distinction matters because PR #26419 changes both the AMD WMMA DKQ>128
kernel implementation and the RDNA4 dispatch. BASE->OFF checks the former does
not regress the real mixed gfx1100+gfx1201 workload. OFF->ON then isolates the
new R9700 route using the same patched ELF.

The benchmark input is bench_tensor_ratio_sweep.py because that runner already
provides the useful balanced A B C C B A load order, fixed-length TG, retrieval
needle, cold model-specific unload and MTP acceptance checks. Its name is
historical; the schema is perfectly adequate for a three-arm kernel experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from analyze_tensor_ratio_sweep import collect


def med(xs):
    vals = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(vals) if vals else None


def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def compare(a: dict, b: dict, deep_from: int) -> dict:
    depths = sorted(set(a.get("depths", {})) | set(b.get("depths", {})))
    rows = {}
    all_pp = []
    deep_pp = []
    all_tg = []
    acc_pp = []
    complete = bool(depths) and a.get("complete") and b.get("complete")
    for d in depths:
        aa = a.get("depths", {}).get(d, {})
        bb = b.get("depths", {}).get(d, {})
        pp = pct(bb.get("pp"), aa.get("pp"))
        tg = pct(bb.get("tg"), aa.get("tg"))
        av, bv = aa.get("acceptance"), bb.get("acceptance")
        acc = (bv - av) * 100.0 if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
        rows[str(d)] = {
            "pp_delta_pct": pp,
            "tg_delta_pct": tg,
            "acceptance_delta_pp": acc,
            "a": aa,
            "b": bb,
        }
        if pp is not None:
            all_pp.append(pp)
            if d >= deep_from:
                deep_pp.append(pp)
        if tg is not None:
            all_tg.append(tg)
        if acc is not None:
            acc_pp.append(acc)
        complete &= bool(aa) and bool(bb)
    return {
        "complete": bool(complete),
        "depths": rows,
        "median_pp_delta_pct": med(all_pp),
        "deep_median_pp_delta_pct": med(deep_pp),
        "worst_pp_delta_pct": min(all_pp) if all_pp else None,
        "median_tg_delta_pct": med(all_tg),
        "worst_tg_delta_pct": min(all_tg) if all_tg else None,
        "worst_acceptance_delta_pp": min(acc_pp) if acc_pp else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--base", default="qwen3.8-flash-next-r2-modern-mtp:256k")
    ap.add_argument("--off", default="qwen3.8-flash-next-r2-rdna4-fa256-off:256k")
    ap.add_argument("--on", default="qwen3.8-flash-next-r2-rdna4-fa256-on:256k")
    ap.add_argument("--deep-from", type=int, default=32768)
    ap.add_argument("--min-route-deep-pp-gain", type=float, default=3.0)
    ap.add_argument("--min-net-deep-pp-gain", type=float, default=2.0)
    ap.add_argument("--max-carryover-pp-loss", type=float, default=4.0)
    ap.add_argument("--max-route-pp-loss", type=float, default=2.0)
    ap.add_argument("--max-tg-loss", type=float, default=2.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    models = data.get("models") or []
    required = [args.base, args.off, args.on]
    if data.get("benchmark") != "tensor_ratio_sweep" or data.get("ignore_eos") is not True:
        raise SystemExit("ERROR unsupported Stage-15 benchmark schema")
    if any(m not in models for m in required):
        raise SystemExit(f"ERROR Stage-15 result missing one of required arms: {required}")

    stats = {m: collect(data, m) for m in required}
    base_off = compare(stats[args.base], stats[args.off], args.deep_from)
    off_on = compare(stats[args.off], stats[args.on], args.deep_from)
    base_on = compare(stats[args.base], stats[args.on], args.deep_from)

    all_complete = all(stats[m].get("complete") for m in required)
    carryover_ok = (
        all_complete
        and base_off["complete"]
        and isinstance(base_off["worst_pp_delta_pct"], (int, float))
        and base_off["worst_pp_delta_pct"] >= -args.max_carryover_pp_loss
        and isinstance(base_off["worst_tg_delta_pct"], (int, float))
        and base_off["worst_tg_delta_pct"] >= -args.max_tg_loss
        and isinstance(base_off["worst_acceptance_delta_pp"], (int, float))
        and base_off["worst_acceptance_delta_pp"] >= -args.max_acceptance_drop_pp
    )
    route_ok = (
        all_complete
        and off_on["complete"]
        and isinstance(off_on["deep_median_pp_delta_pct"], (int, float))
        and off_on["deep_median_pp_delta_pct"] >= args.min_route_deep_pp_gain
        and isinstance(off_on["worst_pp_delta_pct"], (int, float))
        and off_on["worst_pp_delta_pct"] >= -args.max_route_pp_loss
        and isinstance(off_on["worst_tg_delta_pct"], (int, float))
        and off_on["worst_tg_delta_pct"] >= -args.max_tg_loss
        and isinstance(off_on["worst_acceptance_delta_pp"], (int, float))
        and off_on["worst_acceptance_delta_pp"] >= -args.max_acceptance_drop_pp
    )
    net_ok = (
        isinstance(base_on["deep_median_pp_delta_pct"], (int, float))
        and base_on["deep_median_pp_delta_pct"] >= args.min_net_deep_pp_gain
        and isinstance(base_on["worst_tg_delta_pct"], (int, float))
        and base_on["worst_tg_delta_pct"] >= -args.max_tg_loss
        and isinstance(base_on["worst_acceptance_delta_pp"], (int, float))
        and base_on["worst_acceptance_delta_pp"] >= -args.max_acceptance_drop_pp
    )

    if not all_complete:
        verdict = "FAIL_CORRECTNESS"
        reason = "one or more arms failed retrieval/fixed-length/MTP completeness"
    elif not carryover_ok:
        verdict = "REJECT_KERNEL_CARRYOVER"
        reason = "patched binary regressed with the new RDNA4 route disabled"
    elif not route_ok:
        verdict = "REJECT_ROUTE"
        reason = "RDNA4 FA256 route did not provide a safe material deep-prefill gain"
    elif not net_ok:
        verdict = "REJECT_NET"
        reason = "route beat patched-OFF but did not beat the untouched base enough overall"
    else:
        verdict = "PASS"
        reason = "patched kernel carryover is safe and RDNA4 FA256 materially improves deep prefill without TG/MTP regression"

    report = {
        "schema_version": 1,
        "arms": {m: stats[m] for m in required},
        "base_to_patched_off": base_off,
        "patched_off_to_on": off_on,
        "base_to_on": base_on,
        "gate": {
            "result": verdict,
            "reason": reason,
            "carryover_ok": carryover_ok,
            "route_ok": route_ok,
            "net_ok": net_ok,
            "deep_from": args.deep_from,
            "min_route_deep_pp_gain_pct": args.min_route_deep_pp_gain,
            "min_net_deep_pp_gain_pct": args.min_net_deep_pp_gain,
            "max_carryover_pp_loss_pct": args.max_carryover_pp_loss,
            "max_route_pp_loss_pct": args.max_route_pp_loss,
            "max_tg_loss_pct": args.max_tg_loss,
            "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
            "winner_alias": args.on if verdict == "PASS" else args.base,
        },
    }

    print("STAGE15 RDNA4 FA256")
    for label, cmp in (
        ("BASE->PATCHED_OFF", base_off),
        ("PATCHED_OFF->ON", off_on),
        ("BASE->ON", base_on),
    ):
        print(
            f"{label}: deepPP={cmp.get('deep_median_pp_delta_pct')}% "
            f"worstPP={cmp.get('worst_pp_delta_pct')}% "
            f"TGmed={cmp.get('median_tg_delta_pct')}% worstTG={cmp.get('worst_tg_delta_pct')}% "
            f"worstAcc={cmp.get('worst_acceptance_delta_pp')}pp complete={cmp.get('complete')}"
        )
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.out or Path(args.result).with_suffix(".stage15-analysis.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
