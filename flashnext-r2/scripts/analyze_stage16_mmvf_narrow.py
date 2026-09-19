#!/usr/bin/env python3
"""Gate the Stage-16 AMD narrow-MMVF candidate.

Input is the strict analyze_mtp_sweep.py report, so every arm has already passed:
  * fixed-length TG,
  * MTP drafted/accepted counters,
  * deterministic protected token prefix,
  * exact protected prefix vs the untouched Phase-3 anchor.

This layer answers the performance question the generic depth analyzer cannot:
  1) does patched-but-upstream-dispatch n=2 stay close to the untouched base?
  2) does enabling narrow MMVF materially help the discriminating n=4 path?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

WORKLOADS = ("zh", "code", "tool")


def med(xs):
    v = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def compare(models: dict, a: str, b: str) -> dict:
    aa, bb = models[a], models[b]
    tg = {w: pct(bb.get("tg", {}).get(w), aa.get("tg", {}).get(w)) for w in WORKLOADS}
    acc = {}
    for w in WORKLOADS:
        av = aa.get("acceptance", {}).get(w)
        bv = bb.get("acceptance", {}).get(w)
        acc[w] = (bv - av) * 100.0 if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
    vals = [tg[w] for w in WORKLOADS if isinstance(tg[w], (int, float))]
    accs = [acc[w] for w in WORKLOADS if isinstance(acc[w], (int, float))]
    return {
        "a": a,
        "b": b,
        "tg_delta_pct": tg,
        "median_tg_delta_pct": med(vals),
        "worst_tg_delta_pct": min(vals) if vals else None,
        "acceptance_delta_pp": acc,
        "worst_acceptance_delta_pp": min(accs) if accs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", help="analyze_mtp_sweep.py JSON")
    ap.add_argument("--base", default="qwen3.8-flash-next-r2-final-pre-sweep:256k")
    ap.add_argument("--off-n2", default="qwen3.8-flash-next-r2-mmvf-off-n2:256k")
    ap.add_argument("--on-n2", default="qwen3.8-flash-next-r2-mmvf-on-n2:256k")
    ap.add_argument("--off-n4", default="qwen3.8-flash-next-r2-mmvf-off-n4:256k")
    ap.add_argument("--on-n4", default="qwen3.8-flash-next-r2-mmvf-on-n4:256k")
    ap.add_argument("--min-n4-gain", type=float, default=3.0)
    ap.add_argument("--max-control-loss", type=float, default=2.0)
    ap.add_argument("--max-workload-loss", type=float, default=2.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    models = data.get("models") or {}
    required = [args.base, args.off_n2, args.on_n2, args.off_n4, args.on_n4]
    missing = [m for m in required if m not in models]
    if missing:
        raise SystemExit(f"ERROR Stage16 analysis missing models: {missing}")
    invalid = [m for m in required if models[m].get("valid") is not True]

    base_off2 = compare(models, args.base, args.off_n2)
    off2_on2 = compare(models, args.off_n2, args.on_n2)
    off4_on4 = compare(models, args.off_n4, args.on_n4)

    control_ok = (
        not invalid
        and isinstance(base_off2["median_tg_delta_pct"], (int, float))
        and base_off2["median_tg_delta_pct"] >= -args.max_control_loss
        and isinstance(base_off2["worst_tg_delta_pct"], (int, float))
        and base_off2["worst_tg_delta_pct"] >= -max(args.max_control_loss + 1.0, args.max_workload_loss)
        and isinstance(off2_on2["worst_tg_delta_pct"], (int, float))
        and off2_on2["worst_tg_delta_pct"] >= -args.max_workload_loss
        and isinstance(off2_on2["worst_acceptance_delta_pp"], (int, float))
        and off2_on2["worst_acceptance_delta_pp"] >= -args.max_acceptance_drop_pp
    )
    n4_ok = (
        not invalid
        and isinstance(off4_on4["median_tg_delta_pct"], (int, float))
        and off4_on4["median_tg_delta_pct"] >= args.min_n4_gain
        and isinstance(off4_on4["worst_tg_delta_pct"], (int, float))
        and off4_on4["worst_tg_delta_pct"] >= -args.max_workload_loss
        and isinstance(off4_on4["worst_acceptance_delta_pp"], (int, float))
        and off4_on4["worst_acceptance_delta_pp"] >= -args.max_acceptance_drop_pp
    )

    if invalid:
        verdict = "FAIL_CORRECTNESS"
        reason = f"strict MTP prefix/completeness gate rejected: {invalid}"
    elif not control_ok:
        verdict = "REJECT_CONTROL"
        reason = "patched n=2 control regressed vs the untouched Phase-3 baseline"
    elif not n4_ok:
        verdict = "REJECT_NO_MATERIAL_N4_GAIN"
        reason = "n=4 narrow-MMVF path did not provide a safe material TG gain"
    else:
        verdict = "PASS"
        reason = "n=2 control is safe and same-ELF narrow-MMVF materially improves n=4 multi-row decode"

    report = {
        "schema_version": 1,
        "strict_source_analysis": str(args.analysis),
        "invalid_models": invalid,
        "base_to_patched_off_n2": base_off2,
        "patched_off_to_on_n2": off2_on2,
        "patched_off_to_on_n4": off4_on4,
        "gate": {
            "result": verdict,
            "reason": reason,
            "control_ok": control_ok,
            "n4_ok": n4_ok,
            "min_n4_gain_pct": args.min_n4_gain,
            "max_control_loss_pct": args.max_control_loss,
            "max_workload_loss_pct": args.max_workload_loss,
            "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
            "winner_source_alias": args.on_n2 if verdict == "PASS" else args.base,
        },
    }

    print("STAGE16 MMVF NARROW")
    for label, cmp in (
        ("BASE->PATCHED_OFF_N2", base_off2),
        ("PATCHED_OFF_N2->ON_N2", off2_on2),
        ("PATCHED_OFF_N4->ON_N4", off4_on4),
    ):
        print(
            f"{label}: medianTG={cmp['median_tg_delta_pct']}% "
            f"worstTG={cmp['worst_tg_delta_pct']}% "
            f"worstAcc={cmp['worst_acceptance_delta_pp']}pp"
        )
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.out or Path(args.analysis).with_suffix(".stage16-analysis.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
