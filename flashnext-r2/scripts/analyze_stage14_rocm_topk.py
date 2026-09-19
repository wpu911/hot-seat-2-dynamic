#!/usr/bin/env python3
"""Analyze Stage-14 ROCm TOP_K long-context A/B results.

Two independent llama-swap ladders are expected:
  1. graphs ON  : modern-MTP base vs ROCm TOP_K candidate
  2. graphs OFF : same two engines with GGML_CUDA_DISABLE_GRAPHS=1

Verdicts:
  PASS                  normal graph-on production path wins and correctness holds
  HIP_GRAPH_INTERACTION graph-off wins materially but graph-on does not
  FAIL                  no material safe end-to-end win

Current Stage-14 data must come from the fixed-length schema-3 ladder. Missing
MTP acceptance, short generation, or needle failure is a hard correctness failure.
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


def schema_ok(data):
    return (
        isinstance(data.get("schema_version"), int)
        and data.get("schema_version") >= 3
        and data.get("ignore_eos") is True
        and isinstance(data.get("requested_predict"), (int, float))
        and data.get("requested_predict") > 0
    )


def collect(data, model):
    rows = {}
    requested = data.get("requested_predict")
    for leg in data.get("legs", []):
        if leg.get("model") != model:
            continue
        for row in leg.get("rows", []):
            d = int(row["target_depth"])
            rows.setdefault(d, []).append(row)

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
            "samples": len(rr),
            "prompt_n": med([x.get("prompt_n") for x in rr]),
            "pp": med([x.get("pp") for x in rr]),
            "tg": med([x.get("tg") for x in rr]),
            "acceptance": med(acc),
            "acceptance_samples": len(acc),
            "marker_ok": all(bool(x.get("marker_hit")) for x in rr),
            "full_generation": full,
        }
    return out


def compare(data, baseline, candidate, deep_from):
    b = collect(data, baseline)
    c = collect(data, candidate)
    depths = sorted(set(b) | set(c))
    deltas = {}
    deep_tg = []
    deep_acc_pp = []
    pp_all = []
    marker_ok = bool(depths)
    full_ok = bool(depths)
    acceptance_ok = bool(depths)
    samples_ok = bool(depths)

    for d in depths:
        bd, cd = b.get(d, {}), c.get(d, {})
        tg = pct(cd.get("tg"), bd.get("tg"))
        pp = pct(cd.get("pp"), bd.get("pp"))
        ba, ca = bd.get("acceptance"), cd.get("acceptance")
        acc_pp = None if ba is None or ca is None else (ca - ba) * 100.0
        mok = bool(bd.get("marker_ok")) and bool(cd.get("marker_ok"))
        fok = bool(bd.get("full_generation")) and bool(cd.get("full_generation"))
        aok = (
            isinstance(ba, (int, float)) and isinstance(ca, (int, float))
            and bd.get("acceptance_samples", 0) == bd.get("samples", 0)
            and cd.get("acceptance_samples", 0) == cd.get("samples", 0)
        )
        sok = bd.get("samples", 0) > 0 and cd.get("samples", 0) > 0
        marker_ok &= mok
        full_ok &= fok
        acceptance_ok &= aok
        samples_ok &= sok
        deltas[str(d)] = {
            "tg_pct": tg,
            "pp_pct": pp,
            "acceptance_delta_pp": acc_pp,
            "marker_ok": mok,
            "full_generation": fok,
            "acceptance_complete": aok,
        }
        if d >= deep_from:
            if tg is not None:
                deep_tg.append(tg)
            if acc_pp is not None:
                deep_acc_pp.append(acc_pp)
        if pp is not None:
            pp_all.append(pp)

    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_metrics": b,
        "candidate_metrics": c,
        "delta": deltas,
        "summary": {
            "schema_ok": schema_ok(data),
            "samples_ok": samples_ok,
            "marker_ok": marker_ok,
            "full_generation_ok": full_ok,
            "acceptance_complete": acceptance_ok,
            "deep_median_tg_gain_pct": med(deep_tg),
            "deep_worst_tg_delta_pct": min(deep_tg) if deep_tg else None,
            "median_pp_delta_pct": med(pp_all),
            "deep_worst_acceptance_delta_pp": min(deep_acc_pp) if deep_acc_pp else None,
        },
    }


def candidate_graph_effect(on_cmp, off_cmp):
    on = on_cmp["candidate_metrics"]
    off = off_cmp["candidate_metrics"]
    out = {}
    for d in sorted(set(on) & set(off)):
        out[str(d)] = pct(on[d].get("tg"), off[d].get("tg"))
    return out


def print_cmp(label, cmp):
    print(f"\n=== {label} ===")
    print("depth     PP base -> cand        TG base -> cand        TG delta    acc dPP   marker full")
    for ds, dlt in cmp["delta"].items():
        d = int(ds)
        b = cmp["baseline_metrics"].get(d, {})
        c = cmp["candidate_metrics"].get(d, {})
        print(
            f"{d:>7}  {str(b.get('pp')):>9} -> {str(c.get('pp')):<9}  "
            f"{str(b.get('tg')):>8} -> {str(c.get('tg')):<8}  "
            f"{str(dlt.get('tg_pct')):>9}  {str(dlt.get('acceptance_delta_pp')):>8}  "
            f"{dlt.get('marker_ok')} {dlt.get('full_generation')}"
        )
    print(json.dumps(cmp["summary"], ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="graphs-ON ladder JSON")
    ap.add_argument("--off", required=True, help="graphs-OFF ladder JSON")
    ap.add_argument("--base-on", required=True)
    ap.add_argument("--r2-on", required=True)
    ap.add_argument("--base-off", required=True)
    ap.add_argument("--r2-off", required=True)
    ap.add_argument("--deep-from", type=int, default=32768)
    ap.add_argument("--min-on-gain", type=float, default=2.0)
    ap.add_argument("--min-off-gain-for-interaction", type=float, default=3.0)
    ap.add_argument("--max-deep-loss", type=float, default=2.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=3.0)
    ap.add_argument("--max-pp-loss", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    on_data = json.loads(Path(args.on).read_text(encoding="utf-8"))
    off_data = json.loads(Path(args.off).read_text(encoding="utf-8"))
    on_cmp = compare(on_data, args.base_on, args.r2_on, args.deep_from)
    off_cmp = compare(off_data, args.base_off, args.r2_off, args.deep_from)

    print_cmp("HIP graphs ON", on_cmp)
    print_cmp("HIP graphs OFF", off_cmp)

    def correctness_ok(cmp):
        s = cmp["summary"]
        acc = s["deep_worst_acceptance_delta_pp"]
        return (
            s["schema_ok"]
            and s["samples_ok"]
            and s["marker_ok"]
            and s["full_generation_ok"]
            and s["acceptance_complete"]
            and s["deep_worst_tg_delta_pct"] is not None
            and s["deep_worst_tg_delta_pct"] >= -args.max_deep_loss
            and s["median_pp_delta_pct"] is not None
            and s["median_pp_delta_pct"] >= -args.max_pp_loss
            and acc is not None
            and acc >= -args.max_acceptance_drop_pp
        )

    on_s = on_cmp["summary"]
    off_s = off_cmp["summary"]
    on_gain = on_s["deep_median_tg_gain_pct"]
    off_gain = off_s["deep_median_tg_gain_pct"]

    on_pass = correctness_ok(on_cmp) and on_gain is not None and on_gain >= args.min_on_gain
    off_pass = correctness_ok(off_cmp) and off_gain is not None and off_gain >= args.min_off_gain_for_interaction

    if on_pass:
        verdict = "PASS"
        reason = "ROCm TOP_K improves the normal graph-ON production path."
    elif off_pass and on_s["schema_ok"] and on_s["marker_ok"] and on_s["full_generation_ok"]:
        verdict = "HIP_GRAPH_INTERACTION"
        reason = (
            "ROCm TOP_K wins with HIP graphs disabled but not enough with graphs enabled; "
            "profile/repair graph interaction before deciding whether to promote the kernel."
        )
    else:
        verdict = "FAIL"
        reason = "No safe material end-to-end long-context win under the Stage-14 gates."

    graph_effect = candidate_graph_effect(on_cmp, off_cmp)
    report = {
        "graphs_on": on_cmp,
        "graphs_off": off_cmp,
        "candidate_graph_on_vs_off_tg_pct": graph_effect,
        "gate": {
            "result": verdict,
            "reason": reason,
            "deep_from": args.deep_from,
            "graphs_on_deep_median_tg_gain_pct": on_gain,
            "graphs_off_deep_median_tg_gain_pct": off_gain,
            "min_graphs_on_gain_pct": args.min_on_gain,
            "min_graphs_off_gain_for_interaction_pct": args.min_off_gain_for_interaction,
            "max_deep_loss_pct": args.max_deep_loss,
            "max_acceptance_drop_pp": args.max_acceptance_drop_pp,
            "max_pp_loss_pct": args.max_pp_loss,
        },
    }

    print("\n=== candidate graph ON vs OFF diagnostic ===")
    print(json.dumps(graph_effect, ensure_ascii=False, indent=2))
    print("\nGATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.out) if args.out else Path(args.on).with_suffix(".stage14-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")

    if verdict == "PASS":
        raise SystemExit(0)
    if verdict == "HIP_GRAPH_INTERACTION":
        raise SystemExit(3)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
