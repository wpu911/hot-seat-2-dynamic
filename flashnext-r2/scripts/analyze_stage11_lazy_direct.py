#!/usr/bin/env python3
"""Analyze --lazy-mode on vs on-direct for Flash Next Stage 11.

Stage 11 targets prompt processing / PLE row I/O. Decode must stay neutral and
deterministic. The same candidate binary is used for both arms.

PP is evaluated pairwise: each mmap/direct pair sees exactly the same prompt
windows, while the next pair uses a fresh set and reverses arm order. This makes
the gate much less vulnerable to OS page-cache warming than pooled medians.
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
    pp_prompt_n = {}
    for k in pp_keys:
        rows = []
        for leg in legs:
            rows += leg.get("pp", {}).get(k, [])
        pp[k] = med([row.get("pp") for row in rows])
        pp_prompt_n[k] = med([row.get("prompt_n") for row in rows])

    tg, outputs, tg_predicted_n = {}, {}, {}
    for workload in ("zh", "code", "tool"):
        vals, outs, counts = [], [], []
        for leg in legs:
            for row in leg.get("tg", []):
                if row.get("workload") == workload:
                    vals.append(row.get("tg"))
                    outs.append(row.get("content", ""))
                    counts.append(row.get("predicted_n"))
        tg[workload] = med(vals)
        outputs[workload] = outs
        tg_predicted_n[workload] = med(counts)
    return {
        "legs": len(legs),
        "pp": pp,
        "pp_prompt_n": pp_prompt_n,
        "tg": tg,
        "tg_predicted_n": tg_predicted_n,
        "outputs": outputs,
    }


def same(xs):
    return bool(xs) and len(set(xs)) == 1


def rows_by_variant(leg, target):
    rows = leg.get("pp", {}).get(str(target), [])
    out = {}
    for i, row in enumerate(rows):
        key = row.get("variant", i)
        out[int(key)] = row
    return out


def pairwise_pp(data, baseline, r2):
    """Return deltas matched by pair_id, target and prompt variant."""
    pairs = {}
    for leg in data.get("legs", []):
        pair_id = leg.get("pair_id")
        model = leg.get("model")
        if pair_id is None or model not in (baseline, r2):
            continue
        pairs.setdefault(int(pair_id), {})[model] = leg

    report = {}
    all_deltas = []
    for pair_id in sorted(pairs):
        arms = pairs[pair_id]
        if baseline not in arms or r2 not in arms:
            continue
        bleg, rleg = arms[baseline], arms[r2]
        targets = sorted(
            set(bleg.get("pp", {})) | set(rleg.get("pp", {})),
            key=lambda x: int(x),
        )
        p_out = {"order": [], "targets": {}}
        # Preserve actual order from result legs for diagnosing warm-cache bias.
        p_out["order"] = [
            x.get("model") for x in data.get("legs", []) if x.get("pair_id") == pair_id
        ]
        for target_s in targets:
            target = int(target_s)
            br = rows_by_variant(bleg, target)
            rr = rows_by_variant(rleg, target)
            variants = sorted(set(br) & set(rr))
            deltas = []
            rows = []
            for v in variants:
                bp, rp = br[v].get("pp"), rr[v].get("pp")
                d = pct(rp, bp)
                if d is not None:
                    deltas.append(d)
                rows.append({
                    "variant": v,
                    "baseline_pp": bp,
                    "r2_pp": rp,
                    "delta_pct": d,
                    "baseline_prompt_n": br[v].get("prompt_n"),
                    "r2_prompt_n": rr[v].get("prompt_n"),
                })
            tmed = med(deltas)
            if tmed is not None:
                all_deltas.append(tmed)
            p_out["targets"][target_s] = {
                "median_delta_pct": tmed,
                "variants": rows,
            }
        report[str(pair_id)] = p_out

    return {
        "pairs": report,
        "median_of_pair_target_deltas_pct": med(all_deltas),
        "worst_pair_target_delta_pct": min(all_deltas) if all_deltas else None,
        "n_pair_target_cells": len(all_deltas),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--min-median-pp-gain", type=float, default=5.0)
    ap.add_argument("--max-pp-loss", type=float, default=3.0)
    ap.add_argument("--max-tg-loss", type=float, default=2.0)
    ap.add_argument("--min-tg-predicted", type=int, default=32)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    b = collect(data, args.baseline)
    r = collect(data, args.r2)

    # Pooled values remain useful diagnostics, but promotion is based on matched
    # pair/prompt deltas below.
    pp_delta_pooled = {k: pct(r["pp"].get(k), b["pp"].get(k))
                       for k in sorted(set(b["pp"]) | set(r["pp"]), key=lambda x: int(x))}
    paired = pairwise_pp(data, args.baseline, args.r2)
    pp_med = paired["median_of_pair_target_deltas_pct"]
    pp_worst = paired["worst_pair_target_delta_pct"]

    tg_delta = {w: pct(r["tg"].get(w), b["tg"].get(w)) for w in ("zh", "code", "tool")}
    tg_vals = [x for x in tg_delta.values() if isinstance(x, (int, float))]
    tg_worst = min(tg_vals) if tg_vals else None

    exact = {}
    for w in ("zh", "code", "tool"):
        bo, ro = b["outputs"].get(w, []), r["outputs"].get(w, [])
        exact[w] = same(bo) and same(ro) and bool(bo) and bo[0] == ro[0]

    corpus = data.get("corpus", {}) or {}
    worst_div = corpus.get("worst_ngram4_unique_ratio")
    min_div = corpus.get("min_required_ngram4_unique_ratio")
    diversity_ok = (
        isinstance(worst_div, (int, float)) and
        isinstance(min_div, (int, float)) and
        worst_div >= min_div
    )

    tg_coverage = {}
    for w in ("zh", "code", "tool"):
        bn = b["tg_predicted_n"].get(w)
        rn = r["tg_predicted_n"].get(w)
        tg_coverage[w] = (
            isinstance(bn, (int, float)) and bn >= args.min_tg_predicted and
            isinstance(rn, (int, float)) and rn >= args.min_tg_predicted
        )

    # Paired design must actually be present. Falling back to pooled medians would
    # silently reintroduce the page-cache order bias this stage is meant to avoid.
    pair_design_ok = paired["n_pair_target_cells"] > 0 and len(paired["pairs"]) >= 2
    reversed_order_seen = False
    orders = [tuple(x.get("order", [])) for x in paired["pairs"].values()]
    if (args.baseline, args.r2) in orders and (args.r2, args.baseline) in orders:
        reversed_order_seen = True
    pair_design_ok = pair_design_ok and reversed_order_seen

    pass_exact = all(exact.values())
    pass_diversity = diversity_ok
    pass_tg_coverage = all(tg_coverage.values())
    pass_pp_gain = pp_med is not None and pp_med >= args.min_median_pp_gain
    pass_pp_worst = pp_worst is not None and pp_worst >= -args.max_pp_loss
    pass_tg = tg_worst is not None and tg_worst >= -args.max_tg_loss
    result = "PASS" if (
        pair_design_ok and pass_exact and pass_diversity and pass_tg_coverage and
        pass_pp_gain and pass_pp_worst and pass_tg
    ) else "FAIL"

    report = {
        "baseline": args.baseline,
        "r2": args.r2,
        "corpus": corpus,
        "baseline_metrics": {k: v for k, v in b.items() if k != "outputs"},
        "r2_metrics": {k: v for k, v in r.items() if k != "outputs"},
        "delta_pct": {"pp_pooled": pp_delta_pooled, "tg": tg_delta},
        "pairwise_pp": paired,
        "bit_exact": exact,
        "tg_coverage": tg_coverage,
        "gate": {
            "result": result,
            "pair_design_ok": pair_design_ok,
            "reversed_order_seen": reversed_order_seen,
            "diversity_ok": diversity_ok,
            "worst_ngram4_unique_ratio": worst_div,
            "required_ngram4_unique_ratio": min_div,
            "tg_coverage_ok": pass_tg_coverage,
            "min_tg_predicted": args.min_tg_predicted,
            "median_pairwise_pp_gain_pct": pp_med,
            "worst_pairwise_pp_delta_pct": pp_worst,
            "worst_tg_delta_pct": tg_worst,
            "min_median_pp_gain_pct": args.min_median_pp_gain,
            "max_pp_loss_pct": args.max_pp_loss,
            "max_tg_loss_pct": args.max_tg_loss,
        },
    }

    print("CORPUS")
    print(f"  source={corpus.get('source')} tokens={corpus.get('token_count')} "
          f"ngram4_worst={worst_div} required={min_div} ok={diversity_ok}")
    print("PAIRWISE PP")
    for pair_id, pr in paired["pairs"].items():
        print(f"  pair={pair_id} order={' -> '.join(pr['order'])}")
        for target, tr in pr["targets"].items():
            print(f"    {target:>6}: median delta={tr['median_delta_pct']}%")
    print(f"  aggregate median={pp_med}% worst={pp_worst}% pair_design_ok={pair_design_ok}")
    print("TG")
    for w in ("zh", "code", "tool"):
        print(f"  {w:>6}: {b['tg'].get(w)} -> {r['tg'].get(w)}  delta={tg_delta[w]}%  "
              f"exact={exact[w]} coverage={tg_coverage[w]}")
    print("GATE")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))

    out = Path(args.result).with_suffix(".lazy-direct-analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
