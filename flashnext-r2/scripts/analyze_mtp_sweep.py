#!/usr/bin/env python3
"""Summarize MTP n-max 2/3/4 Flash Next sweep results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def load_rows(paths):
    by_model = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for leg in data.get("legs", []):
            m = leg.get("model")
            slot = by_model.setdefault(m, {"tg": {}, "outputs": {}, "accept": {}})
            for row in leg.get("tg", []):
                w = row.get("workload")
                slot["tg"].setdefault(w, []).append(row.get("tg"))
                slot["outputs"].setdefault(w, []).append(row.get("content", ""))
                d, a = row.get("drafted"), row.get("accepted")
                if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                    slot["accept"].setdefault(w, []).append(a / d)
    return by_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-mtp-sweep.analysis.json")
    args = ap.parse_args()

    by_model = load_rows(args.results)
    report = {}
    reference_outputs = {}
    for model, v in by_model.items():
        tg = {w: med(v["tg"].get(w, [])) for w in ("zh", "code", "tool")}
        acc = {w: med(v["accept"].get(w, [])) for w in ("zh", "code", "tool")}
        exact_internal = {w: bool(v["outputs"].get(w)) and len(set(v["outputs"].get(w, []))) == 1 for w in ("zh", "code", "tool")}
        report[model] = {"tg": tg, "acceptance": acc, "deterministic_within_model": exact_internal}
        for w in ("zh", "code", "tool"):
            outs = v["outputs"].get(w, [])
            if outs:
                reference_outputs.setdefault(w, outs[0])

    # Cross-model exactness at temperature=0.
    for model, v in by_model.items():
        cross = {}
        for w in ("zh", "code", "tool"):
            outs = v["outputs"].get(w, [])
            cross[w] = bool(outs) and all(x == reference_outputs.get(w) for x in outs)
        report[model]["exact_vs_sweep"] = cross

    # Rank by median of the three workload TG medians, but only exact models qualify.
    ranked = []
    for model, r in report.items():
        exact = all(r["exact_vs_sweep"].values()) and all(r["deterministic_within_model"].values())
        score = med(list(r["tg"].values()))
        ranked.append({"model": model, "median_tg": score, "exact": exact})
    ranked.sort(key=lambda x: (-1 if x["median_tg"] is None else -x["median_tg"]))

    result = {"models": report, "ranking": ranked}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MTP SWEEP")
    for x in ranked:
        r = report[x["model"]]
        print(f"{x['model']}: medianTG={x['median_tg']} exact={x['exact']} acceptance={r['acceptance']}")
    print(f"ANALYSIS={args.out}")

    # Fail if any candidate is not exact; speculative depth must not change final greedy output.
    raise SystemExit(0 if all(x["exact"] for x in ranked) else 2)


if __name__ == "__main__":
    main()
