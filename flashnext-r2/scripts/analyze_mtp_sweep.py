#!/usr/bin/env python3
"""Analyze a Flash Next MTP draft-depth sweep.

This accepts any number of result JSON files and candidate aliases. It is strict
about benchmark validity: fixed-length TG, complete generations and MTP
draft/accept counters are required for every measured TG sample.

Speculative correctness is checked on a protected generated-token prefix rather
than on an entire long string. That is deliberate: late greedy divergence can
come from backend reduction ties even with the same arm, while an early prefix
mismatch is strong evidence that changing draft depth changed semantics.

A bad arm is rejected instead of poisoning the whole sweep. The anchor arm must
remain valid. A faster arm is recommended only when its median gain over the
anchor clears --min-gain; otherwise the simpler anchor is retained.
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


def load_rows(paths):
    by_model = {}
    source_meta = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        source_meta.append({
            "path": str(p),
            "schema_version": data.get("schema_version"),
            "fixed_tg": data.get("fixed_tg"),
            "requested_tg": data.get("requested_tg"),
            "sequence": data.get("sequence"),
            "output_tokens_recorded": data.get("output_tokens_recorded"),
        })
        if data.get("schema_version") != 3 or data.get("fixed_tg") is not True:
            raise SystemExit(f"ERROR stale/unsafe benchmark schema in {p}: require schema_version=3 fixed_tg=true")
        if data.get("output_tokens_recorded") is not True:
            raise SystemExit(f"ERROR {p} lacks tokenized output; rerun with current bench_mtp_depth_sweep.py")
        requested = data.get("requested_tg")
        if not isinstance(requested, int) or requested <= 0:
            raise SystemExit(f"ERROR invalid requested_tg in {p}: {requested!r}")

        for leg in data.get("legs", []):
            m = leg.get("model")
            if not isinstance(m, str) or not m:
                continue
            slot = by_model.setdefault(m, {
                "tg": {}, "tokens": {}, "accept": {}, "full": {}, "mtp_complete": {},
                "leg_count": 0,
            })
            slot["leg_count"] += 1
            for row in leg.get("tg", []):
                w = row.get("workload")
                if w not in WORKLOADS:
                    continue
                slot["tg"].setdefault(w, []).append(row.get("tg"))
                ids = row.get("output_tokens")
                slot["tokens"].setdefault(w, []).append(ids if isinstance(ids, list) else None)
                pred = row.get("predicted_n")
                full = bool(row.get("full_generation")) and isinstance(pred, (int, float)) and pred >= requested
                slot["full"].setdefault(w, []).append(full)
                d, a = row.get("drafted"), row.get("accepted")
                complete = isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)) and 0 <= a <= d
                slot["mtp_complete"].setdefault(w, []).append(complete)
                if complete:
                    slot["accept"].setdefault(w, []).append(a / d)
    return by_model, source_meta


def prefix_tuple(ids, n):
    if not isinstance(ids, list) or len(ids) < n or not all(isinstance(x, int) for x in ids[:n]):
        return None
    return tuple(ids[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--anchor", default="qwen3.8-flash-next-r2-final-mtp2:256k")
    ap.add_argument("--min-gain", type=float, default=1.0,
                    help="minimum median TG gain over anchor before changing draft depth")
    ap.add_argument("--exact-prefix-tokens", type=int, default=128,
                    help="generated token prefix that must match anchor across all runs")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-mtp-sweep.analysis.json")
    args = ap.parse_args()
    if args.exact_prefix_tokens < 32:
        raise SystemExit("ERROR --exact-prefix-tokens must be >= 32")

    by_model, source_meta = load_rows(args.results)
    if args.anchor not in by_model:
        raise SystemExit(f"ERROR anchor model absent from sweep: {args.anchor}")

    # The anchor is the correctness reference, not whichever alias happened to
    # run first. Require its own runs to agree on the protected prefix first.
    ref_prefix = {}
    av = by_model[args.anchor]
    for w in WORKLOADS:
        prefixes = [prefix_tuple(x, args.exact_prefix_tokens) for x in av["tokens"].get(w, [])]
        prefixes = [x for x in prefixes if x is not None]
        if prefixes:
            ref_prefix[w] = prefixes[0]

    report = {}
    rejected = []
    for model, v in by_model.items():
        tg = {w: med(v["tg"].get(w, [])) for w in WORKLOADS}
        acc = {w: med(v["accept"].get(w, [])) for w in WORKLOADS}
        sample_counts = {w: len(v["tg"].get(w, [])) for w in WORKLOADS}
        token_prefixes = {
            w: [prefix_tuple(x, args.exact_prefix_tokens) for x in v["tokens"].get(w, [])]
            for w in WORKLOADS
        }
        token_prefix_complete = {
            w: bool(token_prefixes[w]) and all(x is not None for x in token_prefixes[w])
            for w in WORKLOADS
        }
        deterministic = {
            w: token_prefix_complete[w] and len(set(token_prefixes[w])) == 1
            for w in WORKLOADS
        }
        exact_vs_anchor = {
            w: token_prefix_complete[w] and w in ref_prefix and all(x == ref_prefix[w] for x in token_prefixes[w])
            for w in WORKLOADS
        }
        full_generation = {
            w: bool(v["full"].get(w)) and all(v["full"].get(w, []))
            for w in WORKLOADS
        }
        mtp_complete = {
            w: bool(v["mtp_complete"].get(w)) and all(v["mtp_complete"].get(w, []))
            for w in WORKLOADS
        }
        coverage = all(sample_counts[w] > 0 for w in WORKLOADS)
        valid = (
            coverage
            and all(token_prefix_complete.values())
            and all(deterministic.values())
            and all(exact_vs_anchor.values())
            and all(full_generation.values())
            and all(mtp_complete.values())
            and all(isinstance(tg[w], (int, float)) for w in WORKLOADS)
        )
        score = med(list(tg.values()))
        report[model] = {
            "tg": tg,
            "acceptance": acc,
            "sample_counts": sample_counts,
            "leg_count": v["leg_count"],
            "exact_prefix_tokens": args.exact_prefix_tokens,
            "token_prefix_complete": token_prefix_complete,
            "deterministic_prefix_within_model": deterministic,
            "exact_prefix_vs_anchor": exact_vs_anchor,
            "full_generation": full_generation,
            "mtp_counters_complete": mtp_complete,
            "median_tg": score,
            "valid": valid,
        }
        if not valid:
            rejected.append(model)

    anchor = report[args.anchor]
    if not anchor["valid"]:
        raise SystemExit("ERROR anchor MTP arm failed correctness/completeness gate; sweep cannot be trusted")

    anchor_score = anchor["median_tg"]
    ranked = []
    for model, r in report.items():
        gain = pct(r["median_tg"], anchor_score)
        r["gain_vs_anchor_pct"] = gain
        ranked.append({
            "model": model,
            "median_tg": r["median_tg"],
            "gain_vs_anchor_pct": gain,
            "valid": r["valid"],
        })
    ranked.sort(key=lambda x: (
        0 if x["valid"] else 1,
        float("inf") if x["median_tg"] is None else -x["median_tg"],
    ))

    valid_ranked = [x for x in ranked if x["valid"]]
    if not valid_ranked:
        raise SystemExit("ERROR no valid MTP depth arm")
    best = valid_ranked[0]
    best_gain = best["gain_vs_anchor_pct"]
    if best["model"] != args.anchor and isinstance(best_gain, (int, float)) and best_gain >= args.min_gain:
        recommended = best["model"]
        reason = f"best valid arm clears anchor by {best_gain:.3f}% >= {args.min_gain:.3f}%"
    else:
        recommended = args.anchor
        reason = "best valid arm does not clear the minimum gain threshold; retain anchor"

    result = {
        "sources": source_meta,
        "anchor": args.anchor,
        "min_gain_pct": args.min_gain,
        "exact_prefix_tokens": args.exact_prefix_tokens,
        "models": report,
        "ranking": ranked,
        "rejected_models": rejected,
        "recommended_model": recommended,
        "recommendation_reason": reason,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MTP DEPTH SWEEP")
    for x in ranked:
        r = report[x["model"]]
        print(
            f"{x['model']}: medianTG={x['median_tg']} gain_vs_anchor={x['gain_vs_anchor_pct']}% "
            f"valid={x['valid']} acceptance={r['acceptance']}"
        )
    print(f"RECOMMENDED_MODEL={recommended}")
    print(f"REASON={reason}")
    print(f"ANALYSIS={args.out}")


if __name__ == "__main__":
    main()
