#!/usr/bin/env python3
"""Cached Large-PP / high-LCP MTP regression gate.

This targets the historical failure mode where fresh decode looked normal but a
cached high-LCP branch crossed the Large-PP ownership boundary and speculative
multi-row verification collapsed. TG is fixed-length; current llama-server MTP
counters are read from timings.draft_n / timings.draft_n_accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from bench_llamaswap_ab import http_json, unload_model, exact_prompt, speculative_counts

URL = "http://127.0.0.1:8090"
BASE = "qwen3.8-flash-next-r2-modern-foundation:256k"
R2 = "qwen3.8-flash-next-r2-modern-mtp:256k"


def pick(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def completion(base_url: str, model: str, prompt: str, n_predict: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": True,
        "stream": False,
        "ignore_eos": True,
    }
    t0 = time.time()
    r = http_json("POST", base_url + "/completion", payload, timeout=3600)
    wall = time.time() - t0
    if not isinstance(r, dict):
        raise RuntimeError(f"unexpected completion response: {type(r)}")
    timings = r.get("timings", {}) or {}
    drafted, accepted = speculative_counts(r, timings)
    cache_n = pick(timings, ("tokens_cached", "n_cached", "cache_n", "prompt_cached_n"))
    if cache_n is None:
        cache_n = pick(r, ("tokens_cached", "n_cached", "cache_n", "prompt_cached_n"))
    predicted_n = timings.get("predicted_n")
    full_generation = isinstance(predicted_n, (int, float)) and predicted_n >= n_predict
    if not full_generation:
        raise RuntimeError(
            f"{model}: cached TG requested {n_predict} tokens but server reported predicted_n={predicted_n}; "
            "refusing to trust a short early-EOS timing"
        )
    mtp_valid = isinstance(drafted, (int, float)) and drafted > 0 and isinstance(accepted, (int, float))
    return {
        "wall_s": wall,
        "prompt_n": timings.get("prompt_n"),
        "pp": timings.get("prompt_per_second"),
        "predicted_n": predicted_n,
        "tg": timings.get("predicted_per_second"),
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": (accepted / drafted) if mtp_valid else None,
        "mtp_valid": mtp_valid,
        "cache_n": cache_n,
        "full_generation": full_generation,
        "content": r.get("content", ""),
    }


def suffix_text(base_url: str, model: str, target_tokens: int) -> str:
    seed = "\n新增核对项：请继续检查缓存、专家驻留、MTP 回滚与下一段数据之间是否一致。"
    text = seed * max(8, target_tokens // 8)
    try:
        tok = http_json("POST", base_url + "/tokenize", {"model": model, "content": text}, timeout=600)
        ids = tok.get("tokens", [])
        if len(ids) < target_tokens:
            text *= 4
            tok = http_json("POST", base_url + "/tokenize", {"model": model, "content": text}, timeout=600)
            ids = tok.get("tokens", [])
        det = http_json("POST", base_url + "/detokenize", {"model": model, "tokens": ids[:target_tokens]}, timeout=600)
        return det.get("content", text)
    except Exception:
        return text[: target_tokens * 5]


def run_model(base_url: str, model: str, prefix_tokens: int, suffix_tokens: int, n_predict: int, force: bool):
    unload_model(base_url, model, force)
    time.sleep(2)
    prefix = exact_prompt(base_url, model, prefix_tokens)
    suffix = suffix_text(base_url, model, suffix_tokens)
    fresh_predict = max(64, min(128, n_predict))
    fresh = completion(base_url, model, prefix, fresh_predict)
    branch = completion(base_url, model, prefix + suffix, n_predict)
    result = {
        "model": model,
        "prefix_tokens_target": prefix_tokens,
        "suffix_tokens_target": suffix_tokens,
        "fresh_predict": fresh_predict,
        "branch_predict": n_predict,
        "fresh": fresh,
        "branch": branch,
    }
    unload_model(base_url, model, force)
    return result


def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--baseline", default=BASE)
    ap.add_argument("--r2", default=R2)
    ap.add_argument("--prefix-tokens", type=int, default=16384)
    ap.add_argument("--suffix-tokens", type=int, default=96)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--absolute-tg-floor", type=float, default=5.0)
    ap.add_argument("--min-self-retention", type=float, default=0.50)
    ap.add_argument("--max-vs-baseline-loss", type=float, default=5.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-stage10-cached-largepp.json")
    args = ap.parse_args()

    data = {
        "schema_version": 2,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": args.url,
        "baseline": args.baseline,
        "r2": args.r2,
        "fixed_tg": True,
        "settings": {
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "n_predict": args.n_predict,
            "repeats": args.repeats,
        },
        "runs": [],
    }

    sequence = []
    for i in range(args.repeats):
        sequence.extend((args.baseline, args.r2) if i % 2 == 0 else (args.r2, args.baseline))

    for i, model in enumerate(sequence, 1):
        print(f"=== cached Large-PP run {i}/{len(sequence)}: {model} ===", flush=True)
        row = run_model(args.url, model, args.prefix_tokens, args.suffix_tokens, args.n_predict, args.force)
        data["runs"].append(row)
        print(json.dumps({
            "model": model,
            "fresh_tg": row["fresh"].get("tg"),
            "branch_tg": row["branch"].get("tg"),
            "branch_pp": row["branch"].get("pp"),
            "cache_n": row["branch"].get("cache_n"),
            "acceptance": row["branch"].get("acceptance"),
            "predicted_n": row["branch"].get("predicted_n"),
        }, ensure_ascii=False), flush=True)

    def rows(model):
        return [x for x in data["runs"] if x["model"] == model]

    summary = {}
    for model in (args.baseline, args.r2):
        rr = rows(model)
        fresh_tg = median([x["fresh"].get("tg") for x in rr])
        branch_tg = median([x["branch"].get("tg") for x in rr])
        branch_pp = median([x["branch"].get("pp") for x in rr])
        acceptance = median([x["branch"].get("acceptance") for x in rr])
        self_retention = branch_tg / fresh_tg if isinstance(branch_tg, (int, float)) and isinstance(fresh_tg, (int, float)) and fresh_tg > 0 else None
        summary[model] = {
            "fresh_tg_median": fresh_tg,
            "branch_tg_median": branch_tg,
            "branch_pp_median": branch_pp,
            "branch_acceptance_median": acceptance,
            "self_retention": self_retention,
            "branch_cache_values": [x["branch"].get("cache_n") for x in rr],
            "all_full_generation": bool(rr) and all(x["fresh"].get("full_generation") and x["branch"].get("full_generation") for x in rr),
            "all_mtp_counters": bool(rr) and all(x["fresh"].get("mtp_valid") and x["branch"].get("mtp_valid") for x in rr),
        }

    b = summary[args.baseline]
    r = summary[args.r2]
    vs_base = pct(r["branch_tg_median"], b["branch_tg_median"])
    acc_delta_pp = None
    if isinstance(b["branch_acceptance_median"], (int, float)) and isinstance(r["branch_acceptance_median"], (int, float)):
        acc_delta_pp = (r["branch_acceptance_median"] - b["branch_acceptance_median"]) * 100.0
    candidate_outputs = [x["branch"].get("content", "") for x in rows(args.r2)]
    baseline_outputs = [x["branch"].get("content", "") for x in rows(args.baseline)]
    exact_each = bool(candidate_outputs and baseline_outputs) and len(set(candidate_outputs)) == 1 and len(set(baseline_outputs)) == 1
    cross_exact = exact_each and candidate_outputs[0] == baseline_outputs[0]

    gates = {
        "fixed_generation": b["all_full_generation"] and r["all_full_generation"],
        "all_mtp_counters": b["all_mtp_counters"] and r["all_mtp_counters"],
        "acceptance_delta": isinstance(acc_delta_pp, (int, float)) and acc_delta_pp >= -args.max_acceptance_drop_pp,
        "absolute_tg_floor": isinstance(r["branch_tg_median"], (int, float)) and r["branch_tg_median"] >= args.absolute_tg_floor,
        "self_retention": isinstance(r["self_retention"], (int, float)) and r["self_retention"] >= args.min_self_retention,
        "vs_baseline": isinstance(vs_base, (int, float)) and vs_base >= -args.max_vs_baseline_loss,
        "cross_model_exact": cross_exact,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"

    data["summary"] = summary
    data["candidate_vs_baseline_branch_tg_pct"] = vs_base
    data["candidate_vs_baseline_acceptance_delta_pp"] = acc_delta_pp
    data["gates"] = gates
    data["verdict"] = verdict

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "baseline": b,
        "candidate": r,
        "candidate_vs_baseline_branch_tg_pct": vs_base,
        "candidate_vs_baseline_acceptance_delta_pp": acc_delta_pp,
        "gates": gates,
        "verdict": verdict,
        "result": str(out),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
