#!/usr/bin/env python3
"""Detect Qwen4exp MTP cross-slot contamination before production promotion.

Upstream llama.cpp issue #28286 reports that draft-MTP with --parallel > 1 can
mix plausible content between concurrent slots. This is nastier than a crash: a
response can look coherent while borrowing another request's context.

This gate runs through the real llama-swap :8090 route. It first cold-loads the
selected R2 winner, reads the upstream /slots endpoint, and:

* if the server exposes one slot, records SERIALIZED_SAFE and exits 0;
* if it exposes multiple slots, fires distinct high-entropy requests at exactly
  that concurrency and checks every response for foreign canaries/domain tokens;
* requires MTP draft/accept counters on every measured response;
* never globally unloads llama-swap and never edits production config.

A detected foreign canary is a hard failure. Do not "average it out". Humans
already invented enough ways to launder correctness bugs into benchmark wins.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.parse

from bench_llamaswap_ab import URL, http_json, unload_model, speculative_counts
from run_final_openclaw_regression import resolve_winner

LOG_DIR = "/app/share/openclaw_tools/logs"


def model_id(model: str) -> str:
    return urllib.parse.quote(model, safe=":")


def load_and_slots(base_url: str, model: str) -> list[dict]:
    # A tiny fixed request forces llama-swap to start exactly this alias.
    payload = {
        "model": model,
        "prompt": "MTP parallel isolation preflight. Reply OK.",
        "n_predict": 8,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": True,
    }
    r = http_json("POST", base_url + "/completion", payload, timeout=3600)
    if not isinstance(r, dict):
        raise RuntimeError("preflight completion returned non-JSON response")
    slots = http_json("GET", base_url + f"/upstream/{model_id(model)}/slots", timeout=60)
    if isinstance(slots, dict) and isinstance(slots.get("slots"), list):
        slots = slots["slots"]
    if not isinstance(slots, list):
        raise RuntimeError(f"unexpected /slots response type: {type(slots).__name__}")
    return [x for x in slots if isinstance(x, dict)]


def make_cases(round_idx: int, n: int) -> list[dict]:
    domains = [
        ("sorting", ["pivot", "partition", "recursion", "stable-order"],
         "Explain a production sorting routine that partitions records around a pivot, discusses recursion depth, and contrasts stable and unstable ordering."),
        ("roman", ["consul", "senate", "legion", "aqueduct"],
         "Write a compact historical analysis of late-Republic Roman institutions, military command, public works, and the political role of the senate."),
        ("distributed", ["quorum", "replica", "partition-tolerance", "consensus"],
         "Explain quorum reads and writes, replica divergence, network partitions, and why consensus protocols separate safety from liveness."),
        ("biology", ["ribosome", "transcription", "mitochondria", "enzyme"],
         "Explain gene expression, protein synthesis, cellular energy production, and enzyme specificity to an advanced biology student."),
        ("finance", ["duration", "convexity", "yield-curve", "coupon"],
         "Explain bond duration and convexity, yield-curve shifts, coupon effects, and why price sensitivity is nonlinear."),
        ("database", ["wal", "mvcc", "snapshot", "vacuum"],
         "Explain write-ahead logging, MVCC snapshots, garbage collection/vacuum, and transaction visibility in a relational database."),
        ("compiler", ["ssa", "dominator", "register-allocation", "inlining"],
         "Explain SSA form, dominator trees, register allocation, and the tradeoffs of aggressive function inlining."),
        ("navigation", ["imu", "kalman", "drift", "gnss"],
         "Explain inertial navigation, IMU drift, GNSS correction, and Kalman-filter state estimation for a mobile platform."),
    ]
    out = []
    for i in range(n):
        name, words, task = domains[i % len(domains)]
        rng = random.Random(9001 + round_idx * 101 + i)
        canary = f"R2ISO_R{round_idx:02d}_S{i:02d}_{rng.getrandbits(64):016X}"
        # High-entropy notes make the prompts information-rich; #28286 explicitly
        # reported that overly repetitive control prompts could miss the bug.
        notes = []
        for k in range(48):
            a = rng.randrange(10**6, 10**7)
            b = rng.randrange(10**6, 10**7)
            notes.append(
                f"note{k:02d} {canary} ref={a} check={b} "
                f"topic={words[k % len(words)]} phase={(k * 17 + i) % 97}"
            )
        prompt = (
            f"You are handling an isolated request in domain {name}. Private canary: {canary}.\n"
            f"{task}\n"
            "Use only this request's subject matter. Produce at least 12 dense paragraphs. "
            "Start EVERY paragraph with the private canary exactly as written. Do not invent a different canary.\n"
            + "\n".join(notes)
        )
        out.append({"slot": i, "domain": name, "canary": canary, "words": words, "prompt": prompt})
    return out


def one(base_url: str, model: str, case: dict, n_predict: int) -> dict:
    payload = {
        "model": model,
        "prompt": case["prompt"],
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 1234 + case["slot"],
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": True,
    }
    t0 = time.time()
    r = http_json("POST", base_url + "/completion", payload, timeout=7200)
    wall = time.time() - t0
    if not isinstance(r, dict):
        raise RuntimeError("completion returned non-JSON response")
    tm = r.get("timings", {}) or {}
    drafted, accepted = speculative_counts(r, tm)
    content = r.get("content", "")
    predicted = tm.get("predicted_n")
    return {
        "slot": case["slot"],
        "domain": case["domain"],
        "canary": case["canary"],
        "wall_s": wall,
        "predicted_n": predicted,
        "tg": tm.get("predicted_per_second"),
        "drafted": drafted,
        "accepted": accepted,
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "full_generation": isinstance(predicted, (int, float)) and predicted >= n_predict,
    }


def inspect_round(cases: list[dict], rows: list[dict]) -> tuple[bool, list[dict]]:
    by_slot = {r["slot"]: r for r in rows}
    details = []
    ok = True
    for case in cases:
        row = by_slot.get(case["slot"])
        if not row:
            ok = False
            details.append({"slot": case["slot"], "error": "missing response"})
            continue
        text = row.get("content", "")
        own = case["canary"]
        foreign_canaries = [c["canary"] for c in cases if c["slot"] != case["slot"] and c["canary"] in text]
        foreign_vocab = []
        low = text.lower()
        for other in cases:
            if other["slot"] == case["slot"]:
                continue
            hits = [w for w in other["words"] if w.lower() in low]
            # Require >=2 distinctive foreign keywords so generic overlap does
            # not become a false positive.
            if len(hits) >= 2:
                foreign_vocab.append({"domain": other["domain"], "hits": hits})
        own_count = text.count(own)
        d, a = row.get("drafted"), row.get("accepted")
        mtp = isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float))
        row_ok = (
            row.get("full_generation") is True
            and mtp
            and own_count >= 1
            and not foreign_canaries
            and not foreign_vocab
        )
        ok &= row_ok
        details.append({
            "slot": case["slot"],
            "domain": case["domain"],
            "own_canary_count": own_count,
            "foreign_canaries": foreign_canaries,
            "foreign_vocab": foreign_vocab,
            "full_generation": row.get("full_generation"),
            "mtp_counters": mtp,
            "drafted": d,
            "accepted": a,
            "tg": row.get("tg"),
            "content_sha256": row.get("content_sha256"),
        })
    return ok, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--logs", default=LOG_DIR)
    ap.add_argument("--winner", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=384)
    ap.add_argument("--max-concurrency", type=int, default=4,
                    help="cap destructive test fan-out even if server exposes more slots")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    winner, winner_source = (args.winner, "CLI_OVERRIDE") if args.winner else resolve_winner(log_dir)
    if not winner:
        raise SystemExit("ERROR no current R2 winner")

    unload_model(args.url, winner, False)
    time.sleep(2)
    slots = load_and_slots(args.url, winner)
    n_slots = len(slots)
    if n_slots < 1:
        raise SystemExit("ERROR upstream /slots returned no slots")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or (log_dir / f"flashnext-r2-mtp-parallel-isolation-{stamp}.json"))
    report = {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "winner_alias": winner,
        "winner_source": winner_source,
        "url": args.url,
        "upstream_slots": n_slots,
        "requested_rounds": args.rounds,
        "n_predict": args.n_predict,
        "rounds": [],
        "production_promoted": False,
    }

    try:
        if n_slots == 1:
            report["mode"] = "SERIALIZED_SAFE"
            report["verdict"] = "PASS"
        else:
            concurrency = min(n_slots, args.max_concurrency)
            if concurrency < 2:
                raise RuntimeError("multi-slot server collapsed to concurrency < 2")
            report["mode"] = "CONCURRENT_TESTED"
            report["tested_concurrency"] = concurrency
            all_ok = True
            for rnd in range(1, args.rounds + 1):
                cases = make_cases(rnd, concurrency)
                rows = []
                errors = []
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = {pool.submit(one, args.url, winner, c, args.n_predict): c for c in cases}
                    for fut in as_completed(futs):
                        c = futs[fut]
                        try:
                            rows.append(fut.result())
                        except Exception as e:
                            errors.append({"slot": c["slot"], "domain": c["domain"], "error": f"{type(e).__name__}: {e}"})
                ok, details = inspect_round(cases, rows)
                ok = ok and not errors
                all_ok &= ok
                report["rounds"].append({
                    "round": rnd,
                    "ok": ok,
                    "errors": errors,
                    "details": details,
                })
                print(json.dumps({"round": rnd, "ok": ok, "errors": errors, "details": details}, ensure_ascii=False, indent=2), flush=True)
            report["verdict"] = "PASS" if all_ok else "FAIL"
    finally:
        try:
            unload_model(args.url, winner, False)
        except Exception as e:
            report["cleanup_warning"] = str(e)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_dir = log_dir / f"flashnext-r2-mtp-parallel-isolation-{stamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dir / "summary.env"
    summary.write_text("\n".join([
        f"MTP_PARALLEL_ISOLATION={report['verdict']}",
        f"WINNER_ALIAS={winner}",
        f"UPSTREAM_SLOTS={n_slots}",
        f"MODE={report.get('mode','UNKNOWN')}",
        f"RESULT={out}",
        "PRODUCTION_PROMOTED=NO",
        f"FINISHED={time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
    ]), encoding="utf-8")

    print(json.dumps({
        "winner_alias": winner,
        "upstream_slots": n_slots,
        "mode": report.get("mode"),
        "verdict": report["verdict"],
        "result": str(out),
        "summary": str(summary),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
