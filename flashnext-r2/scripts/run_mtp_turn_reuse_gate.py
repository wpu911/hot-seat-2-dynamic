#!/usr/bin/env python3
"""Detect draft-MTP EOG-tail cache-reuse regression on the current R2 winner.

Upstream llama.cpp issue #28049 reports a subtle hybrid-model failure mode: MTP
may leave accepted draft tokens *behind* the first EOG. The HTTP response looks
perfectly normal, but the slot is no longer an exact prefix of the next chat
turn, so the server re-prefills the previous answer instead of reusing it.

This gate reproduces the real shape of a multi-turn chat while staying below
OpenClaw so the result is attributable to llama-server itself:

  llama-swap :8090 /upstream/<winner>/apply-template
      -> pinned llama-server slot 0 /completion
      -> assistant reaches EOS naturally
      -> same chat + generated assistant answer + next user turn
      -> same slot 0 with cache_prompt=true

Current llama-server exposes `id_slot`, `cache_n` and `prompt_n`, so this test
can pin the same slot and measure exactly how far the second turn was reused.
A healthy run must reuse well into the generated assistant answer, not merely
the first prompt. Full response text is never persisted; only hashes/metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.parse

from bench_llamaswap_ab import URL, http_json, unload_model, speculative_counts
from run_final_openclaw_regression import resolve_winner

LOG_DIR = "/app/share/openclaw_tools/logs"


def model_id(model: str) -> str:
    return urllib.parse.quote(model, safe=":")


def upstream(base_url: str, model: str, path: str) -> str:
    return f"{base_url}/upstream/{model_id(model)}/{path.lstrip('/')}"


def post_upstream(base_url: str, model: str, path: str, obj, timeout=7200):
    return http_json("POST", upstream(base_url, model, path), obj, timeout=timeout)


def get_slots(base_url: str, model: str) -> list[dict]:
    raw = http_json("GET", upstream(base_url, model, "slots"), timeout=60)
    if isinstance(raw, dict) and isinstance(raw.get("slots"), list):
        raw = raw["slots"]
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected /slots response: {type(raw).__name__}")
    return [x for x in raw if isinstance(x, dict)]


def apply_template(base_url: str, model: str, messages: list[dict]) -> str:
    r = post_upstream(base_url, model, "apply-template", {"messages": messages}, timeout=120)
    if not isinstance(r, dict) or not isinstance(r.get("prompt"), str):
        raise RuntimeError("/apply-template did not return prompt string")
    return r["prompt"]


def tokenize(base_url: str, model: str, text: str) -> list[int]:
    r = post_upstream(
        base_url,
        model,
        "tokenize",
        {"content": text, "add_special": False, "parse_special": True},
        timeout=300,
    )
    if not isinstance(r, dict) or not isinstance(r.get("tokens"), list):
        raise RuntimeError("/tokenize did not return tokens")
    out = []
    for x in r["tokens"]:
        out.append(int(x["id"] if isinstance(x, dict) else x))
    return out


def completion(base_url: str, model: str, prompt: str, *, n_predict: int, seed: int, slot: int) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": seed,
        "cache_prompt": True,
        "id_slot": slot,
        "return_tokens": True,
        "stream": False,
        "ignore_eos": False,
    }
    r = post_upstream(base_url, model, "completion", payload, timeout=7200)
    if not isinstance(r, dict):
        raise RuntimeError("completion returned non-JSON response")
    return r


def metrics(r: dict) -> dict:
    tm = r.get("timings", {}) or {}
    drafted, accepted = speculative_counts(r, tm)
    cache_n = tm.get("cache_n")
    if not isinstance(cache_n, (int, float)):
        cache_n = r.get("tokens_cached")
    prompt_n = tm.get("prompt_n")
    if not isinstance(prompt_n, (int, float)):
        prompt_n = r.get("tokens_evaluated")
    tokens = r.get("tokens") if isinstance(r.get("tokens"), list) else []
    return {
        "stop_type": r.get("stop_type"),
        "tokens": tokens,
        "predicted_n": tm.get("predicted_n"),
        "drafted": drafted,
        "accepted": accepted,
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "prompt_ms": tm.get("prompt_ms"),
        "pp": tm.get("prompt_per_second"),
        "tg": tm.get("predicted_per_second"),
        "content": r.get("content", "") if isinstance(r.get("content"), str) else "",
    }


def make_turn(round_idx: int) -> tuple[str, str, str]:
    rng = random.Random(71003 + round_idx * 7919)
    marker = f"R2TURN_{round_idx:02d}_{rng.getrandbits(64):016X}"
    topic = [
        "database MVCC and write-ahead logging",
        "compiler SSA and register allocation",
        "inertial navigation and Kalman filtering",
        "bond duration and convexity",
    ][(round_idx - 1) % 4]
    user1 = (
        f"Private marker: {marker}. Explain {topic} as exactly 20 numbered, one-sentence technical facts. "
        "Use the marker in fact 1 only. After fact 20, end the assistant turn normally. "
        "Do not ask a question and do not continue with a summary."
    )
    follow = f"Second turn marker {marker}. Reply with exactly: TURN2_OK_{marker}"
    return marker, user1, follow


def safe_ratio(a, b):
    return a / b if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--logs", default=LOG_DIR)
    ap.add_argument("--winner", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--first-max", type=int, default=512)
    ap.add_argument("--second-max", type=int, default=64)
    ap.add_argument("--min-first-generated", type=int, default=96)
    ap.add_argument("--min-answer-reuse-fraction", type=float, default=0.50)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.rounds < 1:
        raise SystemExit("ERROR --rounds must be >=1")
    if not (0.0 < args.min_answer_reuse_fraction <= 1.0):
        raise SystemExit("ERROR --min-answer-reuse-fraction must be in (0,1]")

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    winner, winner_source = (args.winner, "CLI_OVERRIDE") if args.winner else resolve_winner(logs)
    if not winner:
        raise SystemExit("ERROR no current R2 winner")

    # Cold-load through upstream passthrough, then pin every measured request to
    # one real llama-server slot. This makes LCP/cache_n evidence unambiguous even
    # when production is configured with --parallel > 1.
    unload_model(args.url, winner, False)
    time.sleep(2)
    _ = post_upstream(
        args.url,
        winner,
        "completion",
        {
            "prompt": "turn reuse preflight",
            "n_predict": 8,
            "temperature": 0,
            "cache_prompt": False,
            "id_slot": args.slot,
            "ignore_eos": True,
            "stream": False,
        },
        timeout=3600,
    )
    slots = get_slots(args.url, winner)
    ids = {int(x.get("id")) for x in slots if isinstance(x.get("id"), (int, float))}
    if args.slot not in ids:
        raise SystemExit(f"ERROR requested slot {args.slot} not exposed by upstream; slots={sorted(ids)}")

    report = {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "winner_alias": winner,
        "winner_source": winner_source,
        "slot": args.slot,
        "rounds": [],
        "production_promoted": False,
    }
    all_ok = True

    try:
        for rnd in range(1, args.rounds + 1):
            marker, user1, follow = make_turn(rnd)
            system = "You are a precise technical assistant. Follow formatting instructions exactly."
            messages1 = [
                {"role": "system", "content": system},
                {"role": "user", "content": user1},
            ]
            p1 = apply_template(args.url, winner, messages1)
            p1_tokens = tokenize(args.url, winner, p1)
            r1 = metrics(completion(args.url, winner, p1, n_predict=args.first_max, seed=5000 + rnd, slot=args.slot))

            first_mtp = (
                isinstance(r1["drafted"], (int, float)) and r1["drafted"] > 0
                and isinstance(r1["accepted"], (int, float))
            )
            first_eos = r1["stop_type"] == "eos"
            first_len = len(r1["tokens"])
            if not first_eos:
                raise RuntimeError(
                    f"round {rnd}: first assistant turn did not end at EOS (stop_type={r1['stop_type']!r}); "
                    "cannot test EOG-tail reuse"
                )
            if first_len < args.min_first_generated:
                raise RuntimeError(
                    f"round {rnd}: first answer too short ({first_len} tokens); need >= {args.min_first_generated}"
                )
            if not first_mtp:
                raise RuntimeError(f"round {rnd}: first turn missing MTP draft/accept counters")

            messages2 = messages1 + [
                {"role": "assistant", "content": r1["content"]},
                {"role": "user", "content": follow},
            ]
            p2 = apply_template(args.url, winner, messages2)
            p2_tokens = tokenize(args.url, winner, p2)
            r2 = metrics(completion(args.url, winner, p2, n_predict=args.second_max, seed=6000 + rnd, slot=args.slot))
            second_mtp = (
                isinstance(r2["drafted"], (int, float)) and r2["drafted"] > 0
                and isinstance(r2["accepted"], (int, float))
            )
            cache_n = r2["cache_n"]
            if not isinstance(cache_n, (int, float)):
                raise RuntimeError(f"round {rnd}: second turn exposes no cache_n/tokens_cached metric")

            # A healthy second turn must reuse beyond the original prompt and
            # into the generated assistant answer. We deliberately do not demand
            # 100% of answer tokens because template re-tokenization around EOG
            # can move the boundary by a few tokens across model templates.
            reuse_extra = cache_n - len(p1_tokens)
            required_extra = max(32, int(first_len * args.min_answer_reuse_fraction))
            cache_ok = reuse_extra >= required_extra
            marker_ok = f"TURN2_OK_{marker}" in r2["content"]
            row_ok = cache_ok and marker_ok and second_mtp
            all_ok &= row_ok

            row = {
                "round": rnd,
                "marker_sha256": hashlib.sha256(marker.encode()).hexdigest(),
                "prompt1_tokens": len(p1_tokens),
                "assistant_tokens": first_len,
                "prompt2_tokens": len(p2_tokens),
                "first": {
                    "stop_type": r1["stop_type"],
                    "drafted": r1["drafted"],
                    "accepted": r1["accepted"],
                    "acceptance": safe_ratio(r1["accepted"], r1["drafted"]),
                    "tg": r1["tg"],
                    "content_sha256": hashlib.sha256(r1["content"].encode()).hexdigest(),
                },
                "second": {
                    "cache_n": cache_n,
                    "prompt_n": r2["prompt_n"],
                    "reuse_extra_beyond_prompt1": reuse_extra,
                    "required_extra": required_extra,
                    "cache_fraction_of_prompt2": safe_ratio(cache_n, len(p2_tokens)),
                    "drafted": r2["drafted"],
                    "accepted": r2["accepted"],
                    "acceptance": safe_ratio(r2["accepted"], r2["drafted"]),
                    "tg": r2["tg"],
                    "content_sha256": hashlib.sha256(r2["content"].encode()).hexdigest(),
                },
                "checks": {
                    "first_eos": first_eos,
                    "first_mtp": first_mtp,
                    "second_mtp": second_mtp,
                    "turn2_marker": marker_ok,
                    "reused_generated_answer": cache_ok,
                },
                "ok": row_ok,
            }
            report["rounds"].append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    finally:
        try:
            unload_model(args.url, winner, False)
        except Exception as e:
            report["cleanup_warning"] = str(e)

    report["verdict"] = "PASS" if all_ok else "FAIL"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or (logs / f"flashnext-r2-mtp-turn-reuse-{stamp}.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_dir = logs / f"flashnext-r2-mtp-turn-reuse-{stamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dir / "summary.env"
    summary.write_text("\n".join([
        f"MTP_TURN_REUSE={report['verdict']}",
        f"WINNER_ALIAS={winner}",
        f"SLOT={args.slot}",
        f"RESULT={out}",
        "PRODUCTION_PROMOTED=NO",
        f"FINISHED={time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
    ]), encoding="utf-8")

    print(json.dumps({
        "winner_alias": winner,
        "verdict": report["verdict"],
        "result": str(out),
        "summary": str(summary),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
