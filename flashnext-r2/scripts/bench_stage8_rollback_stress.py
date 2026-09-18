#!/usr/bin/env python3
"""Stage-8 pooled-key cache rollback/MTP stress through llama-swap :8090.

The pooled-key cache touches exactly the state paths that speculative decoding and
checkpoint rollback exercise. A speed-only test is therefore insufficient.

This script:
  * builds a long prompt with a unique marker around 15% depth;
  * generates a long deterministic continuation with production MTP still enabled;
  * alternates pooled-cache OFF/ON aliases using the same binary;
  * requires marker retrieval on every run;
  * requires the first N generated tokens to be identical OFF vs ON;
  * records full-output hashes, MTP draft/accept counts, PP and TG.

A mismatch is intentionally treated as a hard failure for promotion. It may still
be ordinary GPU nondeterminism, but cache-state code does not get the benefit of
the doubt merely because it is fast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

URL = "http://127.0.0.1:8090"
OFF = "qwen3.8-flash-next-r2-pooled-off:256k"
ON = "qwen3.8-flash-next-r2-pooled-on:256k"
MARKER = "POOL_ROLLBACK_20260918_A91D"


def http_json(method: str, url: str, obj=None, timeout=14400):
    data = None if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        ctype = r.headers.get("Content-Type", "")
        if "json" in ctype or raw[:1] in (b"{", b"["):
            return json.loads(raw.decode("utf-8"))
        return raw.decode("utf-8", errors="replace")


def mid(model: str) -> str:
    return urllib.parse.quote(model, safe=":")


def optional(url: str, path: str):
    try:
        return http_json("GET", url + path, timeout=30)
    except Exception as e:
        return {"unavailable": str(e)}


def busy(x) -> bool:
    if isinstance(x, list):
        return any(busy(v) for v in x)
    if not isinstance(x, dict):
        return False
    if x.get("is_processing") is True:
        return True
    state = x.get("state")
    if isinstance(state, str) and state.lower() not in ("", "idle", "none"):
        return True
    return any(busy(v) for k, v in x.items()
               if k not in ("params", "prompt", "generated", "timings") and isinstance(v, (dict, list)))


def unload(url: str, model: str, force: bool):
    s = optional(url, f"/upstream/{mid(model)}/slots")
    if not (isinstance(s, dict) and "unavailable" in s) and busy(s) and not force:
        raise RuntimeError(f"refusing to unload busy model: {model}")
    try:
        return http_json("POST", url + "/api/models/unload/" + mid(model), {}, timeout=180)
    except Exception:
        running = optional(url, "/running")
        if model not in json.dumps(running, ensure_ascii=False):
            return {"already_unloaded": True}
        raise


def tokenize(url: str, model: str, text: str) -> list[int]:
    r = http_json("POST", url + "/tokenize", {"model": model, "content": text}, timeout=1800)
    ids = r.get("tokens") if isinstance(r, dict) else None
    if not isinstance(ids, list):
        raise RuntimeError("/tokenize did not return token ids")
    return ids


def detokenize(url: str, model: str, ids: list[int]) -> str:
    r = http_json("POST", url + "/detokenize", {"model": model, "tokens": ids}, timeout=1800)
    if not isinstance(r, dict) or not isinstance(r.get("content"), str):
        raise RuntimeError("/detokenize did not return content")
    return r["content"]


def build_prompt(url: str, model: str, target: int) -> str:
    head = (
        "这是缓存回滚一致性测试。材料中有且只有一个特殊标记。"
        "回答时第一行必须原样输出该标记，第二行开始依次输出整数1到300，"
        "用英文逗号分隔，不要解释。\n材料开始：\n"
    )
    filler = (
        "某仓库记录箱号、日期、温度、湿度、承运信息和流水编号。"
        "本段仅用于形成长上下文，不包含特殊标记，也不包含问题答案。\n"
    )
    needle = f"\n唯一特殊标记：{MARKER}\n"
    tail = "\n材料结束。现在按开头要求作答。"

    h = tokenize(url, model, head)
    f = tokenize(url, model, filler)
    n = tokenize(url, model, needle)
    t = tokenize(url, model, tail)
    if not f:
        raise RuntimeError("empty filler tokenization")

    usable = max(0, target - len(h) - len(n) - len(t))
    before_n = min(usable, max(0, int(target * 0.15) - len(h)))
    before = (f * ((before_n + len(f) - 1) // len(f)))[:before_n]
    after_n = max(0, usable - len(before))
    after = (f * ((after_n + len(f) - 1) // len(f)))[:after_n]
    ids = (h + before + n + after + t)[:target]
    return detokenize(url, model, ids)


def draft_counts(r: dict):
    drafted = accepted = None
    for k in ("drafted_n", "tokens_drafted", "n_drafted"):
        if k in r:
            drafted = r[k]
            break
    for k in ("drafted_n_accepted", "tokens_drafted_accepted", "n_drafted_accepted"):
        if k in r:
            accepted = r[k]
            break
    return drafted, accepted


def run_one(url: str, model: str, prompt: str, n_predict: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.time()
    r = http_json("POST", url + "/completion", payload, timeout=14400)
    wall = time.time() - t0
    if not isinstance(r, dict):
        raise RuntimeError("unexpected completion response")
    tm = r.get("timings", {}) or {}
    content = r.get("content", "")
    drafted, accepted = draft_counts(r)
    out_ids = tokenize(url, model, content)
    return {
        "model": model,
        "wall_s": wall,
        "prompt_n": tm.get("prompt_n"),
        "pp": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n"),
        "tg": tm.get("predicted_per_second"),
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": (accepted / drafted) if isinstance(drafted, (int, float)) and drafted > 0 and isinstance(accepted, (int, float)) else None,
        "marker_hit": MARKER in content,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "out_ids": out_ids,
        "content": content,
    }


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--off", default=OFF)
    ap.add_argument("--on", default=ON)
    ap.add_argument("--depth", type=int, default=65536)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--compare-first", type=int, default=256,
                    help="required identical generated token prefix")
    ap.add_argument("--rounds", type=int, default=4,
                    help="alternating OFF/ON legs; default OFF,ON,OFF,ON")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-pooled-rollback.json")
    args = ap.parse_args()

    # Build prompt through OFF tokenizer once. Both aliases clone the same model/tokenizer.
    unload(args.url, args.off, args.force)
    unload(args.url, args.on, args.force)
    prompt = build_prompt(args.url, args.off, args.depth)
    unload(args.url, args.off, args.force)

    seq = [args.off if i % 2 == 0 else args.on for i in range(args.rounds)]
    rows = []
    for i, model in enumerate(seq, 1):
        unload(args.url, args.off, args.force)
        unload(args.url, args.on, args.force)
        time.sleep(3)
        print(f"=== rollback stress leg {i}/{len(seq)} model={model} ===", flush=True)
        row = run_one(args.url, model, prompt, args.n_predict)
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "model", "prompt_n", "pp", "predicted_n", "tg", "drafted", "accepted",
            "acceptance", "marker_hit", "sha256")}, ensure_ascii=False), flush=True)

    off_rows = [r for r in rows if r["model"] == args.off]
    on_rows = [r for r in rows if r["model"] == args.on]
    if not off_rows or not on_rows:
        raise RuntimeError("need both OFF and ON runs")

    marker_ok = all(r["marker_hit"] for r in rows)
    # Every run must agree with the first OFF run for the protected prefix.
    ref = off_rows[0]["out_ids"]
    lcps = [lcp(ref, r["out_ids"]) for r in rows]
    prefix_ok = all(x >= args.compare_first for x in lcps)
    mtp_seen = any(isinstance(r["drafted"], (int, float)) and r["drafted"] > 0 for r in rows)
    result = "PASS" if marker_ok and prefix_ok and mtp_seen else "FAIL"

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "depth": args.depth,
        "n_predict": args.n_predict,
        "compare_first": args.compare_first,
        "sequence": seq,
        "rows": [{k: v for k, v in r.items() if k not in ("out_ids", "content")} for r in rows],
        "lcp_tokens_vs_first_off": lcps,
        "marker_ok": marker_ok,
        "prefix_ok": prefix_ok,
        "mtp_seen": mtp_seen,
        "result": result,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({
        "result": result,
        "marker_ok": marker_ok,
        "prefix_ok": prefix_ok,
        "mtp_seen": mtp_seen,
        "lcp_tokens": lcps,
        "report": str(out),
    }, ensure_ascii=False, indent=2))

    unload(args.url, args.off, args.force)
    unload(args.url, args.on, args.force)
    raise SystemExit(0 if result == "PASS" else 2)


if __name__ == "__main__":
    main()
