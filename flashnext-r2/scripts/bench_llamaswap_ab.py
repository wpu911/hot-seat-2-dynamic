#!/usr/bin/env python3
"""A/B Flash Next baseline vs R2 through the real llama-swap :8090 path.

This intentionally does NOT start a private llama-server port. It exercises the
same path OpenClaw uses:

  client -> llama-swap :8090 -> selected llama-server

Default sequence is baseline -> R2 -> baseline -> R2. Model switches use the
model-specific llama-swap unload endpoint, never the global unload endpoint, so
an unrelated Ornith/Vision/etc process is not killed by the benchmark.

Before unloading a Flash Next alias, /upstream/<model>/slots is checked. If a slot
looks busy the benchmark refuses to continue unless --force is explicitly used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.parse
import urllib.request

BASE = "qwen3.8-flash-next:256k"
R2 = "qwen3.8-flash-next-r2:256k"
URL = "http://127.0.0.1:8090"


def http_json(method: str, url: str, obj=None, timeout=1800):
    data = None if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        ct = r.headers.get("Content-Type", "")
        if "json" in ct or raw[:1] in (b"{", b"["):
            return json.loads(raw.decode("utf-8"))
        return raw.decode("utf-8", errors="replace")


def model_path_id(model: str) -> str:
    # Current aliases contain ':' but no '/'. Keep ':' readable; encode everything
    # else that could alter the management path.
    return urllib.parse.quote(model, safe=":")


def get_optional(base_url: str, path: str):
    try:
        return http_json("GET", base_url + path, timeout=20)
    except Exception as e:
        return {"unavailable": str(e)}


def slots(base_url: str, model: str):
    return get_optional(base_url, f"/upstream/{model_path_id(model)}/slots")


def slot_is_busy(x) -> bool:
    if isinstance(x, list):
        return any(slot_is_busy(v) for v in x)
    if not isinstance(x, dict):
        return False
    if x.get("is_processing") is True:
        return True
    state = x.get("state")
    if isinstance(state, str) and state.lower() not in ("idle", "", "none"):
        return True
    if isinstance(state, (int, float)) and state != 0:
        return True
    # Some llama-server builds expose slot state inside nested objects.
    for k, v in x.items():
        if k in ("params", "prompt", "generated", "timings"):
            continue
        if isinstance(v, (dict, list)) and slot_is_busy(v):
            return True
    return False


def ensure_not_busy(base_url: str, model: str, force: bool):
    s = slots(base_url, model)
    if isinstance(s, dict) and "unavailable" in s:
        # Not running/not routable is harmless for unload purposes.
        return
    if slot_is_busy(s):
        if force:
            print(f"WARNING: {model} has a busy slot; --force allows benchmark takeover", flush=True)
            return
        raise RuntimeError(
            f"refusing to unload {model}: llama-server slot appears busy. "
            "Wait for the OpenClaw request to finish or rerun with --force only if interruption is intended."
        )


def unload_model(base_url: str, model: str, force: bool = False):
    ensure_not_busy(base_url, model, force)
    url = base_url + "/api/models/unload/" + model_path_id(model)
    try:
        return http_json("POST", url, {}, timeout=120)
    except Exception:
        # An already-unloaded/non-running model may be reported as an HTTP error on
        # some builds. Check /running before deciding this is fatal.
        running = get_optional(base_url, "/running")
        blob = json.dumps(running, ensure_ascii=False)
        if model not in blob:
            return {"already_unloaded": True}
        raise


def exact_prompt(base_url: str, model: str, target_tokens: int) -> str:
    seed = (
        "贵州某地气象站记录温度、湿度、气压和风速。请逐项核对数据，"
        "比较不同日期的变化，并说明可能原因。"
    )
    text = seed * max(64, target_tokens // 10)
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


def one_completion(base_url: str, model: str, prompt: str, n_predict: int):
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
    r = http_json("POST", base_url + "/completion", payload, timeout=3600)
    wall = time.time() - t0
    timings = r.get("timings", {}) if isinstance(r, dict) else {}
    drafted = None
    accepted = None
    if isinstance(r, dict):
        for key in ("drafted_n", "tokens_drafted", "n_drafted"):
            if key in r:
                drafted = r[key]
                break
        for key in ("drafted_n_accepted", "tokens_drafted_accepted", "n_drafted_accepted"):
            if key in r:
                accepted = r[key]
                break
    return {
        "wall_s": wall,
        "prompt_n": timings.get("prompt_n"),
        "pp": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "tg": timings.get("predicted_per_second"),
        "drafted": drafted,
        "accepted": accepted,
        "content": r.get("content", "") if isinstance(r, dict) else str(r),
    }


def run_leg(base_url: str, model: str, pp_tokens: list[int], tg_predict: int, repeats: int):
    result = {
        "model": model,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "running_before": get_optional(base_url, "/running"),
        "pp": {},
        "tg": [],
    }

    for n in pp_tokens:
        prompt = exact_prompt(base_url, model, n)
        rows = []
        for _ in range(repeats):
            rows.append(one_completion(base_url, model, prompt, 1))
        result["pp"][str(n)] = rows

    workloads = [
        ("zh", "解释大型语言模型的 KV Cache、MoE 专家路由和推测解码之间的关系，要求技术准确。"),
        ("code", "写一个 Python 函数读取 CSV，按账号聚合金额，保留正负号并处理空值，给出完整代码。"),
        ("tool", '只输出 JSON：{"action":"inspect","target":"llama_box_714","fields":["status","pid","cmdline"]}'),
    ]
    for name, prompt in workloads:
        rows = []
        for _ in range(repeats):
            row = one_completion(base_url, model, prompt, tg_predict)
            row["workload"] = name
            rows.append(row)
        result["tg"].extend(rows)

    result["running_after"] = get_optional(base_url, "/running")
    result["performance"] = get_optional(base_url, "/api/performance")
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def median(xs):
    ys = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(ys) if ys else None


def summarize(leg):
    pp = {k: median([r.get("pp") for r in v]) for k, v in leg["pp"].items()}
    tg = {}
    acceptance = {}
    for w in ("zh", "code", "tool"):
        rows = [r for r in leg["tg"] if r.get("workload") == w]
        tg[w] = median([r.get("tg") for r in rows])
        ratios = []
        for r in rows:
            d, a = r.get("drafted"), r.get("accepted")
            if isinstance(d, (int, float)) and d > 0 and isinstance(a, (int, float)):
                ratios.append(a / d)
        acceptance[w] = median(ratios)
    return {"model": leg["model"], "pp": pp, "tg": tg, "acceptance": acceptance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--baseline", default=BASE)
    ap.add_argument("--r2", default=R2)
    ap.add_argument("--rounds", type=int, default=4, help="alternating legs, default B/R2/B/R2")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tg", type=int, default=256)
    ap.add_argument("--pp", default="512,2048,8192")
    ap.add_argument("--force", action="store_true", help="allow takeover even if a Flash Next slot appears busy")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-ab.json")
    args = ap.parse_args()

    pp_tokens = [int(x) for x in args.pp.split(",") if x.strip()]
    sequence = [args.baseline if i % 2 == 0 else args.r2 for i in range(args.rounds)]
    all_results = {
        "url": args.url,
        "sequence": sequence,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "legs": [],
    }

    previous = None
    for idx, model in enumerate(sequence, 1):
        print(f"\n=== leg {idx}/{len(sequence)}: {model} ===", flush=True)
        # Only unload the Flash Next model involved in the previous leg. Never use
        # the global unload endpoint because another production model may be live.
        if previous is not None:
            unload_model(args.url, previous, args.force)
            time.sleep(3)
        # Also ensure a stale instance of the upcoming alias is gone before timing.
        unload_model(args.url, model, args.force)
        time.sleep(2)

        leg = run_leg(args.url, model, pp_tokens, args.tg, args.repeat)
        all_results["legs"].append(leg)
        print(json.dumps(summarize(leg), ensure_ascii=False, indent=2), flush=True)
        previous = model

    if previous is not None:
        unload_model(args.url, previous, args.force)
    all_results["summaries"] = [summarize(x) for x in all_results["legs"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT={out}")


if __name__ == "__main__":
    main()
