#!/usr/bin/env python3
"""Long-context llama-swap A/B for Qwen3.8 Flash Next QSA/TOP_K work.

Runs the real :8090 path and measures decode at increasing context depth. The
prompt contains a unique needle around 15% depth; every run must retrieve the
same marker so a throughput win cannot hide a broken sparse-attention path.

TG measurement is intentionally fixed-length: ignore_eos=true forces the server
to generate n_predict tokens. Without this, the old "only output the marker"
prompt often ended after a handful of tokens and produced a very noisy TG number.

Default sequence is OFF -> ON. Use --rounds 4 for OFF/ON/OFF/ON confirmation
once an exploratory pass looks good, because 131k prefills are not free and
apparently electrons also have employment rights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

URL = "http://127.0.0.1:8090"
OFF = "qwen3.8-flash-next-r2-qsa-off:256k"
ON = "qwen3.8-flash-next-r2-qsa-on:256k"
MARKER = "QSA_VERIFY_20260918_7B_7F3C"


def http_json(method: str, url: str, obj=None, timeout=7200):
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


def model_id(model: str) -> str:
    return urllib.parse.quote(model, safe=":")


def get_optional(url: str, path: str):
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
    return any(
        busy(v)
        for k, v in x.items()
        if k not in ("params", "prompt", "generated", "timings") and isinstance(v, (dict, list))
    )


def unload(url: str, model: str, force: bool):
    slots = get_optional(url, f"/upstream/{model_id(model)}/slots")
    if not (isinstance(slots, dict) and "unavailable" in slots) and busy(slots) and not force:
        raise RuntimeError(f"refusing to unload busy model: {model}")
    req_url = url + "/api/models/unload/" + model_id(model)
    try:
        return http_json("POST", req_url, {}, timeout=180)
    except Exception:
        running = get_optional(url, "/running")
        if model not in json.dumps(running, ensure_ascii=False):
            return {"already_unloaded": True}
        raise


def tokenize(url: str, model: str, content: str) -> list[int]:
    r = http_json("POST", url + "/tokenize", {"model": model, "content": content}, timeout=1800)
    ids = r.get("tokens") if isinstance(r, dict) else None
    if not isinstance(ids, list):
        raise RuntimeError("/tokenize did not return tokens")
    return ids


def detokenize(url: str, model: str, ids: list[int]) -> str:
    r = http_json("POST", url + "/detokenize", {"model": model, "tokens": ids}, timeout=1800)
    if not isinstance(r, dict) or not isinstance(r.get("content"), str):
        raise RuntimeError("/detokenize did not return content")
    return r["content"]


def make_prompt(url: str, model: str, target: int) -> str:
    header = (
        "这是一个长上下文检索测试。材料中只有一个 NEEDLE 标记。"
        "请找到它。回答时第一行必须原样输出 NEEDLE 方括号中的字符串；"
        "第一行之后可以继续生成测试文本。\n材料开始：\n"
    )
    filler = (
        "记录显示仓库每天核对温度、湿度、箱号、流水号和装卸时间，"
        "本段只是无关背景材料，不包含答案。\n"
    )
    needle = f"\nNEEDLE[{MARKER}]\n"
    footer = (
        "\n材料结束。第一行必须原样输出唯一 NEEDLE 方括号中的字符串。"
        "随后继续正常生成，测试程序会强制固定输出 token 数。"
    )

    h = tokenize(url, model, header)
    f = tokenize(url, model, filler)
    n = tokenize(url, model, needle)
    t = tokenize(url, model, footer)
    if not f:
        raise RuntimeError("filler tokenization empty")

    usable = max(0, target - len(h) - len(n) - len(t))
    before_n = max(0, int(target * 0.15) - len(h))
    before = (f * ((before_n + len(f) - 1) // len(f)))[:before_n]
    after_n = max(0, usable - len(before))
    after = (f * ((after_n + len(f) - 1) // len(f)))[:after_n]
    ids = (h + before + n + after + t)[:target]
    return detokenize(url, model, ids)


def extract_counts(r: dict):
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


def run_one(url: str, model: str, target: int, n_predict: int):
    prompt = make_prompt(url, model, target)
    payload = {
        "model": model,
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": True,
    }
    t0 = time.time()
    r = http_json("POST", url + "/completion", payload, timeout=14400)
    wall = time.time() - t0
    if not isinstance(r, dict):
        raise RuntimeError(f"unexpected completion response: {type(r)}")
    tm = r.get("timings", {}) or {}
    drafted, accepted = extract_counts(r)
    content = r.get("content", "")
    predicted_n = tm.get("predicted_n")
    full_generation = isinstance(predicted_n, (int, float)) and predicted_n >= n_predict
    return {
        "target_depth": target,
        "requested_predict": n_predict,
        "prompt_n": tm.get("prompt_n"),
        "pp": tm.get("prompt_per_second"),
        "predicted_n": predicted_n,
        "tg": tm.get("predicted_per_second"),
        "wall_s": wall,
        "drafted": drafted,
        "accepted": accepted,
        "marker_hit": MARKER in content,
        "full_generation": full_generation,
        "content": content,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--baseline", default=OFF)
    ap.add_argument("--r2", default=ON)
    ap.add_argument("--depths", default="4096,16384,32768,65536,131072")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-qsa-ladder.json")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seq = [args.baseline if i % 2 == 0 else args.r2 for i in range(args.rounds)]
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": args.url,
        "marker": MARKER,
        "depths": depths,
        "requested_predict": args.tg,
        "ignore_eos": True,
        "sequence": seq,
        "legs": [],
    }

    for li, model in enumerate(seq, 1):
        print(f"\n=== leg {li}/{len(seq)} model={model} ===", flush=True)
        unload(args.url, args.baseline, args.force)
        unload(args.url, args.r2, args.force)
        time.sleep(3)
        leg = {"model": model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": []}
        for d in depths:
            print(f"depth={d}", flush=True)
            row = run_one(args.url, model, d, args.tg)
            leg["rows"].append(row)
            acc = None
            if isinstance(row["drafted"], (int, float)) and row["drafted"]:
                acc = row["accepted"] / row["drafted"] if isinstance(row["accepted"], (int, float)) else None
            print(
                json.dumps(
                    {
                        "target": d,
                        "prompt_n": row["prompt_n"],
                        "predicted_n": row["predicted_n"],
                        "pp": row["pp"],
                        "tg": row["tg"],
                        "marker_hit": row["marker_hit"],
                        "full_generation": row["full_generation"],
                        "acceptance": acc,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not row["full_generation"]:
                raise RuntimeError(
                    f"server returned only {row['predicted_n']} generated tokens at depth {d}; "
                    f"fixed-length TG requires {args.tg}. Check ignore_eos support before trusting this run."
                )
        leg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        leg["running_after"] = get_optional(args.url, "/running")
        result["legs"].append(leg)

    unload(args.url, args.baseline, args.force)
    unload(args.url, args.r2, args.force)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT={out}")


if __name__ == "__main__":
    main()
