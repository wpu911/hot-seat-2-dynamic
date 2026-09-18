#!/usr/bin/env python3
"""Stage-11 PLE A/B using diverse real-world local text instead of repeated filler.

PR #29030 exists because mmap demand paging of the huge qwen4exp PLE table can
look deceptively good on repetitive benchmark prompts while real inputs touch
many more PLE rows. A repeated seed string is therefore the wrong benchmark.

The default four-leg design is two paired experiments with opposite order:

    pair 0: mmap -> direct    (prompt set A)
    pair 1: direct -> mmap    (prompt set B)

Each pair uses the same disjoint prompt windows in both arms, while the second
pair gets completely different windows. This avoids the particularly silly
benchmark where every later leg re-reads the exact pages the first leg just
warmed and then everyone congratulates the page cache.

This runner:
  * reads recent local OpenClaw session text (content never leaves localhost);
  * tokenizes it once through llama-swap;
  * cuts disjoint prompt windows for each pair/repetition;
  * reports 4-gram diversity so low-diversity tests cannot masquerade as proof;
  * alternates mmap/direct aliases and unloads between legs;
  * stores only metrics/hashes, never the session text itself.

Output intentionally matches analyze_stage11_lazy_direct.py's expected shape.
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
BASE = "qwen3.8-flash-next-r2-modern-lazy-mmap:256k"
R2 = "qwen3.8-flash-next-r2-modern-lazy-direct:256k"
SESSIONS = "/app/share/openclaw_data/.openclaw/agents/main/sessions"


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


def collect_strings(obj, out: list[str]):
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 24:
            out.append(s)
        return
    if isinstance(obj, list):
        for x in obj:
            collect_strings(x, out)
        return
    if not isinstance(obj, dict):
        return

    # Prefer fields that actually carry conversation text. Recurse into nested
    # message/content structures without vacuuming every metadata string.
    hit = False
    for key in ("content", "text", "prompt", "input", "output"):
        if key in obj:
            collect_strings(obj[key], out)
            hit = True
    if "message" in obj:
        collect_strings(obj["message"], out)
        hit = True
    if not hit:
        for key in ("messages", "parts", "items", "data"):
            if key in obj:
                collect_strings(obj[key], out)


def load_corpus(root: Path, max_files: int) -> tuple[str, list[str]]:
    if not root.exists():
        return "", []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".jsonl", ".json", ".md", ".txt")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[:max_files]
    chunks: list[str] = []
    used: list[str] = []
    for p in files:
        try:
            if p.suffix.lower() == ".jsonl":
                local: list[str] = []
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            collect_strings(json.loads(line), local)
                        except Exception:
                            continue
                if local:
                    chunks.extend(local)
                    used.append(str(p))
            elif p.suffix.lower() == ".json":
                local = []
                collect_strings(json.loads(p.read_text(encoding="utf-8", errors="ignore")), local)
                if local:
                    chunks.extend(local)
                    used.append(str(p))
            else:
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) >= 128:
                    chunks.append(text)
                    used.append(str(p))
        except Exception:
            continue
    return "\n\n".join(chunks), used


def synthetic_fallback(n: int = 6000, start: int = 0) -> str:
    # Deliberately varied fallback, not one sentence repeated 10,000 times.
    subjects = ["bank-ledger", "container", "kernel", "invoice", "router", "warehouse", "scheduler", "checkpoint"]
    verbs = ["reconciles", "indexes", "validates", "streams", "compares", "restores", "routes", "profiles"]
    attrs = ["timestamp", "amount", "token", "sequence", "expert", "position", "checksum", "latency", "account"]
    rows = []
    for j in range(n):
        i = start + j
        a = subjects[i % len(subjects)]
        b = verbs[(i * 5 + 3) % len(verbs)]
        c = attrs[(i * 7 + 1) % len(attrs)]
        rows.append(
            f"记录{i:05d}: {a} {b} {c}; code={i*2654435761 & 0xffffffff:08x}; "
            f"批次={i%97}; 路径=/data/{a}/{i%313}/{c}; 数值={(i*i*17)%1000003}."
        )
    return "\n".join(rows)


def ngram_unique_ratio(ids: list[int], n: int = 4) -> float:
    if len(ids) < n:
        return 0.0
    total = len(ids) - n + 1
    return len({tuple(ids[i:i+n]) for i in range(total)}) / total


def paired_sequence(base: str, r2: str, rounds: int) -> list[tuple[str, int]]:
    if rounds < 2 or rounds % 2 != 0:
        raise ValueError("--rounds must be an even integer >= 2 for paired A/B")
    out: list[tuple[str, int]] = []
    for pair_id in range(rounds // 2):
        # Reverse every other pair: AB, BA, AB, BA ...
        order = (base, r2) if pair_id % 2 == 0 else (r2, base)
        out.extend((model, pair_id) for model in order)
    return out


def build_windows(ids: list[int], targets: list[int], repeats: int, pair_count: int) -> dict[int, list[list[list[int]]]]:
    need = sum(t * repeats * pair_count for t in targets) + 4096 + 257 * repeats * pair_count * len(targets)
    if len(ids) < need:
        raise RuntimeError(f"corpus has {len(ids)} tokens but {need} are required for disjoint paired windows")
    out: dict[int, list[list[list[int]]]] = {}
    cursor = 1024
    for t in targets:
        pairs: list[list[list[int]]] = []
        for _pair in range(pair_count):
            rows: list[list[int]] = []
            for _ in range(repeats):
                rows.append(ids[cursor:cursor+t])
                cursor += t + 257
            pairs.append(rows)
        out[t] = pairs
    return out


def completion(url: str, model: str, prompt: str, n_predict: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": False,
    }
    t0 = time.time()
    r = http_json("POST", url + "/completion", payload, timeout=14400)
    wall = time.time() - t0
    if not isinstance(r, dict):
        raise RuntimeError("unexpected completion response")
    tm = r.get("timings", {}) or {}
    content = r.get("content", "")
    drafted = accepted = None
    for k in ("drafted_n", "tokens_drafted", "n_drafted"):
        if k in r:
            drafted = r[k]; break
    for k in ("drafted_n_accepted", "tokens_drafted_accepted", "n_drafted_accepted"):
        if k in r:
            accepted = r[k]; break
    return {
        "wall_s": wall,
        "prompt_n": tm.get("prompt_n"),
        "pp": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n"),
        "tg": tm.get("predicted_per_second"),
        "drafted": drafted,
        "accepted": accepted,
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--baseline", default=BASE)
    ap.add_argument("--r2", default=R2)
    ap.add_argument("--sessions", default=SESSIONS)
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--pp", default="512,2048,8192")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4,
                    help="even number of legs; default 4 => AB on pair0, BA on pair1")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--min-ngram4-ratio", type=float, default=0.70)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="/app/share/openclaw_tools/logs/flashnext-r2-modern-lazy-direct-realworld.json")
    args = ap.parse_args()

    targets = [int(x) for x in args.pp.split(",") if x.strip()]
    if not targets:
        raise RuntimeError("no PP targets")

    seq = paired_sequence(args.baseline, args.r2, args.rounds)
    pair_count = args.rounds // 2

    corpus, files = load_corpus(Path(args.sessions), args.max_files)
    source = "openclaw_sessions"
    synth_cursor = 0
    if not corpus:
        corpus = synthetic_fallback(start=synth_cursor)
        synth_cursor += 6000
        source = "synthetic_fallback"

    unload(args.url, args.baseline, args.force)
    unload(args.url, args.r2, args.force)
    ids = tokenize(args.url, args.baseline, corpus)

    need = sum(t * args.repeat * pair_count for t in targets) + 4096 + 257 * args.repeat * pair_count * len(targets)
    extension_rounds = 0
    while len(ids) < need:
        # Grow in bounded chunks rather than interpreting a token deficit as a
        # row count. The old need*2 expression could turn a benchmark helper into
        # an accidental RAM stress test.
        missing = need - len(ids)
        rows = max(1024, min(4096, missing // 8 + 256))
        corpus += "\n" + synthetic_fallback(rows, start=synth_cursor)
        synth_cursor += rows
        extension_rounds += 1
        ids = tokenize(args.url, args.baseline, corpus)
        if extension_rounds > 8:
            raise RuntimeError(f"unable to build enough diverse corpus after {extension_rounds} bounded extensions")
        source += "+synthetic_extension"

    windows = build_windows(ids, targets, args.repeat, pair_count)
    diversity = {
        str(t): [
            [ngram_unique_ratio(w, 4) for w in windows[t][pair_id]]
            for pair_id in range(pair_count)
        ]
        for t in targets
    }
    worst_div = min(v for pairs in diversity.values() for rows in pairs for v in rows)
    if worst_div < args.min_ngram4_ratio:
        raise RuntimeError(
            f"corpus 4-gram diversity too low: worst={worst_div:.4f} < {args.min_ngram4_ratio:.4f}; "
            "refusing a benchmark that can hide PLE I/O"
        )

    # Convert every pair's windows once. The two arms in a pair get byte-identical
    # prompt text; different pairs never reuse the same token window.
    prompts: dict[str, list[list[str]]] = {}
    for t in targets:
        prompts[str(t)] = []
        for pair_id in range(pair_count):
            prompts[str(t)].append([
                detokenize(args.url, args.baseline, w) for w in windows[t][pair_id]
            ])
    unload(args.url, args.baseline, args.force)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": args.url,
        "sequence": [model for model, _ in seq],
        "pair_ids": [pair_id for _, pair_id in seq],
        "pair_order": [
            {"pair_id": p, "order": [args.baseline, args.r2] if p % 2 == 0 else [args.r2, args.baseline]}
            for p in range(pair_count)
        ],
        "corpus": {
            "source": source,
            "files_used": len(files),
            # Paths are intentionally omitted; result logs should not inventory private session filenames.
            "token_count": len(ids),
            "ngram4_unique_ratio": diversity,
            "worst_ngram4_unique_ratio": worst_div,
            "min_required_ngram4_unique_ratio": args.min_ngram4_ratio,
        },
        "legs": [],
    }

    tg_workloads = [
        ("zh", "解释大型语言模型中稀疏注意力、KV Cache、MoE 专家缓存与推测解码的相互影响，连续输出技术说明。"),
        ("code", "写一个 Python 程序：读取多个 xlsx，按账号汇总金额并校验重复记录。给出实现和边界情况。"),
        ("tool", "分析一个本地模型服务出现吞吐下降时，应依次检查哪些运行指标、缓存状态和 GPU 调度信息。"),
    ]

    for li, (model, pair_id) in enumerate(seq, 1):
        unload(args.url, args.baseline, args.force)
        unload(args.url, args.r2, args.force)
        time.sleep(3)
        print(f"\n=== Stage11 real-world leg {li}/{len(seq)} pair={pair_id} model={model} ===", flush=True)
        leg = {
            "model": model,
            "pair_id": pair_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "running_before": optional(args.url, "/running"),
            "pp": {},
            "tg": [],
        }
        for t in targets:
            rows = []
            for ri, prompt in enumerate(prompts[str(t)][pair_id]):
                row = completion(args.url, model, prompt, 1)
                row["variant"] = ri
                row["pair_id"] = pair_id
                row["ngram4_unique_ratio"] = diversity[str(t)][pair_id][ri]
                rows.append(row)
                print(json.dumps({
                    "pair": pair_id,
                    "target": t,
                    "variant": ri,
                    "prompt_n": row["prompt_n"],
                    "pp": row["pp"],
                    "diversity": row["ngram4_unique_ratio"],
                }, ensure_ascii=False), flush=True)
            leg["pp"][str(t)] = rows

        for name, prompt in tg_workloads:
            row = completion(args.url, model, prompt, args.tg)
            row["workload"] = name
            row["pair_id"] = pair_id
            leg["tg"].append(row)

        leg["running_after"] = optional(args.url, "/running")
        leg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["legs"].append(leg)

    unload(args.url, args.baseline, args.force)
    unload(args.url, args.r2, args.force)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT={out}")


if __name__ == "__main__":
    main()
