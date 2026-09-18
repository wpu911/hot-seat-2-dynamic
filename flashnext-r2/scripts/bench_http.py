#!/usr/bin/env python3
"""Deterministic HTTP benchmark for an already-running llama-server.

No server startup logic here on purpose: R2 experiments should launch each variant
with the exact same production-derived command line, changing one variable at a time.

Environment:
  URL=http://127.0.0.1:5820
  LABEL=jmax32
  OUTDIR=/tmp/flashnext-r2
  PP_SIZES=512,2048,8192
  PP_REPEATS=3
  TG_REPEATS=4
  TG_TOKENS=256
"""

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("URL", "http://127.0.0.1:5820").rstrip("/")
LABEL = os.environ.get("LABEL", "run")
OUTDIR = Path(os.environ.get("OUTDIR", "/tmp/flashnext-r2"))
OUTDIR.mkdir(parents=True, exist_ok=True)
PP_SIZES = [int(x) for x in os.environ.get("PP_SIZES", "512,2048,8192").split(",") if x]
PP_REPEATS = int(os.environ.get("PP_REPEATS", "3"))
TG_REPEATS = int(os.environ.get("TG_REPEATS", "4"))
TG_TOKENS = int(os.environ.get("TG_TOKENS", "256"))


def post(path, obj, timeout=1800):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {URL}")


def median(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.mean(xs) if xs else None


def stdev(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


wait_health()

# Deterministic synthetic corpus; /tokenize + /detokenize gives exact token counts.
parts = []
for i in range(6000):
    parts.append(
        f"记录{i}：系统读取传感器、校验字段、更新缓存并记录耗时。"
        f"任务编号{i % 97}，状态正常，下一步继续核对上下文与工具调用。\n"
    )
source = "".join(parts)
tokens = post("/tokenize", {"content": source})["tokens"]


def exact_prompt(n):
    if len(tokens) < n:
        raise RuntimeError(f"synthetic corpus only produced {len(tokens)} tokens, need {n}")
    return post("/detokenize", {"tokens": tokens[:n]})["content"]


result = {
    "label": LABEL,
    "url": URL,
    "pp_sizes": PP_SIZES,
    "pp_repeats": PP_REPEATS,
    "tg_repeats": TG_REPEATS,
    "tg_tokens": TG_TOKENS,
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}

# Warm-up. Keep it small and do not reuse prompt cache.
post("/completion", {
    "prompt": "请用一句话说明缓存命中的意义。",
    "n_predict": 32,
    "temperature": 0,
    "cache_prompt": False,
    "seed": 1234,
})

pp = {}
for n in PP_SIZES:
    values = []
    prompt_ns = []
    p = exact_prompt(n)
    for _ in range(PP_REPEATS):
        r = post("/completion", {
            "prompt": p,
            "n_predict": 1,
            "temperature": 0,
            "cache_prompt": False,
            "seed": 1234,
        })
        values.append(float(r["timings"]["prompt_per_second"]))
        prompt_ns.append(int(r["timings"]["prompt_n"]))
    pp[str(n)] = {
        "median": median(values),
        "mean": mean(values),
        "stdev": stdev(values),
        "all": values,
        "prompt_n": prompt_ns,
    }
    print(f"pp{n}: median={median(values):.2f} t/s all={[round(x,2) for x in values]}")
result["pp"] = pp

prompts = {
    "zh_prose": "解释为什么大型语言模型推理时需要区分提示词预填充和逐 token 解码，并举一个本地部署的例子。",
    "code": "写一个 Python 函数，读取 JSONL 日志，按 model 聚合 prompt_tokens、generated_tokens 和 duration，并计算每个模型的平均 decode tok/s。要求处理缺失字段。",
    "tool_json": "你是自动化代理。请输出一个 JSON 对象，字段为 action、target、reason。任务：检查本地模型服务是否健康，但不要真的执行命令。",
}

tg = {}
for name, p in prompts.items():
    values = []
    generated = []
    for _ in range(TG_REPEATS):
        r = post("/completion", {
            "prompt": p,
            "n_predict": TG_TOKENS,
            "temperature": 0,
            "cache_prompt": False,
            "seed": 1234,
        })
        values.append(float(r["timings"]["predicted_per_second"]))
        generated.append(int(r["timings"]["predicted_n"]))
    tg[name] = {
        "median": median(values),
        "mean": mean(values),
        "stdev": stdev(values),
        "all": values,
        "predicted_n": generated,
    }
    print(f"tg {name}: median={median(values):.3f} t/s all={[round(x,3) for x in values]}")
result["tg"] = tg

out = OUTDIR / f"bench-{LABEL}.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved: {out}")
