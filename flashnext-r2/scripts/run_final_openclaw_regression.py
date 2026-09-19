#!/usr/bin/env python3
"""Final Flash Next R2 regression through the real OpenClaw Gateway.

This is deliberately *after* llama-swap performance/correctness gates. It proves
that the winning experimental alias still works through the path the user
actually uses:

  OpenClaw Gateway :18789 -> main agent -> x-openclaw-model override
      -> local-llm provider -> llama-swap :8090 -> R2 alias

The script does not edit OpenClaw config and does not promote production. It
uses a unique OpenAI `user` value so two HTTP Chat Completions calls share one
fresh OpenClaw session, verifies marker recall across the second turn, and checks
llama-swap telemetry after each turn for the requested experimental alias.

Secrets are never written to the result JSON. Authentication is resolved from
OPENCLAW_GATEWAY_TOKEN / OPENCLAW_GATEWAY_PASSWORD first, then from literal
`gateway.auth.token` / `gateway.auth.password` in openclaw.json. Environment
references of the form ${NAME} are resolved without logging their values.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import urllib.error
import urllib.request

GATEWAY = "http://127.0.0.1:18789"
SWAP = "http://127.0.0.1:8090"
CONFIG = "/app/share/openclaw_data/.openclaw/openclaw.json"
SWAP_CONFIG = "/app/share/llama_box/config/config-rocm714.yaml"
LOG_DIR = "/app/share/openclaw_tools/logs"
PROD = "qwen3.8-flash-next:256k"


def latest(pattern: str) -> Path | None:
    pp = [Path(p) for p in glob.glob(pattern)]
    pp = [p for p in pp if p.is_file()]
    return max(pp, key=lambda p: p.stat().st_mtime) if pp else None


def read_env(path: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path or not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        k, v = raw.split("=", 1)
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def resolve_winner(log_dir: Path) -> tuple[str, str]:
    """Resolve the winner without letting an old Phase-6b outrank a newer Phase-6.

    Re-running Phase-5 after Phase-6 also invalidates the latter. In that case we
    stop rather than testing a stale descendant and producing a very official
    looking JSON file about the wrong binary.
    """
    p5 = latest(str(log_dir / "flashnext-r2-phase5-gdn-*/summary.env"))
    p6 = latest(str(log_dir / "flashnext-r2-phase6-tensor-split-*/summary.env"))
    p6b = latest(str(log_dir / "flashnext-r2-phase6b-ratio-*/summary.env"))
    d5, d6, d6b = read_env(p5), read_env(p6), read_env(p6b)

    for phase, p, d in (("phase5", p5, d5), ("phase6", p6, d6), ("phase6b", p6b, d6b)):
        if p and d.get("PRODUCTION_PROMOTED") not in (None, "NO"):
            raise RuntimeError(f"{phase} summary does not prove production remained untouched: {p}")

    if p6 and p5 and p6.stat().st_mtime < p5.stat().st_mtime:
        raise RuntimeError(
            "Phase-6 result is older than the current Phase-5 winner. Run report_r2_state.py and rebuild/retest downstream stages first."
        )

    p6_alias = d6.get("PHASE6_WINNER_ALIAS")
    p6_mode = d6.get("PHASE6_WINNER_MODE")
    if p6 and p6_alias:
        if p6_mode == "TENSOR_1x1":
            p6b_alias = d6b.get("PHASE6B_WINNER_ALIAS")
            if p6b and p6b_alias and p6b.stat().st_mtime >= p6.stat().st_mtime:
                return p6b_alias, str(p6b)
        return p6_alias, str(p6)

    p5_alias = d5.get("PHASE5_WINNER_ALIAS")
    if p5 and p5_alias:
        return p5_alias, str(p5)
    raise RuntimeError("no current Phase-5/6/6b winner summary found")


def expand_secret(v):
    if not isinstance(v, str) or not v:
        return None
    if v.startswith("${") and v.endswith("}") and len(v) > 3:
        return os.environ.get(v[2:-1])
    return v


def load_openclaw_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise RuntimeError(f"OpenClaw config missing: {config_path}")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"OpenClaw config is not strict JSON at {config_path}: {e}. "
            "Use OPENCLAW_GATEWAY_TOKEN/PASSWORD or export a strict JSON config for this regression; the script will not rewrite config."
        ) from e


def load_auth(cfg: dict) -> tuple[str, str | None]:
    auth = ((cfg.get("gateway") or {}).get("auth") or {})
    mode = str(auth.get("mode") or "token").lower()
    if mode == "none":
        return mode, None
    if mode == "password":
        value = os.environ.get("OPENCLAW_GATEWAY_PASSWORD") or expand_secret(auth.get("password"))
    else:
        value = os.environ.get("OPENCLAW_GATEWAY_TOKEN") or expand_secret(auth.get("token"))
    if not value:
        raise RuntimeError(
            f"OpenClaw gateway auth mode is {mode!r} but no credential was resolvable; "
            "set OPENCLAW_GATEWAY_TOKEN or OPENCLAW_GATEWAY_PASSWORD for this regression run"
        )
    return mode, value


def check_model_policy(cfg: dict, provider_model: str) -> None:
    policy = (((cfg.get("agents") or {}).get("defaults") or {}).get("modelPolicy") or {})
    allow = policy.get("allow")
    if not isinstance(allow, list) or not allow:
        return
    patterns = [x for x in allow if isinstance(x, str) and x]
    if any(fnmatch.fnmatchcase(provider_model, p) for p in patterns):
        return
    raise RuntimeError(
        f"OpenClaw modelPolicy.allow does not permit {provider_model}. "
        "Regression will not mutate the allowlist; add the exact ref or local-llm/* deliberately before retrying."
    )


def http_json(method: str, url: str, obj=None, *, token=None, headers=None, timeout=7200):
    data = None if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            if "json" in ctype or raw[:1] in (b"{", b"["):
                return json.loads(raw.decode("utf-8"))
            return raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e


def response_text(obj) -> str:
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict) and isinstance(x.get("text"), str):
                parts.append(x["text"])
        return "".join(parts)
    return ""


def llama_swap_has_model(models_obj, alias: str) -> bool:
    if not isinstance(models_obj, dict):
        return False
    data = models_obj.get("data")
    if not isinstance(data, list):
        return False
    return any(isinstance(x, dict) and x.get("id") == alias for x in data)


def telemetry_contains(swap: str, alias: str) -> tuple[bool, dict]:
    snap = {}
    for name, path in (("running", "/running"), ("performance", "/api/performance")):
        try:
            snap[name] = http_json("GET", swap + path, timeout=30)
        except Exception as e:
            snap[name] = {"unavailable": str(e)}
    blob = json.dumps(snap, ensure_ascii=False)
    return alias in blob, snap


def unique_stress(marker: str, blocks: int) -> str:
    rows = []
    for i in range(blocks):
        a = (i * 104729 + 17) % 1000003
        b = (i * 130363 + 29) % 1000033
        rows.append(
            f"记录{i:04d}: 箱号BX{i:05d}, 流水A{a:06d}, 校验B{b:06d}, "
            f"温度{18 + (i % 13)}.{i % 10}C, 批次R{i % 37:02d}."
        )
    return (
        "这是一次 OpenClaw 最终回归测试。不要调用工具。阅读下面的独立记录，"
        f"记住唯一标记 {marker}。第一行只输出这个标记，随后用一句话说明已完成读取。\n"
        + "\n".join(rows)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=GATEWAY)
    ap.add_argument("--swap", default=SWAP)
    ap.add_argument("--openclaw-config", default=CONFIG)
    ap.add_argument("--swap-config", default=SWAP_CONFIG)
    ap.add_argument("--logs", default=LOG_DIR)
    ap.add_argument("--winner", default=None, help="override auto-discovered R2 winner alias")
    ap.add_argument("--provider", default="local-llm")
    ap.add_argument("--agent", default="main")
    ap.add_argument("--stress-blocks", type=int, default=220)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.winner:
        winner, winner_source = args.winner, "CLI_OVERRIDE"
    else:
        winner, winner_source = resolve_winner(log_dir)
    if winner == PROD:
        raise SystemExit("ERROR winner resolved to production alias; final regression requires an experimental R2 alias")

    swap_cfg = Path(args.swap_config)
    if not swap_cfg.is_file() or winner not in swap_cfg.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(f"ERROR winner alias is not present in llama-swap config: {winner}")

    swap_models = http_json("GET", args.swap + "/v1/models", timeout=30)
    if not llama_swap_has_model(swap_models, winner):
        raise SystemExit(f"ERROR llama-swap /v1/models does not advertise winner: {winner}")

    cfg = load_openclaw_config(Path(args.openclaw_config))
    auth_mode, credential = load_auth(cfg)
    provider_model = f"{args.provider}/{winner}"
    check_model_policy(cfg, provider_model)

    gateway_models = http_json("GET", args.gateway + "/v1/models", token=credential, timeout=30)
    if not isinstance(gateway_models, dict):
        raise SystemExit("ERROR OpenClaw /v1/models returned a non-JSON object")

    marker = "R2OC_" + secrets.token_hex(8).upper()
    user_key = "flashnext-r2-regression-" + secrets.token_hex(8)
    agent_model = f"openclaw/{args.agent}"
    headers = {"x-openclaw-model": provider_model}

    prompt1 = unique_stress(marker, args.stress_blocks)
    req1 = {
        "model": agent_model,
        "user": user_key,
        "stream": False,
        "messages": [{"role": "user", "content": prompt1}],
    }
    t0 = time.time()
    r1 = http_json("POST", args.gateway + "/v1/chat/completions", req1, token=credential, headers=headers)
    t1 = time.time() - t0
    text1 = response_text(r1)
    marker_first = marker in text1
    routed1, telem1 = telemetry_contains(args.swap, winner)

    req2 = {
        "model": agent_model,
        "user": user_key,
        "stream": False,
        "messages": [{"role": "user", "content": "不要调用工具。刚才要求你记住的唯一标记是什么？只回复该标记。"}],
    }
    t0 = time.time()
    r2 = http_json("POST", args.gateway + "/v1/chat/completions", req2, token=credential, headers=headers)
    t2 = time.time() - t0
    text2 = response_text(r2)
    marker_recall = marker in text2
    routed2, telem2 = telemetry_contains(args.swap, winner)

    user_key2 = "flashnext-r2-structured-" + secrets.token_hex(8)
    req3 = {
        "model": agent_model,
        "user": user_key2,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": '不要调用工具。只输出 JSON，键为 status 和 value，值分别为 "ok" 和 40。',
        }],
    }
    t0 = time.time()
    r3 = http_json("POST", args.gateway + "/v1/chat/completions", req3, token=credential, headers=headers)
    t3 = time.time() - t0
    text3 = response_text(r3)
    structured_ok = '"status"' in text3 and '"ok"' in text3 and '40' in text3
    routed3, telem3 = telemetry_contains(args.swap, winner)

    checks = {
        "gateway_models_ok": True,
        "winner_advertised_by_llamaswap": True,
        "first_turn_marker": marker_first,
        "same_session_marker_recall": marker_recall,
        "structured_output": structured_ok,
        "winner_seen_in_swap_telemetry_after_turn1": routed1,
        "winner_seen_in_swap_telemetry_after_turn2": routed2,
        "winner_seen_in_swap_telemetry_after_turn3": routed3,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or (log_dir / f"flashnext-r2-final-openclaw-{stamp}.json"))
    report = {
        "schema_version": 2,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "winner_alias": winner,
        "winner_source": winner_source,
        "provider_model": provider_model,
        "agent_model": agent_model,
        "gateway": args.gateway,
        "llama_swap": args.swap,
        "auth_mode": auth_mode,
        "stress_blocks": args.stress_blocks,
        "prompt1_sha256": hashlib.sha256(prompt1.encode("utf-8")).hexdigest(),
        "response1_sha256": hashlib.sha256(text1.encode("utf-8")).hexdigest(),
        "response2_sha256": hashlib.sha256(text2.encode("utf-8")).hexdigest(),
        "response3_sha256": hashlib.sha256(text3.encode("utf-8")).hexdigest(),
        "latency_s": {"turn1": t1, "turn2": t2, "structured": t3},
        "response_lengths": {"turn1": len(text1), "turn2": len(text2), "structured": len(text3)},
        "telemetry_hashes": {
            "turn1": hashlib.sha256(json.dumps(telem1, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "turn2": hashlib.sha256(json.dumps(telem2, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "turn3": hashlib.sha256(json.dumps(telem3, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        },
        "checks": checks,
        "verdict": verdict,
        "production_promoted": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_dir = log_dir / f"flashnext-r2-final-openclaw-{stamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dir / "summary.env"
    summary.write_text(
        "\n".join([
            f"OPENCLAW_REGRESSION={verdict}",
            f"WINNER_ALIAS={winner}",
            f"WINNER_SOURCE={winner_source}",
            f"PROVIDER_MODEL={provider_model}",
            f"RESULT={out}",
            "PRODUCTION_PROMOTED=NO",
            f"FINISHED={time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({
        "winner_alias": winner,
        "provider_model": provider_model,
        "checks": checks,
        "verdict": verdict,
        "result": str(out),
        "summary": str(summary),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
