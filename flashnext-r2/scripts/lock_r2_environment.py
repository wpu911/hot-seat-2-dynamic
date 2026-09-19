#!/usr/bin/env python3
"""Freeze/check environment invariants for the long Flash Next R2 campaign.

R2 intentionally edits experimental aliases and builds new runtimes, so hashing
the whole llama-swap YAML would be useless. The invariants that must *not* move
under an A/B campaign are narrower:

* llama-swap executable SHA + reported version;
* exact production alias block SHA;
* production llama-server wrapper / real ELF SHA;
* kernel release;
* ROCm/HIP shared libraries actually resolved by the production real ELF.

Create once near the beginning of the campaign, then check before expensive
phases and before promotion. If llama-swap is upgraded from v255 to v256 halfway
through, that deserves a new baseline, not a footnote pretending the experiment
was still controlled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

CONFIG = "/app/share/llama_box/config/config-rocm714.yaml"
SWAP_BIN = "/app/share/llama_box/bin/llama-swap"
PROD = "qwen3.8-flash-next:256k"
LOCK = "/app/share/openclaw_tools/logs/flashnext-r2-environment-lock.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def alias_block(config: Path, alias: str) -> str:
    lines = config.read_text(encoding="utf-8", errors="replace").splitlines()
    pat = re.compile(r"^(\s*)" + re.escape(alias) + r":\s*(?:#.*)?$")
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        out = [line]
        for s in lines[i + 1:]:
            if s.strip() and not s.lstrip().startswith("#") and len(s) - len(s.lstrip(" ")) <= indent:
                break
            out.append(s)
        return "\n".join(out).rstrip() + "\n"
    raise RuntimeError(f"alias not found: {alias}")


def server_from_block(block: str) -> Path:
    hits = re.findall(r"(?<![A-Za-z0-9_.-])(/[^\s\"']*/llama-server)(?![A-Za-z0-9_.-])", block)
    if not hits:
        raise RuntimeError("could not locate absolute llama-server path in production alias block")
    uniq = list(dict.fromkeys(hits))
    if len(uniq) != 1:
        raise RuntimeError(f"ambiguous production llama-server paths: {uniq}")
    return Path(uniq[0])


def run_version(exe: Path) -> dict:
    attempts = ([str(exe), "-version"], [str(exe), "--version"])
    for cmd in attempts:
        try:
            p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            text = p.stdout.strip()
            if p.returncode == 0 and text:
                return {"command": cmd[1], "text": text}
        except Exception:
            pass
    return {"command": None, "text": "UNAVAILABLE"}


def resolved_rocm_libs(server: Path) -> dict[str, str]:
    try:
        p = subprocess.run(["ldd", str(server)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        return {"__ldd_error__": str(e)}
    out = {}
    for line in p.stdout.splitlines():
        low = line.lower()
        if not any(k in low for k in ("amdhip", "hsa-runtime", "rocblas", "hipblas", "rocwmma", "rocprof", "hiprtc")):
            continue
        m = re.search(r"=>\s+(/\S+)", line)
        if not m:
            m = re.match(r"\s*(/\S+)\s+", line)
        if not m:
            continue
        path = Path(m.group(1))
        try:
            real = path.resolve()
            out[str(real)] = sha(real)
        except Exception as e:
            out[str(path)] = f"ERROR:{e}"
    return dict(sorted(out.items()))


def snapshot(config: Path, swap_bin: Path, prod_alias: str) -> dict:
    if not config.is_file():
        raise RuntimeError(f"config missing: {config}")
    if not swap_bin.is_file():
        raise RuntimeError(f"llama-swap binary missing: {swap_bin}")
    block = alias_block(config, prod_alias)
    wrapper = server_from_block(block)
    if not wrapper.is_file():
        raise RuntimeError(f"production llama-server missing: {wrapper}")
    real = wrapper.with_name("llama-server.real") if wrapper.with_name("llama-server.real").is_file() else wrapper
    return {
        "schema_version": 1,
        "production_alias": prod_alias,
        "llama_swap": {
            "path": str(swap_bin),
            "sha256": sha(swap_bin),
            "version": run_version(swap_bin),
        },
        "production": {
            "alias_block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "server_wrapper": str(wrapper),
            "server_wrapper_sha256": sha(wrapper),
            "server_real": str(real),
            "server_real_sha256": sha(real),
        },
        "system": {
            "kernel_release": platform.release(),
            "machine": platform.machine(),
        },
        "resolved_rocm_libraries": resolved_rocm_libs(real),
    }


def compare(expected: dict, actual: dict) -> list[dict]:
    paths = [
        ("llama_swap.sha256", expected.get("llama_swap", {}).get("sha256"), actual.get("llama_swap", {}).get("sha256")),
        ("llama_swap.version.text", expected.get("llama_swap", {}).get("version", {}).get("text"), actual.get("llama_swap", {}).get("version", {}).get("text")),
        ("production.alias_block_sha256", expected.get("production", {}).get("alias_block_sha256"), actual.get("production", {}).get("alias_block_sha256")),
        ("production.server_wrapper_sha256", expected.get("production", {}).get("server_wrapper_sha256"), actual.get("production", {}).get("server_wrapper_sha256")),
        ("production.server_real_sha256", expected.get("production", {}).get("server_real_sha256"), actual.get("production", {}).get("server_real_sha256")),
        ("system.kernel_release", expected.get("system", {}).get("kernel_release"), actual.get("system", {}).get("kernel_release")),
        ("resolved_rocm_libraries", expected.get("resolved_rocm_libraries"), actual.get("resolved_rocm_libraries")),
    ]
    return [{"field": k, "expected": a, "actual": b} for k, a, b in paths if a != b]


def require_lock(
    config: str | Path = CONFIG,
    swap_bin: str | Path = SWAP_BIN,
    prod_alias: str = PROD,
    lock: str | Path = LOCK,
) -> dict:
    """Return the lock document or raise RuntimeError on any environment drift."""
    config = Path(config)
    swap_bin = Path(swap_bin)
    lock = Path(lock)
    if not lock.is_file():
        raise RuntimeError(f"environment lock missing: {lock}; create it before controlled A/B")
    try:
        expected = json.loads(lock.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"environment lock is unreadable: {lock}: {e}") from e
    actual = snapshot(config, swap_bin, prod_alias)
    diffs = compare(expected, actual)
    if diffs:
        raise RuntimeError("R2 environment drift detected: " + json.dumps(diffs, ensure_ascii=False))
    return expected


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--swap-bin", default=SWAP_BIN)
    ap.add_argument("--production-alias", default=PROD)
    ap.add_argument("--lock", default=LOCK)
    ap.add_argument("--replace", action="store_true", help="allow replacing an existing lock only with --create")
    args = ap.parse_args()

    lock = Path(args.lock)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.create:
        actual = snapshot(Path(args.config), Path(args.swap_bin), args.production_alias)
        if lock.exists() and not args.replace:
            raise SystemExit(f"ERROR lock already exists: {lock}; use --check, not a convenient rewrite of history")
        actual["created"] = now
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        print("R2_ENVIRONMENT_LOCK=CREATED")
        print(f"LOCK={lock}")
        print(f"LLAMA_SWAP_VERSION={actual['llama_swap']['version']['text']}")
        print(f"LLAMA_SWAP_SHA256={actual['llama_swap']['sha256']}")
        print(f"PRODUCTION_ALIAS_BLOCK_SHA256={actual['production']['alias_block_sha256']}")
        return

    try:
        expected = require_lock(args.config, args.swap_bin, args.production_alias, args.lock)
    except RuntimeError as e:
        print(f"R2_ENVIRONMENT_LOCK=FAIL\nERROR={e}")
        raise SystemExit(2)
    print(json.dumps({
        "checked": now,
        "lock": str(lock),
        "result": "PASS",
        "llama_swap_version": expected.get("llama_swap", {}).get("version", {}).get("text"),
        "llama_swap_sha256": expected.get("llama_swap", {}).get("sha256"),
        "production_alias_block_sha256": expected.get("production", {}).get("alias_block_sha256"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
