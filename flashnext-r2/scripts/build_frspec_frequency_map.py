#!/usr/bin/env python3
"""Build a Qwen3.8 Flash Next token-frequency ranking from local OpenClaw sessions.

Nothing leaves the machine. Text is extracted locally from JSON/JSONL session files
and tokenized through the existing llama-swap :8090 endpoint. The output contains
only token ids/counts, not conversation text.

The point is to avoid blindly using somebody else's English-heavy FR-Spec ranking
for a Chinese/tool-heavy OpenClaw workload. Humans have invented enough benchmark
mismatches already.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import urllib.request

DEFAULT_SESSIONS = "/app/share/openclaw_data/.openclaw/agents/main/sessions"
DEFAULT_URL = "http://127.0.0.1:8090"
DEFAULT_MODEL = "qwen3.8-flash-next:256k"
TEXT_KEYS = {
    "content", "text", "prompt", "message", "arguments", "input", "output",
    "reasoning", "tool_input", "tool_output", "query", "response",
}


def http_json(url: str, obj: dict, timeout: int = 7200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def strings_from_obj(obj, parent_key: str | None = None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if isinstance(v, str) and key in TEXT_KEYS:
                if v.strip():
                    yield v
            elif isinstance(v, (dict, list)):
                yield from strings_from_obj(v, key)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str) and parent_key in TEXT_KEYS and v.strip():
                yield v
            elif isinstance(v, (dict, list)):
                yield from strings_from_obj(v, parent_key)


def strings_from_file(path: Path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    # JSONL is the usual session shape. Fall back to whole JSON, then bounded
    # plain-text chunks for unusual records.
    yielded = False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        for text in strings_from_obj(obj):
            yielded = True
            yield text
    if yielded:
        return
    try:
        obj = json.loads(raw)
        for text in strings_from_obj(obj):
            yielded = True
            yield text
    except Exception:
        pass
    if not yielded and path.suffix.lower() in {".txt", ".md"}:
        yield raw


def iter_files(root: Path, limit_files: int):
    exts = {".json", ".jsonl", ".txt", ".md"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    # Newest first so a cap reflects the current workload better than archaeology.
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    yield from files[:limit_files] if limit_files > 0 else files


def chunk_text(text: str, n: int):
    # Character chunking only limits request size; tokenizer boundaries are still
    # exact within each chunk. A small overlap prevents systematic edge loss.
    text = re.sub(r"\x00", "", text)
    if len(text) <= n:
        if text.strip():
            yield text
        return
    overlap = min(256, n // 20)
    step = max(1, n - overlap)
    for i in range(0, len(text), step):
        s = text[i:i+n]
        if s.strip():
            yield s
        if i + n >= len(text):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=DEFAULT_SESSIONS)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-files", type=int, default=300)
    ap.add_argument("--chunk-chars", type=int, default=12000)
    ap.add_argument("--max-chunks", type=int, default=12000)
    args = ap.parse_args()

    root = Path(args.sessions)
    if not root.exists():
        raise SystemExit(f"session root not found: {root}")

    counts: Counter[int] = Counter()
    files_seen = chunks = texts = 0
    chars = 0

    for path in iter_files(root, args.limit_files):
        files_seen += 1
        for text in strings_from_file(path):
            texts += 1
            for chunk in chunk_text(text, args.chunk_chars):
                if args.max_chunks > 0 and chunks >= args.max_chunks:
                    break
                r = http_json(args.url + "/tokenize", {
                    "model": args.model,
                    "content": chunk,
                })
                ids = r.get("tokens") if isinstance(r, dict) else None
                if not isinstance(ids, list):
                    raise RuntimeError(f"/tokenize returned no token list for {path}")
                counts.update(int(x) for x in ids)
                chunks += 1
                chars += len(chunk)
            if args.max_chunks > 0 and chunks >= args.max_chunks:
                break
        if args.max_chunks > 0 and chunks >= args.max_chunks:
            break

    if not counts:
        raise SystemExit("no tokens collected; refusing to create an empty ranking")

    # Stable tie-break by token id. The trimmer appends unseen vocabulary ids.
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    payload = {
        "format": "flashnext-frspec-frequency-v1",
        "model": args.model,
        "session_root": str(root),
        "files_seen": files_seen,
        "texts_seen": texts,
        "chunks": chunks,
        "characters": chars,
        "unique_tokens": len(counts),
        "total_tokens": sum(counts.values()),
        "rank": ranked,
        "counts": {str(t): counts[t] for t in ranked},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({k: payload[k] for k in (
        "format", "model", "files_seen", "texts_seen", "chunks", "characters",
        "unique_tokens", "total_tokens")}, ensure_ascii=False, indent=2))
    print(f"RANK_MAP={out}")


if __name__ == "__main__":
    main()
