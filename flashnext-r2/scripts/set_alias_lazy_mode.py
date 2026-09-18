#!/usr/bin/env python3
"""Set --lazy-mode inside one llama-swap model cmd string.

llama-swap defines cmd as a string; in this project it is normally a YAML block
scalar. This editor changes only the requested model block. If no lazy-mode
option exists, it inserts one into the cmd block. Production aliases are never
passed to this helper by the R2 preparation scripts.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import os
import re
import shutil


def find_block(lines: list[str], alias: str) -> tuple[int, int, int]:
    pat = re.compile(r"^(\s*)" + re.escape(alias) + r":\s*(?:#.*)?$")
    for i, line in enumerate(lines):
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() and not s.lstrip().startswith("#"):
                leading = len(s) - len(s.lstrip(" "))
                if leading <= indent:
                    break
            j += 1
        return i, j, indent
    raise RuntimeError(f"alias not found: {alias}")


def edit_cmd(block: str, model_indent: int, mode: str) -> str:
    lines = block.splitlines(True)
    cmd_i = None
    cmd_re = re.compile(rf"^(\s{{{model_indent+2}}})cmd:\s*(.*)$")
    for i, line in enumerate(lines):
        m = cmd_re.match(line.rstrip("\n"))
        if m:
            cmd_i = i
            tail = m.group(2)
            break
    if cmd_i is None:
        raise RuntimeError("model block has no cmd:")

    # Inline cmd string.
    if tail and tail not in ("|", ">", "|-", ">-"):
        raw = lines[cmd_i].rstrip("\n")
        pat = re.compile(r"(?:(?<=\s)|^)(?:-lzm|--lazy-mode)\s+\S+")
        if pat.search(raw):
            raw = pat.sub(f"--lazy-mode {mode}", raw, count=1)
        else:
            raw += f" --lazy-mode {mode}"
        lines[cmd_i] = raw + "\n"
        return "".join(lines)

    # Block scalar. Determine the cmd content extent, then replace or append.
    cmd_end = cmd_i + 1
    while cmd_end < len(lines):
        s = lines[cmd_end]
        if s.strip() and not s.lstrip().startswith("#"):
            leading = len(s) - len(s.lstrip(" "))
            if leading <= model_indent + 2:
                break
        cmd_end += 1

    opt = re.compile(r"(?:(?<=\s)|^)(?:-lzm|--lazy-mode)\s+\S+")
    hits = []
    for i in range(cmd_i + 1, cmd_end):
        if opt.search(lines[i]):
            hits.append(i)
    if len(hits) > 1:
        raise RuntimeError(f"multiple lazy-mode options in cmd: lines {hits}")
    if hits:
        i = hits[0]
        lines[i] = opt.sub(f"--lazy-mode {mode}", lines[i], count=1)
    else:
        # The cmd scalar content is conventionally indented four spaces deeper
        # than the model key. A standalone option line is whitespace-separated
        # into the same command by llama-swap.
        lines.insert(cmd_end, " " * (model_indent + 4) + f"--lazy-mode {mode}\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/share/llama_box/config/config-rocm714.yaml")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--mode", choices=("on", "on-direct", "auto", "off"), required=True)
    ap.add_argument("--backup-root", default="/app/share/backup")
    args = ap.parse_args()

    p = Path(args.config)
    lines = p.read_text(encoding="utf-8").splitlines(True)
    b0, b1, indent = find_block(lines, args.alias)
    old = "".join(lines[b0:b1])
    new = edit_cmd(old, indent, args.mode)
    if new == old:
        raise RuntimeError("lazy-mode edit made no change")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    bd = Path(args.backup_root) / f"flashnext-r2-lazy-mode-before-{stamp}"
    bd.mkdir(parents=True, exist_ok=False)
    shutil.copy2(p, bd / p.name)

    lines[b0:b1] = [new]
    tmp = p.with_suffix(p.suffix + ".lazytmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, p)

    print(f"OK alias={args.alias}")
    print(f"OK lazy_mode={args.mode}")
    print(f"OK backup={bd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
