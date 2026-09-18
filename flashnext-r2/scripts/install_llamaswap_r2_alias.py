#!/usr/bin/env python3
"""Clone the live Flash Next llama-swap block into an R2 test alias.

No YAML dependency is required. The script works on the existing config text so it
preserves every production model/draft/HotSeat/MTP argument verbatim and changes
only the fields explicitly requested for R2.

Default source config:
  /app/share/llama_box/config/config-rocm714.yaml

Default aliases:
  qwen3.8-flash-next:256k       -> qwen3.8-flash-next-r2:256k

Default R2 runtime:
  /app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-stage1/bin

It creates a timestamped backup under /app/share/backup and refuses to overwrite
an existing R2 alias unless --replace is supplied.

Additional experiment switches may be supplied repeatedly with:
  --env KEY=VALUE

Inherited experiment switches may be explicitly removed from the cloned block with:
  --unset-env KEY

JMAX normally defaults to 32 for Stage-1 compatibility. For experiments where
JMAX must remain byte-for-byte inherited from production, use:
  --jmax keep

MTP sweep aliases can override only the existing production argument with:
  --spec-draft-n-max N

The MTP option must already exist in the cloned production block. The script will
not invent a speculative-decoding command line for a model that does not have one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SRC_ALIAS = "qwen3.8-flash-next:256k"
DST_ALIAS = "qwen3.8-flash-next-r2:256k"
CONFIG = "/app/share/llama_box/config/config-rocm714.yaml"
R2_BIN = "/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-stage1/bin"
BACKUP_ROOT = "/app/share/backup"


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


def replace_runtime(block: str, r2_bin: str) -> tuple[str, str | None]:
    m = re.search(r"(?m)^\s*(/\S*/llama-server)\s*$", block)
    old_server = m.group(1) if m else None
    if old_server:
        old_bin = str(Path(old_server).parent)
        block = block.replace(old_bin, r2_bin)
    else:
        block, n = re.subn(
            r"/app/share/llm/Qwen3\.8-Flash-Next-GGUF/runtime-text/[^\s\"']+/bin",
            r2_bin,
            block,
        )
        if n == 0:
            raise RuntimeError("could not locate Flash Next runtime bin path in source block")
    return block, old_server


def replace_existing_cli_option(block: str, option: str, value: str) -> str:
    pat = re.compile(rf"({re.escape(option)}\s+)(\S+)")
    matches = list(pat.finditer(block))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one existing {option} in cloned block, found {len(matches)}; "
            "refusing to invent or ambiguously edit command line"
        )
    return pat.sub(lambda m: m.group(1) + value, block, count=1)


def inject_env(block: str, indent: int, key: str, value: str) -> str:
    env_pat = re.compile(rf'(?m)^(\s*)-\s*["\']?{re.escape(key)}=[^\n"\']*["\']?\s*$')
    if env_pat.search(block):
        return env_pat.sub(lambda m: f'{m.group(1)}- "{key}={value}"', block, count=1)

    lines = block.splitlines(True)
    env_i = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s{{{indent+2}}}env:\s*$", line.rstrip("\n")):
            env_i = i
            break
    if env_i is None:
        raise RuntimeError("source alias has no env: block; refusing to invent layout")

    item_indent = " " * (indent + 4)
    lines.insert(env_i + 1, f'{item_indent}- "{key}={value}"\n')
    return "".join(lines)


def remove_env(block: str, key: str) -> tuple[str, int]:
    pat = re.compile(rf'(?m)^\s*-\s*["\']?{re.escape(key)}=[^\n"\']*["\']?\s*\n?')
    return pat.subn("", block)


def validate_key(key: str) -> str:
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError(f"invalid environment key: {key!r}")
    return key


def parse_env(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        out.append((validate_key(key), value))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--source-alias", default=SRC_ALIAS)
    ap.add_argument("--alias", default=DST_ALIAS)
    ap.add_argument("--r2-bin", default=R2_BIN)
    ap.add_argument("--jmax", default="32", help="JMAX override, or 'keep' to inherit production unchanged")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE override; repeatable")
    ap.add_argument("--unset-env", action="append", default=[], help="remove inherited KEY from cloned env block; repeatable")
    ap.add_argument("--spec-draft-n-max", type=int, default=None, help="replace existing --spec-draft-n-max value")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--validate", action="store_true", help="run llama-swap -validate after writing")
    args = ap.parse_args()

    if args.spec_draft_n_max is not None and not (1 <= args.spec_draft_n_max <= 16):
        print("ERROR: --spec-draft-n-max must be in 1..16", file=sys.stderr)
        return 1

    if args.jmax.lower() != "keep" and not re.fullmatch(r"-?\d+", args.jmax):
        print("ERROR: --jmax must be an integer or 'keep'", file=sys.stderr)
        return 1

    try:
        extra_env = parse_env(args.env)
        unset_env = [validate_key(x) for x in args.unset_env]
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    p = Path(args.config)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines(True)

    try:
        s0, s1, indent = find_block(lines, args.source_alias)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        d0, d1, _ = find_block(lines, args.alias)
        if not args.replace:
            print(f"ERROR: destination alias already exists: {args.alias}; use --replace", file=sys.stderr)
            return 3
        del lines[d0:d1]
        text = "".join(lines)
        lines = text.splitlines(True)
        s0, s1, indent = find_block(lines, args.source_alias)
    except RuntimeError:
        pass

    block = "".join(lines[s0:s1])
    block = re.sub(
        r"^(\s*)" + re.escape(args.source_alias) + r":",
        lambda m: m.group(1) + args.alias + ":",
        block,
        count=1,
        flags=re.M,
    )

    try:
        block, old_server = replace_runtime(block, args.r2_bin)
        if args.spec_draft_n_max is not None:
            block = replace_existing_cli_option(block, "--spec-draft-n-max", str(args.spec_draft_n_max))
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 5

    # Apply removals first, then explicit overrides. This makes --env win when the
    # same key is supplied in both lists and lets an ON alias guarantee that a
    # presence-based kill switch is truly absent.
    removed = {}
    for key in unset_env:
        block, n = remove_env(block, key)
        removed[key] = n

    if args.jmax.lower() != "keep":
        block = inject_env(block, indent, "GGML_JOHNV8_MMQ_ID_JMAX", args.jmax)
    for key, value in extra_env:
        block = inject_env(block, indent, key, value)

    comment = " " * indent + "# Flash Next R2 experimental alias: cloned from production; only runtime/explicit overrides differ\n"
    block = comment + block

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = Path(BACKUP_ROOT) / f"flashnext-r2-alias-before-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(p, backup_dir / p.name)

    lines[s1:s1] = ["\n", block]
    tmp = p.with_suffix(p.suffix + ".r2tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, p)

    print(f"OK backup={backup_dir}")
    print(f"OK source_alias={args.source_alias}")
    print(f"OK r2_alias={args.alias}")
    print(f"OK r2_bin={args.r2_bin}")
    if args.jmax.lower() == "keep":
        print("OK JMAX=KEEP_FROM_PRODUCTION")
    else:
        print(f"OK JMAX={args.jmax}")
    if args.spec_draft_n_max is not None:
        print(f"OK --spec-draft-n-max={args.spec_draft_n_max}")
    for key in unset_env:
        print(f"OK unset env {key} removed_entries={removed[key]}")
    for key, value in extra_env:
        print(f"OK env {key}={value}")
    if old_server:
        print(f"INFO production_server_preserved_in_source={old_server}")

    if args.validate:
        swap = "/app/share/llama_box/bin/llama-swap"
        if not Path(swap).exists():
            print(f"ERROR: validator not found: {swap}", file=sys.stderr)
            return 4
        r = subprocess.run([swap, "-config", str(p), "-validate"], text=True)
        if r.returncode:
            print("ERROR: llama-swap validation failed; restore backup before reload", file=sys.stderr)
            return r.returncode
        print("OK llama-swap validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
