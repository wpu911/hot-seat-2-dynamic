#!/usr/bin/env python3
"""Build a read-only Flash Next R2 promotion review bundle.

No config is changed. The bundle exists so the final production decision can be
made from recorded machine evidence instead of scrolling through seven terminal
windows and trusting the human hippocampus.

Prerequisite: the latest matching final OpenClaw Gateway regression must PASS.
The script snapshots the current production and candidate llama-swap config
blocks, their hashes, the phase summaries, and a concise promotion checklist.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import time

LOG_DIR = "/app/share/openclaw_tools/logs"
CONFIG = "/app/share/llama_box/config/config-rocm714.yaml"
PROD = "qwen3.8-flash-next:256k"


def latest(pattern: str) -> Path | None:
    xs = [Path(x) for x in glob.glob(pattern)]
    xs = [x for x in xs if x.is_file()]
    return max(xs, key=lambda p: p.stat().st_mtime) if xs else None


def env_file(path: Path | None) -> dict[str, str]:
    out = {}
    if not path or not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            k, v = raw.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
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
    raise RuntimeError(f"alias not found in config: {alias}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=LOG_DIR)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    logs = Path(args.logs)
    config = Path(args.config)
    if not config.is_file():
        raise SystemExit(f"ERROR config missing: {config}")

    oc = latest(str(logs / "flashnext-r2-final-openclaw-*/summary.env"))
    ocd = env_file(oc)
    if not oc or ocd.get("OPENCLAW_REGRESSION") != "PASS":
        raise SystemExit("ERROR latest final OpenClaw regression is not PASS")
    if ocd.get("PRODUCTION_PROMOTED") != "NO":
        raise SystemExit("ERROR OpenClaw summary does not prove production remained untouched")
    winner = ocd.get("WINNER_ALIAS")
    if not winner or winner == PROD:
        raise SystemExit(f"ERROR invalid R2 winner in OpenClaw summary: {winner!r}")

    prod_block = alias_block(config, PROD)
    winner_block = alias_block(config, winner)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (logs / f"flashnext-r2-promotion-review-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "production-alias.yaml").write_text(prod_block, encoding="utf-8")
    (out_dir / "winner-alias.yaml").write_text(winner_block, encoding="utf-8")

    phase_patterns = [
        ("phase1", "flashnext-r2-phase1-*/summary.env"),
        ("phase2", "flashnext-r2-phase2-*/summary.env"),
        ("phase3", "flashnext-r2-phase3-validation-*/summary.env"),
        ("phase4", "flashnext-r2-phase4-*/summary.env"),
        ("phase4b", "flashnext-r2-phase4b-param-validation-*/summary.env"),
        ("phase5", "flashnext-r2-phase5-gdn-*/summary.env"),
        ("phase6", "flashnext-r2-phase6-tensor-split-*/summary.env"),
        ("phase6b", "flashnext-r2-phase6b-ratio-*/summary.env"),
        ("openclaw", "flashnext-r2-final-openclaw-*/summary.env"),
    ]
    phases = {}
    for name, pat in phase_patterns:
        p = latest(str(logs / pat))
        d = env_file(p)
        phases[name] = {"path": str(p) if p else None, "values": d}

    manifest = {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "production_alias": PROD,
        "winner_alias": winner,
        "config": str(config),
        "config_sha256": sha(config),
        "production_block_sha256": hashlib.sha256(prod_block.encode()).hexdigest(),
        "winner_block_sha256": hashlib.sha256(winner_block.encode()).hexdigest(),
        "openclaw_summary": str(oc),
        "phases": phases,
        "production_promoted": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    review = f"""# Flash Next R2 production promotion review

Created: {manifest['created']}

## Candidate

- Production alias: `{PROD}`
- R2 winner alias: `{winner}`
- Current llama-swap config SHA-256: `{manifest['config_sha256']}`
- Final OpenClaw regression: `PASS`

## Hard preconditions before any production mutation

- [ ] No active request is using `{PROD}`.
- [ ] No active request is using `{winner}`.
- [ ] The current config SHA-256 still equals the value above.
- [ ] The latest OpenClaw regression still names exactly `{winner}`.
- [ ] Phase-3 runtime/cached/rollback gates remain PASS.
- [ ] Phase-4b parameter validation selected a safe winner/fallback.
- [ ] Phase-5/6/6b summaries used to derive the winner have not been superseded.
- [ ] A config backup is created under `/app/share/backup` immediately before mutation.
- [ ] Promotion changes only the Flash Next production block; unrelated aliases remain byte-identical.
- [ ] Post-promotion verification checks actual process cmdline, mapped llama/ggml libraries, llama-swap route, MTP counters, one cached-prefix request and one OpenClaw request.

## Files in this review bundle

- `production-alias.yaml`: exact current production block
- `winner-alias.yaml`: exact current winning experimental block
- `manifest.json`: hashes and source summaries

This bundle is read-only. It intentionally does not provide an automatic "replace production now" side effect.
"""
    (out_dir / "PROMOTION_REVIEW.md").write_text(review, encoding="utf-8")

    summary = out_dir / "summary.env"
    summary.write_text(
        "\n".join([
            "PROMOTION_REVIEW=READY",
            f"WINNER_ALIAS={winner}",
            f"CONFIG_SHA256={manifest['config_sha256']}",
            f"REVIEW_DIR={out_dir}",
            "PRODUCTION_PROMOTED=NO",
            f"FINISHED={time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"PROMOTION_REVIEW=READY")
    print(f"WINNER_ALIAS={winner}")
    print(f"REVIEW_DIR={out_dir}")
    print("PRODUCTION_PROMOTED=NO")


if __name__ == "__main__":
    main()
