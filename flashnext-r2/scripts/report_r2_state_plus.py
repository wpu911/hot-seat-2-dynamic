#!/usr/bin/env python3
"""Final resume-state overlay for safety gates added after report_r2_state.py.

The original reporter already reconstructs Phase-1..6b, parallel isolation,
OpenClaw and promotion-review state. This thin overlay preserves every earlier
phase decision but inserts the newer generated-turn reuse gate (#28049) *before*
parallel isolation, and makes a parallel result stale if it predates that gate.

Only one NEXT_ACTION is printed. No model is loaded/unloaded and no config is
mutated.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "report_r2_state.py"
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/share/openclaw_tools/logs"))


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


def newer_or_equal(child: Path | None, parent: Path | None) -> bool:
    return bool(child and child.is_file()) and (not parent or not parent.is_file() or child.stat().st_mtime >= parent.stat().st_mtime)


def winner_parent(winner: str) -> Path | None:
    p5 = latest(str(LOG_DIR / "flashnext-r2-phase5-gdn-*/summary.env"))
    p6 = latest(str(LOG_DIR / "flashnext-r2-phase6-tensor-split-*/summary.env"))
    p6b = latest(str(LOG_DIR / "flashnext-r2-phase6b-ratio-*/summary.env"))
    d5, d6, d6b = env_file(p5), env_file(p6), env_file(p6b)
    if p6b and d6b.get("PHASE6B_WINNER_ALIAS") == winner and newer_or_equal(p6b, p6):
        return p6b
    if p6 and d6.get("PHASE6_WINNER_ALIAS") == winner:
        return p6
    if p5 and d5.get("PHASE5_WINNER_ALIAS") == winner:
        return p5
    return p6b or p6 or p5


def main():
    p = subprocess.run([sys.executable, str(BASE), *sys.argv[1:]], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = p.stdout
    if p.returncode != 0:
        sys.stdout.write(text)
        raise SystemExit(p.returncode)

    winner_m = re.search(r"^effective_final_winner=(.+)$", text, re.M)
    action_m = re.search(r"^NEXT_ACTION=(.+)$", text, re.M)
    why_m = re.search(r"^WHY=(.+)$", text, re.M)
    winner = winner_m.group(1).strip() if winner_m else ""
    base_action = action_m.group(1).strip() if action_m else ""
    base_why = why_m.group(1).strip() if why_m else ""

    # Remove the base reporter's final decision lines so humans and Work see one
    # authoritative NEXT_ACTION instead of choosing whichever one looks cheaper.
    filtered = []
    for line in text.splitlines():
        if line.startswith("WHY=") or line.startswith("NEXT_ACTION=") or line.startswith("PRODUCTION_PROMOTION="):
            continue
        filtered.append(line)
    print("\n".join(filtered).rstrip())

    tail_actions = (
        "python3 flashnext-r2/scripts/run_mtp_parallel_isolation.py",
        "python3 flashnext-r2/scripts/run_final_openclaw_regression.py",
        "python3 flashnext-r2/scripts/prepare_promotion_review.py",
        "READY_FOR_EXPLICIT_PROMOTION",
    )
    in_tail = any(base_action.startswith(x) for x in tail_actions)

    final_action = base_action
    final_why = base_why
    if in_tail and winner and winner != "UNKNOWN":
        parent = winner_parent(winner)
        turn = latest(str(LOG_DIR / "flashnext-r2-mtp-turn-reuse-*/summary.env"))
        td = env_file(turn)
        turn_pass = (
            newer_or_equal(turn, parent)
            and td.get("MTP_TURN_REUSE") == "PASS"
            and td.get("WINNER_ALIAS") == winner
        )
        if not turn_pass:
            final_action = "python3 flashnext-r2/scripts/run_mtp_turn_reuse_gate.py"
            final_why = f"winner {winner} has not proved that a naturally-ended MTP assistant turn is reused on the next pinned-slot chat turn (upstream issue #28049)."
        else:
            parallel = latest(str(LOG_DIR / "flashnext-r2-mtp-parallel-isolation-*/summary.env"))
            pd = env_file(parallel)
            parallel_after_turn = (
                newer_or_equal(parallel, turn)
                and pd.get("MTP_PARALLEL_ISOLATION") == "PASS"
                and pd.get("WINNER_ALIAS") == winner
            )
            if not parallel_after_turn:
                final_action = "python3 flashnext-r2/scripts/run_mtp_parallel_isolation.py"
                final_why = f"turn reuse passed for {winner}; multi-slot MTP isolation must now be rerun after that current gate."
            # Otherwise the original reporter correctly handles OpenClaw/review.

    print("\nDECISION_PLUS")
    print(f"WHY={final_why}")
    print(f"NEXT_ACTION={final_action}")
    print("PRODUCTION_PROMOTION=NO")


if __name__ == "__main__":
    main()
