#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical Flash Next R2 state reporter.

Turn-reuse freshness chaining is now implemented directly in report_r2_state.py.
Keeping this filename as a thin forwarding shim avoids breaking old notes or
half-finished Work sessions while preventing two independent state machines from
drifting apart again.
"""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


def main() -> int:
    target = Path(__file__).resolve().with_name("report_r2_state.py")
    env = os.environ.copy()
    p = subprocess.run([sys.executable, str(target), *sys.argv[1:]], env=env)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
