#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
A2="${A2:-qwen3.8-flash-next-r2-mtp2:256k}"
A3="${A3:-qwen3.8-flash-next-r2-mtp3:256k}"
A4="${A4:-qwen3.8-flash-next-r2-mtp4:256k}"
LOGDIR="${LOGDIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOGDIR"

R23="$LOGDIR/flashnext-r2-mtp2-vs3.json"
R24="$LOGDIR/flashnext-r2-mtp2-vs4.json"
R34="$LOGDIR/flashnext-r2-mtp3-vs4.json"

common=(--rounds "${ROUNDS:-4}" --repeat "${REPEAT:-3}" --tg "${TG:-256}" --pp "${PP:-512}")

python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A2" --r2 "$A3" "${common[@]}" --out "$R23"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A2" --r2 "$A4" "${common[@]}" --out "$R24"
python3 "$SCRIPT_DIR/bench_llamaswap_ab.py" --baseline "$A3" --r2 "$A4" "${common[@]}" --out "$R34"

python3 "$SCRIPT_DIR/analyze_mtp_sweep.py" "$R23" "$R24" "$R34"
