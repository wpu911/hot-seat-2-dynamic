#!/usr/bin/env bash
set -euo pipefail

# Cheap repository sanity check. Run after git pull and before spending hours on
# a 190+ GiB model load. It only reads the checkout and writes Python __pycache__.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R2_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

printf '%s\n' '=== shell syntax ==='
while IFS= read -r -d '' f; do
  bash -n "$f"
  printf 'OK  %s\n' "${f#$R2_DIR/}"
done < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.sh' -print0 | sort -z)

printf '%s\n' '=== python syntax ==='
while IFS= read -r -d '' f; do
  python3 -m py_compile "$f"
  printf 'OK  %s\n' "${f#$R2_DIR/}"
done < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.py' -print0 | sort -z)

printf '%s\n' '=== critical imports ==='
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
mods = [
    'bench_llamaswap_ab',
    'bench_mtp_depth_sweep',
    'analyze_mtp_sweep',
    'run_final_openclaw_regression',
    'prepare_promotion_review',
    'report_r2_state',
]
for name in mods:
    __import__(name)
    print('OK ', name)
PY

printf '%s\n' '=== production-safety markers ==='
if grep -Rns --include='*.sh' --include='*.py' --exclude='selfcheck_r2_repo.sh' \
    'PRODUCTION_PROMOTED=YES' "$SCRIPT_DIR"; then
  echo 'ERROR active R2 script contains a production-promoted YES marker' >&2
  exit 20
fi

# The active A/B and final-regression scripts must retain the real routing ports.
python3 - "$SCRIPT_DIR" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
checks = {
    'bench_llamaswap_ab.py': '127.0.0.1:8090',
    'bench_qsa_context_ladder.py': '127.0.0.1:8090',
    'bench_stage10_cached_largepp.py': '127.0.0.1:8090',
    'bench_stage8_rollback_stress.py': '127.0.0.1:8090',
    'run_final_openclaw_regression.py': '127.0.0.1:18789',
}
for name, needle in checks.items():
    text = (p / name).read_text(encoding='utf-8')
    if needle not in text:
        raise SystemExit(f'ERROR {name} lost required route marker {needle}')
    print('OK ', name, needle)
PY

printf '%s\n' 'R2_REPO_SELFCHECK=PASS'
