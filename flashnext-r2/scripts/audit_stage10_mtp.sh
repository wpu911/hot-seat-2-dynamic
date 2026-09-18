#!/usr/bin/env bash
set -euo pipefail

# Stage 10A: audit the exact production tree before attempting the newer
# Qwen3.8-Flash-Next MTP implementation from ggml-org/llama.cpp PR #28243.
# This stage is read-only.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
ALIAS="${ALIAS:-qwen3.8-flash-next:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/flashnext-stage10-mtp-audit-$STAMP.log"
exec > >(tee "$LOG") 2>&1

if [[ ! -e "$PROD_SRC/.git" ]]; then
  echo "ERROR: source is not a git worktree/repo: $PROD_SRC" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: llama-swap config missing: $CONFIG" >&2
  exit 3
fi

echo "=== Flash Next Stage 10 MTP audit ==="
echo "date=$(date -Is)"
echo "source=$PROD_SRC"
echo "config=$CONFIG"
echo "alias=$ALIAS"
printf 'HEAD='; git -C "$PROD_SRC" rev-parse HEAD
printf 'dirty_entries='; git -C "$PROD_SRC" status --porcelain | wc -l

echo
echo "=== PR28243 feature probes ==="
probe() {
  local label="$1" pattern="$2" path="$3"
  if grep -RqE "$pattern" "$PROD_SRC/$path" 2>/dev/null; then
    echo "PRESENT  $label"
  else
    echo "MISSING  $label"
  fi
}
probe "qwen4exp MTP graph" 'graph_mtp|QWEN4EXP MTP' 'src'
probe "shared MTP embedding CLI" 'mtp-shared-embd|mtp_shared_embd' '.'
probe "nextn HC head tensors" 'NEXTN_HC_HEAD_NORM|nextn\.hc_head_norm' 'src'
probe "nextn layer metadata" 'n_layer_nextn|NEXTN_PREDICT_LAYERS' 'src'
probe "MTP target/draft shared-context guard" 'is_mem_shared|ctx_other' 'common'

echo
echo "=== production llama-swap alias block ==="
python3 - "$CONFIG" "$ALIAS" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1]); alias = sys.argv[2]
lines = p.read_text(encoding='utf-8').splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m:
        continue
    ind = len(m.group(1))
    block = [line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        block.append(s)
    text='\n'.join(block)
    print(text)
    print('\n--- extracted MTP/runtime hints ---')
    for rx in (
        r'/\S*/llama-server',
        r'--spec-[^\s]+(?:\s+[^\s]+)?',
        r'[^\s\"\']+\.gguf',
    ):
        for x in re.findall(rx, text):
            print(x)
    sys.exit(0)
raise SystemExit('ERROR: alias not found: ' + alias)
PY

echo
echo "=== exact-snapshot warning ==="
if [[ -n "$(git -C "$PROD_SRC" status --porcelain)" ]]; then
  echo "LIVE_TREE_DIRTY=1"
  echo "Stage 10 patching must NOT use this dirty tree as a plain HEAD base."
  echo "Create/use an exact committed snapshot first:"
  echo "  bash flashnext-r2/scripts/create_exact_prod_snapshot_repo.sh"
else
  echo "LIVE_TREE_DIRTY=0"
  echo "This tree can be used as an exact committed experiment base if it is the intended production snapshot."
fi

echo
echo "PR28243 currently advertises 1.3x-2x faster Qwen3.8 Flash Next MTP and shared MTP modules."
echo "Do not assume the current production draft tensor layout is compatible until the patched runtime actually loads it."
echo "LOG=$LOG"
