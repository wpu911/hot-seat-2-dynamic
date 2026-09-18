#!/usr/bin/env bash
set -euo pipefail

# Stage 12 audit: inspect the chosen MTP base and the live production draft
# sidecar before attempting any FR-Spec source port or GGUF rewrite.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
ALIAS="${ALIAS:-qwen3.8-flash-next:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/flashnext-stage12-frspec-audit-$STAMP.log"
exec > >(tee "$LOG") 2>&1

[[ -e "$BASE_SRC/.git" ]] || { echo "ERROR: BASE_SRC is not a git repo/worktree: $BASE_SRC" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: llama-swap config missing: $CONFIG" >&2; exit 3; }

echo "=== Flash Next Stage 12 FR-Spec audit ==="
echo "date=$(date -Is)"
echo "base_src=$BASE_SRC"
echo "base_head=$(git -C "$BASE_SRC" rev-parse HEAD)"
echo "base_dirty=$(git -C "$BASE_SRC" status --porcelain | wc -l)"
echo "config=$CONFIG"
echo "alias=$ALIAS"

echo
echo "=== qwen4exp MTP / d2t source probes ==="
grep -RniE 'graph_mtp|LLM_GRAPH_TYPE_DECODER_MTP|n_layer_nextn|NEXTN_' "$BASE_SRC/src/models/qwen4exp.cpp" | head -n 120 || true
echo "--- d2t probes ---"
grep -RniE 'LLM_TENSOR_D2T|model\.d2t|\bd2t\b|draft-vocab|FR-Spec' \
  "$BASE_SRC/src" "$BASE_SRC/common" 2>/dev/null | head -n 160 || true

# Extract the existing draft-model path from the exact alias block. We refuse to
# continue when speculative configuration is absent or ambiguous.
DRAFT_PATH="$(python3 - "$CONFIG" "$ALIAS" <<'PY'
from pathlib import Path
import re, sys
p, alias = Path(sys.argv[1]), sys.argv[2]
lines = p.read_text(encoding='utf-8').splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
block = None
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m: continue
    ind = len(m.group(1)); out=[]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    block='\n'.join(out); break
if block is None:
    raise SystemExit('ERROR: alias not found: ' + alias)
rx = re.compile(r'(?<!\S)(?:--spec-draft-model|--model-draft|--draft-model|-md)\s+("[^"]*"|\'[^\']*\'|\S+)')
vals=[m.group(1).strip('"\'') for m in rx.finditer(block)]
if len(vals) != 1:
    raise SystemExit(f'ERROR: expected exactly one draft-model option, found {len(vals)}')
print(vals[0])
PY
)"

echo
echo "draft_model=$DRAFT_PATH"
[[ -f "$DRAFT_PATH" ]] || { echo "ERROR: draft sidecar does not exist: $DRAFT_PATH" >&2; exit 4; }
ls -lh "$DRAFT_PATH"
sha256sum "$DRAFT_PATH"

# Inspect the GGUF using the base source's Python reader. This avoids guessing
# tensor orientation or quantization from the filename, a hobby best left to less
# expensive mistakes.
PYTHONPATH="$BASE_SRC/gguf-py${PYTHONPATH:+:$PYTHONPATH}" python3 - "$DRAFT_PATH" <<'PY'
import sys
from gguf import GGUFReader
p=sys.argv[1]
r=GGUFReader(p)
print('\n=== draft GGUF ===')
arch = r.fields.get('general.architecture')
print('architecture=', arch.contents() if arch else None)
for name in ('output.weight','token_embd.weight','d2t'):
    ts=[t for t in r.tensors if t.name == name]
    if not ts:
        print(name, '= MISSING')
        continue
    t=ts[0]
    print(name, 'shape=', list(map(int,t.shape)), 'type=', t.tensor_type.name, 'nbytes=', int(t.n_bytes) if hasattr(t,'n_bytes') else getattr(t.data,'nbytes',None))
print('tensor_count=', len(r.tensors))
PY

echo
echo "=== Stage 12 selected reference commits ==="
cat <<'EOF'
7a3aa1dd59f904d8f624afb13460715169838b65  initial qwen4exp d2t reduced-vocab load/scatter
6b7713dd6eb6c91b208d2455a64109dd2e206a25  accept I32/I64 d2t
cd9c998b0f6313d5ca844248e68dfbb47d7e6637  inverse t2d gather path
ff146108d95b63edd345d0fc20970188cc37ce72  preserve MTP graph inputs across continuation steps
ebb3def772fe153b963f9be48cb81e0208a081f9  move FR-Spec expansion into graph_mtp head
fb367b8cf2cf439326c8f77477ff65d01ba4816f  sidecar trimming script + 65k frequency map notes
EOF

echo
echo "AUDIT_LOG=$LOG"
