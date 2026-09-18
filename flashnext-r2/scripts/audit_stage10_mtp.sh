#!/usr/bin/env bash
set -euo pipefail

# Stage 10A: read-only audit before attempting the newer Qwen3.8-Flash-Next MTP
# implementation from ggml-org/llama.cpp PR #28243.
#
# This audit checks both the live production engine and the assumptions baked into
# the Stage-10 experiment. It changes nothing.

PROD_SRC="${PROD_SRC:-/app/share/llama_box/src/llama.cpp-latest-hotseat-prod-20260911}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
ALIAS="${ALIAS:-qwen3.8-flash-next:256k}"
HC_ALIAS="${HC_ALIAS:-qwen3.8-flash-next-r2-upstream-hc:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
MTP_REPO="${MTP_REPO:-https://github.com/danielhanchen/llama.cpp.git}"
MTP_BASE="${MTP_BASE:-911f6cdc8ab8a530b2bee09ee61471a6f3178eeb}"
MTP_HEAD="${MTP_HEAD:-53b1389d0bf98fa367e2a0ce0475008e762ebf28}"
MTP_COMMITS_EXPECTED="${MTP_COMMITS_EXPECTED:-12}"

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
echo "production_alias=$ALIAS"
echo "hc_baseline_alias=$HC_ALIAS"
echo "pr_repo=$MTP_REPO"
echo "pr_base=$MTP_BASE"
echo "pr_head=$MTP_HEAD"
printf 'HEAD='; git -C "$PROD_SRC" rev-parse HEAD
printf 'dirty_entries='; git -C "$PROD_SRC" status --porcelain | wc -l

echo
echo "=== production feature probes ==="
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
probe "MTP target/draft shared-context code" 'is_mem_shared|ctx_other' 'common'

# Important distinction: ctx_other can mean module borrowing without memory
# sharing. In current PR28243 only gemma4-assistant is allowed to take the
# memory-shared shortcut; qwen4exp must catch up / roll back its own state.
if grep -Rq 'gemma4-assistant' "$PROD_SRC/common/speculative.cpp" 2>/dev/null; then
  echo "PRESENT  qwen4exp memory-sharing correctness guard"
else
  echo "MISSING  qwen4exp memory-sharing correctness guard (expected in candidate, not necessarily production)"
fi

echo
echo "=== llama-swap alias presence ==="
python3 - "$CONFIG" "$ALIAS" "$HC_ALIAS" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1]); aliases = sys.argv[2:]
lines = p.read_text(encoding='utf-8').splitlines()
for alias in aliases:
    pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
    print(f"{alias}={'PRESENT' if any(pat.match(x) for x in lines) else 'MISSING'}")
PY

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
echo "=== pinned PR lineage check (read-only temporary repo) ==="
TMP_GIT="$(mktemp -d)"
trap 'rm -rf "$TMP_GIT"' EXIT

git -C "$TMP_GIT" init -q
git -C "$TMP_GIT" remote add mtp "$MTP_REPO"
if git -C "$TMP_GIT" fetch -q --no-tags mtp "$MTP_HEAD"; then
  if git -C "$TMP_GIT" cat-file -e "$MTP_HEAD^{commit}" 2>/dev/null && \
     git -C "$TMP_GIT" cat-file -e "$MTP_BASE^{commit}" 2>/dev/null; then
    MB="$(git -C "$TMP_GIT" merge-base "$MTP_BASE" "$MTP_HEAD")"
    COUNT="$(git -C "$TMP_GIT" rev-list --count "$MTP_BASE..$MTP_HEAD")"
    FILES="$(git -C "$TMP_GIT" diff --name-only "$MTP_BASE" "$MTP_HEAD" | wc -l)"
    echo "merge_base=$MB"
    echo "commit_count=$COUNT"
    echo "changed_files=$FILES"
    if [[ "$MB" == "$MTP_BASE" && "$COUNT" == "$MTP_COMMITS_EXPECTED" && "$FILES" == 19 ]]; then
      echo "PR_LINEAGE=OK"
    else
      echo "PR_LINEAGE=CHANGED"
      echo "Do not build Stage 10 until the pinned base/head assumptions are reviewed."
    fi
  else
    echo "PR_LINEAGE=INCOMPLETE_OBJECTS"
  fi
else
  echo "PR_LINEAGE=FETCH_FAILED"
fi

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
echo "=== Stage-10 invariants ==="
echo "1. Stage 12 HC-only must be prepared first."
echo "2. Stage 10 candidate = exact production + identical Stage-12 HC + PR28243 delta."
echo "3. JMAX, HotSeat env and --spec-draft-n-max stay inherited from production."
echo "4. qwen4exp may borrow target modules but must NOT be treated as target-memory-shared."
echo "5. cached Large-PP/high-LCP regression is mandatory after the normal throughput A/B."
echo
echo "LOG=$LOG"
