#!/usr/bin/env bash
set -euo pipefail

# Verify that the selected Flash Next source uses qwen4exp's native recurrent
# rollback (merged upstream as #28123) instead of the old full-state speculative
# checkpoint path. This is a correctness/performance prerequisite for MTP.
#
# It is deliberately read-only.

SRC="${SRC:-/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
ALIAS="${ALIAS:-qwen3.8-flash-next-r2-modern-foundation:256k}"
ROLLBACK_MERGE="${ROLLBACK_MERGE:-0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0}"
SEP18_BASE="${SEP18_BASE:-911f6cdc8ab8a530b2bee09ee61471a6f3178eeb}"

fail(){ echo "ERROR: $*" >&2; exit 2; }
pass(){ echo "PASS: $*"; }

[[ -e "$SRC/.git" ]] || fail "source is not a git tree: $SRC"
[[ -f "$CONFIG" ]] || fail "llama-swap config missing: $CONFIG"

ARCH="$SRC/src/llama-arch.cpp"
Q4="$SRC/src/models/qwen4exp.cpp"
COMMON_H="$SRC/common/common.h"
COMMON_CPP="$SRC/common/common.cpp"
for f in "$ARCH" "$Q4" "$COMMON_H" "$COMMON_CPP"; do [[ -f "$f" ]] || fail "required source file missing: $f"; done

echo "=== qwen4exp native recurrent rollback audit ==="
echo "source=$SRC"
echo "head=$(git -C "$SRC" rev-parse HEAD)"
echo "alias=$ALIAS"

# Provenance. The modern foundation is supposed to descend from Sep18, which in
# turn already contains merged PR #28123. Fetch only the two tiny commit objects
# when the local clone pruned them.
for sha in "$ROLLBACK_MERGE" "$SEP18_BASE"; do
  if ! git -C "$SRC" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch --no-tags https://github.com/ggml-org/llama.cpp.git "$sha" >/dev/null
  fi
done
git -C "$SRC" merge-base --is-ancestor "$ROLLBACK_MERGE" "$SEP18_BASE" \
  || fail "Sep18 base does not descend from merged qwen4exp rollback commit"
pass "Sep18 upstream contains merged #28123 native rollback"

# Make sure the forwarded HotSeat/custom overlay did not accidentally delete the
# architecture capability declaration.
python3 - "$ARCH" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
m=re.search(r'bool\s+llm_arch_supports_rs_rollback\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}',s,re.S)
if not m:
    raise SystemExit('ERROR: llm_arch_supports_rs_rollback() not found')
if 'case LLM_ARCH_QWEN4EXP:' not in m.group('body'):
    raise SystemExit('ERROR: QWEN4EXP missing from llm_arch_supports_rs_rollback()')
print('PASS: QWEN4EXP advertises recurrent-state rollback support')
PY

# #28123's qwen4exp-specific requirement is more than the arch flag: both GDN
# and PLE convolution histories must retain a plane for every rollback slot.
python3 - "$Q4" <<'PY'
from pathlib import Path
import sys,re
s=Path(sys.argv[1]).read_text()
need=[
    r'n_slots\s*=\s*\(int64_t\)\s*cparams\.n_rs_seq\s*\+\s*1',
    r'for\s*\(int64_t\s+slot\s*=\s*0;\s*slot\s*<\s*n_slots;',
    r'mem_size\s*=\s*mctx_cur->get_size\(\)',
    r'\(slot\s*\*\s*mem_size\s*\+\s*kv_head\)\s*\*\s*row_size',
]
for pat in need:
    if not re.search(pat,s):
        raise SystemExit('ERROR: qwen4exp rollback-slot convolution-state marker missing: '+pat)
print('PASS: qwen4exp convolution state writes one history plane per rollback slot')
PY

# Verify common CLI -> llama_context plumbing. For draft-mtp the target context
# must request n_rs_seq=draft.n_max; otherwise qwen4exp would still fall back to
# expensive full checkpoints despite supporting rollback.
python3 - "$COMMON_H" "$COMMON_CPP" <<'PY'
from pathlib import Path
import re,sys
h=Path(sys.argv[1]).read_text(); c=Path(sys.argv[2]).read_text()
m=re.search(r'uint32_t\s+need_n_rs_seq\s*\(\)\s*const\s*\{(?P<body>.*?)\n\s*\}',h,re.S)
if not m:
    raise SystemExit('ERROR: speculative need_n_rs_seq() missing')
b=m.group('body')
if 'COMMON_SPECULATIVE_TYPE_DRAFT_MTP' not in b or not re.search(r'return\s+needs_rs_seq\s*\?\s*draft\.n_max\s*:\s*0u?',b):
    raise SystemExit('ERROR: draft-mtp no longer maps n_rs_seq to draft.n_max')
if not re.search(r'cparams\.n_rs_seq\s*=\s*params\.speculative\.need_n_rs_seq\(\)',c):
    raise SystemExit('ERROR: common context params no longer propagate need_n_rs_seq()')
print('PASS: draft-mtp target context requests n_rs_seq=draft.n_max')
PY

# Read the selected llama-swap alias exactly as deployed. It must actually be a
# draft-mtp configuration with a positive n-max, not merely have source support.
BLOCK="$(python3 - "$CONFIG" "$ALIAS" <<'PY'
import re,sys
p,alias=sys.argv[1:]
ls=open(p,encoding='utf-8').read().splitlines()
pat=re.compile(r'^(\s*)'+re.escape(alias)+r':\s*(?:#.*)?$')
for i,line in enumerate(ls):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); out=[line]
    for s in ls[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind: break
        out.append(s)
    print('\n'.join(out)); raise SystemExit(0)
raise SystemExit(3)
PY
)" || fail "alias not found: $ALIAS"

python3 - "$BLOCK" <<'PY'
import re,sys
b=sys.argv[1]
if not re.search(r'(?<!\S)--spec-type\s+draft-mtp(?:\s|$)',b):
    raise SystemExit('ERROR: selected alias is not --spec-type draft-mtp')
m=re.findall(r'(?<!\S)--spec-draft-n-max\s+(\d+)',b)
if len(m)!=1 or int(m[0]) < 1:
    raise SystemExit('ERROR: expected exactly one positive --spec-draft-n-max')
print(f'PASS: llama-swap alias uses draft-mtp with n-max={m[0]}')
PY

# The draft context itself may have n_rs_seq=0; rollback slots are required on
# the TARGET context. Keep this distinction explicit so nobody "fixes" the
# intentional draft setting later and allocates memory for nothing.
if grep -q 'cparams_dft.n_rs_seq = 0' "$COMMON_CPP"; then
  pass "draft context n_rs_seq=0 remains intentional; target owns rollback slots"
else
  echo "WARN: could not find the historical draft-context n_rs_seq=0 assignment; inspect speculative init before changing anything"
fi

echo
echo "NATIVE_RS_ROLLBACK=PASS"
echo "CHECKPOINT_HOTPATH_EXPECTED=NO_FOR_QWEN4EXP_DRAFT_MTP"
echo "No files/config/runtime were modified."
