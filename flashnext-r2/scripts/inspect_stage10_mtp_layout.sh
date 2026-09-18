#!/usr/bin/env bash
set -euo pipefail

# Stage 10 preflight for upstream Qwen3.8-Flash-Next MTP PR #28243.
# It does NOT modify the config. It discovers the production draft path and
# checks whether the existing GGUF looks like the new qwen4exp NextN layout.

CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
ALIAS="${ALIAS:-qwen3.8-flash-next:256k}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/flashnext-stage10-mtp-layout-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "=== Flash Next Stage 10 MTP layout preflight ==="
echo "date=$(date -Is)"
echo "config=$CONFIG"
echo "alias=$ALIAS"

test -f "$CONFIG" || { echo "ERROR: config missing: $CONFIG" >&2; exit 2; }

# Extract only this YAML model block without requiring PyYAML.
BLOCK="$(python3 - "$CONFIG" "$ALIAS" <<'PY'
import re, sys
p, alias = sys.argv[1:]
lines = open(p, encoding='utf-8').read().splitlines()
pat = re.compile(r'^(\s*)' + re.escape(alias) + r':\s*(?:#.*)?$')
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m:
        continue
    ind = len(m.group(1))
    out = [line]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    print('\n'.join(out))
    raise SystemExit(0)
raise SystemExit(3)
PY
)" || { echo "ERROR: alias not found: $ALIAS" >&2; exit 3; }

printf '%s\n' "$BLOCK"

# llama-server has used both --model-draft and -md spellings across versions.
DRAFT="$(printf '%s\n' "$BLOCK" | python3 -c '
import re,sys,shlex
s=sys.stdin.read()
# remove yaml list punctuation/quotes enough for shell-like tokenization
flat=" ".join(x.strip().lstrip("-").strip() for x in s.splitlines())
try: toks=shlex.split(flat)
except Exception: toks=flat.replace("\"","").replace("\x27","").split()
for i,t in enumerate(toks):
    if t in ("--model-draft","-md") and i+1 < len(toks):
        print(toks[i+1]); break
else:
    # fallback regex for a quoted/unquoted path
    m=re.search(r"(?:--model-draft|-md)\\s+([^\\s\\\"\\x27]+)", s)
    if m: print(m.group(1))
' | head -n1)"

if [[ -z "$DRAFT" ]]; then
  echo "ERROR: production alias contains no --model-draft/-md path" >&2
  exit 4
fi

echo
echo "draft=$DRAFT"
test -f "$DRAFT" || { echo "ERROR: draft GGUF missing: $DRAFT" >&2; exit 5; }
ls -lh "$DRAFT"
sha256sum "$DRAFT" || true

echo
echo "=== GGUF marker scan ==="
# strings is intentionally only a preflight. It avoids depending on one exact
# gguf-py API revision while still exposing metadata/tensor names embedded in GGUF.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
strings -a "$DRAFT" | grep -Ei \
  'general\.architecture|qwen4exp|qwen4_exp|nextn|next_n|mtp|eh_proj|hc_head|embed_tokens|shared_head|predict_layers' \
  | sort -u | tee "$TMP" | head -n 240 || true

HAVE_QWEN=0
HAVE_NEXTN=0
HAVE_EH=0
HAVE_HC=0

grep -qiE 'qwen4exp|qwen4_exp' "$TMP" && HAVE_QWEN=1 || true
grep -qiE 'nextn|next_n|predict_layers' "$TMP" && HAVE_NEXTN=1 || true
grep -qiE 'eh_proj|fc_embedding|fc_hidden' "$TMP" && HAVE_EH=1 || true
grep -qiE 'hc_head|hyper_connection_mixer' "$TMP" && HAVE_HC=1 || true

echo
echo "=== verdict ==="
echo "qwen4exp_marker=$HAVE_QWEN"
echo "nextn_marker=$HAVE_NEXTN"
echo "eh_proj_or_fc_marker=$HAVE_EH"
echo "hc_head_marker=$HAVE_HC"

if [[ "$HAVE_QWEN" == 1 && "$HAVE_NEXTN" == 1 && "$HAVE_EH" == 1 ]]; then
  echo "MTP_LAYOUT=UPSTREAM_28243_CANDIDATE"
  echo "ACTION=TRY_ISOLATED_BUILD_AND_LOAD"
  RC=0
else
  echo "MTP_LAYOUT=OLD_OR_CUSTOM"
  echo "ACTION=DO_NOT_REPOINT_PRODUCTION_DRAFT"
  echo "A new PR28243-compatible draft conversion may be required before benchmarking."
  RC=10
fi

echo "LOG=$LOG"
exit "$RC"
