#!/usr/bin/env bash
set -euo pipefail

# Phase-6b: only after tensor 1,1 wins Phase-6, derive *bidirectional*
# heterogeneous ratios from the actual ROCm device order and reported VRAM.
#
# Capacity-only tuning is not enough for mixed gfx1100 + gfx1201: the larger
# card is not guaranteed to be the faster llama.cpp device for every kernel.
# Therefore we prepare five arms around 1:1:
#   EVEN     1,1
#   MID      geometric half-way toward VRAM capacity ratio
#   CAP      VRAM-proportional ratio
#   INV_MID  reciprocal of MID
#   INV_CAP  reciprocal of CAP
#
# For 24 GiB -> 32 GiB this is roughly:
#   1,1 ; 13,15 ; 3,4 ; 15,13 ; 4,3
# so the benchmark can discover whether performance wants work biased toward
# either device instead of assuming "more VRAM = more tensors".

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
LOG_DIR="${LOG_DIR:-/app/share/openclaw_tools/logs}"
PHASE6_SRC="${PHASE6_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-tensor-split-20260919}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-tensor-split}"
EVEN_ALIAS="${EVEN_ALIAS:-qwen3.8-flash-next-r2-split-tensor-1x1:256k}"
MID_ALIAS="${MID_ALIAS:-qwen3.8-flash-next-r2-split-tensor-mid:256k}"
CAP_ALIAS="${CAP_ALIAS:-qwen3.8-flash-next-r2-split-tensor-cap:256k}"
INV_MID_ALIAS="${INV_MID_ALIAS:-qwen3.8-flash-next-r2-split-tensor-inv-mid:256k}"
INV_CAP_ALIAS="${INV_CAP_ALIAS:-qwen3.8-flash-next-r2-split-tensor-inv-cap:256k}"

latest_phase6_summary() {
  find "$LOG_DIR" -maxdepth 2 -type f -path '*/flashnext-r2-phase6-tensor-split-*/summary.env' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
}
summary_value(){ awk -F= -v k="$2" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$1"; }

P6="${PHASE6_SUMMARY:-$(latest_phase6_summary || true)}"
[[ -n "$P6" && -f "$P6" ]] || { echo "ERROR: Phase-6 summary missing" >&2; exit 2; }
[[ "$(summary_value "$P6" PHASE6_WINNER_MODE)" == "TENSOR_1x1" ]] || {
  echo "SKIP: Phase-6 did not select tensor 1,1; heterogeneous ratio sweep is not justified."
  exit 0
}
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing" >&2; exit 3; }
[[ -f "$PHASE6_SRC/r2-meta/phase6-device-list.txt" ]] || {
  echo "ERROR: Phase-6 device list missing: $PHASE6_SRC/r2-meta/phase6-device-list.txt" >&2
  exit 4
}
BASE_BIN="$RUNTIME_ROOT/tensor-1x1/bin"
[[ -x "$BASE_BIN/llama-server.real" ]] || { echo "ERROR: Phase-6 real binary missing" >&2; exit 5; }
grep -qE "^[[:space:]]*${EVEN_ALIAS//./\\.}:[[:space:]]*(#.*)?$" "$CONFIG" || {
  echo "ERROR: Phase-6 tensor 1,1 alias missing" >&2; exit 6;
}

readarray -t DETECT < <(python3 - "$CONFIG" "$EVEN_ALIAS" "$PHASE6_SRC/r2-meta/phase6-device-list.txt" <<'PY'
import math,re,sys
from fractions import Fraction
cfg, alias, devfile = sys.argv[1:]
text=open(cfg,encoding='utf-8').read().splitlines()
pat=re.compile(r'^(\s*)'+re.escape(alias)+r':\s*(?:#.*)?$')
block=None
for i,line in enumerate(text):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); out=[line]
    for s in text[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    block='\n'.join(out); break
if block is None: raise SystemExit('ERROR alias block not found')
rows={}
for line in open(devfile,encoding='utf-8',errors='replace'):
    m=re.match(r'^\s*([^:\s]+):\s+(.+?)\s+\((\d+) MiB,\s*(\d+) MiB free\)\s*$',line)
    if m and (m.group(1).lower().startswith('rocm') or 'radeon' in m.group(2).lower() or 'amd' in m.group(2).lower()):
        rows[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)))
md=re.search(r'(?<!\S)(?:--device|-dev)\s+("[^"]+"|\'[^\']+\'|\S+)',block)
if md:
    order=[x.strip() for x in md.group(1).strip('"\'').split(',') if x.strip()]
else:
    order=list(rows)
if len(order) != 2:
    raise SystemExit(f'ERROR expected exactly two GPU devices, got order={order}, detected={list(rows)}')
for d in order:
    if d not in rows:
        raise SystemExit(f'ERROR device {d!r} from --device not found in --list-devices output: {list(rows)}')
t0,t1=rows[order[0]][1],rows[order[1]][1]
if min(t0,t1) <= 0: raise SystemExit('ERROR invalid VRAM totals')
cap=Fraction(t0,t1).limit_denominator(16)
mid=Fraction(math.sqrt(t0/t1)).limit_denominator(16)
inv_cap=Fraction(cap.denominator, cap.numerator)
inv_mid=Fraction(mid.denominator, mid.numerator)
print('DEVICE0='+order[0])
print('DEVICE1='+order[1])
print('DESC0='+rows[order[0]][0])
print('DESC1='+rows[order[1]][0])
print('TOTAL0_MIB='+str(t0))
print('TOTAL1_MIB='+str(t1))
print(f'MID_RATIO={mid.numerator},{mid.denominator}')
print(f'CAP_RATIO={cap.numerator},{cap.denominator}')
print(f'INV_MID_RATIO={inv_mid.numerator},{inv_mid.denominator}')
print(f'INV_CAP_RATIO={inv_cap.numerator},{inv_cap.denominator}')
PY
)
for kv in "${DETECT[@]}"; do export "$kv"; done

for v in MID_RATIO CAP_RATIO INV_MID_RATIO INV_CAP_RATIO; do
  [[ -n "${!v:-}" ]] || { echo "ERROR ratio detection failed: $v" >&2; exit 7; }
done
[[ "$MID_RATIO" != "1,1" || "$CAP_RATIO" != "1,1" ]] || {
  echo "SKIP: devices report equal total VRAM; no heterogeneous ratio to test."
  exit 0
}

echo "Detected order: $DEVICE0 [$DESC0, ${TOTAL0_MIB} MiB] -> $DEVICE1 [$DESC1, ${TOTAL1_MIB} MiB]"
echo "Ratios: EVEN=1,1 MID=$MID_RATIO CAP=$CAP_RATIO INV_MID=$INV_MID_RATIO INV_CAP=$INV_CAP_RATIO"

PLE_DIRECT=0
grep -q -- '--lazy-mode on-direct' "$BASE_BIN/llama-server" && PLE_DIRECT=1 || true
BASE_SHA="$(sha256sum "$BASE_BIN/llama-server.real" | awk '{print $1}')"

make_ratio_runtime() {
  local name="$1" ratio="$2"
  local dst="$RUNTIME_ROOT/$name/bin"
  if [[ -e "$dst" ]]; then
    [[ -x "$dst/llama-server" && -x "$dst/llama-server.real" ]] || {
      echo "ERROR incomplete existing runtime: $dst" >&2; exit 8;
    }
    local old_sha
    old_sha="$(sha256sum "$dst/llama-server.real" | awk '{print $1}')"
    [[ "$old_sha" == "$BASE_SHA" ]] || {
      echo "ERROR stale existing runtime ELF: $dst" >&2; exit 8;
    }
    grep -Fq -- "--tensor-split \"$ratio\"" "$dst/llama-server" || {
      echo "ERROR existing wrapper has wrong ratio: $dst expected=$ratio" >&2; exit 8;
    }
    echo "$dst"
    return 0
  fi
  mkdir -p "$dst"
  cp -a "$BASE_BIN/." "$dst/"
  cat > "$dst/llama-server" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
out=()
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    -sm|--split-mode) shift; [[ \$# -gt 0 ]] && shift ;;
    -sm=*|--split-mode=*) shift ;;
    -ts|--tensor-split) shift; [[ \$# -gt 0 ]] && shift ;;
    -ts=*|--tensor-split=*) shift ;;
    -fit|--fit) shift; if [[ \$# -gt 0 && ( "\$1" == on || "\$1" == off ) ]]; then shift; fi ;;
    -fit=*|--fit=*) shift ;;
EOF
  if [[ "$PLE_DIRECT" == 1 ]]; then
    cat >> "$dst/llama-server" <<'EOF'
    -lzm|--lazy-mode) shift; [[ $# -gt 0 ]] && shift ;;
    -lzm=*|--lazy-mode=*) shift ;;
EOF
  fi
  cat >> "$dst/llama-server" <<EOF
    *) out+=("\$1"); shift ;;
  esac
done
extra=(--split-mode tensor --tensor-split "$ratio" --fit off)
EOF
  if [[ "$PLE_DIRECT" == 1 ]]; then
    echo 'extra+=(--lazy-mode on-direct)' >> "$dst/llama-server"
  fi
  cat >> "$dst/llama-server" <<'EOF'
exec "$SELF_DIR/llama-server.real" "${extra[@]}" "${out[@]}"
EOF
  chmod +x "$dst/llama-server"
  echo "$dst"
}

MID_BIN="$(make_ratio_runtime tensor-mid "$MID_RATIO")"
CAP_BIN="$(make_ratio_runtime tensor-cap "$CAP_RATIO")"
INV_MID_BIN="$(make_ratio_runtime tensor-inv-mid "$INV_MID_RATIO")"
INV_CAP_BIN="$(make_ratio_runtime tensor-inv-cap "$INV_CAP_RATIO")"
for d in "$MID_BIN" "$CAP_BIN" "$INV_MID_BIN" "$INV_CAP_BIN"; do
  s="$(sha256sum "$d/llama-server.real" | awk '{print $1}')"
  [[ "$s" == "$BASE_SHA" ]] || { echo "ERROR real binaries differ: $d" >&2; exit 9; }
done

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" --config "$CONFIG" \
  --source-alias "$EVEN_ALIAS" --alias "$MID_ALIAS" --r2-bin "$MID_BIN" --jmax keep --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" --config "$CONFIG" \
  --source-alias "$EVEN_ALIAS" --alias "$CAP_ALIAS" --r2-bin "$CAP_BIN" --jmax keep --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" --config "$CONFIG" \
  --source-alias "$EVEN_ALIAS" --alias "$INV_MID_ALIAS" --r2-bin "$INV_MID_BIN" --jmax keep --replace
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" --config "$CONFIG" \
  --source-alias "$EVEN_ALIAS" --alias "$INV_CAP_ALIAS" --r2-bin "$INV_CAP_BIN" --jmax keep --replace --validate

cat > "$PHASE6_SRC/r2-meta/phase6b-ratios.env" <<EOF
DEVICE0=$DEVICE0
DEVICE1=$DEVICE1
DESC0=$DESC0
DESC1=$DESC1
TOTAL0_MIB=$TOTAL0_MIB
TOTAL1_MIB=$TOTAL1_MIB
EVEN_RATIO=1,1
MID_RATIO=$MID_RATIO
CAP_RATIO=$CAP_RATIO
INV_MID_RATIO=$INV_MID_RATIO
INV_CAP_RATIO=$INV_CAP_RATIO
REAL_BINARY_SHA256=$BASE_SHA
PLE_DIRECT_PRESERVED=$PLE_DIRECT
EOF

cat <<EOF
PHASE6B_RATIO_SWEEP_READY=1
EVEN_ALIAS=$EVEN_ALIAS
MID_ALIAS=$MID_ALIAS
CAP_ALIAS=$CAP_ALIAS
INV_MID_ALIAS=$INV_MID_ALIAS
INV_CAP_ALIAS=$INV_CAP_ALIAS
RATIOS=1,1;$MID_RATIO;$CAP_RATIO;$INV_MID_RATIO;$INV_CAP_RATIO
DEVICE_ORDER=$DEVICE0,$DEVICE1
PRODUCTION_PROMOTED=NO
Next: bash $SCRIPT_DIR/run_phase6b_verified.sh
EOF
