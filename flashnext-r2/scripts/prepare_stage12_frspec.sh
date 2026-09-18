#!/usr/bin/env bash
set -euo pipefail

# Flash Next R2 Stage 12: workload-ranked FR-Spec reduced-vocabulary MTP head.
#
# Intended stack:
#   exact production HotSeat snapshot
#     + Stage-10 modern qwen4exp MTP candidate (PR #28243)
#     + this narrow FR-Spec semantic port
#
# A/B aliases use the SAME Stage-12 binary. FULL uses the original draft sidecar;
# FR65 uses a copied/trimmed 65,536-row sidecar. Target GGUF is never modified.

BASE_SRC="${BASE_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-mtp-upstream-20260918}"
R2_SRC="${R2_SRC:-/app/share/llama_box/src/llama.cpp-flashnext-r2-frspec-20260918}"
RUNTIME="${RUNTIME:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-frspec}"
CONFIG="${CONFIG:-/app/share/llama_box/config/config-rocm714.yaml}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen3.8-flash-next-r2-mtp-upstream:256k}"
FULL_ALIAS="${FULL_ALIAS:-qwen3.8-flash-next-r2-frspec-full:256k}"
FR_ALIAS="${FR_ALIAS:-qwen3.8-flash-next-r2-frspec-65k:256k}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSIONS="${SESSIONS:-/app/share/openclaw_data/.openclaw/agents/main/sessions}"
FR_DIR="${FR_DIR:-/app/share/llm/Qwen3.8-Flash-Next-GGUF/mtp-frspec}"
K="${K:-65536}"
RANK_MAP="${RANK_MAP:-$FR_DIR/rank-openclaw-current.json}"

[[ -e "$BASE_SRC/.git" ]] || {
  echo "ERROR: Stage-12 BASE_SRC is not a git repo/worktree: $BASE_SRC" >&2
  echo "Expected the clean Stage-10 MTP candidate by default." >&2
  exit 2
}
[[ -z "$(git -C "$BASE_SRC" status --porcelain)" ]] || {
  echo "ERROR: Stage-12 base must be clean/committed: $BASE_SRC" >&2
  exit 3
}
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 4; }
[[ ! -e "$R2_SRC" && ! -e "$RUNTIME" ]] || {
  echo "ERROR: Stage-12 source/runtime already exists; refusing overwrite" >&2
  echo "source=$R2_SRC runtime=$RUNTIME" >&2
  exit 5
}

grep -q 'llama_model_qwen4exp::graph_mtp::graph_mtp' "$BASE_SRC/src/models/qwen4exp.cpp" || {
  echo "ERROR: BASE_SRC has no modern qwen4exp graph_mtp; Stage 12 should sit on Stage 10." >&2
  exit 6
}

# Find the draft sidecar actually used by the source alias. Do not assume a file
# name from memory; those are how three-hour benchmark sessions become folklore.
DRAFT_PATH="$(python3 - "$CONFIG" "$SOURCE_ALIAS" <<'PY'
from pathlib import Path
import re, sys
p, alias = Path(sys.argv[1]), sys.argv[2]
lines=p.read_text(encoding='utf-8').splitlines()
pat=re.compile(r'^(\s*)'+re.escape(alias)+r':\s*(?:#.*)?$')
block=None
for i,line in enumerate(lines):
    m=pat.match(line)
    if not m: continue
    ind=len(m.group(1)); out=[]
    for s in lines[i+1:]:
        if s.strip() and not s.lstrip().startswith('#') and len(s)-len(s.lstrip(' ')) <= ind:
            break
        out.append(s)
    block='\n'.join(out); break
if block is None: raise SystemExit('ERROR: alias not found: '+alias)
rx=re.compile(r'(?<!\S)(?:--spec-draft-model|--model-draft|--draft-model|-md)\s+("[^"]*"|\'[^\']*\'|\S+)')
vals=[m.group(1).strip('"\'') for m in rx.finditer(block)]
if len(vals)!=1: raise SystemExit(f'ERROR: expected one draft option, found {len(vals)}')
print(vals[0])
PY
)"
[[ -f "$DRAFT_PATH" ]] || { echo "ERROR: original draft missing: $DRAFT_PATH" >&2; exit 7; }

echo "Base source       : $BASE_SRC"
echo "Base HEAD         : $(git -C "$BASE_SRC" rev-parse HEAD)"
echo "Source alias      : $SOURCE_ALIAS"
echo "Original draft    : $DRAFT_PATH"
echo "FR-Spec K         : $K"
echo "Frequency map     : $RANK_MAP"

# Ensure this is a sidecar we can actually trim. Shared-embedding MTP files may
# intentionally omit output.weight; FR-Spec then has nothing to reduce.
PYTHONPATH="$BASE_SRC/gguf-py${PYTHONPATH:+:$PYTHONPATH}" python3 - "$DRAFT_PATH" <<'PY'
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
outs=[t for t in r.tensors if t.name=='output.weight']
if len(outs)!=1:
    raise SystemExit(f'ERROR: FR-Spec requires exactly one draft output.weight; found {len(outs)}')
if any(t.name=='d2t' for t in r.tensors):
    raise SystemExit('ERROR: source draft is already FR-Spec/d2t trimmed')
t=outs[0]
print('draft output.weight', list(map(int,t.shape)), t.tensor_type.name)
PY

BASE_HEAD="$(git -C "$BASE_SRC" rev-parse HEAD)"
git -C "$BASE_SRC" worktree add --detach "$R2_SRC" "$BASE_HEAD"
mkdir -p "$R2_SRC/r2-meta"
{
  echo "created=$(date -Is)"
  echo "base_source=$BASE_SRC"
  echo "base_head=$BASE_HEAD"
  echo "source_alias=$SOURCE_ALIAS"
  echo "original_draft=$DRAFT_PATH"
  echo "frspec_k=$K"
  echo "reference_initial=7a3aa1dd59f904d8f624afb13460715169838b65"
  echo "reference_graph_mtp=ebb3def772fe153b963f9be48cb81e0208a081f9"
} > "$R2_SRC/r2-meta/stage12-frspec-base.txt"

python3 "$SCRIPT_DIR/port_stage12_frspec_qwen4exp.py" "$R2_SRC/src/models/qwen4exp.cpp"
git -C "$R2_SRC" diff --check
git -C "$R2_SRC" diff > "$R2_SRC/r2-meta/stage12-frspec-port.diff"

# Build a workload-specific token frequency ranking unless the caller supplied
# an existing map. This uses local OpenClaw histories and stores only token ids.
mkdir -p "$FR_DIR"
if [[ ! -f "$RANK_MAP" ]]; then
  python3 "$SCRIPT_DIR/build_frspec_frequency_map.py" \
    --sessions "$SESSIONS" \
    --model "$SOURCE_ALIAS" \
    --out "$RANK_MAP"
fi

DRAFT_BASE="$(basename "$DRAFT_PATH")"
DRAFT_STEM="${DRAFT_BASE%.gguf}"
FR_DRAFT="$FR_DIR/${DRAFT_STEM}-frspec-${K}.gguf"
if [[ -e "$FR_DRAFT" ]]; then
  echo "ERROR: FR-Spec draft already exists; refusing overwrite: $FR_DRAFT" >&2
  exit 8
fi

PYTHONPATH="$R2_SRC/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$SCRIPT_DIR/trim_frspec_sidecar.py" \
    "$DRAFT_PATH" "$FR_DRAFT" --k "$K" --rank "$RANK_MAP"
sha256sum "$DRAFT_PATH" "$FR_DRAFT" | tee "$R2_SRC/r2-meta/stage12-draft-sha256.txt"
ls -lh "$DRAFT_PATH" "$FR_DRAFT" | tee "$R2_SRC/r2-meta/stage12-draft-sizes.txt"

if [[ -z "${ROCM_PATH:-}" ]]; then
  if [[ -x /opt/host-rocm/core-10.0/lib/llvm/bin/clang++ ]]; then
    ROCM_PATH=/opt/host-rocm/core-10.0
  else
    ROCM_PATH=/opt/rocm
  fi
fi
export ROCM_PATH
HIP_CXX="${CMAKE_HIP_COMPILER:-$ROCM_PATH/lib/llvm/bin/clang++}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1100;gfx1201}"
BUILD="${BUILD:-$R2_SRC/build-r2-frspec}"

cmake -S "$R2_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX"
cmake --build "$BUILD" -j"${JOBS:-$(nproc)}" --target llama-server

test -x "$BUILD/bin/llama-server" || { echo "ERROR: llama-server missing" >&2; exit 20; }
mkdir -p "$RUNTIME"
cp -a "$BUILD/bin/." "$RUNTIME/"
sha256sum "$RUNTIME/llama-server" | tee "$R2_SRC/r2-meta/stage12-llama-server.sha256"
"$RUNTIME/llama-server" --version | tee "$R2_SRC/r2-meta/stage12-version.txt" || true

# Same binary and source alias. FULL keeps the original sidecar; FR65 changes only
# the draft file path. Preserve JMAX and every speculative knob from Stage 10.
python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$FULL_ALIAS" --r2-bin "$RUNTIME" --jmax keep --replace

python3 "$SCRIPT_DIR/install_llamaswap_r2_alias.py" \
  --config "$CONFIG" --source-alias "$SOURCE_ALIAS" \
  --alias "$FR_ALIAS" --r2-bin "$RUNTIME" --jmax keep \
  --draft-model "$FR_DRAFT" --replace --validate

echo
echo "Stage-12 FR-Spec aliases ready."
echo "FULL : $FULL_ALIAS"
echo "FR65 : $FR_ALIAS"
echo "draft: $FR_DRAFT"
echo "same binary: $RUNTIME/llama-server"
echo "Next: bash $SCRIPT_DIR/run_stage12_frspec_ab.sh"
