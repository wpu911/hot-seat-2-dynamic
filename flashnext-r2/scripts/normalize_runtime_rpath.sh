#!/usr/bin/env bash
set -euo pipefail

# Normalize only staged EXPERIMENT runtimes. This never touches production.
# Historical model-isolation deployments rewrote build-tree RUNPATH entries to
# $ORIGIN so llama/ggml libraries load from the model's own bin directory. Keep
# non-build entries (for example system/ROCm paths) intact.
#
# Usage:
#   bash normalize_runtime_rpath.sh /path/to/runtime/bin
#   OUT=/path/audit.txt bash normalize_runtime_rpath.sh /path/to/runtime/bin

RUNTIME="${1:-${RUNTIME:-}}"
[[ -n "$RUNTIME" ]] || { echo "ERROR runtime bin directory required" >&2; exit 2; }
RUNTIME="$(readlink -f "$RUNTIME")"
[[ -d "$RUNTIME" ]] || { echo "ERROR runtime dir missing: $RUNTIME" >&2; exit 3; }
command -v readelf >/dev/null 2>&1 || { echo "ERROR readelf missing" >&2; exit 4; }
command -v patchelf >/dev/null 2>&1 || {
  echo "ERROR patchelf missing; refusing to leave build-tree RUNPATH in staged runtime" >&2
  exit 5
}

OUT="${OUT:-}"
if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  exec > >(tee "$OUT") 2>&1
fi

echo "=== normalize staged runtime RUNPATH ==="
echo "runtime=$RUNTIME"
CHANGED=0
SCANNED=0

while IFS= read -r -d '' f; do
  # readelf is the authority; `file` is not required in minimal containers.
  readelf -h "$f" >/dev/null 2>&1 || continue
  SCANNED=$((SCANNED + 1))
  old="$(patchelf --print-rpath "$f" 2>/dev/null || true)"
  [[ -n "$old" ]] || continue

  new="$(python3 - "$old" <<'PY'
import sys
parts=[p for p in sys.argv[1].split(':') if p]
out=[]
replaced=False
for p in parts:
    # CMake build-tree locations are not allowed in a relocatable staged bundle.
    if '/app/share/llama_box/src/' in p or '/build-r2-' in p or '/build/' in p:
        p='$ORIGIN'; replaced=True
    if p not in out:
        out.append(p)
print(':'.join(out) if replaced else sys.argv[1])
PY
)"
  if [[ "$new" != "$old" ]]; then
    before="$(sha256sum "$f" | awk '{print $1}')"
    patchelf --set-rpath "$new" "$f"
    after="$(sha256sum "$f" | awk '{print $1}')"
    echo "RPATH_REWRITE file=$(basename "$f")"
    echo "  old=$old"
    echo "  new=$new"
    echo "  sha256_before=$before"
    echo "  sha256_after=$after"
    CHANGED=$((CHANGED + 1))
  fi
done < <(find "$RUNTIME" -maxdepth 1 -type f -print0)

# Local inference-library symlinks must remain inside the staged bin directory.
while IFS= read -r -d '' link; do
  name="$(basename "$link")"
  case "$name" in
    libllama*.so*|libggml*.so*|libmtmd*.so*) ;;
    *) continue ;;
  esac
  target="$(readlink "$link")"
  if [[ "$target" = /* ]]; then
    resolved="$(readlink -f "$link" || true)"
    case "$resolved" in
      "$RUNTIME"/*) ;;
      *) echo "ERROR local inference symlink escapes runtime: $link -> $target" >&2; exit 10 ;;
    esac
  fi
done < <(find "$RUNTIME" -maxdepth 1 -type l -print0)

SERVER="$RUNTIME/llama-server"
[[ -x "$RUNTIME/llama-server.real" ]] && SERVER="$RUNTIME/llama-server.real"
[[ -x "$SERVER" ]] || { echo "ERROR no real llama-server ELF in $RUNTIME" >&2; exit 11; }

LDD="$(mktemp)"
trap 'rm -f "$LDD"' EXIT
ldd "$SERVER" | tee "$LDD"
if grep -q 'not found' "$LDD"; then
  echo "ERROR unresolved dependency after RUNPATH normalization" >&2
  exit 12
fi
if grep -E '/app/share/llama_box/src/.*/build[^/]*/' "$LDD" >/dev/null; then
  echo "ERROR build-tree dependency remains after normalization" >&2
  grep -E '/app/share/llama_box/src/.*/build[^/]*/' "$LDD" >&2 || true
  exit 13
fi

echo "RPATH_NORMALIZE=PASS"
echo "ELF_SCANNED=$SCANNED"
echo "ELF_REWRITTEN=$CHANGED"
echo "SERVER_SHA256=$(sha256sum "$SERVER" | awk '{print $1}')"
