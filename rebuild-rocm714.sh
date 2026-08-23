#!/usr/bin/env bash
set -euo pipefail

# ROCm 7.14 / RX 7900 XTX (gfx1100)
# Usage:
#   ./rebuild-rocm714.sh /path/to/patched/llama.cpp

SRC="${1:?usage: $0 /path/to/patched/llama.cpp}"
BUILD="$SRC/build-hip-rocm714"

unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++

cmake --build "$BUILD" -j"$(nproc)" --target llama-server

"$BUILD/bin/llama-server" --version || true
