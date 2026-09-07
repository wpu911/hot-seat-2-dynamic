#!/usr/bin/env bash
set -euo pipefail
# Build an already patched source tree; does not alter running services.
ornith_src="${1:?usage: $0 /path/to/patched/llama.cpp [build-directory]}"
ornith_build="${2:-$ornith_src/build-ornith-rocm}"
ornith_rocm="${ROCM_ROOT:-/opt/rocm}"
ornith_compiler="$ornith_rocm/lib/llvm/bin/clang++"
test -x "$ornith_compiler"
cmake -S "$ornith_src" -B "$ornith_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_CPU=ON \
  -DGGML_NATIVE=ON -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER="$ornith_compiler" \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$ornith_build" --parallel "${BUILD_JOBS:-8}" --target llama-server
"$ornith_build/bin/llama-server" --version
