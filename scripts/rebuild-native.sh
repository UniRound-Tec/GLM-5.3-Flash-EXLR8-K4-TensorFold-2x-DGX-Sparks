#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${OUTPUT_DIR:-$root/vendor/tensorfold/families/glm5_next/cuda}"
mkdir -p "$out"
nvcc -O3 --shared -Xcompiler=-fPIC -gencode=arch=compute_120f,code=sm_121 \
  "$root/vendor/native/source_mx.cu" -o "$out/source_mx_kernel.so"
for item in 'bridge.cu libatlas_experts.so' 'mla.cu libatlas_mla.so'; do
  read -r source library <<< "$item"
  nvcc -O3 --shared -Xcompiler=-fPIC --fmad=false -lineinfo \
    -gencode=arch=compute_121a,code=sm_121a \
    "$root/vendor/atlas-kernels/$source" -o "$out/$library"
done
native="$root/vendor/native"
dsl=/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/cu13/lib
nvcc -O3 --shared -std=c++17 -Xcompiler=-fPIC --fmad=false \
  -gencode=arch=compute_121a,code=sm_121a "$native/mx_prefill.cu" \
  "$native/aot/mx_0.o" "$native/aot/mx_3.o" "$native/aot/mx_6.o" \
  -lcuda -L"$dsl" -lcute_dsl_runtime -Xlinker=-rpath -Xlinker="$dsl" \
  -o "$out/mx_prefill_kernel.so"
