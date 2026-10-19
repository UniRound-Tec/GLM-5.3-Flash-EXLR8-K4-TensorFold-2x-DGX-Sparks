# Source and native libraries

The default build copies the ARM64 `.so` files captured from the running CUDA13.3/NVIDIA26.07 service. This avoids introducing fresh numerical changes while packaging. These libraries are not generic x86/CUDA-version-independent binaries.

Corresponding source:

- `vendor/native/source_mx.cu` → `source_mx_kernel.so`, original MXFP8 ≤32-row path.
- `vendor/atlas-kernels/bridge.cu` and `mla.cu` → `libatlas_experts.so` and `libatlas_mla.so`.
- `vendor/native/mx_prefill.cu` + `vendor/native/aot/mx_{0,3,6}.{h,o}` → disabled `mx_prefill_kernel.so`. B12X source for the exported kernels is in `vendor/b12x`; `original_mx_export_probe.py` preserves the original specialization/export procedure and shapes. It references the original research mounts and is archival source, not a quickstart command or an automatic benchmark.
- Remaining PyTorch/CUDA/Triton extensions compile from `vendor/tensorfold` at startup.
- `vendor/native/expert-atomic.{cpp,cu}` → `prefill_atomic_kernel.so` (2026-10-19). The exact running snapshot imports this extension at startup, but the candidate is disabled by `prefill_atomic=false`. The copied source preserves the upstream header; its prompt down epilogue experimentally uses FP32 atomics, overriding the header's original no-atomics description. `scripts/rebuild-atomic.py` uses the original PyTorch extension name and SM121 flags; `rebuild-native.sh` invokes it. The runtime check validates import and the `prompt_atomic` entry point without initializing CUDA.

`scripts/rebuild-native.sh` rebuilds these libraries inside a matching ARM64 CUDA development container. It is optional and not called by default build. The frozen AOT object inputs are included; regenerating those objects requires the matching CuTe DSL/B12X export setup. A rebuilt binary is not certified bitwise identical merely because compilation succeeds. Follow with numerical and service checks before replacing a deployment.

Package pins and captured installed versions are in `requirements-runtime.txt` and `evidence/live-source-manifest.json`. The unused legacy `libatlas_prefill.so` was omitted after confirming no source reference. `evidence/package-sha256.json` records the delivered artifacts; original runtime hashes remain separately recorded.
