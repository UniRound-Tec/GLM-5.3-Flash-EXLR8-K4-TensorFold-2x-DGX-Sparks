"""CPU-only package and shared-library contract, no CUDA initialization."""
import ast,ctypes,importlib.metadata as md,json,platform
from pathlib import Path
root=Path('/usr/local/lib/python3.12/dist-packages/tensorfold')
assert platform.machine() in ('aarch64','arm64'),'ARM64 DGX Spark runtime required'
assert md.version('torch').startswith('2.13.0a0'), 'Use the pinned NVIDIA 26.07 PyTorch stack'
for name,version in [('nvidia-cutlass-dsl','4.7.1'),('safetensors','0.8.0'),('transformers','5.18.0')]:
    assert md.version(name)==version,(name,md.version(name),version)
for p in root.rglob('*.py'):ast.parse(p.read_text())
base=root/'families/glm5_next/cuda'
for lib,symbols in [('source_mx_kernel.so',['launch','launch_tiled']),('mx_prefill_kernel.so',['mxp_load','mxp_run','mxp_repack','mxp_bytes'])]:
    obj=ctypes.CDLL(str(base/lib))
    for name in symbols:getattr(obj,name)
print(json.dumps({'cpu_runtime_contract':True,'gpu_initialized':False}))
