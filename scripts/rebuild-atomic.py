"""Optional ARM64/SM121 rebuild of the disabled, captured expert-reduction extension."""
import argparse
from pathlib import Path
import shutil
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
native = root / 'vendor/native'
module = load(
    name='glm_prefill2000_expert_atomic',
    sources=[str(native/'expert-atomic.cpp'), str(native/'expert-atomic.cu')],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_121,code=sm_121'],
    verbose=True,
)
args.output.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(module.__file__, args.output)
