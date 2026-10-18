"""Same arithmetic, alternate CTA order; graph capture only selects the path."""
from functools import lru_cache
from pathlib import Path
@lru_cache(maxsize=2)
def extension(order):
    from tensorfold.cuda.build import load
    assert order in (1,2)
    name = 'order_old' if order == 1 else 'expert_major'
    p=Path(__file__).parent
    return load(name='glm_rrs_'+name+'_20261018',sources=[str(p/(name+'.cpp')),str(p/(name+'.cu'))],
                extra_cuda_cflags=['-O3'],verbose=False)
