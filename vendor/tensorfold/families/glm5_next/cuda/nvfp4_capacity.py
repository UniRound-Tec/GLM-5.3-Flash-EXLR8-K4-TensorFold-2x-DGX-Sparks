"""Conservative CSF storage accounting; actual allocation is checked at loading."""
import re


def wrap_weight_estimate(base, cfg, world):
    pattern = re.compile(r'\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(.+)$')
    def estimate(name, info):
        match = pattern.search(name)
        if match is None or int(match[1]) >= cfg.layers:
            return base(name, info)
        size = info['data_offsets'][1] - info['data_offsets'][0]
        part = match[3]
        if part == 'weight':
            return size//world, 0
        if part.endswith('.nvfp4_csf_fixed'):
            extra = cfg.hidden*(world-1)//world if match[2]=='down' else 0
            return (size+world-1)//world+extra, 0
        if part.endswith('.nvfp4_csf_exceptions'):
            # A rank cannot retain more exceptions than the original matrix.
            return size, 0
        if part in ('weight_scale_2','input_scale'):
            return size, 0
        raise ValueError('Unrecognized NVFP4-CSF tensor '+name)
    return estimate
