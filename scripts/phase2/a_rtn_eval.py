"""Evaluate W16-trained A with current-R per-channel RTN scales."""
import argparse
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--a-rtn-scales-output', type=Path, required=True)
options, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / 'repos/SpinQuant'))
import torch
from eval_utils import gptq_utils
from utils import quant_utils

original = gptq_utils.rtn_fwrd

@torch.no_grad()
def rtn_with_current_scales(model, dev, args, **kwargs):
    if kwargs.get('static_weight_scales') != {}:
        raise ValueError('A supplement expects an empty W16-training SW dictionary')
    scales = {}
    for i, layer in enumerate(model.model.layers):
        for name, linear in quant_utils.find_qlayers(layer, layers=[torch.nn.Linear]).items():
            scale = linear.weight.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
            if not torch.isfinite(scale).all():
                raise ValueError(f'Nonfinite scale: {i}.{name}')
            scales[f'model.layers.{i}.{name}.quantizer'] = scale.cpu()
    paired = torch.load(args.static_scale_path, map_location='cpu', weights_only=True)
    paired['weight'] = scales
    if options.a_rtn_scales_output.exists():
        raise FileExistsError(options.a_rtn_scales_output)
    torch.save(paired, options.a_rtn_scales_output)
    args.static_scale_path = str(options.a_rtn_scales_output)
    print(f'A RTN: computed {len(scales)} current-R SW tensors; retained {len(paired["activation"])} SA tensors', flush=True)
    kwargs['static_weight_scales'] = scales
    return original(model, dev, args, **kwargs)

gptq_utils.rtn_fwrd = rtn_with_current_scales
try:
    runpy.run_path(str(project / 'scripts/phase2/w4aware_ab_ptq.py'), run_name='__main__')
finally:
    gptq_utils.rtn_fwrd = original
