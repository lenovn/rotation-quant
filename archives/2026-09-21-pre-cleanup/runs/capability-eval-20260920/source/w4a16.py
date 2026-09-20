"""Load fixed W4 weights at A16 without recalibration or requantization."""
from pathlib import Path
import torch
from experiments.phase3.architecture import load_model


def load_w4a16(model_path, package, device='cuda'):
    checkpoint = torch.load(package, map_location='cpu', weights_only=False)
    if 'weights' in checkpoint and 'high_precision' in checkpoint:
        del checkpoint
        from experiments.phase3.common import MODEL_PATH, wrappers
        from experiments.phase3.postprocess import load_static
        if Path(model_path).resolve() != MODEL_PATH.resolve():
            raise ValueError('Frozen loader model path differs from requested model')
        model, records = load_static(Path(package), device=device)
        wrapped = list(wrappers(model).items())
        if len(wrapped) != 7 * model.config.num_hidden_layers:
            raise ValueError('Qwen W4 coverage mismatch')
        for name, wrapper in wrapped:
            if (wrapper.online_full_had or wrapper.online_partial_had
                    or wrapper.out_quantizer.bits != 16
                    or type(wrapper.module) is not torch.nn.Linear):
                raise ValueError('Cannot remove non-input-quantization behavior: ' + name)
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent), child, wrapper.module)
        if wrappers(model):
            raise ValueError('Backbone activation wrappers remain')
        return model, dict(backbone_w4_linears=len(records), activation_bits=16,
            package=str(package), method='FIRON weights / A16',
            loading='exact final FIRON packed W4 and high-precision tensors; input quantization removed; no calibration or requantization',
            w4_module_names=list(records))
    model = load_model(model_path, untie=True)
    source = checkpoint['model']
    mapped = {}
    for name, tensor in model.state_dict().items():
        prefix, _, suffix = name.rpartition('.')
        wrapped = prefix + '.module.' + suffix
        key = wrapped if wrapped in source else name
        value = source[key]
        if value.shape != tensor.shape:
            raise ValueError('Shape mismatch: ' + name)
        if wrapped in source and name in source and not torch.equal(value, source[name]):
            raise ValueError('Weight aliases disagree: ' + name)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
    quantizers = checkpoint['w_quantizers']
    if len(quantizers) != 7 * model.config.num_hidden_layers:
        raise ValueError('W4 coverage mismatch')
    for name, q in quantizers.items():
        plain = name.replace('.module', '')
        weight = model.get_submodule(plain).weight.detach()
        if q.bits != 4 or not q.sym or not q.perchannel or q.weight_groupsize != -1:
            raise ValueError('Historical W4 format differs: ' + name)
        if not torch.equal(weight, source[name + '.weight']):
            raise ValueError('Loaded W4 weight differs: ' + name)
    model.requires_grad_(False).eval().to(device)
    return model, dict(backbone_w4_linears=len(quantizers), activation_bits=16,
        package=str(package), loading='exact saved GPTQ weights, embedding/head/norm; no rotation recomputation; no requantization',
        w4_module_names=[name.replace('.module', '') for name in quantizers])
