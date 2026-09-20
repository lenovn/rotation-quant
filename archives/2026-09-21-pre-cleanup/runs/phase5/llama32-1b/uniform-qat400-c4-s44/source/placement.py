"""Sequential layer placement for QAT memory; no data or gradient replication."""
import torch


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _align_inputs(module, args, kwargs):
    device = next(module.parameters()).device
    return _move(args, device), _move(kwargs, device)


def place_student(model, device_count):
    if device_count < 1 or device_count > torch.cuda.device_count():
        raise ValueError("Requested QAT layer devices are not visible")
    model.model.embed_tokens.to("cuda:0")
    model.model.norm.to("cuda:0")
    model.lm_head.to("cuda:0")
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to("cuda:0")
    count = len(model.model.layers)
    assignment = {}
    for index, layer in enumerate(model.model.layers):
        device = min(index * device_count // count, device_count - 1)
        layer.to(f"cuda:{device}")
        assignment[str(index)] = device
        if device_count > 1 and not getattr(layer, "_phase5_alignment", False):
            layer.register_forward_pre_hook(_align_inputs, with_kwargs=True)
            layer._phase5_alignment = True
    if device_count > 1 and not getattr(model.model.norm, "_phase5_alignment", False):
        model.model.norm.register_forward_pre_hook(_align_inputs, with_kwargs=True)
        model.model.norm._phase5_alignment = True
    return dict(layer_devices=assignment, embedding_head_norm_device=0,
                global_batch_unchanged=True, tensor_parallel=False)
