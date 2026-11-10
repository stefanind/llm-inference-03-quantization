"""Static-group GPTQ: use calibration inputs to compensate INT4 rounding error.

Weights use the same group-128 scales and packed forward as ordinary rounding.
Only code selection changes; calibration statistics are discarded afterward.
Algorithm reference: https://github.com/IST-DASLab/gptq/blob/main/gptq.py
"""

import torch
from torch import nn

from .symmetric import GROUP_SIZE, NaiveQuantizedLinear, quantize_weight


DAMPING = 0.01


@torch.inference_mode()
def quantize_weight_gptq(weight, second_moment):
    """Quantize W[out, in] using the mean X.T @ X for calibration inputs X[tokens, in]."""
    # Keep exactly the baseline's scales. Reuse its code buffer, replacing each
    # rounded code below after compensating for errors in earlier columns.
    codes, scale = quantize_weight(weight, bits=4, grouped=True)
    values = weight.detach().float().clone()
    columns = values.shape[1]
    assert second_moment.shape == (columns, columns), "Statistics must cover every input channel."
    hessian = second_moment.float().clone()
    assert torch.isfinite(hessian).all(), "Calibration statistics must be finite."

    # An unseen channel contributes no calibration output. Give it a safe diagonal
    # and a zero weight, so it cannot make the factorization singular.
    unseen = hessian.diagonal() == 0
    hessian[unseen, unseen] = 1
    values[:, unseen] = 0
    hessian.diagonal().add_(DAMPING * hessian.diagonal().mean())
    factor = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(factor)
    inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    assert torch.isfinite(inverse_factor).all(), "The damped inverse factor must be finite."

    # Work on 128 columns at a time. Within a block, update immediately; apply
    # accumulated errors to later blocks with one matrix multiplication.
    for start in range(0, columns, GROUP_SIZE):
        end = start + GROUP_SIZE
        block = values[:, start:end].clone()
        errors = torch.empty_like(block)
        inverse_block = inverse_factor[start:end, start:end]
        group_scale = scale[:, start // GROUP_SIZE, 0]
        for column in range(GROUP_SIZE):
            current = block[:, column]
            code = torch.round(current / group_scale).clamp(-7, 7)
            codes[:, start + column] = code.to(torch.int8)
            error = (current - code * group_scale) / inverse_block[column, column]
            block[:, column:] -= error[:, None] * inverse_block[column, column:][None, :]
            errors[:, column] = error
        values[:, end:] -= errors @ inverse_factor[start:end, end:]

    assert codes.min() >= -7 and codes.max() <= 7, "GPTQ must preserve the symmetric INT4 code range."
    return codes, scale


@torch.inference_mode()
def quantize_model(model, calibration_tokens):
    """Calibrate and replace Qwen block projections, propagating quantized outputs."""
    assert not model.training, "Calibrate in evaluation mode."
    assert calibration_tokens.ndim == 2 and calibration_tokens.dtype == torch.long
    assert calibration_tokens.numel() > 0, "Calibration needs fixed-length token sequences."
    assert not model.model.has_sliding_layers, "This recipe expects Qwen's full-attention blocks."
    device = next(model.parameters()).device

    # Ask Qwen to prepare its own mask and rotary positions once. Every sample has
    # the same length and starts at position zero, so all block calls can reuse them.
    block_kwargs = {}

    def capture_kwargs(module, args, kwargs):
        block_kwargs.update(kwargs)

    handle = model.model.layers[0].register_forward_pre_hook(capture_kwargs, with_kwargs=True)
    try:
        model.model(input_ids=calibration_tokens[:1].to(device), use_cache=False)
    finally:
        handle.remove()
    hidden_states = model.model.embed_tokens(calibration_tokens.to(device))

    converted = 0
    for index, block in enumerate(model.model.layers):
        projections = {name: layer for name, layer in block.named_modules() if isinstance(layer, nn.Linear)}
        assert len(projections) == 7, "Expected all seven projections in each Qwen block."
        moments = {
            name: torch.zeros((layer.in_features, layer.in_features), device=device, dtype=torch.float32)
            for name, layer in projections.items()
        }

        def collect_statistics(name):
            def hook(module, inputs, output):
                # Every position is real text: there are no padding tokens to mask.
                inputs = inputs[0].reshape(-1, module.in_features).float()
                moments[name].addmm_(inputs.T, inputs)
            return hook

        handles = [layer.register_forward_hook(collect_statistics(name)) for name, layer in projections.items()]
        try:
            for sample in hidden_states.split(1):
                block(sample, **block_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        for name, layer in projections.items():
            moment = moments.pop(name)
            moment.div_(calibration_tokens.numel())
            codes, scale = quantize_weight_gptq(layer.weight, moment)
            replacement = NaiveQuantizedLinear(
                layer, bits=4, grouped=True, packed=True, codes_and_scale=(codes, scale)
            )
            parent_name, _, child_name = name.rpartition(".")
            setattr(block.get_submodule(parent_name), child_name, replacement)
            converted += 1
            del moment, codes, scale
        del projections, moments

        # Feed the next block the outputs it will actually see during inference.
        # Reuse this cache in place instead of retaining every block's activations.
        for sample in hidden_states.split(1):
            sample.copy_(block(sample, **block_kwargs))
        assert torch.isfinite(hidden_states).all(), "Quantized calibration outputs must be finite."
        print(f"GPTQ: calibrated block {index + 1}/{len(model.model.layers)}.", flush=True)

    return converted
