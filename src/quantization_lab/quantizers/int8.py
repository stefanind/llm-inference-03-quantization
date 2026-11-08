"""Symmetric per-tensor INT8: one FP32 scale for the entire weight matrix."""

import torch
from torch import nn
from torch.nn import functional as F


def quantize_int8(weight):
    assert weight.is_floating_point() and weight.numel() > 0
    values = weight.detach().float()
    assert torch.isfinite(values).all(), "Weights must be finite."
    # Map the largest magnitude to 127. Every other value shares this step size.
    scale = values.abs().max() / 127
    # Zero (or a value too small to represent after division) needs a safe scale.
    if scale == 0:
        scale = torch.ones_like(scale)
    # Round to the nearest code, clamp to the chosen symmetric range, then store.
    qweight = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
    assert scale > 0 and qweight.dtype == torch.int8
    return qweight, scale


def dequantize_int8(qweight, scale):
    assert qweight.dtype == torch.int8
    assert scale.numel() == 1 and torch.isfinite(scale).all() and scale > 0
    return qweight.float() * scale


class NaiveInt8Linear(nn.Module):
    """Store compressed weights; deliberately reconstruct them on every forward.

    This teaches the difference between compressed storage and fast execution.
    F.linear still performs floating-point computation, not an INT8 matmul.
    """

    def __init__(self, linear):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        qweight, scale = quantize_int8(linear.weight)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        weight = dequantize_int8(self.qweight, self.scale).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


@torch.inference_mode()
def run_demo():
    """Hold ordinary weights fixed, then show what a single outlier changes."""
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(8, 128, generator=generator)
    weights = torch.randn(64, 128, generator=generator) * 0.1
    outlier_weights = weights.clone()
    outlier_weights[0, 0] = 10.0
    results = []
    for name, weight in (("ordinary", weights), ("one_outlier", outlier_weights)):
        qweight, scale = quantize_int8(weight)
        reconstructed = dequantize_int8(qweight, scale)
        error = (weight - reconstructed).abs()
        assert error.max() <= scale / 2 + 1e-6, "Nearest rounding error should be <= half a step."
        output_error = F.linear(inputs, weight) - F.linear(inputs, reconstructed)
        stored_bytes = qweight.numel() * qweight.element_size() + scale.element_size()
        results.append({
            "example": name,
            "source_dtype": str(weight.dtype),
            "scale": scale.item(),
            "original_weight_bytes": weight.numel() * weight.element_size(),
            "int8_weight_and_scale_bytes": stored_bytes,
            "weight_mse": error.square().mean().item(),
            "output_mse": output_error.square().mean().item(),
        })
    return results
