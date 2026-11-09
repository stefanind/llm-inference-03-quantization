"""Symmetric INT8 with one FP32 scale per matrix or per output row."""

import torch
from torch import nn
from torch.nn import functional as F


def quantize_int8(weight, per_channel=False):
    assert weight.is_floating_point() and weight.numel() > 0
    values = weight.detach().float()
    assert torch.isfinite(values).all(), "Weights must be finite."
    # Linear weights are [output rows, input columns]. Keeping the column
    # dimension makes each row's scale broadcast across that row's weights.
    if per_channel:
        assert values.ndim == 2, "Per-channel quantization expects a weight matrix."
        max_abs = values.abs().amax(dim=1, keepdim=True)
    else:
        max_abs = values.abs().max()
    scale = max_abs / 127
    # A zero matrix or row needs a safe scale. This also handles underflow to zero.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Round to the nearest code, clamp to the chosen symmetric range, then store.
    qweight = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
    assert torch.isfinite(scale).all() and (scale > 0).all(), "Scales must be finite and positive."
    assert qweight.dtype == torch.int8
    return qweight, scale


def dequantize_int8(qweight, scale):
    # Values were checked during quantization. These metadata checks do not
    # synchronize the GPU on every forward, unlike checking a tensor's contents.
    assert qweight.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.ndim == 0 or (
        qweight.ndim == 2 and scale.shape == (qweight.shape[0], 1)
    ), "Expected one scale per matrix or one scale per output row."
    return qweight.float() * scale


class NaiveInt8Linear(nn.Module):
    """Store compressed weights; deliberately reconstruct them on every forward.

    This teaches the difference between compressed storage and fast execution.
    F.linear still performs floating-point computation, not an INT8 matmul.
    Both scale granularities use the same reconstruction and forward logic.
    """

    def __init__(self, linear, per_channel=False):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        qweight, scale = quantize_int8(linear.weight, per_channel=per_channel)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)
        self.train(linear.training)

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        weight = dequantize_int8(self.qweight, self.scale).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


@torch.inference_mode()
def run_demo():
    """Compare both scale granularities before and after adding one outlier."""
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(8, 128, generator=generator)
    weights = torch.randn(64, 128, generator=generator) * 0.1
    outlier_weights = weights.clone()
    outlier_weights[0, 0] = 10.0
    results = []
    for name, weight in (("ordinary", weights), ("one_outlier", outlier_weights)):
        for per_channel in (False, True):
            qweight, scale = quantize_int8(weight, per_channel=per_channel)
            reconstructed = dequantize_int8(qweight, scale)
            error = (weight - reconstructed).abs()
            # Broadcasting checks each weight against its own scale's half-step.
            assert (error <= scale / 2 + 1e-6).all(), "Rounding error should be <= half a step."
            output_error = F.linear(inputs, weight) - F.linear(inputs, reconstructed)
            weight_bytes = qweight.numel() * qweight.element_size()
            scale_bytes = scale.numel() * scale.element_size()
            results.append({
                "example": name,
                "variant": "int8_per_channel" if per_channel else "int8_per_tensor",
                "source_dtype": str(weight.dtype),
                "num_scales": scale.numel(),
                "scale_min": scale.min().item(),
                "scale_max": scale.max().item(),
                "original_weight_bytes": weight.numel() * weight.element_size(),
                "int8_weight_and_scale_bytes": weight_bytes + scale_bytes,
                "weight_mse": error.square().mean().item(),
                "output_mse": output_error.square().mean().item(),
            })
    return results
