"""Symmetric INT8 and INT4, with FP32 scales and optional INT4 packing."""

import torch
from torch import nn
from torch.nn import functional as F

from .packing import pack_int4, unpack_int4


GROUP_SIZE = 128


def quantize_weight(weight, bits=8, per_channel=False, grouped=False):
    assert bits in (4, 8), "This experiment supports four-bit or eight-bit precision."
    assert weight.is_floating_point() and weight.numel() > 0
    values = weight.detach().float()
    assert torch.isfinite(values).all(), "Weights must be finite."
    # Grouping is fixed at 128: [rows, columns] -> [rows, groups, 128].
    # Qwen's input widths divide evenly, so no padding is needed.
    if grouped:
        assert values.ndim == 2 and values.shape[1] % GROUP_SIZE == 0, "Rows must divide into groups of 128."
        values = values.reshape(values.shape[0], -1, GROUP_SIZE)
        max_abs = values.abs().amax(dim=2, keepdim=True)
    elif per_channel:
        # Keeping the column dimension broadcasts each row's scale correctly.
        assert values.ndim == 2, "Per-channel quantization expects a weight matrix."
        max_abs = values.abs().amax(dim=1, keepdim=True)
    else:
        max_abs = values.abs().max()
    # The symmetric ranges are [-127, 127] for INT8 and [-7, 7] for INT4.
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    # A zero matrix or row needs a safe scale. This also handles underflow to zero.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Start with one code per byte. Optional INT4 packing happens afterward.
    qweight = torch.round(values / scale).clamp(-qmax, qmax).to(torch.int8)
    assert torch.isfinite(scale).all() and (scale > 0).all(), "Scales must be finite and positive."
    assert qweight.dtype == torch.int8
    # Store the original matrix shape even when scales were chosen by group.
    return qweight.reshape(weight.shape), scale


def dequantize_weight(qweight, scale):
    # Values were checked during quantization. These metadata checks do not
    # synchronize the GPU on every forward, unlike checking a tensor's contents.
    assert qweight.dtype == torch.int8
    assert scale.dtype == torch.float32
    if scale.ndim == 3:
        # Restore the group view for broadcasting, then return the original matrix shape.
        values = qweight.float().reshape(qweight.shape[0], -1, GROUP_SIZE)
        assert scale.shape == (*values.shape[:2], 1), "Expected one scale per 128-weight group."
        return (values * scale).reshape(qweight.shape)
    assert scale.ndim == 0 or (
        qweight.ndim == 2 and scale.shape == (qweight.shape[0], 1)
    ), "Expected one scale per matrix or one scale per output row."
    return qweight.float() * scale


class NaiveQuantizedLinear(nn.Module):
    """Store compressed weights; deliberately reconstruct them on every forward.

    This teaches the difference between compressed storage and fast execution.
    F.linear still performs floating-point computation. Packed INT4 first unpacks
    its codes, then uses the same reconstruction as the unpacked variants.
    """

    def __init__(self, linear, bits=8, per_channel=False, grouped=False, packed=False):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        assert not packed or bits == 4, "Packing requires four-bit codes."
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.packed = packed
        qweight, scale = quantize_weight(
            linear.weight, bits=bits, per_channel=per_channel, grouped=grouped
        )
        # Keep only the packed buffer; the unpacked codes are temporary.
        if packed:
            qweight = pack_int4(qweight)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)
        self.train(linear.training)

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        codes = unpack_int4(self.qweight) if self.packed else self.qweight
        weight = dequantize_weight(codes, self.scale).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


@torch.inference_mode()
def run_demo():
    """Compare precision, scale granularity, and storage on two weight matrices."""
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(8, 256, generator=generator)
    weights = torch.randn(64, 256, generator=generator) * 0.1
    outlier_weights = weights.clone()
    outlier_weights[0, 0] = 10.0
    variants = (
        ("int8_per_tensor", 8, False, False, False),
        ("int8_per_channel", 8, True, False, False),
        ("int4_per_channel_unpacked", 4, True, False, False),
        ("int4_group128_unpacked", 4, False, True, False),
        ("int4_group128_packed", 4, False, True, True),
    )
    results = []
    for name, weight in (("ordinary", weights), ("one_outlier", outlier_weights)):
        for variant, bits, per_channel, grouped, packed in variants:
            qweight, scale = quantize_weight(
                weight, bits=bits, per_channel=per_channel, grouped=grouped
            )
            stored = pack_int4(qweight) if packed else qweight
            codes = unpack_int4(stored) if packed else stored
            reconstructed = dequantize_weight(codes, scale)
            error = (weight - reconstructed).abs()
            # Match the error's shape to its scales before checking half-step error.
            error_view = error.reshape(weight.shape[0], -1, GROUP_SIZE) if grouped else error
            assert (error_view <= scale / 2 + 1e-6).all(), "Rounding error should be <= half a step."
            output_error = F.linear(inputs, weight) - F.linear(inputs, reconstructed)
            weight_bytes = stored.numel() * stored.element_size()
            scale_bytes = scale.numel() * scale.element_size()
            results.append({
                "example": name,
                "variant": variant,
                "bits": bits,
                "source_dtype": str(weight.dtype),
                "storage_dtype": str(stored.dtype),
                "num_scales": scale.numel(),
                "scale_min": scale.min().item(),
                "scale_max": scale.max().item(),
                "original_weight_bytes": weight.numel() * weight.element_size(),
                "quantized_weight_and_scale_bytes": weight_bytes + scale_bytes,
                "weight_mse": error.square().mean().item(),
                "output_mse": output_error.square().mean().item(),
            })
    return results
