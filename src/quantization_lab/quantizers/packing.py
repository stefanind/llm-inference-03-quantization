"""Two signed four-bit codes per unsigned byte, first code in the high nibble."""

import torch


def pack_int4(codes):
    assert codes.dtype == torch.int8 and codes.ndim == 2 and codes.shape[1] % 2 == 0, (
        "Expected an INT8 matrix with even row width."
    )
    assert ((codes >= -7) & (codes <= 7)).all(), "Codes must be in [-7, 7]."
    # Keep each code's four-bit two's-complement representation, including its sign.
    nibbles = (codes & 0x0F).to(torch.uint8)
    return (nibbles[:, ::2] << 4) | nibbles[:, 1::2]


def unpack_int4(packed):
    assert packed.dtype == torch.uint8 and packed.ndim == 2
    high = packed >> 4
    low = packed & 0x0F
    # Interleave high/low nibbles to restore the original column order.
    codes = torch.stack((high, low), dim=-1).reshape(packed.shape[0], -1).to(torch.int8)
    # Nibbles 8..15 represent -8..-1. Our symmetric quantizer never emits -8.
    return torch.where(codes >= 8, codes - 16, codes)
