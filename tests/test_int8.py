import unittest

import torch
from torch import nn
from torch.nn import functional as F

from quantization_lab.models import model_tensor_bytes
from quantization_lab.quantizers.int8 import (
    NaiveInt8Linear,
    dequantize_int8,
    quantize_int8,
)


class Int8Tests(unittest.TestCase):
    def test_known_rounding_and_zero_weights(self):
        # The nonzero case has scale 1, so expected codes can be checked by hand.
        cases = (
            ([[-127, -1.6, -0.4, 0.4, 1.6, 127]], [[-127, -2, 0, 0, 2, 127]]),
            ([[0, 0, 0]], [[0, 0, 0]]),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                codes, scale = quantize_int8(torch.tensor(values, dtype=torch.float32))
                self.assertEqual(codes.tolist(), expected)
                torch.testing.assert_close(
                    dequantize_int8(codes, scale), torch.tensor(expected, dtype=torch.float32)
                )

    def test_linear_matches_known_reconstructed_weights(self):
        inputs = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]])
        expected_weights = torch.tensor([[-127.0, 0.0, 1.0], [1.0, 3.0, 127.0]])
        for bias in (False, True):
            original = nn.Linear(3, 2, bias=bias)
            with torch.no_grad():
                original.weight.copy_(torch.tensor([[-127.0, -0.4, 0.6], [1.4, 2.6, 127.0]]))
                if bias:
                    original.bias.copy_(torch.tensor([0.25, -0.5]))
            layer = NaiveInt8Linear(original)
            expected = F.linear(inputs, expected_weights, original.bias)
            torch.testing.assert_close(layer(inputs), expected)
            self.assertEqual(layer.qweight.dtype, torch.int8)

    def test_shared_weights_and_views_are_counted_once(self):
        model = nn.Module()
        model.embedding = nn.Embedding(5, 4)
        model.head = nn.Linear(4, 5, bias=False)
        model.head.weight = model.embedding.weight
        model.register_buffer("view", model.embedding.weight.detach().flatten())
        model.register_buffer("codes", torch.tensor([1, 2], dtype=torch.int8))
        self.assertEqual(model_tensor_bytes(model), 5 * 4 * 4 + 2)


if __name__ == "__main__":
    unittest.main()
