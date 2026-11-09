import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from quantization_lab.models import model_tensor_bytes, quantize_model_int8
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

    def test_qwen_conversion_preserves_excluded_layers_and_cached_decoding(self):
        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = Qwen2ForCausalLM(config).eval()
        embeddings = model.model.embed_tokens
        head = model.lm_head
        norm_weights = {
            name: weight.detach().clone()
            for name, weight in model.named_parameters() if "norm" in name
        }
        original_bytes = model_tensor_bytes(model)

        # One Qwen block has four attention projections and three MLP projections.
        self.assertEqual(quantize_model_int8(model), 7)
        converted = [m for m in model.model.layers.modules() if isinstance(m, NaiveInt8Linear)]
        self.assertEqual(len(converted), 7)
        self.assertFalse(any(isinstance(m, nn.Linear) for m in model.model.layers.modules()))
        for layer in converted:
            self.assertEqual(layer.qweight.dtype, torch.int8)
            self.assertEqual(layer.scale.dtype, torch.float32)
            self.assertFalse(layer.training)
        self.assertIs(model.model.embed_tokens, embeddings)
        self.assertIs(model.lm_head, head)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        for name, expected in norm_weights.items():
            torch.testing.assert_close(model.get_parameter(name), expected)
        self.assertLess(model_tensor_bytes(model), original_bytes)

        with torch.inference_mode():
            inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
            prefill = model(input_ids=inputs, use_cache=True)
            token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
            decode = model(input_ids=token, past_key_values=prefill.past_key_values, use_cache=True)
        self.assertEqual(prefill.logits.shape, (2, 3, 32))
        self.assertEqual(decode.logits.shape, (2, 1, 32))
        self.assertEqual(decode.past_key_values.get_seq_length(), 4)
        self.assertTrue(torch.isfinite(prefill.logits).all())
        self.assertTrue(torch.isfinite(decode.logits).all())


if __name__ == "__main__":
    unittest.main()
