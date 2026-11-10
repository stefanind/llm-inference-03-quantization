import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from quantization_lab.models import model_tensor_bytes, quantize_model_weights
from quantization_lab.data import load_calibration_tokens
from quantization_lab.quantizers.gptq import quantize_model as quantize_model_gptq, quantize_weight_gptq
from quantization_lab.quantizers.packing import pack_int4, unpack_int4
from quantization_lab.quantizers.symmetric import (
    NaiveQuantizedLinear,
    dequantize_weight,
    quantize_weight,
)


class QuantizationTests(unittest.TestCase):
    def test_known_rounding_and_zero_weights(self):
        # The nonzero case has scale 1, so expected codes can be checked by hand.
        cases = (
            ([[-127, -1.6, -0.4, 0.4, 1.6, 127]], [[-127, -2, 0, 0, 2, 127]]),
            ([[0, 0, 0]], [[0, 0, 0]]),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                codes, scale = quantize_weight(torch.tensor(values, dtype=torch.float32))
                self.assertEqual(codes.tolist(), expected)
                torch.testing.assert_close(
                    dequantize_weight(codes, scale), torch.tensor(expected, dtype=torch.float32)
                )

    def test_per_channel_scales_with_an_outlier_and_zero_row(self):
        # Only the second row has a large outlier. Its scale must not affect the first row.
        weight = torch.tensor([[1.27, 0.016, -0.016], [127.0, 1.6, -1.6], [0.0, 0.0, 0.0]])
        codes, scale = quantize_weight(weight, per_channel=True)
        self.assertEqual(codes.tolist(), [[127, 2, -2], [127, 2, -2], [0, 0, 0]])
        torch.testing.assert_close(scale, torch.tensor([[0.01], [1.0], [1.0]]))
        expected = torch.tensor([[1.27, 0.02, -0.02], [127.0, 2.0, -2.0], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(dequantize_weight(codes, scale), expected)

    def test_unpacked_int4_rounding_scales_and_zero_row(self):
        # Each nonzero row uses the same [-7, 7] code range at a different step size.
        weight = torch.tensor([
            [7.0, -7.0, 1.6, -0.4],
            [0.7, -0.7, 0.16, -0.04],
            [0.0, 0.0, 0.0, 0.0],
        ])
        codes, scale = quantize_weight(weight, bits=4, per_channel=True)
        self.assertEqual(codes.tolist(), [[7, -7, 2, 0], [7, -7, 2, 0], [0, 0, 0, 0]])
        torch.testing.assert_close(scale, torch.tensor([[1.0], [0.1], [1.0]]))
        expected = torch.tensor([[7.0, -7.0, 2.0, 0.0], [0.7, -0.7, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(dequantize_weight(codes, scale), expected)

    def test_group128_scales_rounding_and_zero_group(self):
        # Repeat four known values to fill each 128-weight group.
        group = torch.tensor([7.0, -7.0, 1.6, -0.4]).repeat(32)
        weight = torch.cat((group * 0.1, group, torch.zeros_like(group))).unsqueeze(0)
        codes, scale = quantize_weight(weight, bits=4, grouped=True)
        expected_codes = torch.tensor([7, -7, 2, 0], dtype=torch.int8).repeat(32)
        expected = torch.cat((expected_codes, expected_codes, torch.zeros_like(expected_codes)))
        torch.testing.assert_close(codes, expected.unsqueeze(0))
        torch.testing.assert_close(scale, torch.tensor([[[0.1], [1.0], [1.0]]]))
        small = torch.tensor([0.7, -0.7, 0.2, 0.0]).repeat(32)
        large = torch.tensor([7.0, -7.0, 2.0, 0.0]).repeat(32)
        reconstructed = torch.cat((small, large, torch.zeros_like(large))).unsqueeze(0)
        torch.testing.assert_close(dequantize_weight(codes, scale), reconstructed)

    def test_linear_matches_known_reconstructed_weights(self):
        inputs = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]])
        expected_weights = torch.tensor([[-127.0, 0.0, 1.0], [1.0, 3.0, 127.0]])
        for bias in (False, True):
            original = nn.Linear(3, 2, bias=bias)
            with torch.no_grad():
                original.weight.copy_(torch.tensor([[-127.0, -0.4, 0.6], [1.4, 2.6, 127.0]]))
                if bias:
                    original.bias.copy_(torch.tensor([0.25, -0.5]))
            layer = NaiveQuantizedLinear(original)
            expected = F.linear(inputs, expected_weights, original.bias)
            torch.testing.assert_close(layer(inputs), expected)
            self.assertEqual(layer.qweight.dtype, torch.int8)

    def test_int4_packing_known_bytes_and_round_trip(self):
        # Cover every emitted code, including negatives and both ends of [-7, 7].
        codes = torch.tensor([
            [-7, -6, -5, -4, -3, -2, -1, 0],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ], dtype=torch.int8)
        expected = torch.tensor([
            [0x9A, 0xBC, 0xDE, 0xF0],
            [0x12, 0x34, 0x56, 0x70],
        ], dtype=torch.uint8)
        packed = pack_int4(codes)
        torch.testing.assert_close(packed, expected)
        torch.testing.assert_close(unpack_int4(packed), codes)

    def test_shared_weights_and_views_are_counted_once(self):
        model = nn.Module()
        model.embedding = nn.Embedding(5, 4)
        model.head = nn.Linear(4, 5, bias=False)
        model.head.weight = model.embedding.weight
        model.register_buffer("view", model.embedding.weight.detach().flatten())
        model.register_buffer("codes", torch.tensor([1, 2], dtype=torch.int8))
        self.assertEqual(model_tensor_bytes(model), 5 * 4 * 4 + 2)

    def test_gptq_compensates_error_across_groups(self):
        # Both active channels always have the same input. Rounding 0.4 + 0.4
        # independently gives 0; compensating the second group produces 0 + 1.
        weight = torch.zeros(2, 256)
        weight[0, [0, 128]] = 0.4
        weight[0, [127, 255]] = 7  # Fix each group's scale at 1; these inputs are unseen.
        inputs = torch.zeros(2, 256)
        inputs[0, [0, 128]] = 1
        inputs[1, [0, 128]] = -1
        moment = inputs.T @ inputs / len(inputs)

        codes, scale = quantize_weight_gptq(weight, moment)
        expected = torch.zeros_like(codes)
        expected[0, 128] = 1
        torch.testing.assert_close(codes, expected)
        torch.testing.assert_close(scale, torch.ones(2, 2, 1))
        rounded, rounded_scale = quantize_weight(weight, bits=4, grouped=True)
        original = F.linear(inputs, weight)
        gptq_error = (F.linear(inputs, dequantize_weight(codes, scale)) - original).square().mean()
        rounding_error = (F.linear(inputs, dequantize_weight(rounded, rounded_scale)) - original).square().mean()
        self.assertLess(gptq_error.item(), rounding_error.item())

    def test_qwen_conversion_preserves_excluded_layers_and_cached_decoding(self):
        config = Qwen2Config(
            vocab_size=32, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        # Use identical original weights so packing's effect can be isolated.
        initial_state = Qwen2ForCausalLM(config).state_dict()
        # Exercise the actual sampler without downloading a corpus. Calibration
        # always overrides the evaluation split with train, and repeats exactly.
        dataset_config = {"id": "fixture", "subset": "text", "split": "test", "revision": "main"}
        calibration_config = {"split": "train", "num_sequences": 2, "sequence_length": 8}
        tokenizer = object()
        with patch("quantization_lab.data.load_corpus", return_value=(list(range(32)), "fixture")) as corpus:
            calibration, metadata = load_calibration_tokens(dataset_config, tokenizer, calibration_config, seed=0)
            repeated, repeated_metadata = load_calibration_tokens(dataset_config, tokenizer, calibration_config, seed=0)
            corpus.assert_called_with({**dataset_config, "split": "train"}, tokenizer)
        torch.testing.assert_close(calibration, repeated)
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(len(set(metadata["token_offsets"])), 2)
        self.assertEqual(metadata["num_tokens"], 16)
        storage_bytes = {}
        outputs = {}
        modes = (
            (8, False, False, False), (8, True, False, False),
            (4, True, False, False), (4, False, True, False), (4, False, True, True),
        )
        packed_scales = {}
        for mode in (*modes, "gptq"):
            bits, per_channel, grouped, packed = (4, False, True, True) if mode == "gptq" else mode
            with self.subTest(mode=mode):
                model = Qwen2ForCausalLM(config).eval()
                model.load_state_dict(initial_state)
                embeddings = model.model.embed_tokens
                head = model.lm_head
                preserved_weights = {
                    name: weight.detach().clone()
                    for name, weight in model.named_parameters() if "norm" in name or name.endswith(".bias")
                }
                original_bytes = model_tensor_bytes(model)

                if mode == "gptq":
                    second_block_inputs = []
                    handle = model.model.layers[1].register_forward_pre_hook(
                        lambda module, args: second_block_inputs.append(args[0].clone())
                    )
                    try:
                        count = quantize_model_gptq(model, calibration)
                        with torch.inference_mode():
                            model.model(input_ids=calibration[:1], use_cache=False)
                    finally:
                        handle.remove()
                    # The initial setup probe saw the original model. The next
                    # call collected statistics after block 1 was quantized;
                    # replaying the finished model must give that same input.
                    torch.testing.assert_close(second_block_inputs[1], second_block_inputs[-1], rtol=0, atol=0)
                else:
                    count = quantize_model_weights(
                        model, bits=bits, per_channel=per_channel, grouped=grouped, packed=packed
                    )
                # Each block has four attention projections and three MLP projections.
                self.assertEqual(count, 14)
                converted = [m for m in model.model.layers.modules() if isinstance(m, NaiveQuantizedLinear)]
                self.assertEqual(len(converted), 14)
                self.assertFalse(any(isinstance(m, nn.Linear) for m in model.model.layers.modules()))
                for layer in converted:
                    self.assertEqual(layer.qweight.dtype, torch.uint8 if packed else torch.int8)
                    columns = layer.in_features // 2 if packed else layer.in_features
                    self.assertEqual(layer.qweight.shape, (layer.out_features, columns))
                    self.assertEqual(layer.bits, bits)
                    self.assertEqual(layer.scale.dtype, torch.float32)
                    if grouped:
                        expected_shape = (layer.out_features, layer.in_features // 128, 1)
                    else:
                        expected_shape = (layer.out_features, 1) if per_channel else ()
                    self.assertEqual(layer.scale.shape, expected_shape)
                    self.assertFalse(layer.training)
                self.assertIs(model.model.embed_tokens, embeddings)
                self.assertIs(model.lm_head, head)
                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
                for name, expected in preserved_weights.items():
                    torch.testing.assert_close(model.state_dict()[name], expected)
                for name, layer in model.named_modules():
                    if isinstance(layer, NaiveQuantizedLinear) and grouped and packed:
                        if mode == "gptq":
                            torch.testing.assert_close(layer.scale, packed_scales[name], rtol=0, atol=0)
                        else:
                            packed_scales[name] = layer.scale.clone()
                storage_bytes[mode] = model_tensor_bytes(model)
                self.assertLess(storage_bytes[mode], original_bytes)
                if grouped and not packed:
                    unpacked_code_bytes = sum(m.qweight.numel() * m.qweight.element_size() for m in converted)

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
                outputs[mode] = torch.cat((prefill.logits, decode.logits), dim=1)

        # Four-bit codes are still unpacked: they save no bytes over per-channel INT8.
        self.assertEqual(storage_bytes[4, True, False, False], storage_bytes[8, True, False, False])
        self.assertGreater(storage_bytes[4, False, True, False], storage_bytes[4, True, False, False])
        unpacked = (4, False, True, False)
        packed = (4, False, True, True)
        # Only code storage halves. Scales and all unconverted tensors stay the same.
        self.assertEqual(storage_bytes[packed], storage_bytes[unpacked] - unpacked_code_bytes // 2)
        torch.testing.assert_close(outputs[packed], outputs[unpacked], rtol=0, atol=0)
        self.assertEqual(storage_bytes["gptq"], storage_bytes[packed])


if __name__ == "__main__":
    unittest.main()
