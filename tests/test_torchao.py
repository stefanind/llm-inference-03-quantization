import unittest

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from quantization_lab.models import model_tensor_bytes
from quantization_lab.quantizers.torchao_int4 import quantize_model, verify_kernel


class TorchAOTests(unittest.TestCase):
    def test_storage_counts_packed_fields_and_shared_views(self):
        from torchao.quantization.quantize_.workflows import Int4TilePackedTo4dTensor

        # Logical BF16 shape: [8, 896]. Physical INT4 input width is padded to 1024.
        codes = torch.zeros((1, 8, 32, 4), dtype=torch.int32)
        qparams = torch.ones((8, 8, 2), dtype=torch.bfloat16)
        wrapped = Int4TilePackedTo4dTensor(codes, qparams, [1, 128], torch.Size([8, 896]))
        model = nn.Module()
        model.weight = nn.Parameter(wrapped, requires_grad=False)
        model.register_buffer("shared_codes", codes.flatten())
        # 8 * 1024 / 2 code bytes + 64 groups * (BF16 scale + BF16 offset).
        self.assertEqual(model_tensor_bytes(model), 4096 + 256)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        "Packed kernel integration requires an SM80+ CUDA GPU.",
    )
    def test_qwen_packed_kernel_preserves_layers_and_cached_decoding(self):
        config = Qwen2Config(
            vocab_size=32, hidden_size=896, intermediate_size=4864,
            num_hidden_layers=1, num_attention_heads=14, num_key_value_heads=2,
            max_position_embeddings=64, tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        embeddings, head = model.model.embed_tokens, model.lm_head
        protected = {name: value.detach().clone() for name, value in model.named_parameters() if "norm" in name or "bias" in name}
        original_bytes = model_tensor_bytes(model)
        converted = quantize_model(model)
        self.assertEqual(converted, 7)
        self.assertIs(model.model.embed_tokens, embeddings)
        self.assertIs(model.lm_head, head)
        self.assertIs(head.weight, embeddings.weight)
        for name, expected in protected.items():
            torch.testing.assert_close(model.get_parameter(name), expected)
        self.assertLess(model_tensor_bytes(model), original_bytes)
        # Also checks finite prefill/decode logits, cache length, and actual packed dispatch.
        self.assertEqual(verify_kernel(model, converted)["verified_calls"], 14)


if __name__ == "__main__":
    unittest.main()
