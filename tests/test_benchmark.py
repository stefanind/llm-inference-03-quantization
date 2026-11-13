import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from quantization_lab.benchmark import benchmark_case
from quantization_lab.profiling import DECODE_STEPS, profile_decode
from quantization_lab.results import summarize


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Thread scheduling dominates these deliberately tiny CPU forwards.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_actual_batch_sizes_token_counts_and_fresh_caches(self):
        # Same architecture family as the real experiment, with no downloads.
        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = Qwen2ForCausalLM(config).eval()
        observed_cache_lengths = []

        def inspect_cache(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            observed_cache_lengths.append(cache.get_seq_length() if cache is not None else 0)

        handle = model.register_forward_pre_hook(inspect_cache, with_kwargs=True)
        try:
            records = benchmark_case(model, torch.arange(20).reshape(5, 4), list("abcde"), 2, 4, 1, 2)
        finally:
            handle.remove()
        self.assertEqual([r["batch_size"] for r in records], [2, 2, 1, 2, 2, 1])
        for record in records:
            self.assertEqual(record["generated_tokens"], record["batch_size"] * 4)
            self.assertEqual(record["decode_tokens"], record["batch_size"] * 3)
        # Two warmup shapes and six measured batches, each with a fresh prefill followed
        # by three single-token updates to the same batch's growing cache.
        self.assertEqual(observed_cache_lengths, [0, 4, 5, 6] * 8)

    def test_summary_uses_total_tokens_over_total_time(self):
        records = [
            {"status": "ok", "decode_tokens": 100, "decode_ms": 1000, "prefill_ms": 2, "peak_gpu_allocated_bytes": None},
            {"status": "ok", "decode_tokens": 100, "decode_ms": 10000, "prefill_ms": 4, "peak_gpu_allocated_bytes": None},
        ]
        summary = summarize(records)
        self.assertAlmostEqual(summary["decode_tokens_per_second"], 200 / 11)
        self.assertEqual(summary["mean_prefill_ms"], 3)

    def test_decode_trace_excludes_prefill_and_advances_cache(self):
        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = Qwen2ForCausalLM(config).eval()
        cache_lengths = []

        def inspect_cache(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            cache_lengths.append(cache.get_seq_length() if cache is not None else 0)

        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        scratch.mkdir(exist_ok=True)
        handle = model.register_forward_pre_hook(inspect_cache, with_kwargs=True)
        try:
            with TemporaryDirectory(dir=scratch) as directory:
                profile_decode(model, torch.tensor([[1, 2, 3, 4]]), directory)
                events = json.loads((Path(directory) / "trace.json").read_text(encoding="utf-8"))["traceEvents"]
                steps = [event for event in events if event.get("name") == "decode_step"]
                self.assertEqual(len(steps), DECODE_STEPS)
                # If prefill accidentally enters capture, an embedding input has
                # shape [1, 4]. Every recorded model input must instead be [1, 1].
                embeddings = [event for event in events if event.get("name") == "aten::embedding"]
                self.assertEqual(len(embeddings), DECODE_STEPS)
                self.assertTrue(all(event["args"]["Input Dims"][1] == [1, 1] for event in embeddings))
                self.assertIn("self CPU time", (Path(directory) / "operators.txt").read_text(encoding="utf-8"))
        finally:
            handle.remove()
        self.assertEqual(cache_lengths, [0] + list(range(4, 4 + DECODE_STEPS)))

if __name__ == "__main__":
    unittest.main()
