import unittest

from quantization_lab.plotting import prepare_results


class PlottingTests(unittest.TestCase):
    def test_deduplicates_quality_and_keeps_generation_lengths_separate(self):
        rows = []
        for generated in (32, 128):
            for variant, perplexity, size in (("baseline", "14", "1024"), ("int4_group128_packed", "19", "512")):
                row = {
                    "variant": variant, "perplexity": perplexity, "model_tensor_bytes": size,
                    "batch_size_requested": "4", "prompt_length": "512", "new_tokens": str(generated),
                    "status": "ok", "mean_prefill_ms": "20", "decode_tokens_per_second": "100",
                    "peak_gpu_allocated_bytes": "2048",
                }
                if generated == 128:
                    row["peak_gpu_allocated_bytes"] = ""  # CPU results have no GPU allocation metric.
                    if variant == "int4_group128_packed":
                        row.update(status="oom", mean_prefill_ms="", decode_tokens_per_second="")
                rows.append(row)

        quality, workloads = prepare_results(rows)
        self.assertEqual(quality, {
            "baseline": {"perplexity": 14.0, "model_tensor_bytes": 1024},
            "int4_group128_packed": {"perplexity": 19.0, "model_tensor_bytes": 512},
        })
        self.assertEqual(set(workloads), {(4, 512, 32), (4, 512, 128)})
        self.assertEqual(workloads[4, 512, 32]["baseline"]["decode_tokens_per_second"], 100.0)
        self.assertIsNone(workloads[4, 512, 128]["baseline"]["peak_gpu_allocated_bytes"])
        failed = workloads[4, 512, 128]["int4_group128_packed"]
        self.assertEqual(failed["status"], "oom")
        self.assertIsNone(failed["mean_prefill_ms"])


if __name__ == "__main__":
    unittest.main()
