"""Plain JSON and CSV files, with throughput aggregated by tokens / time."""

import csv
import json
import statistics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_measurements(path, records):
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")


def summarize(records):
    """Summarize one workload case. Raw records retain actual partial batch sizes."""
    assert records and all(record["status"] == "ok" for record in records)
    decode_tokens = sum(record["decode_tokens"] for record in records)
    decode_seconds = sum(record["decode_ms"] for record in records) / 1000
    peaks = [r["peak_gpu_allocated_bytes"] for r in records if r["peak_gpu_allocated_bytes"] is not None]
    return {
        "status": "ok",
        "measured_batches": len(records),
        "mean_prefill_ms": statistics.mean(r["prefill_ms"] for r in records),
        # Average rates directly would give unequal-duration batches equal weight.
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "peak_gpu_allocated_bytes": max(peaks) if peaks else None,
    }


def write_summary(path, rows):
    fields = [
        "variant", "batch_size_requested", "prompt_length", "new_tokens", "status",
        "measured_batches", "mean_prefill_ms", "decode_tokens_per_second",
        "peak_gpu_allocated_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path, run_dirs):
    """Join quality and storage to each workload; never average unlike workloads."""
    rows = []
    reference = None
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        quality = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        config.pop("variant", None)
        # Quality and speed can change with these settings independently of quantization.
        environment_keys = (
            "python", "torch", "transformers", "datasets", "cuda_runtime",
            "device", "device_name", "dtype", "attention_implementation",
            "model_resolved_revision", "tokenizer_class",
            "torchao", "execution",
        )
        signature = (
            config, {key: environment[key] for key in environment_keys},
            environment["dataset"], quality["scored_tokens"],
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Cannot compare {run_dir}: settings, environment, or evaluation inputs differ.")

        with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append({
                    "variant": row.pop("variant"),
                    "perplexity": quality["perplexity"],
                    "model_tensor_bytes": environment["model_tensor_bytes"],
                    **row,
                    "run_dir": run_dir.name,
                })

    # Stable sorting puts variants side by side within each matching workload.
    workload_keys = ("batch_size_requested", "prompt_length", "new_tokens")
    rows.sort(key=lambda row: tuple(int(row[key]) for key in workload_keys))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
