"""Two commands: a numerical demo and a measured floating-point baseline."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform


def validate_config(config):
    """Reject invalid user settings before downloading a model or dataset."""
    evaluation = config["evaluation"]
    benchmark = config["benchmark"]
    for name in ("max_tokens", "context_length", "stride"):
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer.")
    if evaluation["max_tokens"] < 2:
        raise ValueError("Perplexity needs at least two tokens.")
    if not 0 < evaluation["stride"] < evaluation["context_length"]:
        raise ValueError("Require 0 < stride < context_length.")
    for name in ("batch_sizes", "prompt_lengths", "new_tokens"):
        values = benchmark[name]
        if not values or any(type(value) is not int or value < 1 for value in values):
            raise ValueError(f"benchmark.{name} must contain positive integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"benchmark.{name} contains duplicate cases.")
    if min(benchmark["new_tokens"]) < 2:
        raise ValueError("new_tokens must be >= 2 to measure decode throughput.")
    for name in ("num_prompts_per_length", "warmup_batches", "repeats"):
        if type(benchmark[name]) is not int or benchmark[name] < 1:
            raise ValueError(f"benchmark.{name} must be a positive integer.")
    if benchmark["num_prompts_per_length"] < max(benchmark["batch_sizes"]):
        raise ValueError("Each batch size needs at least that many prompts.")


def run_experiment(config, output_root):
    # Keep --help and config validation lightweight; load ML packages for a run.
    import gc
    import torch

    from .benchmark import benchmark_case
    from .data import load_tokens, make_prompts
    from .evaluate import evaluate_perplexity
    from .models import load_model, model_tensor_bytes
    from .results import append_measurements, summarize, write_json, write_summary

    validate_config(config)
    torch.manual_seed(config["seed"])
    print("Loading the floating-point baseline...", flush=True)
    model, tokenizer = load_model(config["model"])
    benchmark = config["benchmark"]
    evaluation = config["evaluation"]
    context_limit = model.config.max_position_embeddings
    if max(benchmark["prompt_lengths"]) + max(benchmark["new_tokens"]) > context_limit:
        raise ValueError("Prompt plus generation length exceeds the model context limit.")
    if evaluation["context_length"] > context_limit:
        raise ValueError("Evaluation context length exceeds the model context limit.")

    token_budget = max(
        evaluation["max_tokens"],
        max(benchmark["prompt_lengths"]) * benchmark["num_prompts_per_length"],
    )
    print("Preparing the fixed text sample...", flush=True)
    tokens, dataset_metadata = load_tokens(config["dataset"], tokenizer, token_budget)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-baseline"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    device = next(model.parameters()).device
    write_json(run_dir / "environment.json", {
        "schema_version": 1,
        "variant": "baseline",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": model.config._attn_implementation,
        "model_resolved_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "model_tensor_bytes": model_tensor_bytes(model),
        "dataset": dataset_metadata,
    })
    print(f"Results: {run_dir}", flush=True)

    quality = evaluate_perplexity(
        model, tokens[:evaluation["max_tokens"]],
        evaluation["context_length"], evaluation["stride"],
    )
    write_json(run_dir / "evaluation.json", quality)
    print(f"Perplexity: {quality['perplexity']:.4f}", flush=True)

    summaries = []
    for length in benchmark["prompt_lengths"]:
        prompts, prompt_ids = make_prompts(tokens, length, benchmark["num_prompts_per_length"])
        for batch_size, new_tokens in product(benchmark["batch_sizes"], benchmark["new_tokens"]):
            case = {
                "variant": "baseline",
                "batch_size_requested": batch_size,
                "prompt_length": length,
                "new_tokens": new_tokens,
            }
            print(f"Timing batch={batch_size}, prompt={length}, new_tokens={new_tokens}", flush=True)
            try:
                records = benchmark_case(
                    model, prompts, prompt_ids, batch_size, new_tokens,
                    benchmark["warmup_batches"], benchmark["repeats"],
                )
                summary = summarize(records)
            except torch.cuda.OutOfMemoryError:
                # An oversized workload should not discard the rest of a sweep.
                records = [{"status": "oom"}]
                summary = {"status": "oom"}
            if summary["status"] == "oom":
                # Leave the exception scope first so its traceback releases tensors.
                gc.collect()
                torch.cuda.empty_cache()
            append_measurements(run_dir / "measurements.jsonl", [{**case, **r} for r in records])
            summaries.append({**case, **summary})
            write_summary(run_dir / "summary.csv", summaries)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the INT8 matrix exercise offline on CPU.")
    run = commands.add_parser("run", help="Evaluate and benchmark the floating-point model.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()

    if args.command == "demo":
        from .quantizers.int8 import run_demo

        print(json.dumps(run_demo(), indent=2, allow_nan=False))
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["model"]["device"] = args.device
    if args.dtype:
        config["model"]["dtype"] = args.dtype
    run_experiment(config, args.output)
