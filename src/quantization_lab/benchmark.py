"""Fixed-length, greedy generation for one batch at a time, without serving."""

from time import perf_counter

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def time_batch(model, inputs, new_tokens):
    assert inputs.ndim == 2 and inputs.dtype == torch.long
    assert new_tokens >= 2, "Need at least one decode step after the first token."
    assert not model.training, "Benchmarking requires model.eval()."
    device = inputs.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start = perf_counter()
    output = model(input_ids=inputs, use_cache=True)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    synchronize(device)
    prefill_seconds = perf_counter() - start
    cache = output.past_key_values
    assert cache is not None, "Decode timing requires a working KV cache."
    del output

    # Prefill produces token 1. Only the remaining N-1 tokens require decode
    # forwards. Ignore EOS so every variant performs the same amount of work.
    start = perf_counter()
    for _ in range(new_tokens - 1):
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        del output
    synchronize(device)
    decode_seconds = perf_counter() - start
    assert prefill_seconds > 0 and decode_seconds > 0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    batch_size, prompt_length = inputs.shape
    return {
        "batch_size": batch_size,
        "input_tokens": batch_size * prompt_length,
        "generated_tokens": batch_size * new_tokens,
        "decode_tokens": batch_size * (new_tokens - 1),
        "prefill_ms": prefill_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "peak_gpu_allocated_bytes": peak,
    }


def benchmark_case(model, prompts, prompt_ids, batch_size, new_tokens, warmups, repeats):
    assert prompts.ndim == 2 and len(prompts) == len(prompt_ids)
    assert 0 < batch_size <= len(prompts)
    assert warmups >= 1 and repeats >= 1
    device = next(model.parameters()).device
    warmup_sizes = [batch_size]
    remainder = len(prompts) % batch_size
    if remainder:
        # A partial batch has a different matrix shape; warm that shape up too.
        warmup_sizes.append(remainder)
    for size in warmup_sizes:
        warmup_inputs = prompts[:size].to(device)
        for _ in range(warmups):
            time_batch(model, warmup_inputs, new_tokens)
        del warmup_inputs

    records = []
    for repeat in range(repeats):
        for start in range(0, len(prompts), batch_size):
            inputs = prompts[start : start + batch_size].to(device)
            measurement = time_batch(model, inputs, new_tokens)
            records.append({
                "status": "ok",
                "repeat": repeat,
                "prompt_ids": prompt_ids[start : start + batch_size],
                **measurement,
            })
            del inputs
    return records
