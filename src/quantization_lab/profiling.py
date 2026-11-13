"""A short decode trace, captured separately from ordinary throughput measurements."""

from pathlib import Path

import torch

from .benchmark import synchronize


DECODE_STEPS = 8


@torch.inference_mode()
def profile_decode(model, inputs, output_dir):
    """Profile eight cached forwards after the caller has warmed up generation."""
    assert not model.training and inputs.ndim == 2 and inputs.shape[0] == 1
    assert inputs.shape[1] + DECODE_STEPS <= model.config.max_position_embeddings
    device = inputs.device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiling is unavailable. Check CUPTI in the RunPod image.")
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Build a fresh cache outside capture. Prefill also produces the first token;
    # the traced forwards consume one token at a time, just like the benchmark.
    output = model(input_ids=inputs, use_cache=True)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    del output
    synchronize(device)

    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for _ in range(DECODE_STEPS):
            with torch.profiler.record_function("decode_step"):
                output = model(input_ids=token, past_key_values=cache, use_cache=True)
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                del output
        # Finish GPU work before capture ends. There is no synchronization between
        # decode steps, which would change the execution pattern we want to inspect.
        synchronize(device)

    assert cache.get_seq_length() == inputs.shape[1] + DECODE_STEPS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_dir / "trace.json"))
    events = profiler.key_averages()
    text = (
        f"Diagnostic capture: {DECODE_STEPS} cached decode steps; batch 1; prompt {inputs.shape[1]} tokens.\n"
        f"Activities: {', '.join(activity.name for activity in activities)}.\n"
        "Profiler and shape recording add overhead. Use summary.csv for throughput.\n"
        "CPU and GPU work can overlap; do not add their times together.\n\n"
        "Operators sorted by self CPU time\n"
        + events.table(sort_by="self_cpu_time_total", row_limit=30)
    )
    if device.type == "cuda":
        text += "\nOperators sorted by self GPU time\n" + events.table(sort_by="self_device_time_total", row_limit=30)
    (output_dir / "operators.txt").write_text(text, encoding="utf-8")

    # CUDA can fall back to CPU-side timing when CUPTI cannot collect kernels.
    # Keep the files for diagnosis, but do not report a CPU-only trace as success.
    if device.type == "cuda" and not any(
        event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events()
    ):
        raise RuntimeError("No CUDA kernel activity was captured. Check CUPTI; the diagnostic files were saved.")
    return output_dir
