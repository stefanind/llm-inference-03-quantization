"""One optimized INT4 recipe: TorchAO's group-128 BF16 / tiled-weight CUDA path."""

from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn


KERNEL = "aten::_weight_int4pack_mm"


def check_support(config):
    """Reject unsupported setups before downloading weights or running benchmarks."""
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("This TorchAO recipe requires PyTorch 2.9.1; keep the image's CUDA build.")
    try:
        installed = version("torchao")
    except PackageNotFoundError:
        raise RuntimeError("Install the project dependencies with: python -m pip install -e .") from None
    if installed.split("+")[0] != "0.15.0":
        raise RuntimeError("This recipe requires torchao==0.15.0, paired with PyTorch 2.9.1.")
    device = torch.device(config["device"])
    if device.type != "cuda" or config["dtype"] != "bfloat16":
        raise ValueError("The TorchAO variant requires --device cuda --dtype bfloat16.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the optimized INT4 variant cannot run on CPU.")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("The tiled INT4 kernel requires an Ampere or newer NVIDIA GPU (SM80+).")
    # TorchAO 0.15 queries the current device while packing, so honor cuda:N too.
    if device.index is not None:
        torch.cuda.set_device(device.index)


def int4_config():
    # Import only when requested; CPU demos and ordinary CLI help stay lightweight.
    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows import Int4ChooseQParamsAlgorithm, Int4PackingFormat

    return Int4WeightOnlyConfig(
        group_size=128, version=2,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.TINYGEMM,
        set_inductor_config=False,  # Keep all variants eager, with unchanged compiler settings.
    )


@torch.inference_mode()
def quantize_model(model):
    """Quantize only Qwen's block projections; return the converted layer count."""
    parameter = next(model.parameters())
    check_support({"device": str(parameter.device), "dtype": str(parameter.dtype).removeprefix("torch.")})
    from torchao.quantization import quantize_
    from torchao.quantization.quantize_.workflows import Int4TilePackedTo4dTensor

    projections = [layer for layer in model.model.layers.modules() if isinstance(layer, nn.Linear)]
    assert len(projections) == 7 * len(model.model.layers), "Expected all seven projections in each Qwen block."
    quantize_(model.model.layers, int4_config(), filter_fn=lambda layer, name: isinstance(layer, nn.Linear))
    for layer in projections:
        assert isinstance(layer.weight, Int4TilePackedTo4dTensor), "A projection was left unquantized."
        assert layer.weight.block_size == [1, 128], "Expected one quantization group per 128 input weights."
    return len(projections)


@torch.inference_mode()
def verify_kernel(model, converted_layers):
    """Check a full prefill and cached decode outside the experiment's timing."""
    device = next(model.parameters()).device
    inputs = torch.zeros((1, 2), dtype=torch.long, device=device)
    model(input_ids=inputs, use_cache=False)  # Warm up before profiling.
    torch.cuda.synchronize(device)
    # CPU operator events show dispatch without requiring a working CUPTI setup.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        prefill = model(input_ids=inputs, use_cache=True)
        token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode = model(input_ids=token, past_key_values=prefill.past_key_values, use_cache=True)
        torch.cuda.synchronize(device)
    calls = sum(event.count for event in profile.key_averages() if event.key == KERNEL)
    assert calls == 2 * converted_layers, "Every projection must use the packed kernel in both forwards."
    assert torch.isfinite(prefill.logits).all() and torch.isfinite(decode.logits).all()
    assert decode.past_key_values.get_seq_length() == 3
    return {"operator": KERNEL, "verified_calls": calls, "verified_forwards": 2}


def recipe_metadata(model):
    """Describe the actual format and kernel padding, rather than assuming INT4 layouts match."""
    widths = sorted({layer.in_features for layer in model.model.layers.modules() if isinstance(layer, nn.Linear)})
    return {
        "backend": "torchao", "bits": 4, "group_size": 128,
        "packing_format": "TILE_PACKED_TO_4D", "qparams_algorithm": "TINYGEMM",
        "mapping": "asymmetric", "code_range": [0, 15],
        "scale_and_zero_dtype": "torch.bfloat16",
        "input_padding": {str(width): ((width + 1023) // 1024) * 1024 for width in widths},
    }


@torch.inference_mode()
def preflight(config):
    """Exercise every Qwen2.5-0.5B projection shape before a full comparison starts."""
    check_support(config)
    from torchao.quantization import quantize_

    # Unique [output, input] shapes: q/o, k/v, gate/up, and down projections.
    for output_width, input_width in ((896, 896), (128, 896), (4864, 896), (896, 4864)):
        layer = nn.Linear(input_width, output_width, bias=False, device=config["device"], dtype=torch.bfloat16)
        quantize_(layer, int4_config())
        for length in (1, 128):
            inputs = torch.ones((1, length, input_width), device=config["device"], dtype=torch.bfloat16)
            assert torch.isfinite(layer(inputs)).all(), "Packed execution produced non-finite values."
        del layer, inputs
    print("TorchAO INT4 preflight passed: all four projection shapes, prefill and decode.", flush=True)


if __name__ == "__main__":
    import json
    from pathlib import Path
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m quantization_lab.quantizers.torchao_int4 CONFIG.json")
    preflight(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["model"])
