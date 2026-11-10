"""Model loading and the storage that stays resident between forward calls."""

from itertools import chain
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file, extract_commit_hash

from .quantizers.symmetric import NaiveQuantizedLinear


def resolve_model_revision(config):
    """Pin a Hub revision using its cached config path, even when Transformers drops _commit_hash."""
    config_path = cached_file(config["id"], "config.json", revision=config["revision"])
    revision = extract_commit_hash(config_path, None)
    if revision is None and not Path(config["id"]).is_dir():
        raise RuntimeError("Could not resolve the model's commit. Use an exact Hugging Face commit revision.")
    if revision is not None:
        config["revision"] = revision
    return revision


def load_model(config):
    device = torch.device(config["device"])
    dtype_name = config["dtype"]
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda for these experiments.")
    if dtype_name not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16, or bfloat16.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check your WSL PyTorch installation.")
        if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16. Choose float16 explicitly.")

    revision = resolve_model_revision(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["id"],
        revision=config["revision"],
        dtype=getattr(torch, dtype_name),
        attn_implementation="sdpa",
    )
    model.to(device).eval()

    # Record our resolved commit explicitly and load the tokenizer from the same one.
    # Local fixture directories have no Hub commit and intentionally retain None.
    model.config._commit_hash = revision
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=config["revision"])
    return model, tokenizer


def quantize_model_weights(model, bits=8, per_channel=False, grouped=False, packed=False):
    """Replace Qwen's Transformer linear layers in place; return their count.

    Restricting the walk to the blocks leaves embeddings, the final norm, and
    the output head outside the conversion. Norms inside blocks are not Linear
    modules, so they are also preserved. Tied embedding/head weights stay tied.
    """
    converted = 0
    for parent in model.model.layers.modules():
        # Snapshot each parent's children before replacing any of them.
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                replacement = NaiveQuantizedLinear(
                    child, bits=bits, per_channel=per_channel, grouped=grouped, packed=packed
                )
                setattr(parent, name, replacement)
                converted += 1
    assert converted > 0, "No linear layers found. Start from the original Qwen checkpoint."
    return converted


def model_tensor_bytes(model):
    """Count unique backing storage, including buffers such as scales and qweights.

    Tied embeddings and an output head can share storage. Counting their tensor
    sizes separately would count the same allocation twice. Views also share it.
    This measures resident model tensors, not checkpoint size or the KV cache.
    """
    seen = set()
    total = 0
    tensors = list(chain(model.parameters(), model.buffers()))
    while tensors:
        tensor = tensors.pop()
        if hasattr(tensor, "__tensor_flatten__"):
            # A quantized wrapper's logical BF16 shape is not its physical storage.
            # Visit its actual packed codes and qparams, including any padded entries.
            names, _ = tensor.__tensor_flatten__()
            tensors.extend(getattr(tensor, name) for name in names)
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total
