"""Model loading and the storage that stays resident between forward calls."""

from itertools import chain

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .quantizers.symmetric import NaiveQuantizedLinear


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

    model = AutoModelForCausalLM.from_pretrained(
        config["id"],
        revision=config["revision"],
        dtype=getattr(torch, dtype_name),
        attn_implementation="sdpa",
    )
    model.to(device).eval()

    # Use the model's resolved commit for the tokenizer too, even if revision was main.
    revision = getattr(model.config, "_commit_hash", None) or config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=revision)
    return model, tokenizer


def quantize_model_weights(model, bits=8, per_channel=False):
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
                setattr(parent, name, NaiveQuantizedLinear(child, bits=bits, per_channel=per_channel))
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
    for tensor in chain(model.parameters(), model.buffers()):
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total
