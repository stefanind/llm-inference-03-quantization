"""Fixed evaluation inputs and a separate training-text sample for calibration."""

import hashlib
import json

import torch
from datasets import load_dataset


def load_corpus(config, tokenizer):
    dataset = load_dataset(
        config["id"],
        config["subset"],
        split=config["split"],
        revision=config["revision"],
    )
    # Keep a documented corpus order and boundary policy. No labels are needed:
    # for perplexity, each token is the target for the preceding position.
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    return token_ids, dataset._fingerprint


def load_tokens(config, tokenizer, max_tokens):
    token_ids, fingerprint = load_corpus(config, tokenizer)
    if len(token_ids) < max_tokens:
        raise ValueError(f"Requested {max_tokens} tokens, but the corpus has {len(token_ids)}.")
    tokens = torch.tensor(token_ids[:max_tokens], dtype=torch.long)
    digest = hashlib.sha256(json.dumps(tokens.tolist()).encode("utf-8")).hexdigest()
    metadata = {
        **config,
        "fingerprint": fingerprint,
        "selection": "corpus prefix, rows joined with two newlines, no special tokens",
        "num_tokens": len(tokens),
        "token_sha256": digest,
    }
    return tokens, metadata


def load_calibration_tokens(dataset_config, tokenizer, calibration, seed):
    """Sample distinct fixed-length chunks from train, independently of evaluation."""
    config = {**dataset_config, "split": calibration["split"]}
    assert config["split"] == "train", "Calibration must use the training split."
    token_ids, fingerprint = load_corpus(config, tokenizer)
    length = calibration["sequence_length"]
    count = calibration["num_sequences"]
    num_chunks = len(token_ids) // length
    if num_chunks < count:
        raise ValueError(f"Calibration needs {count} distinct chunks, but only {num_chunks} fit.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_chunks, generator=generator)[:count]
    # Drop the partial tail. Chunk indices sample without replacement, so windows
    # never overlap. A local generator leaves the model's RNG state untouched.
    corpus = torch.tensor(token_ids[:num_chunks * length], dtype=torch.long)
    tokens = corpus.reshape(num_chunks, length)[indices]
    metadata = {
        **config,
        "fingerprint": fingerprint,
        "selection": "seeded sample of non-overlapping chunks, rows joined with two newlines, no special tokens",
        "seed": seed,
        "num_sequences": count,
        "sequence_length": length,
        "num_tokens": tokens.numel(),
        "token_offsets": (indices * length).tolist(),
        "token_sha256": hashlib.sha256(json.dumps(tokens.tolist()).encode("utf-8")).hexdigest(),
    }
    return tokens, metadata


def make_prompts(tokens, length, count):
    """Choose disjoint, equally spaced windows; every prompt has an exact length.

    Exact lengths avoid padding in the first benchmark. Prompt IDs identify the
    token offsets so different model variants can use precisely the same inputs.
    """
    assert tokens.ndim == 1 and tokens.dtype == torch.long
    assert length > 0 and count > 0
    assert len(tokens) >= length * count, "Not enough tokens for distinct prompt windows."
    starts = torch.linspace(0, len(tokens) - length, count).long().tolist()
    prompts = torch.stack([tokens[start : start + length] for start in starts])
    ids = [f"tokens:{start}:{start + length}" for start in starts]
    assert prompts.shape == (count, length)
    return prompts, ids
