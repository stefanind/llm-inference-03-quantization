"""One text corpus for both quality evaluation and reproducible timing inputs."""

import hashlib
import json

import torch
from datasets import load_dataset


def load_tokens(config, tokenizer, max_tokens):
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
    if len(token_ids) < max_tokens:
        raise ValueError(f"Requested {max_tokens} tokens, but the corpus has {len(token_ids)}.")
    tokens = torch.tensor(token_ids[:max_tokens], dtype=torch.long)
    digest = hashlib.sha256(json.dumps(tokens.tolist()).encode("utf-8")).hexdigest()
    metadata = {
        **config,
        "fingerprint": dataset._fingerprint,
        "selection": "corpus prefix, rows joined with two newlines, no special tokens",
        "num_tokens": len(tokens),
        "token_sha256": digest,
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
