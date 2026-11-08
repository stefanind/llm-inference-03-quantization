"""Perplexity with overlapping context windows and no double-counted targets."""

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def evaluate_perplexity(model, tokens, context_length, stride):
    assert tokens.ndim == 1 and tokens.dtype == torch.long
    assert len(tokens) >= 2
    assert 0 < stride < context_length, "Overlap preserves context for every new target."
    assert not model.training, "Evaluation requires model.eval()."
    device = next(model.parameters()).device
    total_nll = 0.0
    scored_tokens = 0
    previous_end = 0

    for start in range(0, len(tokens), stride):
        end = min(start + context_length, len(tokens))
        inputs = tokens[start:end].unsqueeze(0).to(device)
        outputs = model(input_ids=inputs, use_cache=False)

        # A logit at position t predicts token t+1. Keep overlapping tokens as
        # context, but only score targets that earlier windows did not score.
        first_target = max(previous_end, start + 1)
        local_target = first_target - start
        logits = outputs.logits[:, local_target - 1 : -1].float()
        targets = inputs[:, local_target:]
        assert logits.shape[:2] == targets.shape
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
        )
        assert torch.isfinite(nll), "Non-finite loss: inspect the model's outputs."
        total_nll += nll.item()
        scored_tokens += targets.numel()
        previous_end = end
        # Release this window's logits before allocating those for the next one.
        del outputs, logits, nll
        if end == len(tokens):
            break

    # The first corpus token has no preceding context; every other token is scored.
    assert scored_tokens == len(tokens) - 1
    mean_nll = total_nll / scored_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "scored_tokens": scored_tokens,
        "context_length": context_length,
        "stride": stride,
    }
