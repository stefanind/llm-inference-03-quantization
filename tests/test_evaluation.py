from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from quantization_lab.evaluate import evaluate_perplexity


class BigramModel(nn.Module):
    """Known next-token logits, determined only by the current input token."""

    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([
            [-2.0, 0.0, 1.0, 3.0],
            [2.0, -1.0, 0.0, 1.0],
            [1.0, 2.0, -2.0, 0.0],
            [0.0, 1.0, 3.0, -1.0],
        ]), requires_grad=False)

    def forward(self, input_ids, use_cache):
        return SimpleNamespace(logits=self.scores[input_ids])


class EvaluationTests(unittest.TestCase):
    def test_windowed_loss_matches_direct_loss_including_a_partial_tail(self):
        model = BigramModel().eval()
        tokens = torch.tensor([0, 1, 3, 2, 3, 3, 1, 0, 2, 3, 2, 0, 1, 1, 3, 0, 2, 1, 3, 3])
        # Input-dependent logits catch an incorrect prediction/target shift too.
        expected = F.cross_entropy(model.scores[tokens[:-1]], tokens[1:]).item()
        for context, stride in ((7, 3), (4, 1), (32, 16)):
            report = evaluate_perplexity(model, tokens, context, stride)
            self.assertEqual(report["scored_tokens"], len(tokens) - 1)
            self.assertAlmostEqual(report["mean_nll"], expected, places=6)
            self.assertAlmostEqual(report["perplexity"], torch.exp(torch.tensor(expected)).item(), places=5)

if __name__ == "__main__":
    unittest.main()
