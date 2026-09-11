import unittest

import torch
from torch.nn import functional as F

from experiments.auf_vllm.losses import token_loss


class AUFLossTests(unittest.TestCase):
    def test_first_failure_is_supervised_and_later_logits_have_zero_gradient(self):
        logits = torch.tensor([[[3., 0.], [2., 0.], [0., 4.], [5., 0.], [0., 2.]]], requires_grad=True)
        labels = torch.zeros((1, 5), dtype=torch.long)
        valid = torch.ones_like(labels, dtype=torch.bool)
        result = token_loss(logits, labels, valid, "auf")
        self.assertEqual(result.active.tolist(), [[True, True, True, False, False]])
        self.assertFalse(result.active.requires_grad)
        reference = F.cross_entropy(logits[0, :3], labels[0, :3])
        torch.testing.assert_close(result.loss, reference)
        result.loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, 3:]).item(), 0)
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)

    def test_invalid_anchor_padding_and_independent_blocks(self):
        logits = torch.tensor([[[0., 3.], [0., 2.], [4., 0.], [3., 0.]],
                               [[0., 3.], [4., 0.], [3., 0.], [2., 0.]]])
        labels = torch.tensor([[-100, 0, 0, -100], [-100, 0, 0, 0]])
        valid = labels.ne(-100)
        result = token_loss(logits, labels, valid, "auf")
        self.assertEqual(result.active.tolist(), [[False, True, False, False], [False, True, True, True]])
        expected = F.cross_entropy(torch.stack([logits[0, 1], *logits[1, 1:]]), torch.zeros(4, dtype=torch.long))
        torch.testing.assert_close(result.loss, expected)

    def test_all_correct_matches_uniform_ce(self):
        x = torch.tensor([[[3., 0.], [2., 0.]]])
        y = torch.zeros((1, 2), dtype=torch.long)
        m = torch.ones_like(y, dtype=torch.bool)
        torch.testing.assert_close(token_loss(x, y, m, "auf").loss, token_loss(x, y, m, "ce").loss)

    def test_mask_changes_with_current_predictions(self):
        y = torch.zeros((1, 3), dtype=torch.long)
        m = torch.ones_like(y, dtype=torch.bool)
        a = torch.tensor([[[0., 3.], [3., 0.], [3., 0.]]])
        b = a.clone(); b[0, 0] = torch.tensor([3., 0.])
        self.assertEqual(token_loss(a, y, m, "auf").active.sum().item(), 1)
        self.assertEqual(token_loss(b, y, m, "auf").active.sum().item(), 3)

    def test_per_microbatch_mean_is_not_global_active_token_mean(self):
        x = torch.tensor([[[0., 4.], [2., 0.], [2., 0.]], [[2., 0.], [2., 0.], [2., 0.]]], requires_grad=True)
        y = torch.zeros((2, 3), dtype=torch.long)
        m = torch.ones_like(y, dtype=torch.bool)
        separate = sum(token_loss(x[i:i+1], y[i:i+1], m[i:i+1], "auf").loss for i in range(2))/2
        combined = token_loss(x, y, m, "auf").loss
        self.assertGreater(abs((separate-combined).item()), .1)
        expected = (F.cross_entropy(x[0, :1], y[0, :1])+F.cross_entropy(x[1], y[1]))/2
        torch.testing.assert_close(separate, expected)

    def test_rejects_empty_invalid_and_unsupported_inputs(self):
        x = torch.zeros((1, 3, 2)); y = torch.zeros((1, 3), dtype=torch.long)
        with self.assertRaises(ValueError): token_loss(x, y, torch.zeros_like(y, dtype=torch.bool), "auf")
        with self.assertRaises(ValueError): token_loss(x, y, torch.ones_like(y), "auf")
        with self.assertRaises(ValueError): token_loss(x, y+2, torch.ones_like(y, dtype=torch.bool), "auf")
        with self.assertRaises(ValueError): token_loss(x, y, torch.ones_like(y, dtype=torch.bool), "mixed")

    def test_gradient_matches_independent_hand_masked_reference(self):
        torch.manual_seed(8)
        x = torch.randn(3, 6, 5, requires_grad=True)
        y = torch.randint(5, (3, 6))
        valid = torch.ones_like(y, dtype=torch.bool); valid[:, 0] = False
        expected_mask = torch.zeros_like(valid)
        for b in range(3):
            for j in range(6):
                if not valid[b, j]: continue
                expected_mask[b, j] = True
                if x[b, j].detach().argmax() != y[b, j]: break
        result = token_loss(x, y, valid, "auf")
        reference = F.cross_entropy(x[expected_mask].float(), y[expected_mask])
        torch.testing.assert_close(result.active, expected_mask)
        torch.testing.assert_close(torch.autograd.grad(result.loss, x)[0], torch.autograd.grad(reference, x)[0])


if __name__ == "__main__":
    unittest.main()
