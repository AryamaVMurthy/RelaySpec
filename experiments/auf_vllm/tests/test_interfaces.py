import unittest

import torch
from torch.nn import functional as F

from experiments.auf_vllm.interfaces import FusionLoRA, LayerContextMapper, LowRankContext


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_folded_interface_and_zip_feature_objective(self):
        fusion = torch.randn(3, 6); norm = torch.randn(3)
        mapper = LayerContextMapper(fusion, norm, target_width=4, source_width=3, num_taps=2)
        x = torch.randn(2, 5, 8); y = torch.randn(2, 5, 6)
        context, z = mapper(x)
        folded = mapper.normalize(F.linear(x, mapper.folded()))
        torch.testing.assert_close(context, folded, rtol=1e-5, atol=1e-6)
        def rel(a, b): return (a.float()-b.float()).square().sum(-1)/(b.float().square().sum(-1)+1e-6)
        expected = rel(context, mapper.normalize(F.linear(y, fusion))) + rel(z.reshape(2, 5, 2, 3), y.reshape(2, 5, 2, 3)).mean(-1)
        torch.testing.assert_close(mapper.feature_loss(x, y), expected)
        mapper.feature_loss(x, y).mean().backward()
        self.assertEqual(set(dict(mapper.named_parameters())), {"maps.0.weight", "maps.1.weight"})
        self.assertTrue(all(p.grad is not None for p in mapper.parameters()))
        torch.testing.assert_close(mapper.fusion, fusion)
        torch.testing.assert_close(mapper.norm, norm)

    def test_lora_initial_function_and_freezing(self):
        weight = torch.randn(3, 8)
        layer = FusionLoRA(weight, rank=2, alpha=2)
        x = torch.randn(5, 8)
        torch.testing.assert_close(layer(x), F.linear(x, weight))
        self.assertEqual(set(dict(layer.named_parameters())), {"A", "B"})
        layer(x).square().mean().backward()
        self.assertGreater(layer.B.grad.abs().sum().item(), 0)
        # Zero B intentionally makes A's first-step gradient zero.
        torch.testing.assert_close(layer.A.grad, torch.zeros_like(layer.A))
        self.assertIsNone(layer.base.grad)
        with torch.no_grad(): layer.B.add_(.01)
        torch.testing.assert_close(layer(x), F.linear(x, layer.folded()), rtol=1e-5, atol=1e-6)

    def test_context_lora_keeps_norm_and_fusion_frozen(self):
        weight,norm=torch.randn(3,8),torch.randn(3)
        layer=LowRankContext(weight,norm,rank=2,alpha=2)
        x=torch.randn(5,8)
        raw=F.linear(x,weight)
        expected=norm*raw*torch.rsqrt(raw.square().mean(-1,keepdim=True)+1e-6)
        torch.testing.assert_close(layer(x)[0],expected)
        layer(x)[0].square().mean().backward()
        self.assertEqual(set(dict(layer.named_parameters())),{"A","B"})
        self.assertIsNone(layer.fusion.grad)
        self.assertIsNone(layer.norm.grad)
        torch.testing.assert_close(layer.A.grad,torch.zeros_like(layer.A))
        with torch.no_grad():
            layer.B.add_(.01)
        torch.testing.assert_close(layer(x)[0],layer.normalize(F.linear(x,layer.folded())),rtol=1e-5,atol=1e-6)


if __name__ == "__main__":
    unittest.main()
