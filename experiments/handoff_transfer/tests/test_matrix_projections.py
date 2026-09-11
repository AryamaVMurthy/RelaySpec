import torch
from experiments.handoff_transfer.matrix.projections import FiveLowRankMaps,DenseFusion

def test_five_ba_initialization_gradients_and_folding():
    torch.manual_seed(2)
    fusion=torch.randn(4,20);weights=[torch.randn(4,6) for _ in range(5)]
    m=FiveLowRankMaps(fusion,weights,rank=3);x=torch.randn(2,7,30)
    native=torch.cat([fusion[:,i*4:(i+1)*4]@weights[i] for i in range(5)],dim=1)
    torch.testing.assert_close(m.folded(),native)
    m(x).square().mean().backward()
    assert all(b.grad.abs().sum()>0 for b in m.B)
    assert not m.base.requires_grad and not m.fusion.requires_grad
    with torch.no_grad():
        for b in m.B:b.add_(torch.randn_like(b)*.1)
    torch.testing.assert_close(m(x),torch.nn.functional.linear(x,m.folded()),rtol=1e-5,atol=1e-5)
    assert sum(p.numel() for p in m.parameters())==5*3*(4+6)

def test_normalized_interface_preserves_normalization():
    m=DenseFusion(torch.randn(4,30),normalize_input=True)
    x=torch.randn(3,30)
    torch.testing.assert_close(m(x),m(x*2),rtol=1e-4,atol=1e-4)
