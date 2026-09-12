import torch
from experiments.handoff_transfer.feature_objectives import feature_distribution_loss


def test_forward_kl_equals_ce_minus_teacher_entropy_and_same_gradient():
    torch.manual_seed(4)
    x = torch.randn(7, 11, requires_grad=True)
    y = torch.randn(7, 11, requires_grad=True)
    ce = feature_distribution_loss(x,y,'feature_ce')
    kl = feature_distribution_loss(x,y,'forward_kl')
    lp = y.detach().log_softmax(-1)
    torch.testing.assert_close(ce-kl, -(lp.exp()*lp).sum(-1))
    g1 = torch.autograd.grad(ce.sum(),x,retain_graph=True)[0]
    g2 = torch.autograd.grad(kl.sum(),x)[0]
    torch.testing.assert_close(g1,g2)
    assert y.grad is None


def test_kl_directions_and_identity():
    x = torch.tensor([[3.,-2.,1.]],requires_grad=True)
    y = torch.tensor([[-1.,2.,0.]])
    forward = feature_distribution_loss(x,y,'forward_kl')
    reverse = feature_distribution_loss(x,y,'reverse_kl')
    assert forward.item() > 0 and reverse.item() > 0
    assert not torch.allclose(forward,reverse)
    torch.testing.assert_close(reverse,feature_distribution_loss(y,x,'forward_kl'))
    for objective in ['forward_kl','reverse_kl']:
        torch.testing.assert_close(feature_distribution_loss(x,x,objective),torch.zeros(1))
        assert torch.isfinite(torch.autograd.grad(feature_distribution_loss(x,y,objective).sum(),x)[0]).all()
